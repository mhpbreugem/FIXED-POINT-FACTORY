"""Direct quadrature test: on a FIXED nailed ladder surface, compare the
evidence integral A_v(p) (and the resulting one-step Phi output) computed by:
  (K)   kernel with small h  (the co-area target)
  (b)   naive unweighted scan
  (c1)  scan weighted by 1/|grad P|        (spec's prescription)
  (c2)  scan weighted by 1/|along-axis dP| (textbook co-area / layer-cake)

This isolates the QUADRATURE question from the fixed-point iteration: which
weighting makes the strict (h=0) scan reproduce the kernel/co-area integral?
"""
import sys, json
sys.path.insert(0, '/tmp/rezn-source'); sys.path.insert(0, '/tmp')
import numpy as np
from numba import njit
from code.signals import f_signal
from code.contour_K3_halo import _agent_evidence_K3_smooth, init_no_learning_K3

UMAX = 4.0; pad = 2; TAU = 2.0; GAMMA = 0.1
tv = np.full(3, TAU); gv = np.full(3, GAMMA); wv = np.full(3, 1.0)


@njit(cache=True)
def evid_naive(P_slice, p, u_full, t0, t1):
    G = u_full.size; a0 = 0.0; a1 = 0.0
    for axis in range(2):
        for a_idx in range(G):
            ta = t1 if axis == 0 else t0
            to = t0 if axis == 0 else t1
            ua = u_full[a_idx]
            f0a = f_signal(ua, 0, ta); f1a = f_signal(ua, 1, ta)
            prev = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
            for i in range(G - 1):
                nx = P_slice[i+1, a_idx] if axis == 0 else P_slice[a_idx, i+1]
                dp = prev - p; dn = nx - p
                if dp == 0.0 and dn == 0.0:
                    prev = nx; continue
                if dp * dn <= 0.0:
                    den = nx - prev
                    if den == 0.0:
                        prev = nx; continue
                    fr = (p - prev) / den
                    fr = 0.0 if fr < 0 else (1.0 if fr > 1 else fr)
                    uo = (1-fr)*u_full[i] + fr*u_full[i+1]
                    a0 += f0a * f_signal(uo, 0, to)
                    a1 += f1a * f_signal(uo, 1, to)
                prev = nx
    return a0/2.0, a1/2.0


@njit(cache=True)
def evid_weight(P_slice, p, u_full, t0, t1, du, mode):
    """mode=0: 1/|grad P| (full).  mode=1: 1/|along-axis dP|."""
    G = u_full.size; a0 = 0.0; a1 = 0.0
    for axis in range(2):
        for a_idx in range(G):
            ta = t1 if axis == 0 else t0
            to = t0 if axis == 0 else t1
            ua = u_full[a_idx]
            f0a = f_signal(ua, 0, ta); f1a = f_signal(ua, 1, ta)
            prev = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
            for i in range(G - 1):
                nx = P_slice[i+1, a_idx] if axis == 0 else P_slice[a_idx, i+1]
                dp = prev - p; dn = nx - p
                if dp == 0.0 and dn == 0.0:
                    prev = nx; continue
                if dp * dn <= 0.0:
                    den = nx - prev
                    if den == 0.0:
                        prev = nx; continue
                    fr = (p - prev) / den
                    fr = 0.0 if fr < 0 else (1.0 if fr > 1 else fr)
                    uo = (1-fr)*u_full[i] + fr*u_full[i+1]
                    g_off = den / du
                    if mode == 1:
                        gmag = abs(g_off)
                    else:
                        # transverse slope from neighbour lines
                        if axis == 0:
                            ip = a_idx+1; im = a_idx-1
                            if 0 <= ip < G and 0 <= im < G:
                                vp = (1-fr)*P_slice[i, ip]+fr*P_slice[i+1, ip]
                                vm = (1-fr)*P_slice[i, im]+fr*P_slice[i+1, im]
                                g_tr = (vp-vm)/(2*du)
                            else:
                                g_tr = 0.0
                        else:
                            ip = a_idx+1; im = a_idx-1
                            if 0 <= ip < G and 0 <= im < G:
                                vp = (1-fr)*P_slice[ip, i]+fr*P_slice[ip, i+1]
                                vm = (1-fr)*P_slice[im, i]+fr*P_slice[im, i+1]
                                g_tr = (vp-vm)/(2*du)
                            else:
                                g_tr = 0.0
                        gmag = np.sqrt(g_off*g_off + g_tr*g_tr)
                    if gmag < 1e-12:
                        gmag = 1e-12
                    w = 1.0/gmag
                    a0 += w * f0a * f_signal(uo, 0, to)
                    a1 += w * f1a * f_signal(uo, 1, to)
                prev = nx
    return a0/2.0, a1/2.0


def main():
    COA = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
           'solved_fixed_points/k3_coarea_limit')
    out = {}
    for Gi in (13, 17, 21):
        du = 2*UMAX/(Gi-1); Gf = Gi+2*pad
        uf = np.array([-UMAX+(q-pad)*du for q in range(Gf)])
        lo, hi = pad, pad+Gi
        P_inner = np.load(f'{COA}/P_inner_G{Gi}.npy')
        halo = init_no_learning_K3(uf, tv, gv, wv)
        Pf = halo.copy(); Pf[lo:hi, lo:hi, lo:hi] = P_inner
        # kernel reference at the SMALL h used in the ladder
        h = 0.45*du**0.5
        acc = np.empty(2)
        # sample agent-0 evidence ratio A1/A0 across interior cells; compare
        # the kernel target to each scan method (relative L2 over cells)
        rng = range(lo, hi)
        errs = {'naive': [], 'w_grad': [], 'w_along': []}
        ref = []
        for i in rng:
            for j in rng:
                for l in rng:
                    p = Pf[i, j, l]
                    _agent_evidence_K3_smooth(Pf[i, :, :], p, uf,
                                              tv[1], tv[2], h, acc)
                    rk = acc[1]/(acc[0]+acc[1]+1e-30)
                    ref.append(rk)
                    a0, a1 = evid_naive(Pf[i, :, :], p, uf, tv[1], tv[2])
                    errs['naive'].append(a1/(a0+a1+1e-30))
                    a0, a1 = evid_weight(Pf[i, :, :], p, uf, tv[1], tv[2],
                                         du, 0)
                    errs['w_grad'].append(a1/(a0+a1+1e-30))
                    a0, a1 = evid_weight(Pf[i, :, :], p, uf, tv[1], tv[2],
                                         du, 1)
                    errs['w_along'].append(a1/(a0+a1+1e-30))
        ref = np.array(ref)
        row = {'G': Gi, 'h': float(h)}
        for k, v in errs.items():
            v = np.array(v)
            rel = float(np.sqrt(np.mean((v-ref)**2)) /
                        (np.sqrt(np.mean(ref**2))+1e-30))
            row[k+'_relL2_vs_kernel'] = rel
        out[str(Gi)] = row
        print(f"G={Gi} h={h:.3f}  relL2(posterior ratio) vs kernel: "
              f"naive={row['naive_relL2_vs_kernel']:.4f} "
              f"w_grad={row['w_grad_relL2_vs_kernel']:.4f} "
              f"w_along={row['w_along_relL2_vs_kernel']:.4f}", flush=True)
    json.dump(out, open('/tmp/quadrature_test.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
