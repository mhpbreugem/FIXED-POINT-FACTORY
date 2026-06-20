"""Phase 4: confirm the fold at G=21 (9261 unknowns), tau=2.0.

March gamma up at G=21 from a safely-converged warm start, adaptive step
halving to locate gamma*_G21; compute ARPACK leading eigenvalues of J
(FD matvec LinearOperator) at selected converged points.
"""
import json, time, numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from dd_k3_stall_diag_lib import *

TAU = 2.0
F_OK = 1e-10


def lead_eigs_arpack(prob, x, k=4, eps=1e-6):
    phi0 = prob.phi(x)
    nev = [0]
    def mv(v):
        nv = np.linalg.norm(v)
        if nv == 0:
            return np.zeros_like(v)
        nev[0] += 1
        return (prob.phi(x + eps*(v/nv)) - phi0) * (nv/eps)
    Jop = LinearOperator((prob.n, prob.n), matvec=mv, dtype=float)
    w, V = eigs(Jop, k=k, which='LM', maxiter=2000, tol=1e-8)
    order = np.argsort(-np.abs(w))
    w = w[order]; V = V[:, order]
    out = []
    for i in range(k):
        vv = np.real(V[:, i]) if abs(np.imag(w[i])) < 1e-10 else np.abs(V[:, i])
        vb, vp = symmetry_split_vec(vv, prob.Gi)
        out.append(dict(re=float(np.real(w[i])), im=float(np.imag(w[i])),
                        mag=float(np.abs(w[i])), sym_break=vb, sym_pres=vp))
    return out, nev[0]


def main():
    res = dict(tau=TAU, G=21, points=[])
    # start: certified t2.0 g0.3769 at G=21 directly
    x = np.load(f"{EMIN15}/P_ld_t{TAU}_g0.3769.npy").ravel()
    g = 0.3769
    prob = Problem(TAU, g, 21)
    x, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
    print(f"G=21 start g={g}: F={Fn:.2e}", flush=True)

    # spectra checkpoints requested at these gammas (taken when passed)
    ph3 = json.load(open(f"{OUT}/phase3_foldcurve.json"))
    gstar13 = [p['gamma_star'] for p in ph3['fold_points'] if p['tau'] == 2.0][0]
    marks = [0.5278, round(0.9*gstar13, 4)]
    step = 0.05
    while step >= 0.002:
        gt = round(g + step, 10)
        prob = Problem(TAU, gt, 21)
        t0 = time.time()
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        ok = Fn < F_OK
        print(f"G=21 gamma={gt:.5f} F={Fn:.2e} {'ok' if ok else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if ok:
            g, x = gt, xs
            res['points'].append(dict(gamma=g, F=Fn, ok=True))
            while marks and g >= marks[0] - 1e-9:
                m = marks.pop(0)
                t0 = time.time()
                le, nev = lead_eigs_arpack(prob, x)
                print(f"  [spectrum at g={g:.5f}] rho={le[0]['mag']:.6f} "
                      f"symB={le[0]['sym_break']:.3f} ({nev} matvecs, "
                      f"{time.time()-t0:.0f}s)", flush=True)
                res['points'][-1]['lead_eigs'] = le
        else:
            res['points'].append(dict(gamma=gt, F=Fn, ok=False))
            step *= 0.5
        checkpoint(res, 'phase4_G21.json')
    res['gamma_star_G21'] = g
    np.save(f"{OUT}/P21_fold_t2.0.npy", x.reshape(21, 21, 21))
    # final spectrum at the G=21 fold approach
    prob = Problem(TAU, g, 21)
    t0 = time.time()
    le, nev = lead_eigs_arpack(prob, x)
    res['final_lead_eigs'] = le
    print(f"G=21 gamma* = {g:.5f}; final rho={le[0]['mag']:.8f} "
          f"symB={le[0]['sym_break']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    checkpoint(res, 'phase4_G21.json')
    print("phase 4 done")


if __name__ == '__main__':
    main()
