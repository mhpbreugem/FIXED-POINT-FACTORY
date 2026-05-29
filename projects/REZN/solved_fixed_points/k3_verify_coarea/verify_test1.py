"""TEST 1: does the h->0 kernel/co-area LADDER limit equal the STRICT (h=0)
no-kernel co-area-WEIGHTED CONTOUR method, and does the naive unweighted scan
differ?

Three methods on a common grid family G_inner in {9,13,17,21}:
 (a) h->0 kernel ladder (co-area): nail phi_K3_halo_smooth with h=0.45*du^0.5
     via newton_krylov.
 (b) naive strict contour scan: phi_K3_halo to its plateau (damped Picard).
 (c) co-area-weighted strict contour scan: phi_K3_halo_weighted, nailed via
     newton_krylov (with damped-Picard warm start).

Compare deficit = 1 - R^2 of logit(P) on T=tau*sum(u); d_FR; max|P diff|.
"""
import os, sys, json, time, warnings
sys.path.insert(0, '/tmp/rezn-source')
sys.path.insert(0, '/tmp')
import numpy as np
from code.contour_K3_halo import (init_no_learning_K3, phi_K3_halo,
                                  phi_K3_halo_smooth)
from verify_weighted_contour import phi_K3_halo_weighted
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_verify_coarea'
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'run.log'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

K = 3; pad = 2; UMAX = 4.0
TAU = 2.0; GAMMA = 0.1
C = 0.45
G_LIST = [9, 13, 17, 21]
tv = np.full(K, TAU); gv = np.full(K, GAMMA); wv = np.full(K, 1.0)


def build_grid(Gi):
    du = 2 * UMAX / (Gi - 1)
    Gf = Gi + 2 * pad
    uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def metrics(Pin, T, P_FR, Xreg):
    Pc = np.clip(Pin, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    defi = np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    coef, _, _, _ = np.linalg.lstsq(Xreg, y, rcond=None)
    return dict(deficit=float(defi),
                d_FR=float(np.sqrt(np.mean((Pin - P_FR) ** 2))),
                b1=float(coef[0]), slope_T=float(a[0]))


def nail(resid, x0, tol=1e-10, maxiter=300):
    cnt = {'n': 0}
    def cb(x, fx): cnt['n'] += 1
    conv = True
    try:
        sol = newton_krylov(resid, x0, f_tol=tol, maxiter=maxiter,
                            callback=cb, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    return sol, conv, cnt['n']


def run():
    t0 = time.time()
    log(f"\n=== TEST1 three-way verification  tau={TAU} gamma={GAMMA} ===")
    log(f"date 2026-05-29  G_LIST={G_LIST}  h=C*du^0.5 C={C}")
    rows = []
    for Gi in G_LIST:
        du, uf, lo, hi = build_grid(Gi)
        h = C * du ** 0.5
        slc = (slice(lo, hi),) * K
        ui = uf[lo:hi]
        U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
        T = TAU * (U1 + U2 + U3)
        P_FR = 1 / (1 + np.exp(-T))
        Xreg = np.column_stack([U1.ravel(), U2.ravel(), U3.ravel(),
                                np.ones(U1.size)])
        halo = init_no_learning_K3(uf, tv, gv, wv)
        t1 = time.time()

        # ---- (a) kernel ladder (co-area) ----
        def resid_a(xflat):
            Pf = halo.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
            return (phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
                    - Pf)[slc].ravel()
        # reuse nailed surface from k3_coarea_limit if present
        warm = os.path.join('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                            'solved_fixed_points/k3_coarea_limit',
                            f'P_inner_G{Gi}.npy')
        x0a = (np.load(warm).ravel() if os.path.exists(warm)
               else halo[slc].ravel().copy())
        sol_a, conv_a, it_a = nail(resid_a, x0a)
        F_a = float(np.max(np.abs(resid_a(sol_a))))
        Pa = sol_a.reshape((Gi,) * K)
        m_a = metrics(Pa, T, P_FR, Xreg)

        # ---- (b) naive strict contour: damped Picard plateau ----
        Ps = halo.copy()
        for _ in range(120):
            Pn = phi_K3_halo(Ps, uf, lo, hi, tv, gv, wv)
            Ps = 0.75 * Ps + 0.25 * Pn
        F_b = float(np.max(np.abs(
            (phi_K3_halo(Ps, uf, lo, hi, tv, gv, wv) - Ps)[slc])))
        Pb = Ps[slc].copy()
        m_b = metrics(Pb, T, P_FR, Xreg)

        # ---- (c) co-area-weighted strict contour ----
        # The weighted operator is the consistent co-area quadrature. Its
        # transverse-gradient estimate is a finite-difference (mildly
        # non-smooth), so we drive it to its damped-Picard plateau (robust)
        # and ALSO attempt a capped Newton nail from there.
        def resid_c(xflat):
            Pf = halo.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
            return (phi_K3_halo_weighted(Pf, uf, lo, hi, tv, gv, wv, du)
                    - Pf)[slc].ravel()
        Pw = halo.copy()
        for _ in range(400):
            Pn = phi_K3_halo_weighted(Pw, uf, lo, hi, tv, gv, wv, du)
            Pw = 0.6 * Pw + 0.4 * Pn
        F_c_pic = float(np.max(np.abs(resid_c(Pw[slc].ravel()))))
        # capped Newton nail attempt
        sol_c, conv_c, it_c = nail(resid_c, Pw[slc].ravel().copy(),
                                   tol=1e-9, maxiter=60)
        F_c = float(np.max(np.abs(resid_c(sol_c))))
        # keep whichever has smaller residual as the representative surface
        if F_c < F_c_pic:
            Pc = sol_c.reshape((Gi,) * K)
        else:
            Pc = Pw[slc].copy(); F_c = F_c_pic
        m_c = metrics(Pc, T, P_FR, Xreg)
        m_c_pic = metrics(Pw[slc], T, P_FR, Xreg)

        # ---- cross-method surface diffs ----
        diff_ac = float(np.max(np.abs(Pa - Pc)))
        diff_ab = float(np.max(np.abs(Pa - Pb)))
        diff_bc = float(np.max(np.abs(Pb - Pc)))

        row = dict(
            G_inner=Gi, du=float(du), h=float(h),
            a_deficit=m_a['deficit'], a_d_FR=m_a['d_FR'], a_b1=m_a['b1'],
            a_slopeT=m_a['slope_T'], a_Finf=F_a, a_conv=conv_a, a_it=it_a,
            b_deficit=m_b['deficit'], b_d_FR=m_b['d_FR'], b_b1=m_b['b1'],
            b_slopeT=m_b['slope_T'], b_Finf=F_b,
            c_deficit=m_c['deficit'], c_d_FR=m_c['d_FR'], c_b1=m_c['b1'],
            c_slopeT=m_c['slope_T'], c_Finf=F_c, c_conv=conv_c, c_it=it_c,
            c_picard_deficit=m_c_pic['deficit'],
            diff_ac=diff_ac, diff_ab=diff_ab, diff_bc=diff_bc,
            bias_b=abs(m_b['deficit'] - m_a['deficit']),
            gap_c=abs(m_c['deficit'] - m_a['deficit']),
            walltime_s=round(time.time() - t1, 1))
        rows.append(row)
        np.save(os.path.join(OUT, f'Pa_G{Gi}.npy'), Pa)
        np.save(os.path.join(OUT, f'Pc_G{Gi}.npy'), Pc)
        log(f"G={Gi:3d} du={du:.3f} h={h:.3f} | "
            f"(a)def={m_a['deficit']:.5f} F={F_a:.1e} conv={conv_a} | "
            f"(b)def={m_b['deficit']:.5f} F={F_b:.4f} | "
            f"(c)def={m_c['deficit']:.5f} F={F_c:.1e} conv={conv_c} || "
            f"|a-c|={diff_ac:.2e} |a-b|={diff_ab:.2e} "
            f"bias_b={row['bias_b']:.4f} gap_c={row['gap_c']:.4f} "
            f"({time.time()-t0:.0f}s)")
        json.dump(dict(tau=TAU, gamma=GAMMA, C=C, G_LIST=G_LIST,
                       rows=rows), open(os.path.join(OUT, 'test1.json'), 'w'),
                  indent=2)
    return rows


if __name__ == '__main__':
    rows = run()
    log("TEST1 done.")
