"""CRRA UMAX-collapse diagnostic at G=17 (the paper's headline resolution).

Same protocol as crra_umax_diagnostic.py, but at G=17 so the UMAX=4 cell
matches the published sweep2d.json deficit ~0.13 directly. Expected:
deficit stays ~0.13-0.3 at every UMAX -- intrinsic Jensen gap.
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/tmp/rezn-source")
import numpy as np
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

TAU = 2.0; GAMMA = 0.1; G = 17; PAD = 2; C_H = 0.45

def nail(umax):
    du = 2 * umax / (G - 1); h = C_H * np.sqrt(du)
    Gf = G + 2*PAD
    uf = np.array([-umax + (q-PAD)*du for q in range(Gf)])
    lo, hi = PAD, PAD + G; slc = (slice(lo, hi),)*3
    tv = np.full(3, TAU); gv = np.full(3, GAMMA); W = np.full(3, 1.0)
    halo = init_no_learning_K3(uf, tv, gv, W)
    def resid(x):
        P = halo.copy(); P[slc] = x.reshape((G,)*3)
        return (phi_K3_halo_smooth(P, uf, lo, hi, tv, gv, W, h) - P)[slc].ravel()
    x0 = halo[slc].ravel().copy()
    conv = True
    try:
        sol = newton_krylov(resid, x0, f_tol=1e-9, maxiter=200, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(resid(sol))))
    Pin = sol.reshape((G,)*3)
    ui = uf[lo:hi]; U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU*(U1+U2+U3)
    Pc = np.clip(Pin, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel()+aa[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2), 1e-30))
    d_FR = float(np.sqrt(np.mean((Pin - 1/(1+np.exp(-T)))**2)))
    return defi, d_FR, float(aa[0]), Finf, conv, du, h

print(f"CRRA UMAX-collapse at gamma={GAMMA}, tau={TAU}, G={G} (paper's headline G).", flush=True)
rows = []
for umax in [4, 6, 8, 12]:
    t = time.time()
    defi, d_FR, slope, F, conv, du, h = nail(umax)
    dt = time.time() - t
    rec = dict(umax=umax, du=float(du), h=float(h), deficit=defi, d_FR=d_FR, slope=slope,
               Finf=F, converged=conv, sec=round(dt, 1))
    rows.append(rec)
    print(f"  UMAX={umax:2d}  du={du:.3f}  h={h:.3f}: deficit={defi:.4f}  d_FR={d_FR:.3f}  slope={slope:.3f}  "
          f"||F||={F:.1e}  {'NAIL' if F<1e-8 else 'soft'}  ({dt:.0f}s)", flush=True)
    json.dump({'gamma': GAMMA, 'tau': TAU, 'G': G, 'rows': rows},
              open(os.path.join(HERE, 'crra_umax_diag_G17.json'), 'w'), indent=2)
print('\nDONE.', flush=True)
