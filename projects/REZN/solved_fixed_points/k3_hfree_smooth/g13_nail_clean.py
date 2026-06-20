"""Clean G=13 nail of hfree_smooth strict-h=0 PR FP.

Previous attempt with 20 Picard preconditioning steps escaped the PR basin
(drift slope 0.36 -> 0.53). Try LESS preconditioning to stay in basin:
  attempt 1: 0 Picard iters (pure interp from G=9)
  attempt 2: 5 Picard iters
  attempt 3: Anderson(m=8) instead of LGMRES NK

Save the converged G=13 FP for downstream σ-δ V11 comparison.
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp")
import numpy as np
import hfree_operator as H
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from scipy.interpolate import RegularGridInterpolator

UMAX = 4.0; TAU = 2.0; GAMMA = 0.1; NQ = 40; SUB = 4
tau = np.full(3, TAU); gam = np.full(3, GAMMA); W = np.full(3, 1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)

def metrics(P, ui):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU*(U1+U2+U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    deficit = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = 1/(1+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))

def F_of(x, G, ui):
    P = x.reshape((G, G, G))
    return (H.phi_hfree(P, ui, gnodes, gweights, tau, gam, W, SUB) - P).ravel()

# Warmup
print('JIT warmup...', flush=True); t = time.time()
ui9 = np.linspace(-UMAX, UMAX, 9)
P9 = np.load(os.path.join(HERE, 'P_nailed_G9.npy'))
m9 = metrics(P9, ui9)
print(f'  G=9 FP: deficit={m9["deficit"]:.4f} slope={m9["slope_T"]:.4f} d_FR={m9["d_FR"]:.4f}', flush=True)
_ = H.phi_hfree(P9, ui9, gnodes, gweights, tau, gam, W, SUB)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

# Interp G=9 → G=13
ui13 = np.linspace(-UMAX, UMAX, 13)
rgi = RegularGridInterpolator((ui9, ui9, ui9), P9, bounds_error=False, fill_value=None)
A, B, C = np.meshgrid(ui13, ui13, ui13, indexing='ij')
P13_raw = rgi(np.column_stack([A.ravel(), B.ravel(), C.ravel()])).reshape((13,)*3)
m13_raw = metrics(P13_raw, ui13)
F0 = float(np.max(np.abs(F_of(P13_raw.ravel(), 13, ui13))))
print(f'\nG=13 RAW interp warm-start: deficit={m13_raw["deficit"]:.4f} slope={m13_raw["slope_T"]:.4f} '
      f'd_FR={m13_raw["d_FR"]:.4f} ||F||={F0:.3e}', flush=True)

# === Attempt 1: NK from raw interp (no Picard) ===
def try_nk(P_init, label, f_tol=1e-7, maxiter=40):
    print(f'\n=== {label} ===', flush=True)
    m0 = metrics(P_init, ui13)
    print(f'  IC: deficit={m0["deficit"]:.4f} slope={m0["slope_T"]:.4f}', flush=True)
    cnt = {'n': 0, 'hist': []}
    def cb(x, fx):
        cnt['n'] += 1
        cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            m = metrics(x.reshape((13,)*3), ui13)
            print(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}', flush=True)
    conv = True; t = time.time()
    try:
        sol = newton_krylov(lambda x: F_of(x, 13, ui13), P_init.ravel(),
                              f_tol=f_tol, maxiter=maxiter, method='lgmres',
                              callback=cb, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, 13, ui13))))
    m = metrics(sol.reshape((13,)*3), ui13)
    print(f'  RESULT: conv={conv} ||F||={Finf:.3e} iters={cnt["n"]} '
          f'deficit={m["deficit"]:.4f} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}'
          f' ({time.time()-t:.0f}s)', flush=True)
    return sol.reshape((13,)*3), dict(conv=conv, Finf=Finf, iters=cnt['n'], hist=cnt['hist'], **m)

# === Attempt 1 ===
P_a1, r_a1 = try_nk(P13_raw, 'A1: NK raw interp (0 Picard)', f_tol=1e-7, maxiter=40)
np.save(os.path.join(HERE, 'P_G13_attempt1.npy'), P_a1)

# === Attempt 2: 5 Picard then NK ===
P13_p5 = P13_raw.copy()
for _ in range(5):
    P13_p5 = 0.7*P13_p5 + 0.3*H.phi_hfree(P13_p5, ui13, gnodes, gweights, tau, gam, W, SUB)
P_a2, r_a2 = try_nk(P13_p5, 'A2: NK after 5 Picard', f_tol=1e-7, maxiter=40)
np.save(os.path.join(HERE, 'P_G13_attempt2.npy'), P_a2)

# === Attempt 3: Anderson(m=8) Picard ===
print(f'\n=== A3: Anderson(m=8) Picard from raw interp ===', flush=True)
def anderson(P_init, m_max=8, max_iter=60, tol=1e-7):
    x = P_init.ravel().copy()
    G_h = []; F_h = []; hist = []
    best = dict(ferr=np.inf, x=x.copy())
    for it in range(1, max_iter+1):
        ts = time.time()
        g_x = (F_of(x, 13, ui13) + x)  # = phi(x)
        f_x = g_x - x
        ferr = float(np.max(np.abs(f_x)))
        hist.append(ferr)
        if ferr < best['ferr']:
            best = dict(ferr=ferr, x=g_x.copy(), it=it)
        m = metrics(x.reshape((13,)*3), ui13)
        if it % 5 == 0 or it == 1:
            print(f'  A it {it:3d} ||F||={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
        if ferr < tol: break
        G_h.append(g_x.copy()); F_h.append(f_x.copy())
        if len(F_h) > m_max + 1:
            G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0:
            x_new = g_x
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f_x, rcond=None)
                x_new = g_x - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g_x
        x = np.clip(x_new, 1e-12, 1-1e-12)
    return best['x'].reshape((13,)*3), hist, best
P_a3, hist_a3, best_a3 = anderson(P13_raw)
m_a3 = metrics(P_a3, ui13)
print(f'  RESULT A3: best ||F||={best_a3["ferr"]:.3e} at it {best_a3["it"]} '
      f'deficit={m_a3["deficit"]:.4f} slope={m_a3["slope_T"]:.4f} d_FR={m_a3["d_FR"]:.4f}', flush=True)
np.save(os.path.join(HERE, 'P_G13_attempt3.npy'), P_a3)
r_a3 = dict(best_ferr=best_a3['ferr'], best_it=best_a3['it'], hist=hist_a3, **m_a3)

# Pick best
best_attempt = min(['attempt1', 'attempt2', 'attempt3'],
                    key=lambda k: {'attempt1': r_a1['Finf'], 'attempt2': r_a2['Finf'], 'attempt3': r_a3['best_ferr']}[k])
print(f'\n>>> BEST G=13 attempt: {best_attempt} <<<', flush=True)
print(f'  attempt1: ||F||={r_a1["Finf"]:.3e} deficit={r_a1["deficit"]:.4f} slope={r_a1["slope_T"]:.4f}')
print(f'  attempt2: ||F||={r_a2["Finf"]:.3e} deficit={r_a2["deficit"]:.4f} slope={r_a2["slope_T"]:.4f}')
print(f'  attempt3: best ||F||={r_a3["best_ferr"]:.3e} deficit={r_a3["deficit"]:.4f} slope={r_a3["slope_T"]:.4f}')

# Save best
P_best = {'attempt1': P_a1, 'attempt2': P_a2, 'attempt3': P_a3}[best_attempt]
np.save(os.path.join(HERE, 'P_nailed_G13_BEST.npy'), P_best)
json.dump({'best_attempt':best_attempt,'attempt1':r_a1,'attempt2':r_a2,'attempt3':r_a3,
           'G9_metrics':m9,'G13_raw_metrics':m13_raw},
          open(os.path.join(HERE, 'g13_nail_clean.json'),'w'), indent=2, default=str)
print('saved P_nailed_G13_BEST.npy and g13_nail_clean.json')
