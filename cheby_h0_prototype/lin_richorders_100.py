"""100-gamma sweeps at Lin-CDF Richardson 3/4/5-pt orders.
Each sweep uses warm-start chain; results saved incrementally.
"""
import sys, time, json, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from lin_cdf_richardson import phi_lin_richardson, richardson_weights
from lin_cdf_kern_tab import make_cdf_uniform_grid

G = 7
u_grid = make_cdf_uniform_grid(G)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
TAU = 1.0
T_full = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2 = 1 - float(np.sum((L-pred)**2) / np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2, w/ss

def solve_one(HS, gamma, x_init=None, n_anderson=50):
    def F(x):
        return (phi_lin_richardson(x.reshape(G,G,G), u_grid, hs=HS,
                                        gamma=gamma, tau=TAU)
                  - x.reshape(G,G,G)).ravel()
    x = x_init.copy() if x_init is not None else sg(0.5*T_full).ravel().copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_anderson):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-15: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    if min(Fs) > 1e-13:
        try:
            x_nk = newton_krylov(F, x_best, f_tol=1e-15, maxiter=15, verbose=False)
            f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs): return x_nk, f_nk, Fs
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs): return x_nk, f_nk, Fs
    return x_best, min(Fs), Fs

CONFIGS = [
    ('R3', (0.5, 0.4, 0.3)),
    ('R4', (0.5, 0.4, 0.3, 0.2)),
    ('R5', (0.6, 0.5, 0.4, 0.3, 0.2)),
]

print('JIT warmup...', flush=True)
for label, hs in CONFIGS:
    t0 = time.time()
    _ = phi_lin_richardson(sg(0.5*T_full), u_grid, hs=hs, gamma=1.0)
    w = richardson_weights(hs)
    print(f'  {label} hs={hs}, weights range [{w.min():.3f}, {w.max():.3f}], '
          f'sum |w|={np.sum(np.abs(w)):.2f} ({time.time()-t0:.1f}s)', flush=True)


GAMMAS = np.geomspace(0.01, 1000, 100)

for label, HS in CONFIGS:
    OUT = f'/tmp/cheby_h0/lin_{label}_100.json'
    FPS = f'/tmp/cheby_h0/fps_lin_{label}'
    os.makedirs(FPS, exist_ok=True)
    results = {}
    if os.path.exists(OUT): results = json.load(open(OUT))

    print(f'\n=== Lin-CDF {label} hs={HS} ===')
    print(f'  weights={richardson_weights(HS)}')
    x_warm = None
    skipped = []
    for i, gamma in enumerate(GAMMAS):
        key = f'{gamma:.6e}'
        if key in results and results[key]['F'] < 1e-12:
            continue
        t0 = time.time()
        try:
            x_sol, F_best, _ = solve_one(HS, gamma, x_init=x_warm, n_anderson=40)
        except Exception as e:
            print(f'  [{i+1:3d}/100] gamma={gamma:.3g}: ERROR {e}', flush=True)
            skipped.append(i); continue
        P_fp = x_sol.reshape(G, G, G)
        s, def_l, def_1to1 = fit(P_fp, T_full)
        np.save(f'{FPS}/P_FP_g{gamma:.6e}.npy', P_fp)
        results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                              deficit_lin=def_l, deficit_oneToOne=def_1to1,
                              t=time.time()-t0)
        if i % 10 == 0:
            print(f'  [{i+1:3d}/100] gamma={gamma:8.3g}: F={F_best:.2e} '
                  f'slope={s:.4f} def={def_1to1:.4f} ({time.time()-t0:.1f}s)',
                  flush=True)
        if F_best < 1e-10:
            x_warm = x_sol.copy()
        else:
            skipped.append(i)
        json.dump(results, open(OUT, 'w'), indent=2, default=str)
    # Pass 2 from nearest neighbor
    if skipped:
        print(f'  Pass 2: retry {len(skipped)} skipped', flush=True)
        for i in skipped:
            gamma = GAMMAS[i]
            key = f'{gamma:.6e}'
            nbs_below = [j for j in range(i-1, -1, -1)
                            if f'{GAMMAS[j]:.6e}' in results
                            and results[f'{GAMMAS[j]:.6e}']['F'] < 1e-10]
            nbs_above = [j for j in range(i+1, 100)
                            if f'{GAMMAS[j]:.6e}' in results
                            and results[f'{GAMMAS[j]:.6e}']['F'] < 1e-10]
            nearest = None
            if nbs_below: nearest = nbs_below[0]
            if nbs_above and (nearest is None or
                                abs(np.log(GAMMAS[nbs_above[0]]/gamma)) <
                                abs(np.log(GAMMAS[nearest]/gamma))):
                nearest = nbs_above[0]
            if nearest is None: continue
            g_nb = GAMMAS[nearest]
            P_nb = np.load(f'{FPS}/P_FP_g{g_nb:.6e}.npy')
            t0 = time.time()
            x_sol, F_best, _ = solve_one(HS, gamma, x_init=P_nb.ravel(),
                                              n_anderson=60)
            P_fp = x_sol.reshape(G, G, G)
            s, def_l, def_1to1 = fit(P_fp, T_full)
            np.save(f'{FPS}/P_FP_g{gamma:.6e}.npy', P_fp)
            results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                                  deficit_lin=def_l, deficit_oneToOne=def_1to1,
                                  t=time.time()-t0)
            json.dump(results, open(OUT, 'w'), indent=2, default=str)
    eps = sum(1 for v in results.values() if v['F'] < 1e-12)
    conv = sum(1 for v in results.values() if v['F'] < 1e-10)
    print(f'  {label} DONE: {eps}/100 machine eps, {conv}/100 converged',
          flush=True)

print('\nAll sweeps complete.')
