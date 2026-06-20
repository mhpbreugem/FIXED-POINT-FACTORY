"""100-gamma sweep with Linear-CDF Richardson 2-pt {0.5, 0.3}.
Warm-start chain + pass-2 retry from nearest converged neighbor.
Live Ferr updates."""
import sys, time, json, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_cdf_uniform_grid

G = 7
u_grid = make_cdf_uniform_grid(G)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
TAU = 1.0
T_full = TAU*(U1+U2+U3)
HS = (0.5, 0.3)  # 2-pt: only option that converges on Lin-CDF
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

def solve_one(gamma, x_init=None, n_anderson=40, log_every=10):
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
        if (it+1) % log_every == 0 or it < 2:
            print(f'    And it {it+1:3d}: |F|={f:.3e}', flush=True)
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
            if f_nk < min(Fs):
                print(f'    NK -> |F|={f_nk:.3e}', flush=True)
                return x_nk, f_nk, Fs
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs):
                return x_nk, f_nk, Fs
    return x_best, min(Fs), Fs

print(f'=== Lin-CDF Richardson 2-pt hs={HS}, 100-gamma sweep ===')
print('JIT warmup...', flush=True)
t0 = time.time()
_ = phi_lin_richardson(sg(0.5*T_full), u_grid, hs=HS, gamma=1.0)
print(f'  done ({time.time()-t0:.1f}s)\n', flush=True)

OUT = '/tmp/cheby_h0/lin_r2_100.json'
FPS = '/tmp/cheby_h0/fps_lin_r2'
os.makedirs(FPS, exist_ok=True)
results = {}
if os.path.exists(OUT): results = json.load(open(OUT))

GAMMAS = np.geomspace(0.01, 1000, 100)
x_warm = None
skipped = []
for i, gamma in enumerate(GAMMAS):
    key = f'{gamma:.6e}'
    if key in results and results[key].get('F', 1) < 1e-12:
        continue
    print(f'\n[{i+1:3d}/100] gamma={gamma:.4g}{" (warm)" if x_warm is not None else " (cold)"}',
          flush=True)
    t0 = time.time()
    try:
        x_sol, F_best, Fs = solve_one(gamma, x_init=x_warm, n_anderson=30, log_every=10)
    except Exception as e:
        print(f'    ERROR: {e}', flush=True)
        skipped.append(i); continue
    P_fp = x_sol.reshape(G, G, G)
    s, def_lin, def_1to1 = fit(P_fp, T_full)
    print(f'    final |F|={F_best:.3e}, slope={s:.4f}, def_1to1={def_1to1:.4f} ({time.time()-t0:.1f}s)',
          flush=True)
    np.save(f'{FPS}/P_FP_g{gamma:.6e}.npy', P_fp)
    results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                          deficit_lin=def_lin, deficit_oneToOne=def_1to1,
                          t=time.time()-t0)
    if F_best < 1e-10:
        x_warm = x_sol.copy()
    else:
        skipped.append(i)
        print(f'    -> SKIP', flush=True)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

# Pass 2 retry from nearest neighbor
print(f'\n=== PASS 2: retry {len(skipped)} skipped from nearest neighbor ===')
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
    if nearest is None:
        print(f'[retry {i}] gamma={gamma:.4g}: no neighbor, skip')
        continue
    g_nb = GAMMAS[nearest]
    P_nb = np.load(f'{FPS}/P_FP_g{g_nb:.6e}.npy')
    print(f'\n[retry {i}] gamma={gamma:.4g} from gamma={g_nb:.4g}', flush=True)
    t0 = time.time()
    x_sol, F_best, Fs = solve_one(gamma, x_init=P_nb.ravel(), n_anderson=60, log_every=15)
    P_fp = x_sol.reshape(G, G, G)
    s, def_lin, def_1to1 = fit(P_fp, T_full)
    print(f'    final |F|={F_best:.3e}, slope={s:.4f}, def_1to1={def_1to1:.4f}',
          flush=True)
    np.save(f'{FPS}/P_FP_g{gamma:.6e}.npy', P_fp)
    results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                          deficit_lin=def_lin, deficit_oneToOne=def_1to1,
                          t=time.time()-t0)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

eps = sum(1 for v in results.values() if v['F'] < 1e-12)
conv = sum(1 for v in results.values() if v['F'] < 1e-10)
print(f'\n{len(results)}/100 done, {eps} machine eps, {conv} converged')
print('Done.')
