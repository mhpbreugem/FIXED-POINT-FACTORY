"""Paper-quality overnight sweep: 2D tau x gamma with adaptive Richardson order.

Strategy:
- Lin-CDF R4 Richardson (h={0.5, 0.4, 0.3, 0.2}) as MAIN solver
  (best balance: 99/100 reach machine eps, slope α* converged)
- Also compute R5 estimate as ROMBERG ERROR BAR (|μ_R5 - μ_R4|)
- 2D grid: tau in [0.01, 5] (avoid logit clipping); gamma in [0.01, 1000]
- 30 x 30 = 900 cells
- Warm-start chain: sweep gamma at each tau, using previous as warm start

Saved per-cell: F, slope, deficit_lin, deficit_oneToOne, FP array,
                Romberg error bound, time
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
from lin_cdf_richardson import (phi_lin_richardson, richardson_weights,
                                    phi_lin_richardson_jit)
from lin_cdf_kern_tab import (make_cdf_uniform_grid, make_p_grid,
                                  build_mu_table_lin_kern, make_gl_for_u)
from cheby_numba import C_STRETCH

G = 7
u_grid = make_cdf_uniform_grid(G)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
HS_R4 = (0.5, 0.4, 0.3, 0.2)
HS_R5 = (0.6, 0.5, 0.4, 0.3, 0.2)
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

def romberg_error_bound(P_fp, tau, gamma):
    """Compute |mu_R5 - mu_R4| as error bound on Richardson estimate.
    Just one extra Phi evaluation at R5."""
    P_R4 = phi_lin_richardson(P_fp, u_grid, hs=HS_R4, gamma=gamma, tau=tau)
    P_R5 = phi_lin_richardson(P_fp, u_grid, hs=HS_R5, gamma=gamma, tau=tau)
    return float(np.max(np.abs(P_R5 - P_R4)))

def solve_one(tau, gamma, x_init=None, n_anderson=40):
    T_full = tau*(U1+U2+U3)
    def F(x):
        return (phi_lin_richardson(x.reshape(G,G,G), u_grid, hs=HS_R4,
                                        gamma=gamma, tau=tau)
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
            if f_nk < min(Fs): return x_nk, f_nk, T_full
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs): return x_nk, f_nk, T_full
    return x_best, min(Fs), T_full


print('=== Paper-quality 2D (tau, gamma) sweep with Lin-CDF R4 + Romberg ===')
print('JIT warmup...', flush=True)
t0 = time.time()
for hs in [HS_R4, HS_R5]:
    _ = phi_lin_richardson(sg(0.5*(U1+U2+U3)), u_grid, hs=hs, gamma=1.0, tau=1.0)
print(f'  warmup done ({time.time()-t0:.1f}s)', flush=True)

OUT = '/tmp/cheby_h0/paper_2d.json'
FPS = '/tmp/cheby_h0/fps_paper_2d'
os.makedirs(FPS, exist_ok=True)
results = {}
if os.path.exists(OUT): results = json.load(open(OUT))

# Grid
TAUS = np.geomspace(0.01, 5.0, 30)       # valid range (no logit clipping)
GAMMAS = np.geomspace(0.01, 1000.0, 30)

print(f'\nGrid: {len(TAUS)} x {len(GAMMAS)} = {len(TAUS)*len(GAMMAS)} cells')
print(f'tau in [{TAUS[0]:.3g}, {TAUS[-1]:.3g}]')
print(f'gamma in [{GAMMAS[0]:.3g}, {GAMMAS[-1]:.3g}]\n')

n_total = len(TAUS) * len(GAMMAS)
n_done = 0
t_start = time.time()

# For warm-start chains: walk through tau, then gamma at each tau
prev_x_per_tau = {}  # tau idx -> last x_warm for that tau row
for i, tau in enumerate(TAUS):
    x_warm = None
    for j, gamma in enumerate(GAMMAS):
        key = f't{tau:.6e}_g{gamma:.6e}'
        n_done += 1
        if key in results and results[key]['F'] < 1e-12:
            continue
        t0 = time.time()
        try:
            x_sol, F_best, T_full = solve_one(tau, gamma, x_init=x_warm)
        except Exception as e:
            print(f'  [{n_done:>4d}/{n_total}] (t={tau:.3g}, g={gamma:.3g}): ERROR {e}',
                  flush=True)
            continue
        P_fp = x_sol.reshape(G, G, G)
        s, def_l, def_1to1 = fit(P_fp, T_full)
        # Romberg error bound
        rb_err = romberg_error_bound(P_fp, tau, gamma)
        np.save(f'{FPS}/P_FP_t{tau:.6e}_g{gamma:.6e}.npy', P_fp)
        results[key] = dict(tau=float(tau), gamma=float(gamma),
                              F=float(F_best), slope=s,
                              deficit_lin=def_l, deficit_oneToOne=def_1to1,
                              romberg_err=rb_err, t=time.time()-t0)
        # Compact print
        if n_done % 30 == 0 or F_best > 1e-10:
            flag = 'eps' if F_best < 1e-12 else 'OK' if F_best < 1e-10 else 'FAIL'
            elapsed = time.time() - t_start
            eta = elapsed * (n_total - n_done) / n_done
            print(f'  [{n_done:>4d}/{n_total}] tau={tau:8.3g} gamma={gamma:8.3g}: '
                  f'F={F_best:.2e}[{flag}] s={s:.3f} d={def_1to1:.4f} '
                  f'rb={rb_err:.2e} ({time.time()-t0:.1f}s, ETA {eta/60:.1f}min)',
                  flush=True)
        if F_best < 1e-10:
            x_warm = x_sol.copy()
        json.dump(results, open(OUT, 'w'), indent=2, default=str)

n_eps = sum(1 for v in results.values() if v['F'] < 1e-12)
n_conv = sum(1 for v in results.values() if v['F'] < 1e-10)
print(f'\n=== Phase 1 done ===')
print(f'  {len(results)}/{n_total} cells')
print(f'  {n_eps} at machine eps')
print(f'  {n_conv} converged')

# Pass 2: retry skipped from any nearest converged neighbor
print(f'\n=== Phase 2: retry failed cells ===')
failed = [k for k, v in results.items() if v['F'] >= 1e-10]
print(f'  {len(failed)} failed cells')
for key in failed:
    parts = key.split('_g')
    tau = float(parts[0][1:])
    gamma = float(parts[1])
    # Find nearest converged in same tau or gamma row
    best_dist = float('inf'); best_warm = None
    for k2, v in results.items():
        if v['F'] < 1e-10:
            p = k2.split('_g')
            t2 = float(p[0][1:]); g2 = float(p[1])
            d = (np.log(t2/tau))**2 + (np.log(g2/gamma))**2
            if d < best_dist:
                best_dist = d; best_warm_key = k2
    if not best_warm: continue
    p = best_warm_key.split('_g')
    t2 = float(p[0][1:]); g2 = float(p[1])
    P_warm = np.load(f'{FPS}/P_FP_t{t2:.6e}_g{g2:.6e}.npy')
    t0 = time.time()
    x_sol, F_best, T_full = solve_one(tau, gamma, x_init=P_warm.ravel(),
                                          n_anderson=80)
    P_fp = x_sol.reshape(G, G, G)
    s, def_l, def_1to1 = fit(P_fp, T_full)
    rb_err = romberg_error_bound(P_fp, tau, gamma)
    np.save(f'{FPS}/P_FP_t{tau:.6e}_g{gamma:.6e}.npy', P_fp)
    results[key] = dict(tau=float(tau), gamma=float(gamma),
                          F=float(F_best), slope=s,
                          deficit_lin=def_l, deficit_oneToOne=def_1to1,
                          romberg_err=rb_err, t=time.time()-t0)
    print(f'  retry t={tau:.3g} g={gamma:.3g}: F={F_best:.2e}', flush=True)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

n_eps_final = sum(1 for v in results.values() if v['F'] < 1e-12)
n_conv_final = sum(1 for v in results.values() if v['F'] < 1e-10)
print(f'\n=== FINAL ===')
print(f'  {len(results)}/{n_total} cells in {(time.time()-t_start)/60:.1f}min')
print(f'  {n_eps_final} machine eps')
print(f'  {n_conv_final} converged (F<1e-10)')
