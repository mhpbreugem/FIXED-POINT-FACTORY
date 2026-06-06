"""100-gamma sweep with 10-point Richardson + warm-start chain.

10 h-points: logarithmic between 0.5 and 0.1 -> Richardson cancels O(h^18)
Per-gamma: try cold (sigmoid), else warm from previous-gamma FP
If Ferr > 1e-10 after Anderson+NK: SKIP, mark for second pass
Second pass: solve from nearest converged neighbor

Live updates: every Anderson iter's Ferr; per-gamma summary.
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
from cheby_richardson_tab import phi_richardson, richardson_weights
from cheby_numba import U_NODES, TAU
from cheby_sym2 import expand, contract

G = 7
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_full = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# 10-point Richardson: log-spaced h from 0.5 down to 0.1
HS = tuple(np.geomspace(0.5, 0.1, 10).tolist())
print(f'10-pt Richardson hs: {[f"{h:.3f}" for h in HS]}')
print(f'Vandermonde-derived weights:')
W = richardson_weights(HS)
print(f'  weights: {W}')
print(f'  sum(w) = {W.sum():.6e} (should be 1)')
print(f'  sum(w*h^2) = {np.sum(W * np.array(HS)**2):.3e} (should be 0)')
print(f'  Vandermonde condition: {np.linalg.cond(np.array([[h**(2*k) for k in range(10)] for h in HS])):.2e}')


def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2_l = 1 - float(np.sum((L-pred)**2) / np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2_l, w/ss


def solve_one(gamma, x_init=None, n_anderson=80, log_every=10):
    """Solve at gamma; return (best_x, best_F, anderson_Fs)."""
    def F(x):
        return contract(phi_richardson(expand(x), hs=HS, gamma=gamma, tau=TAU)) - x
    x0 = x_init.copy() if x_init is not None else contract(sg(0.5*T_full))
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_anderson):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if (it+1) % log_every == 0 or it < 3 or f < 1e-13:
            print(f'    Anderson it {it+1:3d}: |F|={f:.3e}', flush=True)
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
    # NK
    if min(Fs) > 1e-13:
        try:
            x_nk = newton_krylov(F, x_best, f_tol=1e-15, maxiter=20, verbose=False)
            f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs):
                print(f'    NK improved to |F|={f_nk:.3e}', flush=True)
                return x_nk, f_nk, Fs
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
            if f_nk < min(Fs):
                print(f'    NK noConv but improved to |F|={f_nk:.3e}', flush=True)
                return x_nk, f_nk, Fs
    return x_best, min(Fs), Fs


# Warmup
print('\nJIT warmup...', flush=True)
t0 = time.time()
_ = phi_richardson(sg(0.5*T_full), hs=HS, gamma=1.0, tau=TAU)
_ = phi_richardson(sg(0.5*T_full), hs=HS, gamma=1.0, tau=TAU)
print(f'  done ({time.time()-t0:.1f}s)', flush=True)
# Time one Phi
t0 = time.time()
_ = phi_richardson(sg(0.5*T_full), hs=HS, gamma=1.0, tau=TAU)
print(f'  per-Phi: {(time.time()-t0)*1000:.1f} ms', flush=True)


OUT = '/tmp/cheby_h0/r10_100gamma.json'
FPS = '/tmp/cheby_h0/fps_r10'
os.makedirs(FPS, exist_ok=True)
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

GAMMAS = np.geomspace(0.01, 1000, 100)
print(f'\n=== 100-point gamma sweep, gamma in [{GAMMAS[0]:.4f}, {GAMMAS[-1]:.4f}] ===\n')

# ===== PASS 1: solve each gamma in order, warm-start from neighbor =====
x_warm = None  # warm from previous gamma
skipped = []
for i, gamma in enumerate(GAMMAS):
    key = f'{gamma:.6e}'
    if key in results and results[key].get('F', 1) < 1e-12:
        print(f'[{i+1:3d}/100] gamma={gamma:.4g} (already converged, skip)',
              flush=True)
        if results[key].get('x_arr'):
            x_warm = np.array(results[key]['x_arr'])
        continue
    print(f'\n[{i+1:3d}/100] gamma={gamma:.4g}{" (warm)" if x_warm is not None else " (cold)"}',
          flush=True)
    t0 = time.time()
    try:
        x_sol, F_best, Fs = solve_one(gamma, x_init=x_warm, n_anderson=40,
                                          log_every=10)
    except Exception as e:
        print(f'    ERROR: {e}', flush=True)
        skipped.append(i)
        continue
    s, def_lin, def_1to1 = fit(expand(x_sol), T_full)
    print(f'    final |F|={F_best:.3e}, slope={s:.4f}, def_1to1={def_1to1:.4f} '
          f'({time.time()-t0:.1f}s)', flush=True)
    P_fp = expand(x_sol)
    np.save(f'{FPS}/P_FP_gamma{gamma:.6e}.npy', P_fp)
    results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                          deficit_lin=def_lin, deficit_oneToOne=def_1to1,
                          t=time.time()-t0)
    if F_best < 1e-10:
        x_warm = x_sol.copy()  # use as warm for next
    else:
        skipped.append(i)
        print(f'    -> SKIPPED (no convergence)', flush=True)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

# ===== PASS 2: retry skipped using NEAREST converged neighbor =====
print(f'\n\n=== PASS 2: retry {len(skipped)} skipped gammas ===')
print(f'skipped indices: {skipped}', flush=True)
for i in skipped:
    gamma = GAMMAS[i]
    key = f'{gamma:.6e}'
    # Find nearest converged neighbor
    candidates_below = [j for j in range(i-1, -1, -1)
                          if f'{GAMMAS[j]:.6e}' in results
                          and results[f'{GAMMAS[j]:.6e}']['F'] < 1e-10]
    candidates_above = [j for j in range(i+1, len(GAMMAS))
                          if f'{GAMMAS[j]:.6e}' in results
                          and results[f'{GAMMAS[j]:.6e}']['F'] < 1e-10]
    nearest = None
    if candidates_below: nearest = candidates_below[0]
    if candidates_above and (nearest is None or
                                abs(np.log(GAMMAS[candidates_above[0]]/gamma)) <
                                abs(np.log(GAMMAS[nearest]/gamma))):
        nearest = candidates_above[0]
    if nearest is None:
        print(f'[retry {i}] gamma={gamma:.4g}: no neighbor converged, skip',
              flush=True)
        continue
    g_nb = GAMMAS[nearest]
    P_nb = np.load(f'{FPS}/P_FP_gamma{g_nb:.6e}.npy')
    x_warm = contract(P_nb)
    print(f'\n[retry {i}] gamma={gamma:.4g} from gamma={g_nb:.4g}', flush=True)
    t0 = time.time()
    x_sol, F_best, Fs = solve_one(gamma, x_init=x_warm, n_anderson=80, log_every=20)
    s, def_lin, def_1to1 = fit(expand(x_sol), T_full)
    print(f'    final |F|={F_best:.3e}, slope={s:.4f}, '
          f'def_1to1={def_1to1:.4f}', flush=True)
    np.save(f'{FPS}/P_FP_gamma{gamma:.6e}.npy', expand(x_sol))
    results[key] = dict(gamma=float(gamma), F=float(F_best), slope=s,
                          deficit_lin=def_lin, deficit_oneToOne=def_1to1,
                          t=time.time()-t0)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

# Summary
print('\n\n=== Final summary ===')
n_conv = sum(1 for k, v in results.items() if v['F'] < 1e-10)
n_eps = sum(1 for k, v in results.items() if v['F'] < 1e-13)
print(f'  {n_eps}/100 reach machine eps (F<1e-13)')
print(f'  {n_conv}/100 converge (F<1e-10)')
print('Done.', flush=True)
