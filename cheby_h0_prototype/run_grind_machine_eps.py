"""Drive ||F||_inf as low as the float64 architecture allows.

Approach: start from the lowest-residual point found so far (warm + chained
solvers), then refine with pure Newton using a sweep of FD epsilon values.
Newton's finite-difference Jacobian has noise floor ~ |dF| * eps, so
Newton's reachable floor is roughly eps_FD * |F| / |dF/dx| ~ eps_FD.

Strategy:
  1. Identify best (start, operator, recipe) from warmstart_push.json
  2. Pure Newton with eps_FD in {1e-6, 1e-7, 1e-8, sqrt(eps_64)=1e-8}
  3. Multiple Newton restarts (line-search + small step + tight tol)
  4. Compare ||F||_inf trajectory under each
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import least_squares
from cheby_numba import phi as phi_chebroots, U_NODES, TAU, GAMMA, N_GRID
from cheby_numba_bisect import phi_bisect
from cheby_sym2 import expand, contract

REPO = '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype'
G = N_GRID

U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# Load best warm
print('Loading P_final_vbasis.npy as warm start (best found so far)...')
P_warm = np.load(f'{REPO}/P_final_vbasis.npy')

# Warmup
_ = phi_chebroots(P_warm); _ = phi_bisect(P_warm)

x0 = contract(P_warm)
print(f'Warm-start F (chebroots): {float(np.max(np.abs(contract(phi_chebroots(P_warm)) - x0))):.3e}')

# Step 1: Anderson to get into basin
def anderson(op, x0, n_iter=200, m=12):
    x = x0.copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        gx = contract(op(expand(x)))
        F = gx - x; Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > m: Xh.pop(0); Gh.pop(0)
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
    return Fs, x_best

print('\nStage A: 200 iters Anderson (chebroots) from warm...')
F_and, x_and = anderson(phi_chebroots, x0, n_iter=200)
print(f'  Anderson best: {min(F_and):.3e}')

print('Stage B: LM from Anderson output (chebroots, 500 fevals)...')
hist = []; x_lm_best = x_and.copy(); f_lm_best = float('inf')
def F_func(xs):
    f = contract(phi_chebroots(expand(xs))) - xs
    err = float(np.max(np.abs(f)))
    hist.append(err)
    global x_lm_best, f_lm_best
    if err < f_lm_best: f_lm_best = err; x_lm_best = xs.copy()
    return f
sol_lm = least_squares(F_func, x_and, method='lm', max_nfev=500,
                        xtol=1e-16, ftol=1e-16, gtol=1e-16)
print(f'  LM best: {f_lm_best:.3e}')

# Stage C: Pure Newton with multiple FD epsilons
def newton(op, x0, n_iter=40, eps=1e-7, stall_patience=10):
    x = x0.copy()
    Fs = []; x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F0 = contract(op(expand(x))) - x
        Ferr = float(np.max(np.abs(F0))); Fs.append(Ferr)
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
        if Ferr < 1e-14: break
        if it >= stall_patience and Fs[-1] > 0.95*Fs[-(stall_patience+1)]:
            break
        # Jacobian (forward diff)
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += eps
            Fp = contract(op(expand(xp))) - xp
            J[:, j] = (Fp - F0) / eps
        try: dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError: dx = -np.linalg.pinv(J) @ F0
        # Armijo
        alpha = 1.0
        for _ in range(30):
            xn = x + alpha*dx
            Fn = contract(op(expand(xn))) - xn
            if np.max(np.abs(Fn)) < (1-1e-4*alpha)*Ferr: break
            alpha *= 0.5
        x = xn
    return Fs, x_best

print('\nStage C: Pure Newton from LM output, sweep FD epsilon (chebroots)')
print(f'{"eps_FD":>10} {"Newton best":>14}')
nw_results = {}
x_input = x_lm_best.copy()
for eps_fd in [1e-5, 1e-6, 1e-7, 1e-8, 5e-9]:
    F_nw, x_nw = newton(phi_chebroots, x_input, n_iter=40, eps=eps_fd)
    nw_results[eps_fd] = dict(min_F=float(min(F_nw)), Fs=F_nw)
    print(f'{eps_fd:>10.0e} {min(F_nw):>14.3e}')

# Stage D: Cross-Newton (bisect Newton refinement after chebroots Newton)
print('\nStage D: bisect-Newton on top of chebroots-Newton best')
best_eps = min(nw_results, key=lambda e: nw_results[e]['min_F'])
print(f'  best eps_FD was {best_eps:.0e}, residual = {nw_results[best_eps]["min_F"]:.3e}')
# Re-run from that to get the x_best
F_nw_best, x_nw_best = newton(phi_chebroots, x_input, n_iter=40, eps=best_eps)

# Try bisect refinement
F_bi_nw, x_bi_nw = newton(phi_bisect, x_nw_best, n_iter=40, eps=1e-7)
print(f'  bisect Newton on top: {min(F_bi_nw):.3e}')

# Also try the saved-bisect-Anderson FP at this point
F_bi_and, _ = anderson(phi_bisect, x_nw_best, n_iter=100)
print(f'  bisect Anderson on top: {min(F_bi_and):.3e}')

# Save full history
data = dict(
    cold_F_warm=float(np.max(np.abs(contract(phi_chebroots(P_warm)) - x0))),
    anderson_best=min(F_and),
    lm_best=f_lm_best,
    newton_chebroots_sweep={f'{e:.0e}': r['min_F'] for e, r in nw_results.items()},
    bisect_refine=float(min(F_bi_nw)),
    bisect_anderson_refine=float(min(F_bi_and)),
    overall_best=float(min(min(F_and), f_lm_best,
                            *[r['min_F'] for r in nw_results.values()],
                            min(F_bi_nw), min(F_bi_and))),
)
json.dump(data, open('/tmp/cheby_h0/grind_results.json', 'w'),
            indent=2, default=str)

print(f'\n=== OVERALL BEST FLOOR: {data["overall_best"]:.3e} ===')
print('saved grind_results.json')
