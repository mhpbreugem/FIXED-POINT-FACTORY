"""Pure dense Newton on symmetric subspace at N=7 (40 DOFs).

J = dPhi/dP via forward differences, dense 40x40.
Solve (J - I) dx = -F at each step, Armijo line search.

Newton converges quadratically near a FP regardless of spectral radius,
provided J - I is well-conditioned and a FP actually exists.
"""
import sys, time
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from cheby_numba_bisect import phi_bisect
from cheby_numba_smooth import phi_smooth
from cheby_numba import U_NODES, TAU, GAMMA
from cheby_sym2 import expand, contract

U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def F_of(x, op):
    return contract(op(expand(x))) - x

def jacobian(x, op, eps=1e-6):
    n = x.size
    F0 = F_of(x, op)
    J = np.empty((n, n))
    for j in range(n):
        xp = x.copy(); xp[j] += eps
        Fp = F_of(xp, op)
        J[:, j] = (Fp - F0) / eps
    return J, F0

def pure_newton(op, n_iter=30, tol=1e-13, x0=None, label='', stall_patience=5):
    x = contract(sg(0.5*T)) if x0 is None else x0.copy()
    Ferrs = []
    cond_history = []
    print(f'\n--- Pure dense Newton: {label} ---')
    for it in range(n_iter):
        t0 = time.time()
        J, F = jacobian(x, op, eps=1e-7)
        Ferr = float(np.max(np.abs(F)))
        Ferrs.append(Ferr)
        # J here is dF/dx already (returned by jacobian()). Newton: J dx = -F.
        cond = np.linalg.cond(J)
        cond_history.append(cond)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print(f'  iter {it+1}: singular, switching to pinv')
            dx = -np.linalg.pinv(J) @ F
        # Armijo line search (weak: c1 = 1e-4)
        alpha = 1.0
        for ls in range(30):
            x_new = x + alpha * dx
            F_new = F_of(x_new, op)
            if np.max(np.abs(F_new)) < (1 - 1e-4*alpha) * Ferr or ls == 29:
                break
            alpha *= 0.5
        x = x_new
        dt = time.time() - t0
        print(f'  iter {it+1:2d}  Ferr={Ferr:.3e}  cond(J-I)={cond:.2e}  '
              f'alpha={alpha:.2e}  ({dt:.1f}s)')
        if Ferr < tol:
            print(f'  CONVERGED at iter {it+1}')
            break
        if it >= stall_patience and Ferrs[-1] > 0.95 * Ferrs[-(stall_patience+1)]:
            print(f'  STALLED (no progress in {stall_patience} iters)')
            break
    return x, Ferrs, cond_history

# Warmup
_ = phi_bisect(sg(0.5*T))
_ = phi_smooth(sg(0.5*T), h_bw=1e-3)

results = {}
# Test 1: Newton on bisect from cold start
x_bi, F_bi, c_bi = pure_newton(phi_bisect, n_iter=40, label='phi_bisect cold',
                                  stall_patience=8)
results['bisect_cold'] = dict(Ferrs=F_bi, cond=c_bi, final=float(F_bi[-1]))

# Test 2: Newton on smoothed (broad bandwidth, expect smoothest landscape)
op_sm = lambda P: phi_smooth(P, h_bw=1e-1)
x_sm, F_sm, c_sm = pure_newton(op_sm, n_iter=40, label='phi_smooth h_bw=1e-1 cold',
                                  stall_patience=8)
results['smooth_h1e-01'] = dict(Ferrs=F_sm, cond=c_sm, final=float(F_sm[-1]))

# Test 3: Newton WARM-STARTED from the LM-converged point on bisect
print('\n--- Warm-starting Newton from LM-converged point ---')
from scipy.optimize import least_squares
def F_func(x): return contract(phi_bisect(expand(x))) - x
sol = least_squares(F_func, contract(sg(0.5*T)), method='lm',
                      max_nfev=300, xtol=1e-15, ftol=1e-15, gtol=1e-15)
print(f'LM warm point: Ferr = {float(np.max(np.abs(sol.fun))):.3e}')
x_warm, F_warm, c_warm = pure_newton(phi_bisect, n_iter=30, x0=sol.x,
                                       label='phi_bisect WARM from LM',
                                       stall_patience=10)
results['bisect_warm_from_LM'] = dict(Ferrs=F_warm, cond=c_warm,
                                       final=float(F_warm[-1]))

# Test 4: Newton on smoothed h_bw=1e-3, warm from LM (best LM run)
def F_func_sm(x): return contract(phi_smooth(expand(x), h_bw=1e-3)) - x
sol_sm = least_squares(F_func_sm, contract(sg(0.5*T)), method='lm',
                          max_nfev=300, xtol=1e-15, ftol=1e-15, gtol=1e-15)
print(f'LM(smooth h=1e-3) warm point: Ferr = '
      f'{float(np.max(np.abs(sol_sm.fun))):.3e}')
op_sm3 = lambda P: phi_smooth(P, h_bw=1e-3)
x_warm3, F_warm3, c_warm3 = pure_newton(op_sm3, n_iter=30, x0=sol_sm.x,
                                          label='phi_smooth h=1e-3 WARM from LM',
                                          stall_patience=10)
results['smooth_warm_from_LM'] = dict(Ferrs=F_warm3, cond=c_warm3,
                                        final=float(F_warm3[-1]))

print('\n=== Summary ===')
for k, v in results.items():
    print(f'  {k:>20}: final Ferr = {v["final"]:.3e}, '
          f'min cond(J-I) = {min(v["cond"]):.2e}, '
          f'max cond(J-I) = {max(v["cond"]):.2e}')

import json
json.dump(results, open('/tmp/cheby_h0/pure_newton.json', 'w'), indent=2, default=str)
print('\nsaved pure_newton.json')
