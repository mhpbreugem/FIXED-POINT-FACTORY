"""Sweep h_bw and try to drive ||F||_inf below 3e-3 with smoothed operator."""
import sys, time
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import least_squares

from cheby_numba_smooth import phi_smooth
from cheby_numba import U_NODES, TAU, GAMMA
from cheby_sym2 import expand, contract

U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def run_anderson(op_func, omega, m_hist, n_iter, tol=1e-13):
    x = contract(sg(0.5*T))
    Xh, Gh = [], []
    Ferrs = []
    for it in range(n_iter):
        gx = contract(op_func(expand(x)))
        F = gx - x
        Ferr = float(np.max(np.abs(F)))
        Ferrs.append(Ferr)
        if Ferr < tol: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > m_hist:
            Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1:
            x = omega*gx + (1-omega)*x
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = omega*(Gh[k-1] + DG @ ga) + (1-omega)*x
            except: x = gx
    return Ferrs

def run_lm(op_func, n_eval=200):
    history = []
    def F(x):
        f = contract(op_func(expand(x))) - x
        history.append(float(np.max(np.abs(f))))
        return f
    sol = least_squares(F, contract(sg(0.5*T)), method='lm',
                          max_nfev=n_eval, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return history, sol

# Warmup numba
_ = phi_smooth(sg(0.5*T), h_bw=1e-3)

print(f'=== Sweep h_bw and try to get below 3e-3 ===\n')
print(f'{"h_bw":>10}  {"And.floor":>12}  {"LM.floor":>12}  {"LM.cost":>12}')
results = {}
for h_bw in [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9]:
    op = lambda P, h=h_bw: phi_smooth(P, h_bw=h)
    Fa = run_anderson(op, 1.0, 8, 60)
    Fl, sol = run_lm(op, n_eval=200)
    results[h_bw] = dict(anderson_floor=min(Fa), lm_floor=min(Fl),
                          lm_cost=float(sol.cost), n_lm_eval=int(sol.nfev))
    print(f'{h_bw:>10.0e}  {min(Fa):>12.3e}  {min(Fl):>12.3e}  {sol.cost:>12.3e}')

import json
json.dump(results, open('/tmp/cheby_h0/smooth_sweep.json', 'w'), indent=2, default=str)
print('\nsaved smooth_sweep.json')
