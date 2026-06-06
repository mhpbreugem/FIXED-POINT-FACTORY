"""Solve POU with chebroots-based all-roots. Slow but should be more accurate."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

from cheby_pou_chebroots import phi_pou_cr
from cheby_numba import U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def F_sym(x):
    return contract(phi_pou_cr(expand(x), G_p=121)) - x
def Ferr(x): return float(np.max(np.abs(F_sym(x))))

def anderson(x0, n_iter=20, m=8):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        t0 = time.time()
        F = F_sym(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        print(f'  iter {it+1:2d}: F={f:.3e}  ({time.time()-t0:.1f}s)', flush=True)
        if f < 1e-14: break
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

print('=== POU+chebroots solve from cold ===')
x0 = contract(sg(0.5*T))
print(f'Initial F: {Ferr(x0):.3e}')
print('Anderson 20 iters:')
F_a, x_a = anderson(x0, n_iter=20)
print(f'Anderson best: {min(F_a):.3e}')

print('\nNewton-Krylov from Anderson best:')
t0 = time.time()
try:
    x_nk = newton_krylov(F_sym, x_a, f_tol=1e-14, maxiter=15, verbose=False)
    f_nk = Ferr(x_nk)
except NoConvergence as e:
    x_nk = e.args[0]; f_nk = Ferr(x_nk)
print(f'  Result: {f_nk:.3e} ({time.time()-t0:.1f}s)')

json.dump(dict(anderson_floor=min(F_a), nk_floor=float(f_nk)),
            open('/tmp/cheby_h0/pou_cr_results.json', 'w'),
            indent=2, default=str)
print(f'\n=== Best: {min(min(F_a), f_nk):.3e} ===')
