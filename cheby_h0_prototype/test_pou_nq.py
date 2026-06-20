"""Test POU+chebroots floor as a function of NQ (quadrature order).
Hypothesis: NQ=12 isn't enough at the corner cells of the cube."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from numba import njit
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import (V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH,
                            N_GRID, NQ as NQ_default)
from cheby_sym2 import expand, contract
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def phi_pou_nq(P, NQ, G_p=121):
    p_grid = make_p_grid(G_p)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)

print('=== POU+chebroots: floor vs NQ (GL quadrature order) ===\n')
# Warmup at each NQ
for nq in [12, 16, 24, 32]:
    _ = phi_pou_nq(sg(0.5*T), nq, G_p=121)

def anderson(F_func, x0, n_iter=60):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-14: break
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
    return Fs, x_best

results = {}
print(f'{"NQ":>4} {"And floor":>12} {"NK floor":>12} {"t/iter":>10}')
x_cold = contract(sg(0.5*T))
for nq in [12, 16, 24, 32]:
    F_func = lambda x, nq=nq: contract(phi_pou_nq(expand(x), nq, G_p=121)) - x
    t0 = time.time()
    Fs, x_a = anderson(F_func, x_cold, n_iter=40)
    t_iter = (time.time() - t0) / len(Fs)
    try:
        x_nk = newton_krylov(F_func, x_a, f_tol=1e-14, maxiter=20, verbose=False)
        f_nk = float(np.max(np.abs(F_func(x_nk))))
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
    results[nq] = dict(anderson=float(min(Fs)), nk=f_nk, t_iter=t_iter)
    print(f'{nq:>4d} {min(Fs):>12.3e} {f_nk:>12.3e} {t_iter*1000:>7.1f} ms')

json.dump(results, open('/tmp/cheby_h0/pou_nq_sweep.json', 'w'),
            indent=2, default=str)
print('\nsaved')
