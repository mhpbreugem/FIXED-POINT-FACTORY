"""Solve POU+chebroots at N=6, NQ=16 to machine eps and save the FP."""
import sys, time
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

NQ_BEST = 16; G_p = 121
p_grid = make_p_grid(G_p)
gl_n, gl_w = np.polynomial.legendre.leggauss(NQ_BEST)
def F_sym(x):
    P = expand(x)
    Pn = phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                          TAU, GAMMA, C_STRETCH, G, NQ_BEST)
    return contract(Pn) - x

# Warmup
_ = phi_pou_cr_jit(sg(0.5*T), V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                       TAU, GAMMA, C_STRETCH, G, NQ_BEST)

print(f'=== POU+chebroots NAIL at N=6, NQ={NQ_BEST} ===\n')
x0 = contract(sg(0.5*T))
print(f'Initial F: {float(np.max(np.abs(F_sym(x0)))):.3e}')

# Anderson
print('Anderson 80 iters:')
Xh, Gh = [], []; Fs = []
x = x0.copy(); x_best = x.copy(); f_best = float('inf')
for it in range(80):
    F = F_sym(x); gx = F + x
    f = float(np.max(np.abs(F))); Fs.append(f)
    if f < f_best: f_best = f; x_best = x.copy()
    if f < 1e-15: break
    Xh.append(x.copy()); Gh.append(gx.copy())
    if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
    k = len(Xh)
    if k <= 1: x = gx
    else:
        DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
        R_k = Gh[k-1] - Xh[k-1]
        A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
        ga = np.linalg.solve(A, -DR.T @ R_k)
        DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
        x = Gh[k-1] + DG @ ga
print(f'  best Anderson: {min(Fs):.3e}')

print('Newton-Krylov nail:')
x_nk = newton_krylov(F_sym, x_best, f_tol=1e-15, maxiter=30, verbose=False)
f_nk = float(np.max(np.abs(F_sym(x_nk))))
print(f'  F = {f_nk:.6e}')

# Extract FP
P_fp = expand(x_nk)
np.save('/tmp/cheby_h0/P_FP_pou_nq16_n6.npy', P_fp)
print(f'\nSaved P_FP_pou_nq16_n6.npy, range [{P_fp.min():.4f}, {P_fp.max():.4f}]')

# Compute FP shape
Pc = np.clip(P_fp, 1e-15, 1-1e-15)
L = np.log(Pc/(1-Pc)).ravel()
slope = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
intercept = float(np.mean(L - slope*T.ravel()))
pred = slope*T.ravel() + intercept
res = np.sum((L-pred)**2)
tot = np.sum((L-L.mean())**2)
deficit = float(res/tot)
print(f'\nFP characterization:')
print(f'  slope α* = {slope:.6f}')
print(f'  intercept = {intercept:.6f}')
print(f'  deficit (1-R²) = {deficit:.6f}')
print(f'  ||F||_inf = {f_nk:.3e}')
