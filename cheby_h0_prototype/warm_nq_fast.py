"""Fast version: just check warm residual under each NQ — no NK."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)

p_grid = make_p_grid(121)
P_warm = np.load('/tmp/cheby_h0/P_FP_pou_nq16_n6.npy')
x_warm = contract(P_warm)

def phi(P, NQ):
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)
def F_sym(x, NQ): return contract(phi(expand(x), NQ)) - x

# Quick Anderson 30 iters per NQ (no NK)
def anderson(F_func, x0, n_iter=30):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
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
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    return Fs, x_best

print('=== Warm from NQ=16 FP into each NQ operator ===\n')
print(f'{"NQ":>4} {"|F| warm":>14} {"After And(30)":>14}')
results = {}
for NQ in [12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 32]:
    _ = phi(P_warm, NQ)  # warmup
    f_warm = float(np.max(np.abs(F_sym(x_warm, NQ))))
    F_func = lambda x, NQ=NQ: F_sym(x, NQ)
    Fs, x_a = anderson(F_func, x_warm)
    results[NQ] = dict(warm=f_warm, anderson=float(min(Fs)))
    print(f'{NQ:>4d} {f_warm:>14.3e} {min(Fs):>14.3e}')

json.dump(results, open('/tmp/cheby_h0/warm_nq_fast.json', 'w'),
            indent=2, default=str)
print('\nIf the NQ=16 FP has small |F| (~1e-X) under other NQ, the FPs nearly match.')
print('If much larger, each NQ has a SHIFTED FP (operator depends on NQ).')
