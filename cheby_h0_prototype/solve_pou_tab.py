"""Drive the strict h=0 POU Cheb-tab operator to machine eps."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_numba_pou_tab import phi_pou_tab, make_p_grid
from cheby_numba import U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

p_grid = make_p_grid(121)
def F_sym(x):
    return contract(phi_pou_tab(expand(x), G_p=121, p_grid=p_grid)) - x
def F_err(x):
    return float(np.max(np.abs(F_sym(x))))

# Anderson
def anderson(F_func, x0, n_iter=40, m=10, tol=1e-15):
    x = x0.copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        t0 = time.time()
        F = F_func(x); gx = F + x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
        print(f'  iter {it+1:>2d}: |F| = {Ferr:.3e}  ({time.time()-t0:.2f}s)',
              flush=True)
        if Ferr < tol: break
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

print(f'=== Strict h=0 POU Cheb-tab: solve at G={G}, G_p=121 ===\n')
x_cold = contract(sg(0.5*T))
print(f'COLD start: F = {F_err(x_cold):.3e}')
print('Anderson from cold:')
F_c, x_c = anderson(F_sym, x_cold, n_iter=30)
print(f'  best: {min(F_c):.3e}')

REPO = '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype'
P_warm = np.load(f'{REPO}/P_final_sym2.npy')
x_warm = contract(P_warm)
print(f'\nWARM start (P_final_sym2): F = {F_err(x_warm):.3e}')
print('Anderson from warm:')
F_w, x_w = anderson(F_sym, x_warm, n_iter=20)
print(f'  best: {min(F_w):.3e}')

# Newton-Krylov from the better of the two
best_x = x_c if min(F_c) < min(F_w) else x_w
print(f'\nNewton-Krylov from best point (F={F_err(best_x):.3e})...')
try:
    x_nk = newton_krylov(F_sym, best_x, f_tol=1e-15, maxiter=20, verbose=True)
    f_nk = F_err(x_nk)
    print(f'  Final F: {f_nk:.3e}')
except NoConvergence as e:
    x_nk = e.args[0]
    f_nk = F_err(x_nk)
    print(f'  Final F (NoConv): {f_nk:.3e}')

# Save
data = dict(
    cold_F=min(F_c), cold_Fs=F_c,
    warm_F=min(F_w), warm_Fs=F_w,
    nk_F=float(f_nk),
)
json.dump(data, open('/tmp/cheby_h0/pou_tab_results.json', 'w'),
            indent=2, default=str)
print(f'\nsaved pou_tab_results.json')
print(f'\nBest overall: {min(min(F_c), min(F_w), f_nk):.3e}')
if min(min(F_c), min(F_w), f_nk) < 1e-10:
    P_FP = expand(x_nk if f_nk < min(F_c) else (x_c if min(F_c) < min(F_w) else x_w))
    np.save('/tmp/cheby_h0/P_FP_pou_tab.npy', P_FP)
    print('saved P_FP_pou_tab.npy')
