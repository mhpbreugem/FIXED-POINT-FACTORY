"""Solve the kernel-band Cheb-tab operator to machine eps.

The operator is analytic in P (Gaussian kernel of differences),
so Newton-Krylov should converge quadratically.
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov, least_squares
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

from cheby_numba_kern_tab import phi_kern_tab, make_p_grid, DEFAULT_H_KERN
from cheby_numba import U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# warmup
_ = phi_kern_tab(sg(0.5*T), kernel_h=0.3, G_p=51)

# Pick a kernel bandwidth and Gp. Start moderate.
def F_sym(x_sym, h=0.3, G_p=51):
    P = expand(x_sym)
    Pn = phi_kern_tab(P, kernel_h=h, G_p=G_p)
    return contract(Pn) - x_sym

def F_full(P, h=0.3, G_p=51):
    return phi_kern_tab(P, kernel_h=h, G_p=G_p) - P

def Ferr_sym(x_sym, h=0.3, G_p=51):
    return float(np.max(np.abs(F_sym(x_sym, h, G_p))))


print('=== Solve kernel-band Cheb-tab to machine eps ===\n')
print(f'N=6, G={G}, sym-DOF=40, tau=gamma=1\n')

# Try several h values; for each: Anderson -> Newton-Krylov
def anderson(F_func, x0, n_iter=80, m=10):
    x = x0.copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
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


def nail_NK(F_func, x0, tol=1e-13, maxiter=80):
    try:
        x_sol = newton_krylov(F_func, x0, f_tol=tol, maxiter=maxiter,
                                verbose=False)
        return float(np.max(np.abs(F_func(x_sol)))), x_sol, True
    except NoConvergence as e:
        x_sol = e.args[0]
        return float(np.max(np.abs(F_func(x_sol)))), x_sol, False


# --- Sweep h ---
results = {}
print(f'{"h":>6} {"G_p":>5} {"AndFloor":>14} {"NK final":>14} {"NK ok?":>8} {"t_NK":>8}')
print('-'*70)
for h in [0.5, 0.3, 0.2, 0.15]:
    for G_p in [51, 121]:
        x0 = contract(sg(0.5*T))
        F_func = lambda x, h=h, G_p=G_p: F_sym(x, h, G_p)
        F_and, x_and = anderson(F_func, x0, n_iter=60)
        t0 = time.time()
        f_nk, x_nk, ok = nail_NK(F_func, x_and, tol=1e-14, maxiter=80)
        dt = time.time() - t0
        results[f'cold_h{h}_Gp{G_p}'] = dict(
            anderson_floor=float(min(F_and)),
            nk_final=f_nk, nk_converged=ok, time_NK=dt,
        )
        print(f'{h:>6.2f} {G_p:>5d} {min(F_and):>14.3e} {f_nk:>14.3e} '
              f'{"yes" if ok else "no":>8} {dt:>8.1f}s')

# --- Also test warm-start from existing FP at h=0.3, G_p=121 ---
REPO = '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype'
P_warm = np.load(f'{REPO}/P_final_sym2.npy')
x_warm = contract(P_warm)
print(f'\n=== Warm start from P_final_sym2 (h=0.3, G_p=121) ===')
F_func = lambda x: F_sym(x, h=0.3, G_p=121)
print(f'  Initial Ferr at warm: {Ferr_sym(x_warm, 0.3, 121):.3e}')
F_and, x_and = anderson(F_func, x_warm, n_iter=80)
print(f'  After 80 Anderson: {min(F_and):.3e}')
t0 = time.time()
f_nk, x_nk, ok = nail_NK(F_func, x_and, tol=1e-15, maxiter=80)
dt = time.time() - t0
print(f'  After Newton-Krylov: {f_nk:.3e} ({"converged" if ok else "did NOT converge"}) [{dt:.1f}s]')

# Save best result
results['warm_h0.3_Gp121'] = dict(
    anderson_floor=float(min(F_and)),
    nk_final=f_nk, nk_converged=ok, time_NK=dt,
)
# Save the FP itself if good
if f_nk < 1e-10:
    P_FP_kern = expand(x_nk)
    np.save('/tmp/cheby_h0/P_FP_kerntab.npy', P_FP_kern)
    print(f'  saved P_FP_kerntab.npy (range [{P_FP_kern.min():.4f}, {P_FP_kern.max():.4f}])')

# Best
best_recipe = min(results, key=lambda k: results[k]['nk_final'])
print(f'\n=== BEST FLOOR: {results[best_recipe]["nk_final"]:.3e} '
      f'(recipe={best_recipe}) ===')
json.dump(results, open('/tmp/cheby_h0/kern_tab_results.json', 'w'),
            indent=2, default=str)
