"""Sweep G_p (table density) and N (grid) for POU+chebroots.
Goal: determine if floor is (a) p-table interpolation error or
(b) something else."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr, make_p_grid
from cheby_numba import U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def anderson(F_func, x0, n_iter=80, m=10):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-15: break
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

# Warmup
_ = phi_pou_cr(sg(0.5*T), G_p=121)

print('=== Sweep G_p at N=6 (POU+chebroots) ===\n')
print(f'{"G_p":>5} {"And floor":>12} {"NK floor":>12} {"t_iter":>8}')
results = {}
x_cold = contract(sg(0.5*T))
for G_p in [51, 121, 251, 501, 1001]:
    p_grid_l = make_p_grid(G_p)
    F_func = lambda x: contract(phi_pou_cr(expand(x), G_p=G_p, p_grid=p_grid_l)) - x
    t0 = time.time()
    Fs, x_a = anderson(F_func, x_cold, n_iter=50)
    t_iter = (time.time() - t0) / len(Fs)
    try:
        x_nk = newton_krylov(F_func, x_a, f_tol=1e-14, maxiter=20, verbose=False)
        f_nk = float(np.max(np.abs(F_func(x_nk))))
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
    results[f'G_p={G_p}'] = dict(anderson=float(min(Fs)), nk=f_nk,
                                    t_iter=t_iter)
    print(f'{G_p:>5d} {min(Fs):>12.3e} {f_nk:>12.3e} {t_iter*1000:>7.1f} ms')

json.dump(results, open('/tmp/cheby_h0/pou_gp_sweep.json', 'w'),
            indent=2, default=str)
print('saved pou_gp_sweep.json')
