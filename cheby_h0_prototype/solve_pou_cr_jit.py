"""Solve POU+chebroots numba version: Anderson + Newton-Krylov + dense
Newton with FD Jacobian. Should be much cleaner than scan-bisect version
because chebroots finds ALL real roots."""
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

p_grid = make_p_grid(121)
def F_sym(x):
    return contract(phi_pou_cr(expand(x), G_p=121, p_grid=p_grid)) - x
def Ferr(x): return float(np.max(np.abs(F_sym(x))))

_ = phi_pou_cr(sg(0.5*T), G_p=121, p_grid=p_grid)  # warmup

def anderson(x0, n_iter=80, m=10, tol=1e-15, verbose=False):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        t0 = time.time()
        F = F_sym(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if verbose:
            print(f'  it {it+1:2d}: F={f:.3e}  ({time.time()-t0:.1f}s)',
                  flush=True)
        if f < tol: break
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


def newton_fd(x0, eps, n_iter=20, stall=8, verbose=False):
    x = x0.copy(); Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        t0 = time.time()
        F0 = F_sym(x)
        f = float(np.max(np.abs(F0))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-14: break
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += eps
            J[:, j] = (F_sym(xp) - F0) / eps
        try: dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError: dx = -np.linalg.pinv(J) @ F0
        alpha = 1.0
        for ls in range(30):
            xn = x + alpha*dx
            fn = float(np.max(np.abs(F_sym(xn))))
            if fn < (1-1e-4*alpha)*f: break
            alpha *= 0.5
        x = xn
        dt = time.time() - t0
        if verbose:
            print(f'  it {it+1:2d}: F={f:.3e}  alpha={alpha:.1e}  ({dt:.1f}s)',
                  flush=True)
        if it >= stall and Fs[-1] > 0.95*Fs[-(stall+1)]: break
    return Fs, x_best


print('=== POU+chebroots numba: solve to floor ===\n')
x0 = contract(sg(0.5*T))
print(f'Initial F (cold): {Ferr(x0):.3e}')

print('\nStage 1: Anderson 60 iters')
F_a, x_a = anderson(x0, n_iter=60, verbose=True)
print(f'  Anderson best: {min(F_a):.3e}')

print('\nStage 2: Newton-Krylov from Anderson best')
try:
    x_nk = newton_krylov(F_sym, x_a, f_tol=1e-14, maxiter=30, verbose=False)
    f_nk = Ferr(x_nk)
except NoConvergence as e:
    x_nk = e.args[0]; f_nk = Ferr(x_nk)
print(f'  NK: {f_nk:.3e}')

print('\nStage 3: FD-Newton from NK (eps_FD=1e-7)')
F_nf, x_nf = newton_fd(x_nk, eps=1e-7, n_iter=20, verbose=True)
print(f'  FD-Newton best: {min(F_nf):.3e}')

best = min(min(F_a), f_nk, min(F_nf))
print(f'\n=== OVERALL BEST: {best:.3e} ===')

json.dump(dict(anderson_floor=min(F_a), nk_floor=float(f_nk),
                 fdnewton_floor=min(F_nf), overall_best=best),
            open('/tmp/cheby_h0/pou_cr_jit_results.json', 'w'),
            indent=2, default=str)
print('saved pou_cr_jit_results.json')
