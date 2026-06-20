"""Auto-NQ-finder: for each (tau, gamma), try NQ in some range and
report which converges to machine eps. Saves results incrementally."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import V_INV, LOBATTO, U_NODES, C_STRETCH, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')

def sg(x): return 1/(1+np.exp(-x))
def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    return s, float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))

G_p = 121
p_grid = make_p_grid(G_p)

def phi(P, tau, gamma, NQ):
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                             tau, gamma, C_STRETCH, G, NQ)
def F(x, tau, gamma, NQ): return contract(phi(expand(x), tau, gamma, NQ)) - x

# Warmup
for nq in [12, 14, 16, 18, 20, 22, 24]:
    _ = phi(sg(0.5*(U1+U2+U3)), 1.0, 1.0, nq)

def anderson(F_func, x0, n_iter=50):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        Fv = F_func(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
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

OUT = '/tmp/cheby_h0/auto_nq.json'
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

print(f'=== Auto-NQ finder at G={G}, G_p={G_p}, sweep (tau,gamma) ===\n')

TAUS = [0.5, 1.0, 2.0, 3.0]
GAMMAS = [0.5, 1.0, 2.0, 5.0]
NQs = [12, 14, 16, 18, 20, 22, 24, 28]

print(f'{"tau":>5} {"gamma":>6} {"best NQ":>8} {"|F|":>12} {"slope":>10}', flush=True)
for tau in TAUS:
    T_full = tau*(U1+U2+U3)
    for gamma in GAMMAS:
        key = f'tau={tau},gamma={gamma}'
        if key in results: continue
        x0 = contract(sg(0.5*T_full))
        best_NQ = None; best_F = float('inf'); best_x = x0
        for NQ in NQs:
            F_func = lambda x, NQ=NQ: F(x, tau, gamma, NQ)
            Fs, x_a = anderson(F_func, x0, n_iter=40)
            try:
                x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=15,
                                       verbose=False)
                f_nk = float(np.max(np.abs(F_func(x_nk))))
                if f_nk < min(Fs): x_a = x_nk
            except NoConvergence as e:
                x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
            best_for_NQ = min(min(Fs), f_nk)
            if best_for_NQ < best_F:
                best_F = best_for_NQ; best_NQ = NQ; best_x = x_a
        s, d = fit(expand(best_x), T_full)
        results[key] = dict(tau=tau, gamma=gamma, best_NQ=best_NQ,
                              best_F=float(best_F), slope=s, deficit=d)
        json.dump(results, open(OUT, 'w'), indent=2, default=str)
        print(f'{tau:>5.1f} {gamma:>6.2f} {best_NQ:>8d} {best_F:>12.3e} {s:>10.4f}',
              flush=True)

print('\nDone.')
