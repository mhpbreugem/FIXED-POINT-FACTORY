"""Find the NQ sweet spot for POU at G=7. NQ=16 hit machine eps; check
neighbors and verify FP shape."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    return s, float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))

def phi_nq(P, NQ, G_p=121, p_grid=None):
    if p_grid is None: p_grid = make_p_grid(G_p)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)

def anderson(F_func, x0, n_iter=80):
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

print('=== Sweep NQ around sweet spot 16 ===')
for nq in [13, 14, 15, 16, 17, 18, 19, 20]:
    _ = phi_nq(sg(0.5*T), nq)
print(f'{"NQ":>4} {"And":>12} {"NK":>12} {"slope":>9} {"def":>9} {"t/Phi":>9}')
results = {}
x_cold = contract(sg(0.5*T))
for nq in [13, 14, 15, 16, 17, 18, 19, 20]:
    F_func = lambda x, nq=nq: contract(phi_nq(expand(x), nq)) - x
    t0 = time.time(); _ = F_func(x_cold); t_phi = time.time()-t0
    Fs, x_a = anderson(F_func, x_cold, n_iter=50)
    try:
        x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=20, verbose=False)
        f_nk = float(np.max(np.abs(F_func(x_nk))))
        if f_nk < min(Fs): x_a = x_nk
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
    P_fp = expand(x_a)
    s, d = fit(P_fp, T)
    results[nq] = dict(anderson=float(min(Fs)), nk=f_nk, slope=s, deficit=d)
    print(f'{nq:>4d} {min(Fs):>12.3e} {f_nk:>12.3e} {s:>9.4f} {d:>9.4f} {t_phi*1000:>7.1f}ms')

json.dump(results, open('/tmp/cheby_h0/sweet_nq.json', 'w'), indent=2, default=str)

# Save the best FP
best_nq = min(results, key=lambda nq: results[nq]['nk'])
print(f'\nBest NQ = {best_nq}, NK floor = {results[best_nq]["nk"]:.3e}')
print(f'         slope = {results[best_nq]["slope"]:.6f}')
print(f'         deficit = {results[best_nq]["deficit"]:.6f}')
