"""Smaller (N, NQ) sweep that WRITES INCREMENTALLY so timeouts preserve
partial results. Focus on N=8, 10 with NQ values centered on sweet
spots."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import C_STRETCH
from cheby_numba_kern_tab_N import make_grid_N

TAU = 1.0; GAMMA = 1.0
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    return s, float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))

def phi_N_NQ(P, N, NQ, G_p=121):
    G, lob, u_n, V_inv = make_grid_N(N)
    p_grid = make_p_grid(G_p)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_inv, lob, u_n, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)

def anderson(F_func, x0, n_iter=60, m=10):
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

OUT = '/tmp/cheby_h0/pou_N_NQ_v2.json'
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

# Warm up at all N (this triggers numba compilation per (N, NQ) signature)
print('Warming up numba (may take a few minutes)...', flush=True)
for N in [6, 8]:
    G, lob, u_n, V_inv = make_grid_N(N)
    U1, U2, U3 = np.meshgrid(u_n, u_n, u_n, indexing='ij')
    P0 = sg(0.5*TAU*(U1+U2+U3))
    for NQ in [14, 16, 18, 20]:
        _ = phi_N_NQ(P0, N, NQ)
        print(f'  warm N={N}, NQ={NQ}', flush=True)
print('warmup done\n', flush=True)

print(f'{"N":>3} {"G":>3} {"NQ":>3} {"And":>12} {"NK":>12} {"slope":>9} {"def":>9}',
      flush=True)
for N in [6, 8]:
    G, lob, u_n, V_inv = make_grid_N(N)
    U1, U2, U3 = np.meshgrid(u_n, u_n, u_n, indexing='ij')
    T = TAU*(U1+U2+U3)
    P0 = sg(0.5*T)
    for NQ in [14, 16, 18, 20]:
        key = f'N={N},NQ={NQ}'
        if key in results: continue
        F_func = lambda x_flat, N=N, NQ=NQ, G=G: \
            (phi_N_NQ(x_flat.reshape(G,G,G), N, NQ)
             - x_flat.reshape(G,G,G)).ravel()
        Fs, x_a = anderson(F_func, P0.ravel(), n_iter=60)
        try:
            x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=20,
                                   verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
            if f_nk < min(Fs): x_a = x_nk
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
        P_fp = x_a.reshape(G, G, G)
        s, d = fit(P_fp, T)
        results[key] = dict(N=N, G=G, NQ=NQ, anderson=float(min(Fs)),
                              nk=f_nk, slope=s, deficit=d)
        json.dump(results, open(OUT, 'w'), indent=2, default=str)
        print(f'{N:>3} {G:>3} {NQ:>3} {min(Fs):>12.3e} {f_nk:>12.3e} '
              f'{s:>9.4f} {d:>9.4f}', flush=True)

print('\nSweep complete.', flush=True)
print('\n=== Best NQ per N ===')
for N in [6, 8]:
    best_nq = None; best_nk = float('inf')
    for k, r in results.items():
        if r['N'] == N and r['nk'] < best_nk:
            best_nq = r['NQ']; best_nk = r['nk']
    if best_nq:
        r = results[f'N={N},NQ={best_nq}']
        print(f'  N={N} (G={N+1}): best NQ={best_nq}, NK={best_nk:.3e}, '
              f'slope={r["slope"]:.4f}, deficit={r["deficit"]:.4f}')
