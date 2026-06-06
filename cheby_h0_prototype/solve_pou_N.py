"""Solve POU Cheb-tab at various N."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_N import phi_pou_N, make_grid_N

def sg(x): return 1/(1+np.exp(-x))
def fit_slope_deficit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    slope = float(np.sum(L*T.ravel()) / np.sum(T.ravel()**2))
    pred = slope*T.ravel() + np.mean(L - slope*T.ravel())
    res = np.sum((L - pred)**2); tot = np.sum((L - L.mean())**2)
    return slope, float(res/max(tot, 1e-30))

def anderson(F_func, x0, n_iter=80, m=10, tol=1e-15):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
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

results = {}
for N in [6, 8, 10]:
    G, lob, u, V_inv = make_grid_N(N)
    U1, U2, U3 = np.meshgrid(u, u, u, indexing='ij')
    T = 1.0*(U1+U2+U3)
    P0 = sg(0.5*T)
    _ = phi_pou_N(P0, N, G_p=121)
    F_func = lambda x_flat, N=N, G=G: (phi_pou_N(x_flat.reshape(G,G,G), N,
                                                    G_p=121)
                                        - x_flat.reshape(G,G,G)).ravel()
    print(f'\n=== N={N} (G={G}) ===')
    x0 = P0.ravel()
    print(f'Initial F: {float(np.max(np.abs(F_func(x0)))):.3e}')
    t0 = time.time()
    Fs, x_and = anderson(F_func, x0, n_iter=80, tol=1e-14)
    print(f'Anderson best: {min(Fs):.3e} ({time.time()-t0:.1f}s)')
    # Newton-Krylov
    if min(Fs) > 1e-12:
        t0 = time.time()
        try:
            x_nk = newton_krylov(F_func, x_and, f_tol=1e-14, maxiter=40,
                                   verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
        except NoConvergence as e:
            x_nk = e.args[0]
            f_nk = float(np.max(np.abs(F_func(x_nk))))
        print(f'Newton-Krylov: {f_nk:.3e} ({time.time()-t0:.1f}s)')
    else:
        x_nk = x_and; f_nk = min(Fs)
    P_fp = x_nk.reshape(G, G, G) if f_nk < min(Fs) else x_and.reshape(G, G, G)
    slope, deficit = fit_slope_deficit(P_fp, T)
    results[N] = dict(N=N, G=G, anderson_floor=float(min(Fs)),
                        nk_floor=float(f_nk),
                        best=float(min(min(Fs), f_nk)),
                        slope=slope, deficit=deficit)
    print(f'  best: {results[N]["best"]:.3e}, slope={slope:.4f}, deficit={deficit:.4e}')

json.dump(results, open('/tmp/cheby_h0/pou_N_results.json', 'w'), indent=2, default=str)
print('\n=== Summary ===')
for N, r in results.items():
    print(f'N={N}: best={r["best"]:.3e}, slope={r["slope"]:.4f}, deficit={r["deficit"]:.4e}')
