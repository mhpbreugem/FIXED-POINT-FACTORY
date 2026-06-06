"""Push kernel-band joint limit harder. At small h the GL quadrature
needs many more nodes to resolve the narrow kernel band — let NQK grow
with h. Goal: get the kernel-band slope closer to the strict-h=0 value
of 0.366 (from POU)."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_numba_kern_tab_N import phi_kern_tab_N, make_grid_N

TAU = 1.0; GAMMA = 1.0
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    return s, float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))

def anderson(F_func, x0, n_iter=80, m=10):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-14: break
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


# Pin N=8, push h with NQK scaling
N = 8
G, lob, u, V_inv = make_grid_N(N)
U1, U2, U3 = np.meshgrid(u, u, u, indexing='ij')
T = TAU*(U1+U2+U3)
P0 = sg(0.5*T)

# Warmup
_ = phi_kern_tab_N(P0, N, kernel_h=0.5, G_p=121, NQK=16)
print(f'=== Push kernel-band joint limit at N={N} ===\n')
print(f'{"h":>6} {"NQK":>4} {"G_p":>5} {"And floor":>12} {"NK floor":>12} {"slope":>9} {"deficit":>10} {"t":>6}')

x_prev = P0.ravel().copy()
results = {}
H_PATH = [0.50, 0.35, 0.25, 0.18, 0.13, 0.10, 0.08, 0.06, 0.04]
for h in H_PATH:
    # NQK scales as ~ 1/h (more nodes when kernel is narrow)
    NQK = max(16, int(np.ceil(3.0 / h)))
    G_p = max(121, int(np.ceil(50/h)))
    F_func = lambda x_flat, h=h, NQK=NQK, G_p=G_p: \
        (phi_kern_tab_N(x_flat.reshape(G,G,G), N, kernel_h=h, G_p=G_p,
                          NQK=NQK, tau=TAU, gamma=GAMMA)
          - x_flat.reshape(G,G,G)).ravel()
    t0 = time.time()
    Fs, x_a = anderson(F_func, x_prev, n_iter=120)
    f_and = min(Fs)
    if f_and > 1e-12:
        try:
            x_nk = newton_krylov(F_func, x_a, f_tol=1e-14, maxiter=30,
                                   verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
            if f_nk < f_and:
                f_and = f_nk; x_a = x_nk
        except NoConvergence as e:
            x_a = e.args[0]
            f_and = float(np.max(np.abs(F_func(x_a))))
    dt = time.time() - t0
    P_fp = x_a.reshape(G, G, G)
    slope, deficit = fit(P_fp, T)
    results[f'h={h}'] = dict(h=h, NQK=NQK, G_p=G_p, Ferr=float(f_and),
                                slope=slope, deficit=deficit, t=dt)
    print(f'{h:>6.2f} {NQK:>4d} {G_p:>5d} {f_and:>12.3e} {f_and:>12.3e} '
          f'{slope:>9.4f} {deficit:>10.4e} {dt:>5.1f}s', flush=True)
    if f_and < 1e-10:
        x_prev = x_a.copy()

json.dump(results, open('/tmp/cheby_h0/push_kern_h0.json', 'w'),
            indent=2, default=str)
print('saved push_kern_h0.json')
print('\nTarget: slope ~ 0.366 (POU strict h=0 at N=8)')
