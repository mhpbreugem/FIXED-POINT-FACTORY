"""Live-streaming high-order Chebyshev strict h=0 POU solve.

For each N in {6, 8, 10, 12, 14}: print every Anderson iteration's Ferr
+ every NK iteration's residual. All in numba; lookup table architecture.

gamma=tau=1, NQ=16 (proven sweet spot at N=6).
"""
import sys, time, json, os
sys.stdout.reconfigure(line_buffering=True)
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
G_p = 121
p_grid = make_p_grid(G_p)
def sg(x): return 1/(1+np.exp(-x))

OUT = '/tmp/cheby_h0/live_high_order.json'
FPS = '/tmp/cheby_h0/fps_high_order'
os.makedirs(FPS, exist_ok=True)
all_results = {}
if os.path.exists(OUT):
    all_results = json.load(open(OUT))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2 = 1 - float(np.sum((L-pred)**2) / np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 8), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2, w/ss


def anderson_live(F_func, x0, n_iter=80, m=10, label=''):
    x = x0.copy(); Xh, Gh = [], []
    x_best = x.copy(); f_best = float('inf')
    Fs = []
    print(f'  {label}', flush=True)
    for it in range(n_iter):
        t0 = time.time()
        F = F_func(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        print(f'    Anderson it {it+1:3d}: |F|={f:.3e}  ({time.time()-t0:.2f}s)',
              flush=True)
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


def nk_live(F_func, x0, label=''):
    print(f'  {label}', flush=True)
    nk_iter = [0]
    def cb(x, f):
        nk_iter[0] += 1
        F_x = F_func(x)
        print(f'    NK it {nk_iter[0]:3d}: |F|={float(np.max(np.abs(F_x))):.3e}',
              flush=True)
    try:
        x_sol = newton_krylov(F_func, x0, f_tol=1e-15, maxiter=20,
                                callback=cb, verbose=False)
        return x_sol, float(np.max(np.abs(F_func(x_sol))))
    except NoConvergence as e:
        x_sol = e.args[0]
        return x_sol, float(np.max(np.abs(F_func(x_sol))))


NQ = 16
for N in [6, 8, 10, 12]:
    key = f'N={N},NQ={NQ}'
    if key in all_results and all_results[key].get('F', 1) < 1e-12:
        print(f'\n=== N={N} (already converged: F={all_results[key]["F"]:.2e}) — skip ==='); continue
    print(f'\n{"="*60}', flush=True)
    print(f'=== N={N} (G={N+1}), NQ={NQ}, gamma=tau=1 ===', flush=True)
    print(f'{"="*60}', flush=True)
    G, lob, u_n, V_inv = make_grid_N(N)
    U1, U2, U3 = np.meshgrid(u_n, u_n, u_n, indexing='ij')
    T_full = TAU*(U1+U2+U3)
    P0 = sg(0.5*T_full)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    # JIT warmup
    t0 = time.time()
    print(f'  JIT warmup...', flush=True, end='')
    _ = phi_pou_cr_jit(P0, V_inv, lob, u_n, p_grid, gl_n, gl_w,
                          TAU, GAMMA, C_STRETCH, G, NQ)
    print(f' done ({time.time()-t0:.1f}s)', flush=True)

    # Time one Phi
    t0 = time.time()
    _ = phi_pou_cr_jit(P0, V_inv, lob, u_n, p_grid, gl_n, gl_w,
                          TAU, GAMMA, C_STRETCH, G, NQ)
    t_phi = time.time() - t0
    print(f'  Per-Phi: {t_phi*1000:.1f} ms', flush=True)

    F_func = lambda x_flat, G=G: \
        (phi_pou_cr_jit(x_flat.reshape(G,G,G), V_inv, lob, u_n, p_grid,
                           gl_n, gl_w, TAU, GAMMA, C_STRETCH, G, NQ)
          - x_flat.reshape(G,G,G)).ravel()

    # Anderson
    t0_stage = time.time()
    Fs, x_a = anderson_live(F_func, P0.ravel(), n_iter=80,
                              label=f'Anderson 80 iters (Phi={t_phi*1000:.0f}ms ea)')
    print(f'  Anderson done in {time.time()-t0_stage:.1f}s, best |F|={min(Fs):.3e}',
          flush=True)

    # Newton-Krylov
    if min(Fs) > 1e-13:
        t0_stage = time.time()
        x_nk, f_nk = nk_live(F_func, x_a, label=f'Newton-Krylov nail')
        print(f'  NK done in {time.time()-t0_stage:.1f}s, |F|={f_nk:.3e}',
              flush=True)
        if f_nk < min(Fs): x_a = x_nk; best_F = f_nk
        else: best_F = min(Fs)
    else:
        best_F = min(Fs)

    # Final stats
    P_fp = x_a.reshape(G, G, G)
    s, def_l, def_np = fit(P_fp, T_full)
    np.save(f'{FPS}/P_FP_N{N}_gamma{GAMMA}.npy', P_fp)
    all_results[key] = dict(N=N, G=G, NQ=NQ, gamma=GAMMA, F=float(best_F),
                              slope=s, deficit_lin=def_l,
                              deficit_oneToOne=def_np,
                              t_phi_ms=t_phi*1000)
    json.dump(all_results, open(OUT, 'w'), indent=2, default=str)
    print(f'\n  ===> N={N}: |F|={best_F:.3e}, slope={s:.4f}, '
          f'def_lin={def_l:.4f}, def_1to1={def_np:.4f}',
          flush=True)

print('\nDone.', flush=True)
