"""Gamma sweep with auto-NQ per gamma. Saves FPs and per-gamma data."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid, build_mu_table_pou_cr
from cheby_numba import (V_INV, LOBATTO, U_NODES, C_STRETCH, N_GRID,
                            vals_to_coeffs_3d_jit)
from cheby_sym2 import expand, contract

G = N_GRID
TAU = 1.0
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

# JIT warmup
print('JIT warmup...', flush=True)
T_full = TAU*(U1+U2+U3)
P0 = sg(0.5*T_full)
for NQ in [12, 14, 16, 18, 20, 24]:
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    _ = phi_pou_cr_jit(P0, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                          TAU, 1.0, C_STRETCH, G, NQ)
print('warmup done\n', flush=True)


def anderson(F_func, x0, n_iter=60):
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


OUT = '/tmp/cheby_h0/gamma_sweep_data.json'
FPS_DIR = '/tmp/cheby_h0/fps_gamma'
os.makedirs(FPS_DIR, exist_ok=True)
data = {}
if os.path.exists(OUT):
    data = json.load(open(OUT))

GAMMAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
print(f'{"gamma":>6} {"NQ":>4} {"F":>12} {"slope":>8} {"def":>8} {"t":>6}',
      flush=True)
for gamma in GAMMAS:
    key = f'gamma={gamma}'
    if key in data: continue
    best_F = float('inf'); best_NQ = None; best_x = None; best_Fs = None
    t0 = time.time()
    for NQ in [16, 14, 18, 20, 12, 22]:
        gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
        def phi(P, gl_n=gl_n, gl_w=gl_w, NQ=NQ):
            return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n,
                                     gl_w, TAU, gamma, C_STRETCH, G, NQ)
        F_func = lambda x: contract(phi(expand(x))) - x
        x0 = contract(sg(0.5*T_full))
        Fs, x_a = anderson(F_func, x0, n_iter=60)
        try:
            x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=15,
                                   verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
            if f_nk < min(Fs): x_a = x_nk
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
        f_nq = min(min(Fs), f_nk)
        if f_nq < best_F:
            best_F = f_nq; best_NQ = NQ; best_x = x_a; best_Fs = Fs
        if best_F < 1e-13: break
    P_fp = expand(best_x)
    s, d = fit(P_fp, T_full)
    np.save(f'{FPS_DIR}/P_FP_gamma{gamma}.npy', P_fp)
    gl_n, gl_w = np.polynomial.legendre.leggauss(best_NQ)
    coeffs = vals_to_coeffs_3d_jit(P_fp, V_INV)
    mu_table = build_mu_table_pou_cr(coeffs, LOBATTO, U_NODES, p_grid,
                                         gl_n, gl_w, TAU, C_STRETCH, G, best_NQ)
    np.save(f'{FPS_DIR}/mu_table_gamma{gamma}.npy', mu_table)
    data[key] = dict(gamma=gamma, NQ=best_NQ, F_final=float(best_F),
                       slope=s, deficit=d,
                       anderson_history=best_Fs,
                       t_solve=time.time()-t0)
    json.dump(data, open(OUT, 'w'), indent=2, default=str)
    print(f'{gamma:>6.2f} {best_NQ:>4d} {best_F:>12.3e} {s:>8.4f} {d:>8.4f} {time.time()-t0:>5.1f}s',
          flush=True)

print('\nSweep complete.', flush=True)
