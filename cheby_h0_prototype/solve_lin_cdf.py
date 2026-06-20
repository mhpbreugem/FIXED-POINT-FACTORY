"""Solve linear-CDF kernel-band operator + gamma sweep."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from lin_cdf_kern_tab import (phi_lin_kern_tab, make_cdf_uniform_grid,
                                  make_p_grid)

G = 7
u_grid = make_cdf_uniform_grid(G)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    Tf = T.ravel()
    slope_lin = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred_lin = slope_lin*Tf + np.mean(L - slope_lin*Tf)
    R2_lin = 1 - float(np.sum((L - pred_lin)**2) / np.sum((L - L.mean())**2))
    # Nonparam f(T) R²
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss_tot = float(np.sum((L - L.mean())**2)); within = 0.0
    for g in range(len(uT)):
        mask = (inv == g)
        within += float(np.sum((L[mask] - L[mask].mean())**2))
    R2_nonparam = 1 - within / ss_tot
    return slope_lin, 1 - R2_lin, 1 - R2_nonparam

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

TAU = 1.0
T_full = TAU*(U1+U2+U3)

# Warmup
print('JIT warmup...', flush=True)
for h in [0.5, 0.3, 0.2]:
    _ = phi_lin_kern_tab(sg(0.5*T_full), u_grid, kernel_h=h, G_p=121, NQK=16,
                            tau=TAU, gamma=1.0)
print('warmup done\n', flush=True)

OUT = '/tmp/cheby_h0/lin_cdf_gamma.json'
FPS = '/tmp/cheby_h0/fps_lin_cdf'
os.makedirs(FPS, exist_ok=True)
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

GAMMAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
HS = [0.5, 0.3, 0.2]
print(f'{"gamma":>6} {"h_best":>7} {"F":>12} {"slope_lin":>10} '
      f'{"def_lin":>9} {"def_oneToOne":>13} {"t":>6}', flush=True)
for gamma in GAMMAS:
    key = f'gamma={gamma}'
    if key in results: continue
    best_F = float('inf'); best_h = None; best_x = None
    t0 = time.time()
    for h in HS:
        F_func = lambda x_flat, h=h, gamma=gamma: \
            (phi_lin_kern_tab(x_flat.reshape(G,G,G), u_grid, kernel_h=h,
                                G_p=121, NQK=16, tau=TAU, gamma=gamma)
              - x_flat.reshape(G,G,G)).ravel()
        x0 = sg(0.5*T_full).ravel().copy()
        Fs, x_a = anderson(F_func, x0, n_iter=60)
        try:
            x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=20,
                                   verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
            if f_nk < min(Fs): x_a = x_nk
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
        f_h = min(min(Fs), f_nk)
        if f_h < best_F:
            best_F = f_h; best_h = h; best_x = x_a.copy()
        if best_F < 1e-13: break
    P_fp = best_x.reshape(G, G, G)
    slope, def_lin, def_oneToOne = fit(P_fp, T_full)
    np.save(f'{FPS}/P_FP_lin_gamma{gamma}.npy', P_fp)
    results[key] = dict(gamma=gamma, h=best_h, F=float(best_F),
                          slope=slope, deficit_lin=def_lin,
                          deficit_oneToOne=def_oneToOne,
                          t_solve=time.time()-t0)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)
    print(f'{gamma:>6.2f} {best_h:>7.2f} {best_F:>12.3e} {slope:>10.4f} '
          f'{def_lin:>9.4f} {def_oneToOne:>13.4f} {time.time()-t0:>5.1f}s',
          flush=True)

print('\nSweep complete.')
