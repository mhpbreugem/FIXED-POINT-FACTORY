"""Gamma sweep with Richardson-extrapolated kernel-tab Cheb operator.
3-pt Richardson h = {0.5, 0.3, 0.2}, NQK=16, G=7."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_richardson_tab import phi_richardson
from cheby_numba import U_NODES
from cheby_sym2 import expand, contract

G = 7
TAU = 1.0
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_full = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2 = 1 - float(np.sum((L-pred)**2) / np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2, w/ss

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

# Warmup
_ = phi_richardson(sg(0.5*T_full), hs=(0.5, 0.3, 0.2), gamma=1.0, tau=TAU)

OUT = '/tmp/cheby_h0/richardson_gamma.json'
FPS = '/tmp/cheby_h0/fps_richardson'
os.makedirs(FPS, exist_ok=True)
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

GAMMAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
HS = (0.5, 0.3, 0.2)  # 3-pt Richardson, cancels O(h^4)
print(f'=== Richardson gamma sweep, hs={HS} ===')
print(f'{"gamma":>6} {"F":>12} {"slope":>8} {"def_lin":>9} {"def_1to1":>10} {"t":>6}',
      flush=True)
for gamma in GAMMAS:
    key = f'gamma={gamma}'
    if key in results: continue
    def F(x):
        return contract(phi_richardson(expand(x), hs=HS, gamma=gamma, tau=TAU)) - x
    t0 = time.time()
    x0 = contract(sg(0.5*T_full))
    Fs, x_a = anderson(F, x0, n_iter=60)
    try:
        x_nk = newton_krylov(F, x_a, f_tol=1e-15, maxiter=20, verbose=False)
        f_nk = float(np.max(np.abs(F(x_nk))))
        if f_nk < min(Fs): x_a = x_nk
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
    best_F = min(min(Fs), f_nk)
    P_fp = expand(x_a)
    s, def_lin, def_1to1 = fit(P_fp, T_full)
    np.save(f'{FPS}/P_FP_rich_gamma{gamma}.npy', P_fp)
    results[key] = dict(gamma=gamma, F=float(best_F), slope=s,
                          deficit_lin=def_lin, deficit_oneToOne=def_1to1,
                          t_solve=time.time()-t0)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)
    print(f'{gamma:>6.2f} {best_F:>12.3e} {s:>8.4f} {def_lin:>9.4f} '
          f'{def_1to1:>10.4f} {time.time()-t0:>5.1f}s', flush=True)
print('\nDone.')
