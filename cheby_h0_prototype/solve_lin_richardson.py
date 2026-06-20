"""Solve Lin-CDF Richardson at gamma=1 to test."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_cdf_uniform_grid

G = 7
u_grid = make_cdf_uniform_grid(G)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
T_full = U1+U2+U3
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

# JIT warmup
_ = phi_lin_richardson(sg(0.5*T_full), u_grid, hs=(0.5, 0.3, 0.2), gamma=1.0)

for HS in [(0.5, 0.3), (0.5, 0.3, 0.2), (0.7, 0.5, 0.3, 0.2)]:
    print(f'\n{"="*60}')
    print(f'=== Lin-CDF Richardson hs={HS} ===')
    print(f'{"="*60}')
    def F(x):
        return (phi_lin_richardson(x.reshape(G,G,G), u_grid, hs=HS, gamma=1.0)
                 - x.reshape(G,G,G)).ravel()
    x = sg(0.5*T_full).ravel()
    print(f'Initial |F|={float(np.max(np.abs(F(x)))):.3e}')

    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(40):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if (it+1) % 5 == 0 or it < 3 or f < 1e-13:
            print(f'  And it {it+1:3d}: |F|={f:.3e}')
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
    print(f'Anderson best: {min(Fs):.3e}')

    try:
        x_nk = newton_krylov(F, x_best, f_tol=1e-15, maxiter=15, verbose=False)
        f_nk = float(np.max(np.abs(F(x_nk))))
        if f_nk < min(Fs): x_best = x_nk
        print(f'NK final: {f_nk:.3e}')
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F(x_nk))))
        print(f'NK noConv: {f_nk:.3e}')

    s, def_l, def_1to1 = fit(x_best.reshape(G,G,G), T_full)
    print(f'===> slope={s:.4f}, def_lin={def_l:.4f}, def_1to1={def_1to1:.4f}')
