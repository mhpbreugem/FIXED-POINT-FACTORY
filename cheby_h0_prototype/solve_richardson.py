"""Solve Richardson-extrapolated tabulated mu, gamma=1, tau=1.
Compare slope alpha* and one-to-one deficit to strict-h=0 POU baseline."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_richardson_tab import phi_richardson
from cheby_numba import U_NODES, TAU, GAMMA
from cheby_sym2 import expand, contract

G = 7
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_full = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))
P0 = sg(0.5*T_full)


def fit_metrics(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    slope = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = slope*Tf + np.mean(L - slope*Tf)
    R2 = 1 - float(np.sum((L-pred)**2) / np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return slope, 1-R2, w/ss


def solve_at(hs, label, n_anderson=80):
    print(f'\n{"="*60}')
    print(f'=== {label}: hs={hs} ===')
    print(f'{"="*60}')
    _ = phi_richardson(P0, hs=hs, G_p=121, NQK=16)  # warmup
    t0 = time.time()
    _ = phi_richardson(P0, hs=hs, G_p=121, NQK=16)
    t_phi = time.time() - t0
    print(f'  per-Phi: {t_phi*1000:.1f} ms')

    def F(x):
        return contract(phi_richardson(expand(x), hs=hs, G_p=121, NQK=16)) - x

    x = contract(P0).copy()
    print(f'  Initial |F| = {float(np.max(np.abs(F(x)))):.3e}')

    # Anderson
    Xh, Gh = [], []; Fs = []; x_best = x.copy(); f_best = float('inf')
    for it in range(n_anderson):
        t1 = time.time()
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-15:
            print(f'  Anderson it {it+1}: |F|={f:.3e} CONVERGED ({time.time()-t1:.2f}s)')
            break
        if (it+1) % 5 == 0 or it < 5:
            print(f'  Anderson it {it+1:3d}: |F|={f:.3e} ({time.time()-t1:.2f}s)')
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
    print(f'  Anderson best: {min(Fs):.3e}')

    # Newton-Krylov
    print(f'  Newton-Krylov:')
    nk_it = [0]
    def cb(xx, ff):
        nk_it[0] += 1
        print(f'    NK it {nk_it[0]:3d}: |F|={float(np.max(np.abs(F(xx)))):.3e}',
              flush=True)
    try:
        x_sol = newton_krylov(F, x_best, f_tol=1e-15, maxiter=20,
                                callback=cb, verbose=False)
        f_nk = float(np.max(np.abs(F(x_sol))))
    except NoConvergence as e:
        x_sol = e.args[0]
        f_nk = float(np.max(np.abs(F(x_sol))))
    best = min(min(Fs), f_nk)
    P_fp = expand(x_sol if f_nk < min(Fs) else x_best)
    s, def_lin, def_1to1 = fit_metrics(P_fp, T_full)
    print(f'\n  ===> {label}: best |F|={best:.3e}, slope={s:.4f}, '
          f'def_lin={def_lin:.4f}, def_1to1={def_1to1:.4f}')
    return best, s, def_lin, def_1to1


print('=== Richardson extrapolation test at gamma=tau=1 ===\n')
print('Baseline references:')
print('  Cheb strict POU (NQ=16):       slope=0.345, def_1to1=0.029')
print('  Lin-CDF kernel (h=0.5):        slope=0.296, def_1to1=0.0001')

# Test single h (no Richardson) — sanity baseline
solve_at((0.5,), label='SINGLE h=0.5 (no Richardson)')

# 2-point Richardson
solve_at((0.5, 0.3), label='2-pt Richardson (h={0.5, 0.3}, cancels O(h^2))')

# 3-point Richardson
solve_at((0.5, 0.3, 0.2), label='3-pt Richardson (h={0.5, 0.3, 0.2}, cancels O(h^4))')

# 4-point Richardson
solve_at((0.7, 0.5, 0.3, 0.2), label='4-pt Richardson (h={0.7, 0.5, 0.3, 0.2}, cancels O(h^6))')
