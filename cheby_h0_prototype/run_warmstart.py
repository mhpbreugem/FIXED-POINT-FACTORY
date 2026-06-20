"""Warm-start test: load saved FPs from the repo and check their F_err
under the various operators. If they're at machine-eps, the floor is just
a basin issue with the cold start. If they're at 3e-3, the floor is
structural."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import least_squares
from cheby_numba import phi as phi_chebroots, U_NODES, TAU, GAMMA, N_GRID
from cheby_numba_bisect import phi_bisect
from cheby_numba_tab import phi_tab
from cheby_sym2 import expand, contract

REPO = '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype'
G = N_GRID

U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# Load all available FPs of the right shape
FPs = {}
for name in ['P_final_gamma_sweep.npy', 'P_final_lifted_numba.npy',
              'P_final_sym2.npy', 'P_final_vbasis.npy']:
    p = np.load(f'{REPO}/{name}')
    if p.shape == (G, G, G):
        FPs[name] = p
print(f'Loaded {len(FPs)} candidate warm starts at G={G}')
for n, p in FPs.items():
    print(f'  {n}: range=[{p.min():.4f}, {p.max():.4f}]')

# Cold start for reference
P0_cold = sg(0.5*T)

print('\n=== Step 1: F_err of each saved point under chebroots and bisect ===')
print(f'{"warm start":<35s} {"||F_cr||_inf":>14s} {"||F_bi||_inf":>14s} {"||F_tab||_inf":>14s}')
print('-' * 80)
def F_err(P, op_func):
    return float(np.max(np.abs(op_func(P) - P)))

# Cold start
fc_cr = F_err(P0_cold, phi_chebroots)
fc_bi = F_err(P0_cold, phi_bisect)
fc_tab = F_err(P0_cold, lambda P: phi_tab(P, G_p=51))
print(f'{"cold (sigmoid(0.5T))":<35s} {fc_cr:>14.3e} {fc_bi:>14.3e} {fc_tab:>14.3e}')

# Warm starts
warm_stats = {}
for name, P in FPs.items():
    f_cr = F_err(P, phi_chebroots)
    f_bi = F_err(P, phi_bisect)
    f_tab = F_err(P, lambda Q: phi_tab(Q, G_p=51))
    warm_stats[name] = dict(F_chebroots=f_cr, F_bisect=f_bi, F_tab=f_tab)
    print(f'{name:<35s} {f_cr:>14.3e} {f_bi:>14.3e} {f_tab:>14.3e}')

# ===== Step 2: Run Anderson + LM + Newton from the best warm start =====
print('\n=== Step 2: Drive each warm start with LM, Anderson, pure Newton ===')

def anderson_sym(op, x0_sym, n_iter=80, m=8):
    x = x0_sym.copy()
    Xh, Gh = [], []
    Fs = []
    for it in range(n_iter):
        gx = contract(op(expand(x)))
        F = gx - x; Fs.append(float(np.max(np.abs(F))))
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
    return Fs, x

def lm_sym(op, x0_sym, max_nfev=300):
    history = []
    def F(x):
        f = contract(op(expand(x))) - x
        history.append(float(np.max(np.abs(f))))
        return f
    sol = least_squares(F, x0_sym, method='lm', max_nfev=max_nfev,
                          xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return history, sol

def newton_sym(op, x0_sym, n_iter=20, eps=1e-7):
    x = x0_sym.copy()
    Fs = []
    for it in range(n_iter):
        # Jacobian
        F0 = contract(op(expand(x))) - x
        Ferr = float(np.max(np.abs(F0)))
        Fs.append(Ferr)
        if Ferr < 1e-13: break
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += eps
            Fp = contract(op(expand(xp))) - xp
            J[:, j] = (Fp - F0) / eps
        try:
            dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(J) @ F0
        # Armijo
        alpha = 1.0
        for _ in range(30):
            xn = x + alpha*dx
            Fn = contract(op(expand(xn))) - xn
            if np.max(np.abs(Fn)) < (1-1e-4*alpha)*Ferr: break
            alpha *= 0.5
        x = xn
    return Fs, x

# Run from each warm start with phi_bisect
print(f'\n{"warm start":<35s} {"And.floor":>12s} {"LM.floor":>12s} {"Newton.floor":>14s}')
print('-'*80)

# Cold first
xc = contract(P0_cold)
F_and_cold, _ = anderson_sym(phi_bisect, xc, n_iter=80)
F_lm_cold, _ = lm_sym(phi_bisect, xc, max_nfev=300)
F_nw_cold, x_nw_cold = newton_sym(phi_bisect, contract(F_lm_cold and contract(np.zeros_like(xc)) or xc) if False else
                                     contract(expand(xc)), n_iter=15)
# Newton from LM output (better starting point)
x_lm_cold = lm_sym(phi_bisect, xc, max_nfev=300)[1].x
F_nw_lm_cold, _ = newton_sym(phi_bisect, x_lm_cold, n_iter=20)
print(f'{"COLD (sigmoid(0.5T))":<35s} {min(F_and_cold):>12.3e} {min(F_lm_cold):>12.3e} {min(F_nw_lm_cold):>14.3e}')

results = {'cold': dict(anderson=min(F_and_cold), lm=min(F_lm_cold),
                          newton_after_lm=min(F_nw_lm_cold))}

for name, P_warm in FPs.items():
    xw = contract(P_warm)
    F_and, _ = anderson_sym(phi_bisect, xw, n_iter=80)
    F_lm, sol_lm = lm_sym(phi_bisect, xw, max_nfev=300)
    F_nw_lm, _ = newton_sym(phi_bisect, sol_lm.x, n_iter=20)
    F_nw_warm, _ = newton_sym(phi_bisect, xw, n_iter=20)
    results[name] = dict(anderson=float(min(F_and)),
                         lm=float(min(F_lm)),
                         newton_direct=float(min(F_nw_warm)),
                         newton_after_lm=float(min(F_nw_lm)),
                         F_anderson=F_and, F_lm=F_lm, F_newton=F_nw_lm)
    print(f'{name:<35s} {min(F_and):>12.3e} {min(F_lm):>12.3e} {min(F_nw_lm):>14.3e}')

json.dump({k: {kk: vv if not isinstance(vv, list) else vv
               for kk, vv in v.items()}
           for k, v in results.items()},
           open('/tmp/cheby_h0/warmstart_results.json', 'w'),
           indent=2, default=str)
print('\nsaved warmstart_results.json')
