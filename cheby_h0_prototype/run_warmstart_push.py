"""Push the best warm starts as hard as we can:
chain Anderson -> LM -> Newton, try chebroots as the more-accurate operator,
also try cross-operator strategy (warm with one op, refine with another).
"""
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

# Load best warm starts
P_VBASIS = np.load(f'{REPO}/P_final_vbasis.npy')
P_GAMMA = np.load(f'{REPO}/P_final_gamma_sweep.npy')
P_LIFTED = np.load(f'{REPO}/P_final_lifted_numba.npy')
P_SYM2 = np.load(f'{REPO}/P_final_sym2.npy')

starts = {
    'COLD': sg(0.5*T),
    'vbasis': P_VBASIS,
    'gamma_sweep': P_GAMMA,
    'lifted': P_LIFTED,
    'sym2': P_SYM2,
}

# Warmup operators
for op_func in [phi_bisect, phi_chebroots, lambda P: phi_tab(P, G_p=51)]:
    _ = op_func(starts['COLD'])

# Solvers
def anderson_sym(op, x0_sym, n_iter=120, m=10, regul=1e-10):
    x = x0_sym.copy()
    Xh, Gh = [], []
    Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        gx = contract(op(expand(x)))
        F = gx - x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < f_best:
            f_best = Ferr; x_best = x.copy()
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > m: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + regul*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    return Fs, x_best

def lm_sym(op, x0_sym, max_nfev=500):
    history = []; x_best = x0_sym.copy(); f_best = float('inf')
    def F(x):
        f = contract(op(expand(x))) - x
        Ferr = float(np.max(np.abs(f)))
        history.append(Ferr)
        nonlocal x_best, f_best
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
        return f
    sol = least_squares(F, x0_sym, method='lm', max_nfev=max_nfev,
                          xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return history, x_best

def newton_sym(op, x0_sym, n_iter=30, eps=1e-7, stall_patience=8):
    x = x0_sym.copy()
    Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F0 = contract(op(expand(x))) - x
        Ferr = float(np.max(np.abs(F0))); Fs.append(Ferr)
        if Ferr < f_best: f_best = Ferr; x_best = x.copy()
        if Ferr < 1e-13: break
        if it >= stall_patience and Fs[-1] > 0.95*Fs[-(stall_patience+1)]:
            break
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += eps
            Fp = contract(op(expand(xp))) - xp
            J[:, j] = (Fp - F0) / eps
        try: dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError: dx = -np.linalg.pinv(J) @ F0
        alpha = 1.0
        for _ in range(30):
            xn = x + alpha*dx
            Fn = contract(op(expand(xn))) - xn
            if np.max(np.abs(Fn)) < (1-1e-4*alpha)*Ferr: break
            alpha *= 0.5
        x = xn
    return Fs, x_best

# Push protocol: chain Anderson -> LM -> Newton
print('=== Push protocol per warm start (op=phi_bisect) ===\n')
print(f'{"start":>15s} {"And120":>12s} {"+ LM500":>12s} {"+ Newton30":>14s}')
print('-' * 60)
results_bisect = {}
for name, P0 in starts.items():
    x0 = contract(P0)
    F_and, x_and = anderson_sym(phi_bisect, x0, n_iter=120)
    F_lm, x_lm = lm_sym(phi_bisect, x_and, max_nfev=500)
    F_nw, x_nw = newton_sym(phi_bisect, x_lm, n_iter=30)
    results_bisect[name] = dict(anderson=min(F_and), lm=min(F_lm), newton=min(F_nw),
                                  F_and=F_and, F_lm=F_lm, F_nw=F_nw)
    print(f'{name:>15s} {min(F_and):>12.3e} {min(F_lm):>12.3e} {min(F_nw):>14.3e}')

# Same with chebroots (more accurate but slower)
print('\n=== Push protocol per warm start (op=phi_chebroots) ===\n')
print(f'{"start":>15s} {"And80":>12s} {"+ LM200":>12s} {"+ Newton20":>14s}')
print('-' * 60)
results_cr = {}
for name, P0 in starts.items():
    if name == 'vbasis':  # Allow more iterations on the best start
        nA, nL, nN = 150, 500, 30
    else:
        nA, nL, nN = 80, 200, 20
    x0 = contract(P0)
    F_and, x_and = anderson_sym(phi_chebroots, x0, n_iter=nA)
    F_lm, x_lm = lm_sym(phi_chebroots, x_and, max_nfev=nL)
    F_nw, x_nw = newton_sym(phi_chebroots, x_lm, n_iter=nN)
    results_cr[name] = dict(anderson=min(F_and), lm=min(F_lm), newton=min(F_nw),
                              F_and=F_and, F_lm=F_lm, F_nw=F_nw)
    print(f'{name:>15s} {min(F_and):>12.3e} {min(F_lm):>12.3e} {min(F_nw):>14.3e}')

# Cross-operator: warm-and-Anderson with bisect (fast), then refine with chebroots
print('\n=== Cross-operator: bisect-Anderson then chebroots-Newton ===\n')
print(f'{"start":>15s} {"bi-And":>12s} {"+ cr-LM":>12s} {"+ cr-Newton":>14s}')
print('-' * 60)
results_cross = {}
for name, P0 in starts.items():
    x0 = contract(P0)
    F_and, x_and = anderson_sym(phi_bisect, x0, n_iter=120)
    F_lm, x_lm = lm_sym(phi_chebroots, x_and, max_nfev=300)
    F_nw, x_nw = newton_sym(phi_chebroots, x_lm, n_iter=20)
    results_cross[name] = dict(anderson=min(F_and), lm=min(F_lm), newton=min(F_nw),
                                 F_and=F_and, F_lm=F_lm, F_nw=F_nw)
    print(f'{name:>15s} {min(F_and):>12.3e} {min(F_lm):>12.3e} {min(F_nw):>14.3e}')

# Save
all_data = dict(bisect=results_bisect, chebroots=results_cr, cross=results_cross)
# Strip lists if large
for kk in all_data:
    for k, v in all_data[kk].items():
        all_data[kk][k] = {kkk: vvv for kkk, vvv in v.items()
                            if not isinstance(vvv, list) or len(vvv) <= 5000}
json.dump(all_data, open('/tmp/cheby_h0/warmstart_push.json', 'w'),
            indent=2, default=str)
print('\nsaved warmstart_push.json')

# Summary best floor
best_overall = float('inf'); best_recipe = None
for proto, res in all_data.items():
    for start, d in res.items():
        f = d['newton']
        if f < best_overall:
            best_overall = f; best_recipe = (proto, start)
print(f'\n=== BEST FLOOR ACHIEVED: {best_overall:.3e} '
      f'(protocol={best_recipe[0]}, start={best_recipe[1]}) ===')
