"""Pure dense Newton on the POU Cheb-tab operator (40-DOF sym subspace).

Step 1: warm-start from POU Anderson best (~6e-3 from cold)
Step 2: dense Newton with FD Jacobian, sweep eps_FD
Step 3: if FD-Newton hits a noise floor, that justifies escalating to
        complex-step (exact analytic Jacobian via Im[F(x+i*h*e_j)]/h).
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from cheby_numba_pou_tab import phi_pou_tab, make_p_grid
from cheby_numba import U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

p_grid = make_p_grid(121)
def F_sym(x):
    return contract(phi_pou_tab(expand(x), G_p=121, p_grid=p_grid)) - x

def Ferr(x):
    return float(np.max(np.abs(F_sym(x))))

# Warmup
_ = phi_pou_tab(sg(0.5*T), G_p=121, p_grid=p_grid)

# Quick Anderson to warm into the basin
def anderson(x0, n_iter=40, m=10):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_sym(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
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

print('Warming up with Anderson from cold...', flush=True)
x_cold = contract(sg(0.5*T))
F_and, x_warm = anderson(x_cold, n_iter=40)
print(f'Anderson best: {min(F_and):.3e}', flush=True)

def newton(x0, eps_fd, n_iter=30, stall_patience=8):
    x = x0.copy(); Fs = []
    x_best = x.copy(); f_best = float('inf')
    print(f'  Newton (eps_FD={eps_fd:.0e}):', flush=True)
    for it in range(n_iter):
        t0 = time.time()
        F0 = F_sym(x)
        f = float(np.max(np.abs(F0))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-14:
            print(f'    iter {it+1}: F={f:.3e} CONVERGED'); break
        # FD Jacobian
        J = np.empty((x.size, x.size))
        for j in range(x.size):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (F_sym(xp) - F0) / eps_fd
        try: dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError: dx = -np.linalg.pinv(J) @ F0
        # Armijo line search (weak)
        alpha = 1.0
        for ls in range(30):
            xn = x + alpha*dx
            f_new = float(np.max(np.abs(F_sym(xn))))
            if f_new < (1 - 1e-4*alpha)*f: break
            alpha *= 0.5
        x = xn
        dt = time.time() - t0
        print(f'    iter {it+1:2d}: F={f:.3e}  alpha={alpha:.2e}  ({dt:.1f}s)',
              flush=True)
        if it >= stall_patience and Fs[-1] > 0.95*Fs[-(stall_patience+1)]:
            print(f'    STALLED'); break
    return Fs, x_best

results = {}
print('\nFD Newton sweep over eps_FD on POU warm start:')
for eps in [1e-5, 1e-6, 1e-7, 1e-8]:
    Fs, x_nw = newton(x_warm, eps, n_iter=20, stall_patience=6)
    results[f'eps_{eps:.0e}'] = dict(min_F=float(min(Fs)), Fs=Fs)
    print(f'  eps={eps:.0e}: best F = {min(Fs):.3e}', flush=True)

best_eps = min(results, key=lambda k: results[k]['min_F'])
best_F = results[best_eps]['min_F']
print(f'\n=== BEST: F={best_F:.3e} with eps_FD={best_eps} ===')
print(f'(cold-Anderson floor was {min(F_and):.3e};')
print(f' if Newton barely improves, the operator is the bottleneck — escalate to complex-step.)')

json.dump(dict(anderson_floor=min(F_and), newton_results=results),
            open('/tmp/cheby_h0/pou_dense_newton.json', 'w'),
            indent=2, default=str)
print('\nsaved pou_dense_newton.json')
