"""Two-parameter continuation: τ=1, γ=1 PR FP → τ=2, γ=0.1 gold target.

Starts from the converged lifted FP at (τ=1, γ=1) and sweeps:
  Step A: τ: 1.0 → 1.5 → 2.0 (holding γ=1)
  Step B: γ: 1.0 → 0.7 → 0.5 → 0.3 → 0.2 → 0.15 → 0.1 (holding τ=2)
Newton at each stage, lifted form.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_numba import phi as phi_jit, U_NODES, LOBATTO, C_STRETCH, N, N_GRID
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
U_SUM = U1 + U2 + U3

ORBIT_POS, ORBIT_NEG = [], []
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3)); ORBIT_NEG.append(list(z2_s3))

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

def expand(x, tau):
    T_FIELD = tau * U_SUM
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx, rep in enumerate(FREE_REPS):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P, tau):
    T_FIELD = tau * U_SUM
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(FREE_REPS))
    for idx, rep in enumerate(FREE_REPS):
        vals = [h_full[c] for c in ORBIT_POS[idx]] + [-h_full[c] for c in ORBIT_NEG[idx]]
        h_small[idx] = float(np.mean(vals))
    return np.concatenate([[alpha], h_small])

def phi_lift(x, gamma, tau):
    P = expand(x, tau)
    P_new = phi_jit(P, gamma=gamma, tau=tau)
    return contract(P_new, tau)

def metrics(P, tau):
    T = tau * U_SUM
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def stage_newton(x, gamma, tau, n_iter=5, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    F = phi_lift(x, gamma, tau) - x; Fn = float(np.max(np.abs(F)))
    ferrs = [Fn]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_lift(xp, gamma, tau) - xp - F) / eps_fd
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
            Fnn = phi_lift(xn, gamma, tau) - xn
            nn = float(np.max(np.abs(Fnn)))
            if nn < best[1]: best = (xn, nn, Fnn, alpha)
            if nn < (1 - 0.5*alpha)*Fn: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, Fn, F, alpha = best
        ferrs.append(Fn)
        m = metrics(expand(x, tau), tau)
        print(f'      Newt {it+1}  ||F||={Fn:.3e}  aLS={alpha:.3g}  a_slope={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
        if Fn < tol: break
    return x, ferrs

if __name__ == '__main__':
    print('\n=== 2-PARAMETER CONTINUATION ===')
    print('  τ=1,γ=1 PR FP → τ=2,γ=0.1 gold target', flush=True)
    # Load converged lifted FP
    x_init = np.load('/tmp/cheby_h0/x_final_lifted_numba.npy')
    print(f'  start: α={x_init[0]:.4f} (loaded from x_final_lifted_numba.npy)', flush=True)
    print('JIT warmup...', flush=True)
    t0 = time.time()
    _ = phi_jit(np.full((G, G, G), 0.5), gamma=1.0, tau=1.0)
    print(f'  {time.time()-t0:.1f}s', flush=True)

    history = []

    # Stage A: τ continuation at γ=1
    tau_sched = [1.0, 1.2, 1.5, 1.75, 2.0]
    print('\n--- Stage A: τ continuation at γ=1 ---', flush=True)
    x = x_init.copy()
    for tau in tau_sched:
        print(f'  τ={tau}:', flush=True)
        # Re-contract: x is in old-τ coords, expand with new τ then re-contract
        if tau != 1.0:
            P_old = expand(x, 1.0 if tau==tau_sched[0] else tau_prev)
            x = contract(P_old, tau)
        tau_prev = tau
        x, ferrs = stage_newton(x, 1.0, tau, n_iter=5, tol=1e-9)
        m = metrics(expand(x, tau), tau)
        print(f'    DONE: slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  final||F||={ferrs[-1]:.3e}', flush=True)
        history.append(dict(tau=tau, gamma=1.0, x=x.tolist(), metrics=m, final_F=ferrs[-1]))

    # Stage B: γ continuation at τ=2
    gamma_sched = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
    print('\n--- Stage B: γ continuation at τ=2 ---', flush=True)
    for gamma in gamma_sched:
        if gamma == 1.0: continue  # already done in stage A
        print(f'  γ={gamma}:', flush=True)
        x, ferrs = stage_newton(x, gamma, 2.0, n_iter=6, tol=1e-9)
        m = metrics(expand(x, 2.0), 2.0)
        print(f'    DONE: slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  final||F||={ferrs[-1]:.3e}', flush=True)
        history.append(dict(tau=2.0, gamma=gamma, x=x.tolist(), metrics=m, final_F=ferrs[-1]))

    P_final = expand(x, 2.0)
    print(f'\n=== FINAL (τ=2, γ=0.1) ===', flush=True)
    print(f'  metrics: {metrics(P_final, 2.0)}')
    print(f'  prior session at τ=2, γ=0.1: slope ≈ 0.3641')

    np.save('/tmp/cheby_h0/P_final_2dcont.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_2dcont.npy', x)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau_final=2.0, gamma_final=0.1, c=C_STRETCH,
                     tau_schedule=tau_sched, gamma_schedule=gamma_sched),
        history=history,
        final_metrics=metrics(P_final, 2.0),
        prior_session_slope=0.3641,
    ), open('/tmp/cheby_h0/results_2dcont.json','w'), indent=2, default=str)
    print('saved')
