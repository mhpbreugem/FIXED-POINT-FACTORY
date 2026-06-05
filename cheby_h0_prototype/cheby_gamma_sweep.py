"""γ-sweep at τ=1, starting from the converged τ=1, γ=1 PR FP.
Continues γ: 1.0 → 0.7 → 0.5 → 0.3 → 0.2 → 0.15 → 0.1.
This is the "easier" direction — at fixed τ, γ-continuation usually tracks
the PR branch (the prior session did exactly this at τ=2).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

TAU_C = 1.0
GAMMA_SCHED = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]

from cheby_numba import phi as phi_jit, U_NODES, LOBATTO, C_STRETCH, N, N_GRID
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU_C * (U1 + U2 + U3)

ORBIT_POS, ORBIT_NEG = [], []
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3)); ORBIT_NEG.append(list(z2_s3))

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

def expand(x):
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx, rep in enumerate(FREE_REPS):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(FREE_REPS))
    for idx, rep in enumerate(FREE_REPS):
        vals = [h_full[c] for c in ORBIT_POS[idx]] + [-h_full[c] for c in ORBIT_NEG[idx]]
        h_small[idx] = float(np.mean(vals))
    return np.concatenate([[alpha], h_small])

def phi_lift(x, gamma):
    P = expand(x)
    P_new = phi_jit(P, gamma=gamma, tau=TAU_C)
    return contract(P_new)

def metrics(P):
    T = T_FIELD
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def stage_newton(x, gamma, n_iter=5, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    F = phi_lift(x, gamma) - x; Fn = float(np.max(np.abs(F)))
    ferrs = [Fn]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_lift(xp, gamma) - xp - F) / eps_fd
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
            Fnn = phi_lift(xn, gamma) - xn
            nn = float(np.max(np.abs(Fnn)))
            if nn < best[1]: best = (xn, nn, Fnn, alpha)
            if nn < (1 - 0.5*alpha)*Fn: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, Fn, F, alpha = best
        ferrs.append(Fn)
        m = metrics(expand(x))
        print(f'      Newt {it+1}  ||F||={Fn:.3e}  aLS={alpha:.3g}  a_slope={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
        if Fn < tol: break
    return x, ferrs

if __name__ == '__main__':
    print('\n=== γ-SWEEP at τ=1, starting from PR FP ===', flush=True)
    x = np.load('/tmp/cheby_h0/x_final_lifted_numba.npy')
    print(f'  start: α={x[0]:.4f} (loaded τ=1,γ=1 PR FP)', flush=True)
    print('JIT warmup...', flush=True)
    _ = phi_jit(np.full((G, G, G), 0.5), gamma=1.0, tau=TAU_C)

    history = []
    for gi, gamma in enumerate(GAMMA_SCHED):
        print(f'\n--- γ={gamma} ({gi+1}/{len(GAMMA_SCHED)}) ---', flush=True)
        x, ferrs = stage_newton(x, gamma, n_iter=5, tol=1e-9)
        m = metrics(expand(x))
        print(f'  DONE γ={gamma}: slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  final||F||={ferrs[-1]:.3e}', flush=True)
        history.append(dict(gamma=gamma, x=x.tolist(), metrics=m, final_F=ferrs[-1], ferrs=ferrs))
        # Save incremental
        json.dump(dict(
            tau=TAU_C, gamma_sched=GAMMA_SCHED, history=history,
        ), open('/tmp/cheby_h0/results_gamma_sweep.json', 'w'), indent=2, default=str)

    np.save('/tmp/cheby_h0/x_final_gamma_sweep.npy', x)
    np.save('/tmp/cheby_h0/P_final_gamma_sweep.npy', expand(x))
    print('\n=== γ-sweep complete ===')
    print(f'  τ=1 γ-sweep results: {[(h["gamma"], h["metrics"]["slope_T"]) for h in history]}')
    print(f'  prior session γ-sweep at τ=2: g={[0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.225, 0.259]}  slope={[0.3599, 0.3601, 0.3603, 0.3611, 0.3626, 0.3641, 0.3668, 0.3708, 0.3728]}')
