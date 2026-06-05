"""γ-continuation gold test at τ=2.
Track the deep PR branch from γ=1 down to γ=0.1 (where prior session reported slope=0.36).
Uses numba operator + S₃×Z₂ symmetric unlifted Newton.
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

TAU_C = 2.0
GAMMA_SCHEDULE = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]

from cheby_numba import phi as phi_jit_wrap, U_NODES, LOBATTO, C_STRETCH, N, N_GRID, crra_clear
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit, expand, contract

G = N_GRID

def phi_sym(x_small, gamma, tau=TAU_C):
    P = expand(x_small)
    P_new = phi_jit_wrap(P, gamma=gamma, tau=tau)
    P_sym = np.empty_like(P_new)
    for ridx, rep in enumerate(FREE_REPS):
        s3, z2_s3 = full_orbit(*rep)
        vals = [P_new[c] for c in s3] + [1.0 - P_new[c] for c in z2_s3]
        avg = float(np.mean(vals))
        for c in s3: P_sym[c] = avg
        for c in z2_s3: P_sym[c] = 1.0 - avg
    for rep in FIXED_REPS:
        s3, z2_s3 = full_orbit(*rep)
        for c in s3 | z2_s3:
            P_sym[c] = 0.5
    return contract(P_sym)

def metrics(P, tau=TAU_C):
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = tau * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = 1.0/(1.0+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def newton_for_gamma(x_init, gamma, n_iter=8, eps_fd=1e-6, tol=1e-9):
    x = x_init.copy()
    F = phi_sym(x, gamma) - x; Fn = float(np.max(np.abs(F)))
    ferrs = [Fn]
    print(f'    init ||F||={Fn:.3e}', flush=True)
    for it in range(n_iter):
        ts = time.time()
        Nu = x.size
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_sym(xp, gamma) - xp - F) / eps_fd
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = np.clip(x + alpha*dx, 1e-12, 1-1e-12)
            Fnn = phi_sym(xn, gamma) - xn
            nn = float(np.max(np.abs(Fnn)))
            if nn < best[1]: best = (xn, nn, Fnn, alpha)
            if nn < (1 - 0.5*alpha)*Fn: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, Fn, F, alpha = best
        ferrs.append(Fn)
        m = metrics(expand(x))
        print(f'    NewtIter {it+1:2d}  ||F||={Fn:.3e}  aLS={alpha:.3g}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
        if Fn < tol: break
    return x, ferrs

if __name__ == '__main__':
    print(f'\n=== γ-CONTINUATION at τ={TAU_C} ===')
    print(f'Schedule: γ = {GAMMA_SCHEDULE}', flush=True)

    # Warm-start IC: NL Bayes at first γ
    def sg_np(x): return 1.0/(1.0+np.exp(-x))
    gamma0 = GAMMA_SCHEDULE[0]
    P_IC_full = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu0 = sg_np(TAU_C*U_NODES[i])
                mu1 = sg_np(TAU_C*U_NODES[j])
                mu2 = sg_np(TAU_C*U_NODES[k])
                P_IC_full[i,j,k] = crra_clear(mu0, mu1, mu2, gamma0)
    x = contract(P_IC_full)
    print(f'\nIC (NL Bayes, γ={gamma0}): {metrics(expand(x), TAU_C)}', flush=True)

    # JIT warm-up
    print('Triggering JIT...', flush=True)
    t0 = time.time()
    _ = phi_jit_wrap(np.full((G, G, G), 0.5), gamma=gamma0, tau=TAU_C)
    print(f'  JIT: {time.time()-t0:.1f}s', flush=True)

    history = []
    for gi, gamma in enumerate(GAMMA_SCHEDULE):
        print(f'\n--- γ-step {gi+1}/{len(GAMMA_SCHEDULE)}: γ={gamma} ---', flush=True)
        # Picard preconditioning at new gamma
        print(f'  Picard preconditioning (3 iters):', flush=True)
        for it in range(3):
            x_new = phi_sym(x, gamma)
            ferr_p = float(np.max(np.abs(x_new - x)))
            x = 0.5*x + 0.5*x_new
            m = metrics(expand(x), TAU_C)
            print(f'    Picard {it+1}  ||F||={ferr_p:.3e}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}', flush=True)
        # Newton
        print(f'  Newton:', flush=True)
        x, ferrs = newton_for_gamma(x, gamma, n_iter=8, tol=1e-9)
        m = metrics(expand(x), TAU_C)
        print(f'  γ={gamma} DONE: slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  final ||F||={ferrs[-1]:.3e}', flush=True)
        history.append(dict(gamma=gamma, x_final=x.tolist(), metrics=m,
                              final_F=ferrs[-1], ferrs=ferrs))

    np.save('/tmp/cheby_h0/x_final_continuation.npy', x)
    P_final = expand(x)
    np.save('/tmp/cheby_h0/P_final_continuation.npy', P_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU_C, c=C_STRETCH, schedule=GAMMA_SCHEDULE),
        history=history,
        final_gamma=GAMMA_SCHEDULE[-1],
        final_metrics=metrics(P_final, TAU_C),
        prior_session_slope_at_gamma_0p1=0.3641,
    ), open('/tmp/cheby_h0/results_continuation.json','w'), indent=2, default=str)
    print('\n=== ALL γ steps complete. saved ===')
    print(f'Final at γ={GAMMA_SCHEDULE[-1]}: {metrics(P_final, TAU_C)}')
    print(f'Prior session (different operator) at γ=0.1: slope ≈ 0.3641')
