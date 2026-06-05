"""Gold test (UNLIFTED): reproduce prior session's PR FP at τ=2, γ=0.1.

At γ=0.1, equilibrium is NOT well-approximated by σ(α·T + h) form (CRRA
demand at small γ is too steep). So we use the raw cell-by-cell solver
on S₃×Z₂ symmetric subspace (40 free DOFs), with numba'd inner phi.

Prior session: slope=0.3641 (PR), deep PR branch.
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

TAU_GOLD = 2.0
GAMMA_GOLD = 0.1

from cheby_numba import phi as phi_jit_wrap, U_NODES, LOBATTO, C_STRETCH, N, N_GRID, crra_clear
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit, expand, contract, CELL_TO_REP

G = N_GRID

def phi_gold(P_vals, gamma=GAMMA_GOLD, tau=TAU_GOLD):
    return phi_jit_wrap(P_vals, gamma=gamma, tau=tau)

def phi_sym(x_small, gamma=GAMMA_GOLD, tau=TAU_GOLD):
    """Symmetrized phi: expand → phi → project back to symmetric basis."""
    P = expand(x_small)
    P_new = phi_gold(P, gamma=gamma, tau=tau)
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

def metrics(P, tau=TAU_GOLD):
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = tau * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = 1.0/(1.0+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(x, n_iter=10, eps_fd=1e-6, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton SYM (unlifted gold): {Nu} unknowns', flush=True)
    F = phi_sym(x) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; mlist = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_sym(xp) - xp - F) / eps_fd
        t_build = time.time() - ts
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = np.clip(x + alpha*dx, 1e-12, 1-1e-12)
            Fn = phi_sym(xn) - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(expand(x)); ferrs.append(F_norm); mlist.append(m)
        print(f'  NewtIter {it+1:2d}  ||F||={F_norm:.3e}  aLS={alpha:.3g}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
        if F_norm < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, mlist


if __name__ == '__main__':
    print(f'\n=== UNLIFTED GOLD TEST: τ={TAU_GOLD}, γ={GAMMA_GOLD} ===')
    print('Triggering JIT compile...', flush=True)
    t0 = time.time()
    _ = phi_gold(np.full((G, G, G), 0.5))
    print(f'  JIT: {time.time()-t0:.1f}s', flush=True)

    # NL Bayes IC
    def sg_np(x): return 1.0/(1.0+np.exp(-x))
    P_IC_full = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu0 = sg_np(TAU_GOLD*U_NODES[i])
                mu1 = sg_np(TAU_GOLD*U_NODES[j])
                mu2 = sg_np(TAU_GOLD*U_NODES[k])
                P_IC_full[i,j,k] = crra_clear(mu0, mu1, mu2, GAMMA_GOLD)
    x_IC = contract(P_IC_full)
    P_IC = expand(x_IC)
    print(f'\nIC: no-learning Bayes')
    print(f'  IC metrics: {metrics(P_IC)}')

    print('\n=== Picard preconditioning (8 iters, ω=0.5) ===')
    x = x_IC.copy(); pferrs = []; pmetrics = []
    for it in range(8):
        ts = time.time()
        x_new = phi_sym(x)
        ferr = float(np.max(np.abs(x_new - x))); pferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); pmetrics.append(m)
        print(f'  Picard {it+1:2d}  ||F||={ferr:.3e}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)

    print('\n=== Pure dense Newton ===')
    x_final, nferrs, nmetrics = dense_newton(x, n_iter=10, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal: metrics={metrics(P_final)}')
    print(f'Prior session result at τ=2, γ=0.1: slope ≈ 0.3641')

    np.save('/tmp/cheby_h0/P_final_gold_unlifted.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_gold_unlifted.npy', x_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU_GOLD, gamma=GAMMA_GOLD, c=C_STRETCH,
                     n_unk_sym=len(FREE_REPS)),
        x_final=x_final.tolist(),
        picard=dict(ferrs=pferrs, metrics=pmetrics),
        newton=dict(ferrs=nferrs, metrics=nmetrics),
        final_metrics=metrics(P_final),
        prior_session_slope=0.3641,
    ), open('/tmp/cheby_h0/results_gold_unlifted.json','w'), indent=2, default=str)
    print('saved')
