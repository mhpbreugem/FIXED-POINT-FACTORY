"""Picard + pure (dense) Newton wrapper around the Chebyshev h=0 operator."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_h0_solver import (
    phi, crra_clear, N_GRID, U_NODES, LOBATTO,
    TAU, GAMMA, C_STRETCH, N
)

def sg(x): return 1/(1+np.exp(-x))

def metrics(P):
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = 1.0/(1.0+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def build_NL_IC(gamma):
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    P_IC = np.empty_like(U1)
    for i in range(N_GRID):
        for j in range(N_GRID):
            for k in range(N_GRID):
                mu0 = sg(TAU*U_NODES[i]); mu1 = sg(TAU*U_NODES[j]); mu2 = sg(TAU*U_NODES[k])
                P_IC[i,j,k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_IC

def picard(P, gamma, n_iter=8, omega=0.5):
    """Damped Picard, returns trajectory of ||F||."""
    ferrs = []; metrics_list = []
    for it in range(n_iter):
        ts = time.time()
        P_new = phi(P, gamma)
        ferr = float(np.max(np.abs(P_new - P)))
        ferrs.append(ferr)
        P = (1-omega)*P + omega*P_new
        m = metrics(P)
        metrics_list.append(m)
        print(f'  Picard it {it+1:2d}  ||F||_∞={ferr:.3e}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  d_FR={m["d_FR"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
    return P, ferrs, metrics_list

def dense_newton(P, gamma, n_iter=6, eps_fd=1e-6, tol=1e-9):
    """Pure dense Newton with forward-difference Jacobian + Armijo line search."""
    G = N_GRID; N_unk = G**3
    print(f'  Dense Newton: {N_unk} unknowns, FD eps={eps_fd}', flush=True)
    x = P.ravel().copy()
    F = phi(x.reshape((G,G,G)), gamma).ravel() - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_∞={F_norm:.3e}', flush=True)

    ferrs = [F_norm]; metrics_list = [metrics(x.reshape((G,G,G)))]
    for it in range(n_iter):
        ts = time.time()
        # Build Jacobian via forward difference
        J = np.empty((N_unk, N_unk))
        for j in range(N_unk):
            xp = x.copy(); xp[j] += eps_fd
            Fp = phi(xp.reshape((G,G,G)), gamma).ravel() - xp
            J[:, j] = (Fp - F) / eps_fd
        t_build = time.time() - ts
        # Solve
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        # Armijo
        alpha = 1.0; best = (None, 1e100)
        for _ in range(20):
            xn = np.clip(x + alpha*dx, 1e-12, 1-1e-12)
            Fn = phi(xn.reshape((G,G,G)), gamma).ravel() - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(x.reshape((G,G,G)))
        ferrs.append(F_norm); metrics_list.append(m)
        print(f'  NewtIter {it+1:2d}  ||F||_∞={F_norm:.3e}  α={alpha:.3g}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  (J build {t_build:.1f}s, total {time.time()-ts:.1f}s)', flush=True)
        if F_norm < tol:
            print(f'  CONVERGED', flush=True); break
    return x.reshape((G,G,G)), ferrs, metrics_list

# ===== Main =====
if __name__ == '__main__':
    print(f'CHEBYSHEV h=0 MIZN SOLVER — pure Newton')
    print(f'  N={N}  N_unk={N_GRID**3}  τ={TAU}  γ={GAMMA}  c_stretch={C_STRETCH}')
    print(f'  Lobatto nodes ξ: {LOBATTO}\n  u-nodes (clipped at boundary): {U_NODES}')

    # Phase 1: Picard preconditioning
    P_IC = build_NL_IC(GAMMA)
    print(f'\nIC: range=[{P_IC.min():.4f}, {P_IC.max():.4f}], metrics={metrics(P_IC)}')
    print(f'\n=== PHASE 1: Damped Picard (warm-up to get into basin) ===')
    P_warm, picard_ferrs, picard_metrics = picard(P_IC, GAMMA, n_iter=5, omega=0.5)

    # Phase 2: Pure Newton
    print(f'\n=== PHASE 2: Pure Newton ===')
    P_final, newton_ferrs, newton_metrics = dense_newton(P_warm, GAMMA, n_iter=5, tol=1e-9)

    print(f'\nFinal: ||F||_∞={newton_ferrs[-1]:.3e}, metrics={metrics(P_final)}')

    # Save
    np.save('/tmp/cheby_h0/P_final.npy', P_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU, gamma=GAMMA, c=C_STRETCH, lobatto=LOBATTO.tolist(), u_nodes=U_NODES.tolist()),
        picard=dict(ferrs=picard_ferrs, metrics=picard_metrics),
        newton=dict(ferrs=newton_ferrs, metrics=newton_metrics),
        final_metrics=metrics(P_final),
    ), open('/tmp/cheby_h0/results.json','w'), indent=2, default=str)
    print('saved')
