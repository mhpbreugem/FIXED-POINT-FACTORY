"""Chebyshev h=0 solver WITH sigmoid lift + smooth Chebyshev plotting.

P = σ(α·T + h(ξ)) where:
  - T = τ·(u_1 + u_2 + u_3) (the FR direction)
  - α = scalar slope parameter
  - h(ξ) = Chebyshev correction, S_3×Z_2 symmetric (odd in ξ ⟹ h(0,0,0)=0)

Total DOFs: 1 (α) + 40 (symmetric h orbits, with the 4 Z_2-fixed orbits forced to h=0).

Boundary behavior: as u_k → ±∞, T → ±∞, σ saturates to 0/1 exactly.
Spectral expansion of h only needs to capture the smooth correction.
"""
import os, sys, time, math, json, itertools
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_h0_solver import phi, crra_clear, N_GRID, U_NODES, LOBATTO, TAU, GAMMA, C_STRETCH, N
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit, orbit_Z2

G = N_GRID  # 7
EPS_PRICE = 1e-9

# T(u_1, u_2, u_3) = τ·(u_1 + u_2 + u_3) at each cube cell
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)
print(f'Sigmoid lift solver: τ={TAU}, γ={GAMMA}, N={N}, c={C_STRETCH}')
print(f'  T field range: [{T_FIELD.min():.2f}, {T_FIELD.max():.2f}]')

def sigmoid(x): return 1.0/(1.0 + np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

# DOFs: x = [α, h_orbit_1, h_orbit_2, ..., h_orbit_40]
N_DOF = 1 + len(FREE_REPS)
print(f'  DOFs: 1 (α) + {len(FREE_REPS)} (symmetric h orbits) = {N_DOF}')

def expand(x):
    """Expand (α, h_small) into full P cube."""
    alpha = x[0]
    h_small = x[1:]
    h_full = np.zeros((G, G, G))
    rep_to_idx = {rep: i for i, rep in enumerate(FREE_REPS)}
    for rep in FREE_REPS:
        idx = rep_to_idx[rep]
        h_val = h_small[idx]
        s3, z2_s3 = full_orbit(*rep)
        for cell in s3: h_full[cell] = h_val
        for cell in z2_s3: h_full[cell] = -h_val  # h is odd: h(-ξ)=-h(ξ)
    for rep in FIXED_REPS:
        s3, z2_s3 = full_orbit(*rep)
        for cell in s3 | z2_s3:
            h_full[cell] = 0.0  # Z_2-fixed cells must have h=0
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    """Extract (α, h_small) from P assuming sigmoid lift form.
    α = slope of logit(P) vs T regression (with h-orthogonal weighting).
    h = logit(P) - α·T, then project onto symmetric orbits."""
    L = logit(P)
    # Regress L on T (with T_FIELD as predictor). Best α minimizes ||L - α·T - bias||
    # But h orthogonal to constants (h has no constant term) → bias=0
    # So α = <L, T> / <T, T>
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    # Project h_full onto symmetric orbits, taking the orbit-average
    h_small = np.zeros(len(FREE_REPS))
    for idx, rep in enumerate(FREE_REPS):
        s3, z2_s3 = full_orbit(*rep)
        vals_pos = [h_full[c] for c in s3]
        vals_neg = [-h_full[c] for c in z2_s3]  # since h is odd
        h_small[idx] = float(np.mean(vals_pos + vals_neg))
    return np.concatenate([[alpha], h_small])

def phi_lifted(x, gamma):
    P = expand(x)
    P_new = phi(P, gamma)
    return contract(P_new)

def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_FIELD.ravel(), y, 1); pr = a[0]*T_FIELD.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T_FIELD); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(x, gamma, n_iter=8, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton (lifted): {Nu} unknowns', flush=True)
    F = phi_lifted(x, gamma) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_∞={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; metrics_list = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            Fp = phi_lifted(xp, gamma) - xp
            J[:, j] = (Fp - F) / eps_fd
        t_build = time.time() - ts
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100)
        for _ in range(20):
            xn = x + alpha*dx
            # No clip on lifted variables — α and h can be any real
            Fn = phi_lifted(xn, gamma) - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(expand(x)); ferrs.append(F_norm); metrics_list.append(m)
        print(f'  NewtIter {it+1:2d}  ||F||_∞={F_norm:.3e}  α_LS={alpha:.3g}  α_slope={x[0]:.4f}  slope_T={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  (J build {t_build:.1f}s, total {time.time()-ts:.1f}s)', flush=True)
        if F_norm < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, metrics_list

# ===== Main =====
if __name__ == '__main__':
    print(f'\n=== START ===')
    # IC: α = 0.5 (initial guess of slope), h = 0 ⟹ P = σ(0.5·T) (FR-with-slope-0.5)
    x0 = np.zeros(N_DOF)
    x0[0] = 0.5  # initial slope guess
    P_IC = expand(x0)
    print(f'\nIC: P = σ(0.5·T), α=0.5, h=0')
    print(f'  P range: [{P_IC.min():.4f}, {P_IC.max():.4f}]')
    print(f'  IC metrics: {metrics(P_IC)}')

    # Picard preconditioning on lifted vector
    print(f'\n=== PHASE 1: Picard (5 iters, ω=0.5) ===')
    x = x0.copy(); picard_ferrs = []; picard_metrics = []
    for it in range(5):
        ts = time.time()
        x_new = phi_lifted(x, GAMMA)
        ferr = float(np.max(np.abs(x_new - x))); picard_ferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); picard_metrics.append(m)
        print(f'  Picard it {it+1:2d}  ||F||={ferr:.3e}  α_slope={x[0]:.4f}  slope_T={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)

    print(f'\n=== PHASE 2: Pure dense Newton ===')
    x_final, newton_ferrs, newton_metrics = dense_newton(x, GAMMA, n_iter=8, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal: α_slope={x_final[0]:.4f}, metrics={metrics(P_final)}')

    np.save('/tmp/cheby_h0/P_final_lifted.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_lifted.npy', x_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU, gamma=GAMMA, c=C_STRETCH,
                     lobatto=LOBATTO.tolist(), u_nodes=U_NODES.tolist(),
                     n_dof=N_DOF),
        x_final=x_final.tolist(),
        picard=dict(ferrs=picard_ferrs, metrics=picard_metrics),
        newton=dict(ferrs=newton_ferrs, metrics=newton_metrics),
        final_metrics=metrics(P_final),
        final_alpha=float(x_final[0]),
    ), open('/tmp/cheby_h0/results_lifted.json','w'), indent=2, default=str)
    print('saved')
