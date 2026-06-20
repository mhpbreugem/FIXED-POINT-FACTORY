"""Chebyshev h=0 solver: sigmoid lift + numba operator + S₃×Z₂ symmetry.

P = σ(α·T + h(ξ)), where:
  - T = τ·(u_1+u_2+u_3) (FR direction)
  - α = scalar slope DOF (1)
  - h(ξ) = symmetric Chebyshev correction (40 DOFs)
Total: 41 DOFs after symmetry reduction.

Uses cheby_numba.phi (numba-JIT'd, ~12× faster than pure Python).
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_numba import phi, U_NODES, LOBATTO, TAU, GAMMA, C_STRETCH, N, N_GRID
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit, orbit_Z2

G = N_GRID

# Precompute T_FIELD = τ·(u_1+u_2+u_3) on the Lobatto cube
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)
print(f'Lifted+Numba solver: τ={TAU}, γ={GAMMA}, N={N}, c={C_STRETCH}')
print(f'  T field range: [{T_FIELD.min():.2f}, {T_FIELD.max():.2f}]')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

N_DOF = 1 + len(FREE_REPS)
print(f'  DOFs: 1 (α) + {len(FREE_REPS)} (h orbits) = {N_DOF}')

REP_TO_IDX = {rep: i for i, rep in enumerate(FREE_REPS)}

# Precompute, for each free rep: the S3 orbit cells and Z2*S3 orbit cells
ORBIT_POS = []   # list of arrays of cells (positive h sign)
ORBIT_NEG = []   # list of arrays of cells (negative h sign)
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3))
    ORBIT_NEG.append(list(z2_s3))
FIXED_CELLS = []
for rep in FIXED_REPS:
    s3, z2_s3 = full_orbit(*rep)
    FIXED_CELLS.extend(list(s3 | z2_s3))

def expand(x):
    """(α, h_small) → full P cube via sigmoid lift."""
    alpha = x[0]
    h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx, rep in enumerate(FREE_REPS):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    # FIXED cells already 0 by zeros() init
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    """P → (α, h_small).  α = <L,T>/<T,T>, h = symmetric orbit-average of L−αT."""
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(FREE_REPS))
    for idx, rep in enumerate(FREE_REPS):
        vals_pos = [h_full[c] for c in ORBIT_POS[idx]]
        vals_neg = [-h_full[c] for c in ORBIT_NEG[idx]]
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

def dense_newton(x, gamma, n_iter=10, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton (lifted+numba): {Nu} unknowns', flush=True)
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
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
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


if __name__ == '__main__':
    print(f'\n=== START ===')
    print('Triggering numba JIT compile...', flush=True)
    t0 = time.time()
    _ = phi(np.full((G, G, G), 0.5), GAMMA)
    print(f'  JIT compile: {time.time()-t0:.1f}s', flush=True)

    # IC
    x0 = np.zeros(N_DOF); x0[0] = 0.5
    P_IC = expand(x0)
    print(f'\nIC: P = σ(0.5·T), α=0.5, h=0')
    print(f'  P range: [{P_IC.min():.4f}, {P_IC.max():.4f}]')
    print(f'  IC metrics: {metrics(P_IC)}')

    print(f'\n=== PHASE 1: Picard (8 iters, ω=0.5) ===')
    x = x0.copy(); picard_ferrs = []; picard_metrics = []
    for it in range(8):
        ts = time.time()
        x_new = phi_lifted(x, GAMMA)
        ferr = float(np.max(np.abs(x_new - x))); picard_ferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); picard_metrics.append(m)
        print(f'  Picard it {it+1:2d}  ||F||={ferr:.3e}  α_slope={x[0]:.4f}  slope_T={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)

    print(f'\n=== PHASE 2: Pure dense Newton ===')
    x_final, newton_ferrs, newton_metrics = dense_newton(x, GAMMA, n_iter=10, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal: α_slope={x_final[0]:.4f}, metrics={metrics(P_final)}')

    np.save('/tmp/cheby_h0/P_final_lifted_numba.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_lifted_numba.npy', x_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU, gamma=GAMMA, c=C_STRETCH,
                     lobatto=LOBATTO.tolist(), u_nodes=U_NODES.tolist(),
                     n_dof=N_DOF),
        x_final=x_final.tolist(),
        picard=dict(ferrs=picard_ferrs, metrics=picard_metrics),
        newton=dict(ferrs=newton_ferrs, metrics=newton_metrics),
        final_metrics=metrics(P_final),
        final_alpha=float(x_final[0]),
    ), open('/tmp/cheby_h0/results_lifted_numba.json','w'), indent=2, default=str)
    print('saved')
