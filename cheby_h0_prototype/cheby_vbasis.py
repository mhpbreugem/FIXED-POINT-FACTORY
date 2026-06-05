"""Chebyshev h=0 solver with VANISHING-AT-BOUNDARY basis on top of sigmoid lift.

P = σ(α·T(ξ) + B(ξ)·h(ξ))   where  B(ξ) = (1-ξ_1²)(1-ξ_2²)(1-ξ_3²)

By construction, B = 0 at any boundary cell (ξ_k = ±1 for some k), so
σ saturates to FR exactly there regardless of h. The interior h-mode can
freely fit the operator's deviation from σ(αT) without overshooting at ξ=±1.

Uses cheby_numba operator + S₃×Z₂ symmetry (40 free + 4 fixed at 0.5).
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_numba import phi as phi_jit, U_NODES, LOBATTO, C_STRETCH, N, N_GRID
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit

G = N_GRID
TAU = 1.0
GAMMA = 1.0

# T-field
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)

# Boundary factor B(ξ) on the Lobatto grid
XI1, XI2, XI3 = np.meshgrid(LOBATTO, LOBATTO, LOBATTO, indexing='ij')
B_FIELD = (1.0 - XI1**2) * (1.0 - XI2**2) * (1.0 - XI3**2)
INTERIOR_MASK = B_FIELD > 1e-12

print(f'Vanishing-at-boundary solver: τ={TAU}, γ={GAMMA}, N={N}')
print(f'  T-field range: [{T_FIELD.min():.2f}, {T_FIELD.max():.2f}]')
print(f'  Interior cells: {int(INTERIOR_MASK.sum())} / {G**3}  (interior fraction = {INTERIOR_MASK.mean():.3f})')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

# Orbit precomputation
ORBIT_POS, ORBIT_NEG = [], []
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3))
    ORBIT_NEG.append(list(z2_s3))

# How many orbits are "active" (have at least one interior cell)
ACTIVE_ORBIT = np.zeros(len(FREE_REPS), dtype=bool)
for idx in range(len(FREE_REPS)):
    for c in ORBIT_POS[idx] + ORBIT_NEG[idx]:
        if INTERIOR_MASK[c]:
            ACTIVE_ORBIT[idx] = True
            break
n_active = int(ACTIVE_ORBIT.sum())
print(f'  Active orbits (≥1 interior cell): {n_active} / {len(FREE_REPS)}')

N_DOF = 1 + len(FREE_REPS)  # keep same DOF count, inactive orbits forced to 0
print(f'  DOFs: 1 (α) + {len(FREE_REPS)} (h_orbit) = {N_DOF}  ({len(FREE_REPS)-n_active} forced to 0)')

def expand(x):
    """(α, h_small) → P cube via σ(αT + B·h)."""
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx in range(len(FREE_REPS)):
        if not ACTIVE_ORBIT[idx]:
            continue
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + B_FIELD * h_full)

# Design matrix for contract via least-squares (avoids 1/B blow-up near boundary).
# Each row = one cube cell. Cols: [α, g_orbit_1, ..., g_orbit_M].
# L[cell] = α·T[cell] + sign·B[cell]·g_orbit  (for cell in orbit; sign ±1)
# Z₂-fixed cells contribute L[cell] = 0 since FIXED cells are forced to P=0.5 (L=0).
_REP_TO_IDX = {rep: i for i, rep in enumerate(FREE_REPS)}
_CELL_KEY = {}  # cell → (orbit_idx, sign) or 'FIXED'
for idx, rep in enumerate(FREE_REPS):
    for c in ORBIT_POS[idx]: _CELL_KEY[c] = (idx, +1)
    for c in ORBIT_NEG[idx]: _CELL_KEY[c] = (idx, -1)
for rep in FIXED_REPS:
    s3, z2_s3 = full_orbit(*rep)
    for c in s3 | z2_s3: _CELL_KEY[c] = 'FIXED'

_DESIGN = np.zeros((G**3, N_DOF))
_FREE_CELL_MASK = np.zeros(G**3, dtype=bool)  # rows for non-FIXED cells
_row = 0
for i in range(G):
    for j in range(G):
        for k in range(G):
            key = _CELL_KEY[(i,j,k)]
            if key == 'FIXED':
                pass  # row stays zero; L[FIXED] = 0 anyway
            else:
                orb_idx, sign = key
                _DESIGN[_row, 0] = T_FIELD[i,j,k]
                if ACTIVE_ORBIT[orb_idx]:
                    _DESIGN[_row, 1 + orb_idx] = sign * B_FIELD[i,j,k]
                _FREE_CELL_MASK[_row] = True
            _row += 1

def contract(P):
    """P → (α, h_small) via least-squares solve.
    L[c] = α·T[c] + B[c]·h_orbit·sign  ⇒ overdetermined linear system."""
    L = logit(P).ravel()
    # Use only non-FIXED cells (FIXED cells trivially have L=0)
    A = _DESIGN[_FREE_CELL_MASK]
    b = L[_FREE_CELL_MASK]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Zero out inactive orbit slots (they had zero columns; lstsq returns 0 for them)
    return x

def phi_lift(x, gamma):
    P = expand(x)
    P_new = phi_jit(P, gamma=gamma, tau=TAU)
    return contract(P_new)

def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_FIELD.ravel(), y, 1); pr = a[0]*T_FIELD.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T_FIELD); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(x, gamma, n_iter=10, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton (V-basis lift): {Nu} unknowns ({n_active+1} effectively active)', flush=True)
    F = phi_lift(x, gamma) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_∞={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; metrics_list = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            if j > 0 and not ACTIVE_ORBIT[j-1]:
                # Inactive orbit — column zero (no effect on Φ)
                J[:, j] = 0.0
                J[j, j] = -1.0  # so dx[j] = 0 will satisfy this row
                continue
            xp = x.copy(); xp[j] += eps_fd
            Fp = phi_lift(xp, gamma) - xp
            J[:, j] = (Fp - F) / eps_fd
        t_build = time.time() - ts
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
            # Force inactive orbits to 0
            for j in range(1, Nu):
                if not ACTIVE_ORBIT[j-1]:
                    xn[j] = 0.0
            Fn = phi_lift(xn, gamma) - xn
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
    print('JIT warmup...', flush=True)
    t0 = time.time()
    _ = phi_jit(np.full((G, G, G), 0.5), gamma=GAMMA, tau=TAU)
    print(f'  JIT: {time.time()-t0:.1f}s')

    # IC: start from converged lifted_numba FP (same τ, γ)
    if os.path.exists('/tmp/cheby_h0/x_final_lifted_numba.npy'):
        x_old = np.load('/tmp/cheby_h0/x_final_lifted_numba.npy')
        # Old FP was P = σ(αT + h_full). Reproject as P = σ(αT + B·h_new) via contract.
        P_old = sigmoid(x_old[0] * T_FIELD)  # start from old α, h_new from contract of P_old after one phi
        # Actually simpler: just use NL IC and warm up
        x0 = np.zeros(N_DOF); x0[0] = 0.5
    else:
        x0 = np.zeros(N_DOF); x0[0] = 0.5
    P_IC = expand(x0)
    print(f'\nIC: P = σ(0.5·T), α=0.5, h=0')
    print(f'  P range: [{P_IC.min():.4f}, {P_IC.max():.4f}]')
    print(f'  IC metrics: {metrics(P_IC)}')

    print(f'\n=== PHASE 1: Picard (8 iters, ω=0.5) ===')
    x = x0.copy(); picard_ferrs = []; picard_metrics = []
    for it in range(8):
        ts = time.time()
        x_new = phi_lift(x, GAMMA)
        for j in range(1, N_DOF):
            if not ACTIVE_ORBIT[j-1]:
                x_new[j] = 0.0
        ferr = float(np.max(np.abs(x_new - x))); picard_ferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); picard_metrics.append(m)
        print(f'  Picard {it+1:2d}  ||F||={ferr:.3e}  α={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)

    print(f'\n=== PHASE 2: Dense Newton ===')
    x_final, newton_ferrs, newton_metrics = dense_newton(x, GAMMA, n_iter=10, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal: α={x_final[0]:.4f}, metrics={metrics(P_final)}')
    print(f'\nComparison:')
    print(f'  Previous lifted_numba final ||F||: 1.74e-01 (at basis floor)')
    print(f'  V-basis final ||F||: {newton_ferrs[-1]:.3e}')

    np.save('/tmp/cheby_h0/P_final_vbasis.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_vbasis.npy', x_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU, gamma=GAMMA, c=C_STRETCH, n_dof=N_DOF,
                     n_active=n_active),
        x_final=x_final.tolist(),
        picard=dict(ferrs=picard_ferrs, metrics=picard_metrics),
        newton=dict(ferrs=newton_ferrs, metrics=newton_metrics),
        final_metrics=metrics(P_final),
        final_alpha=float(x_final[0]),
    ), open('/tmp/cheby_h0/results_vbasis.json','w'), indent=2, default=str)
    print('saved')
