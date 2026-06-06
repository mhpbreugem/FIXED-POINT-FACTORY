"""1D-in-T solver: P(u) = σ(α·T + h̃(ξ_T)) where T = τΣu_k.

h̃ is a 1D Chebyshev expansion in stretched T-coordinate, odd by Z2 symmetry,
expanded in modes T_3, T_5, T_7, ...  (T_1 is the linear T direction, already
in α; T_0 is constant, forbidden by Z2).

KEY INSIGHT (same as rank-1): the agent inversion of conjecture P̃(T) gives
T_extracted = P̃^{-1}(p_obs) = T(u) exactly when conjecture is self-consistent,
so s_k = u_a + u_b INDEPENDENTLY OF (α, h̃). The operator output is therefore
conjecture-shape-invariant. 1D-in-T is solved in ONE SHOT:
  compute operator output → fit best (α, h̃) by 1D least-squares regression.

No iteration, no Newton. Just rank-1 operator eval + a (m+1)-column lstsq."""

import os, sys, time, math
import numpy as np
from numba import njit
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_rank1 import operator_rank1  # same operator — conjecture-invariant


# ===== T-domain stretching =====
C_STRETCH_U = 2.0   # u = C·atanh(ξ_u)
# Natural T-scale: T_max = K·τ·u_max where u_max ≈ C_STRETCH_U·9.21 (atanh(0.9999))
# We'll compute T_max per τ inside the solver.

@njit(cache=True, inline='always')
def chebT(k, x):
    """Evaluate T_k(x) by recurrence. k >= 0, x ∈ [-1,1]."""
    if k == 0: return 1.0
    if k == 1: return x
    Tk_m2 = 1.0; Tk_m1 = x
    for n in range(2, k+1):
        Tk = 2*x*Tk_m1 - Tk_m2
        Tk_m2 = Tk_m1; Tk_m1 = Tk
    return Tk_m1


def design_matrix(T_flat, T_max, m):
    """Build design matrix:
       Column 0: T (linear, coefficient is α)
       Columns 1..m-1: T_{2j+1}(ξ_T) for j=1..m-1 where ξ_T = T/T_max
    Returns (X, ξ_T)."""
    N = T_flat.shape[0]
    xi_T = T_flat / T_max
    X = np.empty((N, m))
    X[:, 0] = T_flat  # T column for α
    for j in range(1, m):
        k_mode = 2*j + 1  # odd modes T_3, T_5, ...
        for i in range(N):
            X[i, j] = chebT(k_mode, xi_T[i])
    return X, xi_T


def fit_1d_in_T(P, T_grid, m, T_max, weighted=True, edge_trim=0.97):
    """Fit logit(P) ≈ α·T + Σ c_j T_{2j+1}(ξ_T).
    Returns (coeffs, deficit_R2, predicted_logit, residual)."""
    Pc = np.clip(P, 1e-15, 1 - 1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    T_flat = T_grid.ravel()
    xi_T = T_flat / T_max
    # Trim near-saturated cells (|ξ_T| close to 1)
    mask = np.abs(xi_T) < edge_trim
    X, _ = design_matrix(T_flat[mask], T_max, m)
    y = L[mask]
    if weighted:
        # Weight by P(1-P) to undo logit amplification at saturated cells
        w = (Pc.ravel()[mask] * (1 - Pc.ravel()[mask]))**0.5
        Xw = X * w[:, None]
        yw = y * w
        coeffs, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    else:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    deficit = ss_res / max(ss_tot, 1e-30)
    return coeffs, deficit, pred, resid


def evaluate_P_tilde(coeffs, T_vals, T_max):
    """Evaluate P̃(T) = σ(α·T + Σ c_j T_{2j+1}(ξ_T)) at given T values."""
    alpha = coeffs[0]
    h_coeffs = coeffs[1:]
    xi_T = np.clip(T_vals / T_max, -1, 1)
    h = np.zeros_like(T_vals)
    for j, c in enumerate(h_coeffs):
        k_mode = 2*j + 1
        for i in range(len(T_vals)):
            h[i] += c * chebT(k_mode, xi_T[i])
    return 1.0/(1.0 + np.exp(-(alpha*T_vals + h)))


def solve_1d_in_T(tau, gamma, G=15, m=10, weighted=True):
    """One-shot 1D-in-T solve."""
    LOBATTO = -np.cos(np.pi * np.arange(G) / (G-1))
    U_NODES = C_STRETCH_U * np.arctanh(np.clip(LOBATTO, -0.9999, 0.9999))
    u_max = U_NODES.max()
    T_max = 3 * tau * u_max  # K·τ·u_max for K=3
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T_grid = tau * (U1 + U2 + U3)
    P_op = operator_rank1(U_NODES, tau, gamma)
    coeffs, deficit, _, _ = fit_1d_in_T(P_op, T_grid, m, T_max, weighted=weighted)
    alpha = float(coeffs[0])
    return dict(alpha=alpha, h_coeffs=coeffs[1:].tolist(),
                 deficit=deficit, T_max=T_max, P=P_op, T_grid=T_grid,
                 U_NODES=U_NODES, m=m)


if __name__ == '__main__':
    print('=== 1D-in-T solver self-test ===\n')
    print(f'tau=1, gamma=1, G=15, sweep m=1..10')
    print(f'{"m":>3} {"alpha":>10} {"|h_coeffs|":>30} {"deficit":>10} {"time(s)":>8}')
    for m in [1, 2, 3, 5, 7, 10, 15, 20]:
        t = time.time()
        r = solve_1d_in_T(1.0, 1.0, G=15, m=m)
        elapsed = time.time() - t
        h_str = ', '.join(f'{c:+.4f}' for c in r['h_coeffs'][:min(5, len(r['h_coeffs']))])
        if len(r['h_coeffs']) > 5: h_str += f', ... ({len(r["h_coeffs"])} total)'
        print(f'{m:>3} {r["alpha"]:>10.6f} [{h_str:^28}] {r["deficit"]:>10.3e} {elapsed:>8.3f}')

    print('\nCompare: rank-1 (m=1) is the baseline; m>1 captures shape beyond pure σ(αT).')
    print('Deficit should decrease monotonically with m.\n')

    # Test at the deep-PR corner where rank-1 has high deficit
    print('=== Deep-PR corner: tau=3, gamma=0.2, G=15 ===')
    print(f'{"m":>3} {"alpha":>10} {"deficit":>10}')
    for m in [1, 2, 3, 5, 7, 10, 15]:
        r = solve_1d_in_T(3.0, 0.2, G=15, m=m)
        print(f'{m:>3} {r["alpha"]:>10.6f} {r["deficit"]:>10.3e}')
