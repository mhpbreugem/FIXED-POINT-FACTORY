"""Method C v2: clean Bayesian-symmetric monotone parameterization.

P(u_1, u_2, u_3) = sigmoid(L)
L = h(u_1) + h(u_2) + h(u_3)
h(u) = ∫_0^u σ²(s) ds   (odd in u, increasing; σ is even polynomial in u)

Properties (by construction):
  - Permutation symmetric in u_1, u_2, u_3
  - Sign-flip symmetric: P(u) = 1 - P(-u)  (since L(-u) = -L(u))
  - Strictly monotone increasing in each u_i (∂P/∂u_i = sigmoid'(L) · h'(u_i) = sigmoid'(L) · σ²(u_i) ≥ 0)
  - VERY few parameters: d_sigma+1 (e.g., d_sigma=4 → 5 unknowns)

σ(s) = sum_n a_n * T_{2n}(s/U)   (even Cheby basis)
h(u) = ∫_0^u σ²(s) ds              (computed by GL on [0, u])
"""
import os, sys, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_richardson import phi_lin_richardson


G = 7
D_SIGMA = 4         # σ has degree 2*D_SIGMA (even), so D_SIGMA+1 coefs
U_MAX = 2.33
N_QUAD = 16         # GL quad for 1D h integral
GAMMA = 100.0


@njit(cache=True)
def _sigma2(coefs, xi, d_sigma):
    """Evaluate σ²(s) at s where xi = s/U. σ = sum_n coefs[n] T_{2n}(xi)."""
    # Use Cheby recurrence for T_{2n}(xi)
    # Or directly: T_0(xi)=1, T_2(xi)=2xi^2-1, T_4=8xi^4-8xi^2+1, T_6=32xi^6-48xi^4+18xi^2-1
    # General: T_{2n}(xi) = T_n(2*xi^2-1) (composition); compute via Cheby recurrence
    # Easier: compute all T_k for k=0..2*d_sigma then pick even
    T = np.empty(2*d_sigma + 1)
    T[0] = 1.0
    if 2*d_sigma >= 1:
        T[1] = xi
        for k in range(1, 2*d_sigma):
            T[k+1] = 2*xi*T[k] - T[k-1]
    s_val = 0.0
    for n in range(d_sigma + 1):
        s_val += coefs[n] * T[2*n]
    return s_val * s_val


@njit(cache=True)
def _h_at_u(coefs, u_val, d_sigma, U, gl_nodes, gl_weights):
    """h(u) = ∫_0^u σ²(s) ds via GL quadrature on [0, u]."""
    if u_val == 0.0: return 0.0
    sign = 1.0 if u_val > 0 else -1.0
    a = 0.0; b = abs(u_val)
    half = (b - a) / 2.0; mid = (a + b) / 2.0
    integral = 0.0
    n_q = gl_nodes.size
    for q in range(n_q):
        s = mid + half * gl_nodes[q]
        w = half * gl_weights[q]
        xi = s / U
        integral += w * _sigma2(coefs, xi, d_sigma)
    return sign * integral    # h is odd


@njit(cache=True, parallel=False)
def _build_P_v2(coefs, u_grid, d_sigma, U, gl_nodes, gl_weights, P_out):
    """P[i, j, k] = sigmoid(h(u_i) + h(u_j) + h(u_k))."""
    G_loc = u_grid.size
    h_vals = np.empty(G_loc)
    for i in range(G_loc):
        h_vals[i] = _h_at_u(coefs, u_grid[i], d_sigma, U, gl_nodes, gl_weights)
    for i in range(G_loc):
        for j in range(G_loc):
            for k in range(G_loc):
                L = h_vals[i] + h_vals[j] + h_vals[k]
                if L > 30: P_out[i, j, k] = 1.0
                elif L < -30: P_out[i, j, k] = 0.0
                else: P_out[i, j, k] = 1.0 / (1.0 + np.exp(-L))


def build_P(coefs, u_grid):
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(N_QUAD)
    P_out = np.empty((u_grid.size,)*3)
    _build_P_v2(coefs, u_grid, D_SIGMA, U_MAX, gl_nodes, gl_weights, P_out)
    return P_out


def F_residual(coefs, tau, u_grid, p_grid, hs=(0.5, 0.4, 0.3, 0.2), gamma=GAMMA):
    P = build_P(coefs, u_grid)
    P_new = phi_lin_richardson(P, u_grid, hs=hs, gamma=gamma, tau=tau,
                                       G_p=p_grid.size, NQK=16, p_grid=p_grid)
    return (P_new - P).ravel()


def solve(tau, params0=None, gamma=GAMMA, max_nfev=300, verbose=True):
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    if params0 is None:
        params0 = np.zeros(D_SIGMA + 1)
        params0[0] = 0.5    # constant σ ~ 0.5 → σ² ~ 0.25 → h'(u) ~ 0.25 → moderate slope
    if verbose:
        print(f"  initial coefs: {params0}", flush=True)
        F0 = F_residual(params0, tau, u_grid, p_grid, gamma=gamma)
        print(f"  init |F|={np.max(np.abs(F0)):.3e}", flush=True)
    t0 = time.time()
    res = least_squares(F_residual, params0,
                              args=(tau, u_grid, p_grid),
                              kwargs=dict(gamma=gamma),
                              jac="2-point", method="trf",
                              max_nfev=max_nfev, ftol=1e-14, xtol=1e-14,
                              verbose=2 if verbose else 0)
    t_solve = time.time() - t0
    P = build_P(res.x, u_grid)
    # monotonicity
    viol = 0; mind = 0.0
    for ax in range(3):
        dd = np.diff(P, axis=ax)
        viol += int((dd < 0).sum()); mind = min(mind, float(dd.min()))
    if verbose:
        print(f"  TRF done {t_solve:.0f}s, |F|={np.max(np.abs(res.fun)):.3e}", flush=True)
        print(f"  monotonicity: {viol} viol, min dP/du = {mind:.3e}", flush=True)
        print(f"  P range: [{P.min():.6f}, {P.max():.6f}]", flush=True)
    return res.x, P, float(np.max(np.abs(res.fun))), viol, t_solve


if __name__ == "__main__":
    print(f"=== Method C v2 (Bayesian-symmetric monotone) at gamma={GAMMA} ===")
    print(f"D_SIGMA={D_SIGMA}, unknowns={D_SIGMA+1}")

    # JIT warm
    print("\nJIT warmup...", flush=True); t0 = time.time()
    u_grid = make_cdf_uniform_grid(G)
    p0 = np.zeros(D_SIGMA+1); p0[0] = 0.5
    P_test = build_P(p0, u_grid)
    print(f"  done {time.time()-t0:.1f}s, P range [{P_test.min():.4f}, {P_test.max():.4f}]",
          flush=True)
    t0 = time.time()
    P_test = build_P(p0, u_grid)
    print(f"  build_P: {(time.time()-t0)*1000:.0f}ms per call", flush=True)

    # Test at tau=1
    print(f"\n--- tau=1.0 ---")
    coefs, P, F, viol, ts = solve(1.0)

    print(f"\n--- tau=0.5 ---")
    coefs, P, F, viol, ts = solve(0.5, params0=coefs)

    print(f"\n--- tau=0.1 ---")
    coefs, P, F, viol, ts = solve(0.1, params0=coefs)
