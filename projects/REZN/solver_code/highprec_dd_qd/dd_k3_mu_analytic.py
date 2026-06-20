"""Analytic strict-h=0 lookup mu(p, u_k) from Method C v2 ansatz.

Given P = sigmoid(h(u_1) + h(u_2) + h(u_3)) with h odd monotone (Method C v2),
the lookup mu(p, u_k) reduces to a 1D integral via change of variable
v_a = h(u_a). No level-set search, no kernel, no contour integration.

Math:
  L = logit(p) = h(u_k) + h(u_a) + h(u_b)
  M = L - h(u_k); constraint h(u_a) + h(u_b) = M
  v_a = h(u_a), v_b = h(u_b) = M - v_a
  u_a = h^{-1}(v_a), u_b = h^{-1}(M - v_a)
  du_a = dv_a / h'(u_a) = dv_a / sigma^2(u_a)
  dp = sigma'(L) dL = p(1-p) dL, where dL = dM (u_k fixed)

  A_v(p, u_k) = (1/(p(1-p))) * ∫ [f_v(h^{-1}(v_a)) * f_v(h^{-1}(M-v_a))]
                                       / [sigma^2(h^{-1}(v_a)) * sigma^2(h^{-1}(M-v_a))] dv_a
  mu(p, u_k) = f_1(u_k) A_1 / (f_0(u_k) A_0 + f_1(u_k) A_1)

Integration bounds for v_a: [h(u_min), h(u_max)] intersected with
[M - h(u_max), M - h(u_min)] so that both u_a and u_b are in [u_min, u_max].
"""
import os, sys, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import dd_k3_method_c_v2 as MC2
from cheby_numba import f_signal_jit
import math


def make_h_invert(coefs, d_sigma, U_MAX, n_table=201):
    """Precompute h on a fine grid for fast h(u) eval and h^{-1}(v) by interp."""
    u_table = np.linspace(-U_MAX, U_MAX, n_table)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(MC2.N_QUAD)
    h_table = np.array([MC2._h_at_u(coefs, u, d_sigma, U_MAX, gl_nodes, gl_weights)
                              for u in u_table])
    # h is monotone increasing → h_table sorted
    # build interp objects
    def h_func(u):
        return np.interp(u, u_table, h_table)
    def h_inv(v):
        return np.interp(v, h_table, u_table)
    def sigma2_func(u):
        return MC2._sigma2(coefs, u/U_MAX, d_sigma)
    return h_func, h_inv, sigma2_func, h_table[0], h_table[-1]


def mu_analytic(p, u_k, coefs, tau, d_sigma=MC2.D_SIGMA, U_MAX=MC2.U_MAX,
                  n_quad=32):
    """Analytic strict-h=0 lookup from Method C v2 P ansatz."""
    h_func, h_inv, sigma2_func, h_min, h_max = make_h_invert(coefs, d_sigma, U_MAX)
    # logit p
    if p <= 1e-12: p = 1e-12
    if p >= 1 - 1e-12: p = 1 - 1e-12
    L = math.log(p / (1 - p))
    h_uk = h_func(u_k)
    M = L - h_uk
    # bounds for v_a so both u_a, u_b in [-U_MAX, U_MAX]:
    # v_a in [h_min, h_max] AND v_b = M - v_a in [h_min, h_max]
    # → v_a in [max(h_min, M - h_max), min(h_max, M - h_min)]
    v_lo = max(h_min, M - h_max)
    v_hi = min(h_max, M - h_min)
    if v_hi <= v_lo:
        return 0.5    # empty integration domain
    # GL nodes on [v_lo, v_hi]
    gl_n, gl_w = np.polynomial.legendre.leggauss(n_quad)
    half = (v_hi - v_lo) / 2; mid = (v_lo + v_hi) / 2
    A0 = 0.0; A1 = 0.0
    for q in range(n_quad):
        v_a = mid + half * gl_n[q]
        w = half * gl_w[q]
        v_b = M - v_a
        u_a = h_inv(v_a); u_b = h_inv(v_b)
        s2_a = sigma2_func(u_a); s2_b = sigma2_func(u_b)
        if s2_a < 1e-30 or s2_b < 1e-30: continue
        # f_v(u_a), f_v(u_b)
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        # Jacobian factors 1/sigma^2 from du = dv / sigma^2
        jacobian = 1.0 / (s2_a * s2_b)
        # contribution
        A0 += w * f0a * f0b * jacobian
        A1 += w * f1a * f1b * jacobian
    # mu = f_1(u_k) A_1 / (f_0(u_k) A_0 + f_1(u_k) A_1)
    f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
    denom = f0k * A0 + f1k * A1
    if denom < 1e-300: return 0.5
    return f1k * A1 / denom


def main():
    """Verify against strict-h=0 at the Method C v2 ansatz P."""
    from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u
    from lin_cdf_strict import build_mu_table_lin_strict

    print("=== Verify mu_analytic (1D integral from Method C v2) vs strict-h=0 ===")
    coefs = np.array([-0.008746885807678776, -0.03478614425381441,
                          -0.01744997234804044, -0.023238132714746427,
                          -0.01461400447175478])    # Method C v2 fit at tau=0.001
    d_sigma = MC2.D_SIGMA
    tau = 0.001
    u_grid = make_cdf_uniform_grid(7)
    p_grid = make_p_grid(121)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    P = MC2.build_P(coefs, u_grid)
    print(f"P range: [{P.min():.8f}, {P.max():.8f}]")
    # Strict-h=0 reference at same P
    mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u, gl_du,
                                                   tau, 7, 16)
    # Analytic mu at sample (p, u_k) values
    print(f"\n{'p':>10s} {'u_k':>10s} {'mu_strict':>12s} {'mu_analytic':>12s} {'diff':>12s}")
    print("-" * 60)
    max_diff = 0
    n_tested = 0
    for ip in range(0, 121, 12):
        for k in range(0, 7, 2):
            p = p_grid[ip]; u_k = u_grid[k]
            mu_s = mu_strict[ip, k]
            mu_a = mu_analytic(p, u_k, coefs, tau, d_sigma=d_sigma)
            diff = abs(mu_s - mu_a)
            max_diff = max(max_diff, diff); n_tested += 1
            print(f"{p:>10.4f} {u_k:>10.4f} {mu_s:>12.6f} {mu_a:>12.6f} {diff:>12.3e}")
    print(f"\nMax diff across {n_tested} points: {max_diff:.3e}")


if __name__ == "__main__":
    main()
