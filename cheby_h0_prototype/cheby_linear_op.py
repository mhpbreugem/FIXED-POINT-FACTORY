"""Linear-grid operator: a CDF-uniform / u-uniform grid representation of P
with linear interpolation (no Chebyshev). Same tabulated-mu architecture as
cheby_numba_tab.

We use uniform-in-u nodes on [-U_max, U_max]. To compare apples to apples
with the Chebyshev operator at N=6, we use G_lin=7 nodes by default, but
the routine supports any grid size.

Phase A (table build): for each Lobatto-equivalent u_k:
  - slice2 = P_grid[:, :, k_idx] (no transform needed, just read off)
  - For each p in p_grid, find contour roots in xi_b by scanning the
    linear-segment sign changes and interpolating linearly.
  - Co-area weights via piecewise-constant dP/du_b (slope of the segment).

Phase B (clearing): identical to cheby_numba_tab.
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, NQ, GL_NODES, GL_WEIGHTS,
                            f_signal_jit, crra_clear_jit)


# ===== Grid construction =====
def make_uniform_u_grid(G, U_max=4.0):
    """Uniform-in-u grid on [-U_max, U_max]."""
    return np.linspace(-U_max, U_max, G)


def make_cdf_uniform_grid(G, U_max=4.0, std=1.0):
    """Uniform-in-CDF (Gaussian) grid on [-U_max, U_max].
    Equivalent to: u_i = std * Phi^{-1}(F_min + i*(F_max - F_min)/(G-1)),
    bracketed to [-U_max, U_max]."""
    from scipy.stats import norm
    F_lo = norm.cdf(-U_max / std)
    F_hi = norm.cdf( U_max / std)
    return std * norm.ppf(np.linspace(F_lo, F_hi, G))


# ===== Linear interpolation primitives =====
@njit(cache=True, inline='always')
def _find_interval(x, q, n):
    """Find lo such that x[lo] <= q <= x[lo+1]. Clamps to ends."""
    if q <= x[0]: return 0
    if q >= x[n-1]: return n-2
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= q:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, inline='always')
def linear_1d(x, y, q, n):
    """Linear interp of y on grid x at query q."""
    if q <= x[0]: return y[0]
    if q >= x[n-1]: return y[n-1]
    i = _find_interval(x, q, n)
    w = (q - x[i]) / (x[i+1] - x[i])
    return (1.0 - w) * y[i] + w * y[i+1]


@njit(cache=True, inline='always')
def linear_1d_slope(x, y, q, n):
    """Slope of the linear interpolant at query q (constant per segment)."""
    if q <= x[0]:
        return (y[1] - y[0]) / (x[1] - x[0])
    if q >= x[n-1]:
        return (y[n-1] - y[n-2]) / (x[n-1] - x[n-2])
    i = _find_interval(x, q, n)
    return (y[i+1] - y[i]) / (x[i+1] - x[i])


@njit(cache=True)
def slice_eval_2d(u_grid, P_slice2d, ua_q, n_grid):
    """Bilinear interp of P_slice2d (G x G) on (u_grid, u_grid) at (ua_q, ub).
    Returns a 1D array along the second axis (no interp in second axis
    — we want to evaluate at all grid nodes for ub)."""
    out = np.empty(n_grid)
    if ua_q <= u_grid[0]:
        for j in range(n_grid):
            out[j] = P_slice2d[0, j]
        return out
    if ua_q >= u_grid[n_grid-1]:
        for j in range(n_grid):
            out[j] = P_slice2d[n_grid-1, j]
        return out
    i = _find_interval(u_grid, ua_q, n_grid)
    w = (ua_q - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    for j in range(n_grid):
        out[j] = (1.0 - w) * P_slice2d[i, j] + w * P_slice2d[i+1, j]
    return out


# ===== Contour root finding (grid traversal + linear interp) =====
@njit(cache=True)
def find_root_grid(u_grid, P_along_b, p_target, n_grid):
    """Find first u_b where the piecewise-linear interpolant of P_along_b
    crosses p_target.  Returns (u_root, slope) or (NaN, NaN) if none."""
    for j in range(n_grid - 1):
        a = P_along_b[j] - p_target
        b = P_along_b[j+1] - p_target
        if a * b <= 0 and (a != 0 or b != 0):
            # Linear root
            ya = P_along_b[j]; yb = P_along_b[j+1]
            xa = u_grid[j]; xb = u_grid[j+1]
            slope = (yb - ya) / (xb - xa)
            if slope == 0:
                continue
            u_root = xa + (p_target - ya) / slope
            return u_root, slope
    return float('nan'), float('nan')


# ===== Co-area integral on the linear-grid representation =====
@njit(cache=True)
def co_area_linear(u_grid, P_slice2d, p_target, gl_nodes, gl_weights,
                     tau, n_grid, nq, u_max):
    """Co-area integral A_v = (1/2) int f_v(u_a) f_v(u_b) / |dP/du_b| du_a.
    Maps GL nodes xi_a in [-1, 1] to u_a = u_max * xi_a (uniform measure).
    Returns (A0, A1)."""
    A0 = 0.0; A1 = 0.0
    for q in range(nq):
        xi_a = gl_nodes[q]
        w_gl = gl_weights[q]
        u_a = u_max * xi_a
        # Jacobian du/dxi for linear map is just u_max
        du_dxi_a = u_max
        # Evaluate slice along u_b axis at u_a
        P_along_b = slice_eval_2d(u_grid, P_slice2d, u_a, n_grid)
        u_root, slope = find_root_grid(u_grid, P_along_b, p_target, n_grid)
        if math.isnan(u_root):
            continue
        if abs(slope) < 1e-12:
            continue
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        f0b = f_signal_jit(u_root, 0, tau)
        f1b = f_signal_jit(u_root, 1, tau)
        wt = w_gl * du_dxi_a * 1.0 / abs(slope)
        # Note: there's no du_b/dxi_b jacobian factor because we work in
        # u-space directly, not xi-space.
        A0 += wt * f0a * f0b
        A1 += wt * f1a * f1b
    return 0.5 * A0, 0.5 * A1


# ===== mu(p, u_k) table builder (linear grid version) =====
@njit(cache=True)
def build_mu_table_linear(P_vals, u_grid, p_grid, gl_nodes, gl_weights,
                            tau, n_grid, nq, u_max):
    """Build mu(p, u_k) table from P_vals[i,j,k] on (u_grid)^3.
    Returns mu_table[G_p, n_grid] computed by fixing axis 0 = each grid node."""
    G_p = p_grid.shape[0]
    mu_table = np.empty((G_p, n_grid))
    slice2 = np.empty((n_grid, n_grid))
    for k_node in range(n_grid):  # fix axis 0 = u_grid[k_node]
        u_k = u_grid[k_node]
        f0 = f_signal_jit(u_k, 0, tau)
        f1 = f_signal_jit(u_k, 1, tau)
        for jj in range(n_grid):
            for kk in range(n_grid):
                slice2[jj, kk] = P_vals[k_node, jj, kk]
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_linear(u_grid, slice2, p, gl_nodes, gl_weights,
                                       tau, n_grid, nq, u_max)
            den = f0*A0 + f1*A1
            if den > 1e-30:
                mu_table[ip, k_node] = f1*A1 / den
            else:
                mu_table[ip, k_node] = 0.5
    return mu_table


@njit(cache=True, inline='always')
def interp_mu_1d(mu_table, p, p_grid, k_idx, G_p):
    """Linear interpolation of mu_table in p; k_idx is integer."""
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    i = _find_interval(p_grid, p, G_p)
    w = (p - p_grid[i]) / (p_grid[i+1] - p_grid[i])
    return (1.0 - w) * mu_table[i, k_idx] + w * mu_table[i+1, k_idx]


@njit(cache=True)
def phi_linear_jit(P_vals, u_grid, p_grid, gl_nodes, gl_weights,
                     tau, gamma, n_grid, nq, u_max):
    """Linear-grid version of phi_tab. P_vals stored on uniform-u grid."""
    mu_table = build_mu_table_linear(P_vals, u_grid, p_grid, gl_nodes,
                                        gl_weights, tau, n_grid, nq, u_max)
    G_p = p_grid.shape[0]
    P_new = np.empty((n_grid, n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                p_cell = P_vals[i, j, k]
                eps = 1e-9
                if p_cell < eps: p_cell = eps
                elif p_cell > 1 - eps: p_cell = 1 - eps
                mu0 = interp_mu_1d(mu_table, p_cell, p_grid, i, G_p)
                mu1 = interp_mu_1d(mu_table, p_cell, p_grid, j, G_p)
                mu2 = interp_mu_1d(mu_table, p_cell, p_grid, k, G_p)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def phi_linear(P_vals, u_grid, gamma=GAMMA, tau=TAU, G_p=51, u_max=None):
    """Python wrapper for the linear-grid operator."""
    if u_max is None:
        u_max = max(abs(u_grid[0]), abs(u_grid[-1]))
    p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8.0, 8.0, G_p)))
    n_grid = u_grid.shape[0]
    return phi_linear_jit(P_vals, u_grid, p_grid, GL_NODES, GL_WEIGHTS,
                            tau, gamma, n_grid, NQ, float(u_max))


if __name__ == '__main__':
    print('=== Linear-grid operator quick test ===\n')
    # Build a test conjecture on a uniform grid
    G_lin = 11
    U_max = 4.0
    u_grid = make_uniform_u_grid(G_lin, U_max)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = TAU*(U1+U2+U3)
    def sg(x): return 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    # Warmup
    _ = phi_linear(P_in, u_grid, G_p=51)

    print(f'G_lin={G_lin}, U_max={U_max}, G_p=51:')
    for trial in range(3):
        t0 = time.time(); P_out = phi_linear(P_in, u_grid, G_p=51); dt = time.time()-t0
        print(f'  trial {trial+1}: t={dt*1000:.1f} ms, '
              f'P_out range [{P_out.min():.3f}, {P_out.max():.3f}]')
