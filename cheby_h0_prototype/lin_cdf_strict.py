"""STRICT h=0 (NO kernel) co-area operator on linear-CDF grid + POU.

Same approach as the Chebyshev POU operator but with:
  - u-grid: Gaussian quantiles (CDF-uniform)
  - P: stored as grid values, piecewise-linear interp for off-grid eval
  - Slice: extracted as 2D grid of values
  - Roots: piecewise-linear sign-change scan (each segment has one root)
  - Co-area: partition-of-unity over both axes (kills 1/|dP/du| singularity)
  - Lookup table mu(p, u_k) + per-cube CRRA clearing
"""
import os, time, math
import numpy as np
from numba import njit
from scipy.stats import norm

from cheby_numba import f_signal_jit, crra_clear_jit

MAX_ROOTS = 16

def make_cdf_uniform_grid(G, eps_q=0.01):
    qs = np.linspace(eps_q, 1 - eps_q, G)
    return norm.ppf(qs)


@njit(cache=True, inline='always')
def _find_interval(u_grid, q, n):
    if q <= u_grid[0]: return 0
    if q >= u_grid[n-1]: return n-2
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u_grid[mid] <= q: lo = mid
        else: hi = mid
    return lo


@njit(cache=True, inline='always')
def linterp_1d(u_grid, y, q, n):
    if q <= u_grid[0]: return y[0]
    if q >= u_grid[n-1]: return y[n-1]
    i = _find_interval(u_grid, q, n)
    w = (q - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    return (1.0 - w) * y[i] + w * y[i+1]


@njit(cache=True, inline='always')
def linterp_deriv_1d(u_grid, y, q, n):
    """Piecewise-constant derivative (slope of the linear segment)."""
    if q <= u_grid[0]:
        return (y[1] - y[0]) / (u_grid[1] - u_grid[0])
    if q >= u_grid[n-1]:
        return (y[n-1] - y[n-2]) / (u_grid[n-1] - u_grid[n-2])
    i = _find_interval(u_grid, q, n)
    return (y[i+1] - y[i]) / (u_grid[i+1] - u_grid[i])


@njit(cache=True)
def all_roots_linear(u_grid, P_along, p_target, n, roots_out, slopes_out):
    """Find all u where piecewise-linear P_along crosses p_target.
    Stores (u_root, slope) in roots_out, slopes_out. Returns count."""
    nr = 0
    for j in range(n - 1):
        a = P_along[j] - p_target
        b = P_along[j+1] - p_target
        if a == 0:
            if nr < MAX_ROOTS:
                roots_out[nr] = u_grid[j]
                slopes_out[nr] = (P_along[j+1] - P_along[j]) / (u_grid[j+1] - u_grid[j])
                nr += 1
        elif a * b < 0:
            slope = (P_along[j+1] - P_along[j]) / (u_grid[j+1] - u_grid[j])
            if slope == 0: continue
            u_root = u_grid[j] + (p_target - P_along[j]) / slope
            if nr < MAX_ROOTS:
                roots_out[nr] = u_root
                slopes_out[nr] = slope
                nr += 1
    # last endpoint
    if P_along[n-1] - p_target == 0:
        if nr < MAX_ROOTS:
            roots_out[nr] = u_grid[n-1]
            slopes_out[nr] = (P_along[n-1] - P_along[n-2]) / (u_grid[n-1] - u_grid[n-2])
            nr += 1
    return nr


@njit(cache=True)
def linterp_slice_eval(u_grid, P_slice2d, u_a, n_grid):
    """Evaluate slice2d at u_a along axis 0; returns 1D array along axis 1."""
    out = np.empty(n_grid)
    if u_a <= u_grid[0]:
        for j in range(n_grid):
            out[j] = P_slice2d[0, j]
        return out
    if u_a >= u_grid[n_grid-1]:
        for j in range(n_grid):
            out[j] = P_slice2d[n_grid-1, j]
        return out
    i = _find_interval(u_grid, u_a, n_grid)
    w = (u_a - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    for j in range(n_grid):
        out[j] = (1.0 - w) * P_slice2d[i, j] + w * P_slice2d[i+1, j]
    return out


@njit(cache=True)
def linterp_slice_eval_axis_b(u_grid, P_slice2d, u_b, n_grid):
    """Evaluate slice2d at u_b along axis 1; returns 1D array along axis 0."""
    out = np.empty(n_grid)
    if u_b <= u_grid[0]:
        for i in range(n_grid):
            out[i] = P_slice2d[i, 0]
        return out
    if u_b >= u_grid[n_grid-1]:
        for i in range(n_grid):
            out[i] = P_slice2d[i, n_grid-1]
        return out
    j = _find_interval(u_grid, u_b, n_grid)
    w = (u_b - u_grid[j]) / (u_grid[j+1] - u_grid[j])
    for i in range(n_grid):
        out[i] = (1.0 - w) * P_slice2d[i, j] + w * P_slice2d[i, j+1]
    return out


@njit(cache=True)
def co_area_pou_lin(u_grid, slice2d, p_target,
                      gl_u_a, gl_du_a, gl_u_b, gl_du_b,
                      tau, n_grid, nq):
    """POU co-area integral on linear-CDF grid.
    A_v = A^(b) + A^(a), each with partition weight d^2/(d_a^2+d_b^2).
    """
    A0 = 0.0; A1 = 0.0
    roots = np.empty(MAX_ROOTS); slopes = np.empty(MAX_ROOTS)

    # Term A^(b): for each u_a GL node, find u_b roots
    for q in range(nq):
        u_a = gl_u_a[q]
        w_a = gl_du_a[q]
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        # 1D slice along u_b at u_a fixed
        P_along_b = linterp_slice_eval(u_grid, slice2d, u_a, n_grid)
        nr = all_roots_linear(u_grid, P_along_b, p_target, n_grid, roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_b = roots[r]
            dPdu_b = slopes[r]
            # dPdu_a from linear-segment derivative along axis 0 at (u_a, u_b)
            # We need dP/du_a at the root point. Compute via differencing the
            # slice along axis 0: extract row at u_b interp, then linear deriv.
            P_along_a = linterp_slice_eval_axis_b(u_grid, slice2d, u_b, n_grid)
            dPdu_a = linterp_deriv_1d(u_grid, P_along_a, u_a, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau)
            f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a): for each u_b GL node, find u_a roots
    for q in range(nq):
        u_b = gl_u_b[q]
        w_b = gl_du_b[q]
        f0b = f_signal_jit(u_b, 0, tau)
        f1b = f_signal_jit(u_b, 1, tau)
        P_along_a = linterp_slice_eval_axis_b(u_grid, slice2d, u_b, n_grid)
        nr = all_roots_linear(u_grid, P_along_a, p_target, n_grid, roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_a = roots[r]
            dPdu_a = slopes[r]
            P_along_b = linterp_slice_eval(u_grid, slice2d, u_a, n_grid)
            dPdu_b = linterp_deriv_1d(u_grid, P_along_b, u_b, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal_jit(u_a, 0, tau)
            f1a = f_signal_jit(u_a, 1, tau)
            wt = w_b * w_a_pou / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return A0, A1


@njit(cache=True)
def build_mu_table_lin_strict(P_vals, u_grid, p_grid,
                                 gl_u, gl_du, tau, n_grid, nq):
    G = n_grid; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    for k_node in range(G):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau)
        f1k = f_signal_jit(u_k, 1, tau)
        slice2d = np.empty((G, G))
        for jj in range(G):
            for kk in range(G):
                slice2d[jj, kk] = P_vals[k_node, jj, kk]
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pou_lin(u_grid, slice2d, p, gl_u, gl_du,
                                        gl_u, gl_du, tau, G, nq)
            den = f0k*A0 + f1k*A1
            if den > 1e-300:
                mu_table[ip, k_node] = f1k*A1 / den
            else:
                mu_table[ip, k_node] = 0.5
    return mu_table


@njit(cache=True, inline='always')
def _interp_mu(mu_table, p, p_grid, k_idx, G_p):
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    lo = 0; hi = G_p - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_grid[mid] <= p: lo = mid
        else: hi = mid
    w = (p - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    return (1.0 - w) * mu_table[lo, k_idx] + w * mu_table[hi, k_idx]


@njit(cache=True)
def phi_lin_strict_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                          tau, gamma, n_grid, nq):
    mu_table = build_mu_table_lin_strict(P_vals, u_grid, p_grid,
                                            gl_u, gl_du, tau, n_grid, nq)
    G = n_grid; G_p = p_grid.size
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                eps = 1e-9
                if p_cell < eps: p_cell = eps
                elif p_cell > 1 - eps: p_cell = 1 - eps
                mu0 = _interp_mu(mu_table, p_cell, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, p_cell, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, p_cell, p_grid, k, G_p)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def make_p_grid(G_p=121, L=8.0):
    return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))


def make_gl_for_u(u_min, u_max, NQK):
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    u_nodes = 0.5*(u_max + u_min) + 0.5*(u_max - u_min)*n_xi
    du_w = 0.5*(u_max - u_min) * w_xi
    return u_nodes, du_w


def phi_lin_strict(P_vals, u_grid, gamma=1.0, tau=1.0, G_p=121, NQ=16,
                      p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    G = u_grid.size
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ)
    return phi_lin_strict_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                                 tau, gamma, G, NQ)


if __name__ == '__main__':
    print('=== Linear-CDF strict h=0 POU operator self-test ===\n')
    G = 7
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = U1+U2+U3
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)
    _ = phi_lin_strict(P_in, u_grid, NQ=16, G_p=121)
    for trial in range(3):
        t0 = time.time()
        P_out = phi_lin_strict(P_in, u_grid, NQ=16, G_p=121)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  trial {trial+1}: |F|={F:.3e}, t={dt*1000:.1f} ms')
