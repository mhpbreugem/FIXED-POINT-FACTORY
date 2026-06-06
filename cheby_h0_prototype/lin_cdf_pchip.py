"""STRICT h=0 (NO kernel) on linear-CDF grid with PCHIP smoothing.

The Lagrangian "line-shifting smooth" approach for the strict line integral:
  - Same CDF-uniform u-grid
  - Replace piecewise-linear interp with PCHIP (C^1 cubic Hermite,
    Fritsch-Carlson monotone slopes)
  - dP/du is now continuous everywhere -> no jumps -> co-area integrand
    smooth in P -> Newton can converge
  - POU and lookup table unchanged
  - Roots found via sign-change scan on PCHIP-evaluated slice values
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


@njit(cache=True)
def pchip_slopes(x, y, n, slopes_out):
    """Fritsch-Carlson monotone cubic Hermite slopes."""
    # Interior slopes via harmonic mean of secants
    h = np.empty(n - 1)
    delta = np.empty(n - 1)
    for i in range(n - 1):
        h[i] = x[i+1] - x[i]
        delta[i] = (y[i+1] - y[i]) / h[i]
    # Interior
    for i in range(1, n - 1):
        if delta[i-1] * delta[i] <= 0:
            slopes_out[i] = 0.0
        else:
            w1 = 2.0*h[i] + h[i-1]
            w2 = h[i] + 2.0*h[i-1]
            slopes_out[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])
    # End slopes via one-sided 3-point formula, with shape-preserving limits
    # Left
    s = ((2.0*h[0] + h[1])*delta[0] - h[0]*delta[1]) / (h[0] + h[1])
    if s * delta[0] <= 0:
        s = 0.0
    elif delta[0]*delta[1] <= 0 and abs(s) > abs(3.0*delta[0]):
        s = 3.0 * delta[0]
    slopes_out[0] = s
    # Right
    s = ((2.0*h[n-2] + h[n-3])*delta[n-2] - h[n-2]*delta[n-3]) / (h[n-2] + h[n-3])
    if s * delta[n-2] <= 0:
        s = 0.0
    elif delta[n-2]*delta[n-3] <= 0 and abs(s) > abs(3.0*delta[n-2]):
        s = 3.0 * delta[n-2]
    slopes_out[n-1] = s


@njit(cache=True, inline='always')
def _find_interval(x, q, n):
    if q <= x[0]: return 0
    if q >= x[n-1]: return n-2
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= q: lo = mid
        else: hi = mid
    return lo


@njit(cache=True, inline='always')
def pchip_eval(x, y, m, q, n):
    """Evaluate PCHIP at q. Returns (value, derivative)."""
    if q <= x[0]:
        # extrapolate linearly with end slope
        return y[0] + m[0]*(q - x[0]), m[0]
    if q >= x[n-1]:
        return y[n-1] + m[n-1]*(q - x[n-1]), m[n-1]
    i = _find_interval(x, q, n)
    h = x[i+1] - x[i]
    t = (q - x[i]) / h
    t2 = t*t; t3 = t2*t
    # Hermite basis
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    v = h00*y[i] + h10*h*m[i] + h01*y[i+1] + h11*h*m[i+1]
    # Derivative wrt q
    dh00 = (6*t2 - 6*t) / h
    dh10 = (3*t2 - 4*t + 1) / h
    dh01 = (-6*t2 + 6*t) / h
    dh11 = (3*t2 - 2*t) / h
    d = dh00*y[i] + dh10*h*m[i] + dh01*y[i+1] + dh11*h*m[i+1]
    return v, d


@njit(cache=True)
def pchip_all_roots(x, y, m, p_target, n, roots_out, slopes_out):
    """Find all roots of PCHIP(y, m) = p_target in [x[0], x[n-1]].
    Sign-change scan on subdivided intervals + Newton polish."""
    nr = 0
    SUB = 4  # subdivisions per cell
    for i in range(n - 1):
        # Scan SUB sub-intervals
        h_cell = (x[i+1] - x[i]) / SUB
        q_prev = x[i]
        v_prev, _ = pchip_eval(x, y, m, q_prev, n)
        s_prev = v_prev - p_target
        for s in range(1, SUB + 1):
            q = x[i] + s*h_cell
            v, _ = pchip_eval(x, y, m, q, n)
            s_cur = v - p_target
            if s_prev == 0.0:
                if nr < MAX_ROOTS:
                    _, der = pchip_eval(x, y, m, q_prev, n)
                    roots_out[nr] = q_prev; slopes_out[nr] = der
                    nr += 1
            elif s_prev * s_cur < 0:
                # Bisect on PCHIP
                lo = q_prev; hi = q; v_lo = s_prev; v_hi = s_cur
                for _ in range(60):
                    mid = 0.5*(lo + hi)
                    vm, _ = pchip_eval(x, y, m, mid, n)
                    vmm = vm - p_target
                    if v_lo * vmm <= 0:
                        hi = mid; v_hi = vmm
                    else:
                        lo = mid; v_lo = vmm
                    if hi - lo < 1e-15:
                        break
                u_root = 0.5*(lo + hi)
                _, der = pchip_eval(x, y, m, u_root, n)
                if nr < MAX_ROOTS:
                    roots_out[nr] = u_root; slopes_out[nr] = der
                    nr += 1
            q_prev = q; v_prev = v; s_prev = s_cur
    return nr


@njit(cache=True)
def pchip_slice_eval_axis0(u_grid, slice2d, m_axis0, u_a, n_grid):
    """At fixed u_a, evaluate slice2d (PCHIP-interpolated along axis 0)
    over all (a row of size n_grid in axis 1).
    Returns 1D array of length n_grid (slice along axis 1)."""
    out = np.empty(n_grid)
    for jj in range(n_grid):
        # Column slice2d[:, jj], slopes m_axis0[:, jj]
        v, _ = pchip_eval(u_grid, slice2d[:, jj], m_axis0[:, jj], u_a, n_grid)
        out[jj] = v
    return out


@njit(cache=True)
def pchip_slice_eval_axis1(u_grid, slice2d, m_axis1, u_b, n_grid):
    out = np.empty(n_grid)
    for ii in range(n_grid):
        v, _ = pchip_eval(u_grid, slice2d[ii, :], m_axis1[ii, :], u_b, n_grid)
        out[ii] = v
    return out


@njit(cache=True)
def precompute_pchip_slopes_2d(u_grid, slice2d, n_grid):
    """Precompute axis-0 and axis-1 PCHIP slopes for the 2D slice."""
    m_axis0 = np.empty((n_grid, n_grid))  # m_axis0[i, j] = dP/du_a slope along axis 0 at row j
    m_axis1 = np.empty((n_grid, n_grid))
    slopes_tmp = np.empty(n_grid)
    for jj in range(n_grid):
        pchip_slopes(u_grid, slice2d[:, jj], n_grid, slopes_tmp)
        for i in range(n_grid):
            m_axis0[i, jj] = slopes_tmp[i]
    for ii in range(n_grid):
        pchip_slopes(u_grid, slice2d[ii, :], n_grid, slopes_tmp)
        for j in range(n_grid):
            m_axis1[ii, j] = slopes_tmp[j]
    return m_axis0, m_axis1


@njit(cache=True)
def co_area_pou_pchip(u_grid, slice2d, p_target,
                        gl_u, gl_du, tau, n_grid, nq):
    """POU strict h=0 co-area on PCHIP linear-CDF grid."""
    A0 = 0.0; A1 = 0.0
    m_axis0, m_axis1 = precompute_pchip_slopes_2d(u_grid, slice2d, n_grid)
    roots = np.empty(MAX_ROOTS); slopes = np.empty(MAX_ROOTS)
    slopes_1d = np.empty(n_grid)

    # Term A^(b): fix u_a at GL nodes, find u_b roots
    for q in range(nq):
        u_a = gl_u[q]; w_a = gl_du[q]
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        # Build 1D slice P(u_a, u_b) for varying u_b
        # First, evaluate slice2d at u_a along axis 0 -> 1D in axis 1
        # Plus PCHIP slopes for that 1D
        P_along_b = pchip_slice_eval_axis0(u_grid, slice2d, m_axis0, u_a, n_grid)
        # Slopes of P_along_b (along axis 1) via PCHIP
        pchip_slopes(u_grid, P_along_b, n_grid, slopes_1d)
        nr = pchip_all_roots(u_grid, P_along_b, slopes_1d, p_target, n_grid,
                                roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_b = roots[r]
            dPdu_b = slopes[r]
            # dPdu_a at (u_a, u_b): evaluate axis-0 slope through u_b
            # First get 1D in u_a at u_b (using axis-1 slopes)
            P_along_a = pchip_slice_eval_axis1(u_grid, slice2d, m_axis1, u_b, n_grid)
            pchip_slopes(u_grid, P_along_a, n_grid, slopes_1d)
            _, dPdu_a = pchip_eval(u_grid, P_along_a, slopes_1d, u_a, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a): fix u_b at GL nodes, find u_a roots
    for q in range(nq):
        u_b = gl_u[q]; w_b = gl_du[q]
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        P_along_a = pchip_slice_eval_axis1(u_grid, slice2d, m_axis1, u_b, n_grid)
        pchip_slopes(u_grid, P_along_a, n_grid, slopes_1d)
        nr = pchip_all_roots(u_grid, P_along_a, slopes_1d, p_target, n_grid,
                                roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_a = roots[r]
            dPdu_a = slopes[r]
            P_along_b = pchip_slice_eval_axis0(u_grid, slice2d, m_axis0, u_a, n_grid)
            pchip_slopes(u_grid, P_along_b, n_grid, slopes_1d)
            _, dPdu_b = pchip_eval(u_grid, P_along_b, slopes_1d, u_b, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            wt = w_b * w_a_pou / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return A0, A1


@njit(cache=True)
def build_mu_table_pchip(P_vals, u_grid, p_grid, gl_u, gl_du, tau, n_grid, nq):
    G = n_grid; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    for k_node in range(G):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
        slice2d = np.empty((G, G))
        for jj in range(G):
            for kk in range(G):
                slice2d[jj, kk] = P_vals[k_node, jj, kk]
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pou_pchip(u_grid, slice2d, p, gl_u, gl_du,
                                          tau, G, nq)
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
def phi_lin_pchip_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                         tau, gamma, n_grid, nq):
    mu_table = build_mu_table_pchip(P_vals, u_grid, p_grid, gl_u, gl_du,
                                        tau, n_grid, nq)
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


def phi_lin_pchip(P_vals, u_grid, gamma=1.0, tau=1.0, G_p=121, NQ=16, p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    G = u_grid.size
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ)
    return phi_lin_pchip_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                                tau, gamma, G, NQ)


if __name__ == '__main__':
    print('=== Linear-CDF PCHIP strict h=0 POU operator ===\n')
    G = 7
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = U1+U2+U3
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)
    _ = phi_lin_pchip(P_in, u_grid, NQ=16, G_p=121)
    for trial in range(3):
        t0 = time.time(); P_out = phi_lin_pchip(P_in, u_grid, NQ=16, G_p=121)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  trial {trial+1}: |F|={F:.3e}, t={dt*1000:.1f} ms')
