"""STRICT h=0 (NO kernel) on linear-CDF grid with C^2 NATURAL CUBIC SPLINE.

Improvement over PCHIP: globally C^2 (not just C^1).
  - Natural cubic spline: M_0 = M_{n-1} = 0
  - Solve tridiagonal Thomas system for interior second derivatives
  - Evaluate via Hermite-like cubic basis
  - Derivative AND second derivative continuous at every grid line

Same architecture: CDF u-grid + POU + tabulated mu + per-cube CRRA clearing.
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
def natural_cspl_M(x, y, n, M):
    """Compute second-derivatives M[i] for a NATURAL cubic spline.
    Solves tridiagonal system with M[0] = M[n-1] = 0.
    Stores results in M (length n)."""
    if n < 3:
        for i in range(n): M[i] = 0.0
        return
    h = np.empty(n - 1)
    for i in range(n - 1):
        h[i] = x[i+1] - x[i]
    # Build RHS: rhs[i] = 6 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        rhs[i] = 6.0 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    # Tridiagonal: diag[i] = 2*(h[i-1]+h[i]), sub/super = h[i-1] / h[i]
    # M[0] = M[n-1] = 0 (natural BC)
    M[0] = 0.0; M[n-1] = 0.0
    c = np.zeros(n)  # super-diag after elim
    d = np.zeros(n)  # rhs after elim
    # First interior row i=1: diag = 2*(h[0]+h[1]), super = h[1]
    diag0 = 2.0*(h[0] + h[1])
    c[1] = h[1] / diag0
    d[1] = rhs[1] / diag0
    for i in range(2, n - 1):
        m_pre = h[i-1] / 1.0  # sub-diag is h[i-1]
        # Eliminate
        denom = 2.0*(h[i-1] + h[i]) - h[i-1] * c[i-1]
        c[i] = h[i] / denom
        d[i] = (rhs[i] - h[i-1] * d[i-1]) / denom
    M[n-2] = d[n-2]
    for i in range(n - 3, 0, -1):
        M[i] = d[i] - c[i] * M[i+1]


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
def cspl_eval(x, y, M, q, n):
    """Evaluate natural cubic spline at q. Returns (value, derivative)."""
    if q <= x[0]:
        # Linear extrapolation with the natural-spline boundary slope
        # For natural spline M[0]=0, slope at x[0]:
        h0 = x[1] - x[0]
        slope0 = (y[1] - y[0])/h0 - h0*(2.0*M[0] + M[1])/6.0
        return y[0] + slope0*(q - x[0]), slope0
    if q >= x[n-1]:
        hN = x[n-1] - x[n-2]
        slopeN = (y[n-1] - y[n-2])/hN + hN*(M[n-2] + 2.0*M[n-1])/6.0
        return y[n-1] + slopeN*(q - x[n-1]), slopeN
    i = _find_interval(x, q, n)
    h = x[i+1] - x[i]
    a = (x[i+1] - q) / h
    b = (q - x[i]) / h
    val = (a*y[i] + b*y[i+1]
           + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * h*h / 6.0)
    der = ((y[i+1] - y[i]) / h
           - (3.0*a*a - 1.0) / 6.0 * h * M[i]
           + (3.0*b*b - 1.0) / 6.0 * h * M[i+1])
    return val, der


@njit(cache=True)
def cspl_all_roots(x, y, M, p_target, n, roots_out, slopes_out):
    """Find all roots of cspl - p_target = 0 in [x[0], x[n-1]]."""
    nr = 0
    SUB = 4
    for i in range(n - 1):
        h_cell = (x[i+1] - x[i]) / SUB
        q_prev = x[i]
        v_prev, _ = cspl_eval(x, y, M, q_prev, n)
        s_prev = v_prev - p_target
        for s in range(1, SUB + 1):
            q = x[i] + s*h_cell
            v, _ = cspl_eval(x, y, M, q, n)
            s_cur = v - p_target
            if s_prev == 0.0:
                if nr < MAX_ROOTS:
                    _, der = cspl_eval(x, y, M, q_prev, n)
                    roots_out[nr] = q_prev; slopes_out[nr] = der
                    nr += 1
            elif s_prev * s_cur < 0:
                lo = q_prev; hi = q; v_lo = s_prev
                for _ in range(60):
                    mid = 0.5*(lo + hi)
                    vm, _ = cspl_eval(x, y, M, mid, n)
                    vmm = vm - p_target
                    if v_lo * vmm <= 0:
                        hi = mid
                    else:
                        lo = mid; v_lo = vmm
                    if hi - lo < 1e-15:
                        break
                u_root = 0.5*(lo + hi)
                _, der = cspl_eval(x, y, M, u_root, n)
                if nr < MAX_ROOTS:
                    roots_out[nr] = u_root; slopes_out[nr] = der
                    nr += 1
            q_prev = q; v_prev = v; s_prev = s_cur
    return nr


@njit(cache=True)
def cspl_slice_eval_axis0(u_grid, slice2d, M_axis0, u_a, n_grid):
    out = np.empty(n_grid)
    for jj in range(n_grid):
        v, _ = cspl_eval(u_grid, slice2d[:, jj], M_axis0[:, jj], u_a, n_grid)
        out[jj] = v
    return out


@njit(cache=True)
def cspl_slice_eval_axis1(u_grid, slice2d, M_axis1, u_b, n_grid):
    out = np.empty(n_grid)
    for ii in range(n_grid):
        v, _ = cspl_eval(u_grid, slice2d[ii, :], M_axis1[ii, :], u_b, n_grid)
        out[ii] = v
    return out


@njit(cache=True)
def precompute_cspl_M_2d(u_grid, slice2d, n_grid):
    """Precompute axis-0 and axis-1 cubic-spline M arrays."""
    M_axis0 = np.empty((n_grid, n_grid))
    M_axis1 = np.empty((n_grid, n_grid))
    M_tmp = np.empty(n_grid)
    for jj in range(n_grid):
        natural_cspl_M(u_grid, slice2d[:, jj], n_grid, M_tmp)
        for i in range(n_grid):
            M_axis0[i, jj] = M_tmp[i]
    for ii in range(n_grid):
        natural_cspl_M(u_grid, slice2d[ii, :], n_grid, M_tmp)
        for j in range(n_grid):
            M_axis1[ii, j] = M_tmp[j]
    return M_axis0, M_axis1


@njit(cache=True)
def co_area_pou_cspl(u_grid, slice2d, p_target, gl_u, gl_du,
                       tau, n_grid, nq):
    """POU strict h=0 co-area on natural cubic-spline linear-CDF grid."""
    A0 = 0.0; A1 = 0.0
    M_axis0, M_axis1 = precompute_cspl_M_2d(u_grid, slice2d, n_grid)
    roots = np.empty(MAX_ROOTS); slopes = np.empty(MAX_ROOTS)
    M_1d = np.empty(n_grid)

    # Term A^(b)
    for q in range(nq):
        u_a = gl_u[q]; w_a = gl_du[q]
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        P_along_b = cspl_slice_eval_axis0(u_grid, slice2d, M_axis0, u_a, n_grid)
        natural_cspl_M(u_grid, P_along_b, n_grid, M_1d)
        nr = cspl_all_roots(u_grid, P_along_b, M_1d, p_target, n_grid,
                              roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_b = roots[r]
            dPdu_b = slopes[r]
            P_along_a = cspl_slice_eval_axis1(u_grid, slice2d, M_axis1, u_b, n_grid)
            natural_cspl_M(u_grid, P_along_a, n_grid, M_1d)
            _, dPdu_a = cspl_eval(u_grid, P_along_a, M_1d, u_a, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a)
    for q in range(nq):
        u_b = gl_u[q]; w_b = gl_du[q]
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        P_along_a = cspl_slice_eval_axis1(u_grid, slice2d, M_axis1, u_b, n_grid)
        natural_cspl_M(u_grid, P_along_a, n_grid, M_1d)
        nr = cspl_all_roots(u_grid, P_along_a, M_1d, p_target, n_grid,
                              roots, slopes)
        if nr == 0: continue
        for r in range(nr):
            u_a = roots[r]
            dPdu_a = slopes[r]
            P_along_b = cspl_slice_eval_axis0(u_grid, slice2d, M_axis0, u_a, n_grid)
            natural_cspl_M(u_grid, P_along_b, n_grid, M_1d)
            _, dPdu_b = cspl_eval(u_grid, P_along_b, M_1d, u_b, n_grid)
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
def build_mu_table_cspl(P_vals, u_grid, p_grid, gl_u, gl_du,
                          tau, n_grid, nq):
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
            A0, A1 = co_area_pou_cspl(u_grid, slice2d, p, gl_u, gl_du,
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
def phi_lin_cspl_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                        tau, gamma, n_grid, nq):
    mu_table = build_mu_table_cspl(P_vals, u_grid, p_grid, gl_u, gl_du,
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


def phi_lin_cspl(P_vals, u_grid, gamma=1.0, tau=1.0, G_p=121, NQ=16,
                    p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    G = u_grid.size
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ)
    return phi_lin_cspl_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                               tau, gamma, G, NQ)


if __name__ == '__main__':
    print('=== Linear-CDF C^2 cubic-spline strict h=0 POU operator ===\n')
    G = 7
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = U1+U2+U3
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)
    _ = phi_lin_cspl(P_in, u_grid, NQ=16, G_p=121)
    for trial in range(3):
        t0 = time.time(); P_out = phi_lin_cspl(P_in, u_grid, NQ=16, G_p=121)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  trial {trial+1}: |F|={F:.3e}, t={dt*1000:.1f} ms')
