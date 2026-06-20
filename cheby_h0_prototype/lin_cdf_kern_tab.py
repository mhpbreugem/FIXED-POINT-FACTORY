"""CDF-linear-grid kernel-band tabulated-mu operator.

Same architecture as cheby_numba_kern_tab.py:
  - Phase A: mu(p, u_k) table via Gaussian kernel K_h(P-p) over 2D slice
  - Phase B: per-cube cell lookup + CRRA clearing

But the underlying P representation is:
  - u-grid: CDF-uniform (Gaussian quantiles) -- concentrates points
    where signal density is high
  - Interp: bilinear (no Chebyshev)
  - Quadrature: GL nodes on the u-range mapped via CDF
"""
import os, time, math
import numpy as np
from numba import njit
from scipy.stats import norm

from cheby_numba import f_signal_jit, crra_clear_jit, NQ as NQ_default

DEFAULT_H_KERN = 0.30


def make_cdf_uniform_grid(G, mode_spread=2.0, eps_q=0.01):
    """CDF-uniform grid for binary signal model.
    Take quantiles of a Gaussian N(0, 1) covering the [-mode_spread, +mode_spread]
    range, plus a few outside for tails."""
    qs = np.linspace(eps_q, 1 - eps_q, G)
    u = norm.ppf(qs)
    # Optionally rescale to match Cheb's atanh-stretched range
    return u


def _build_interp_helpers(u_grid):
    """Return arrays needed for fast bilinear interp."""
    n = u_grid.size
    du = np.diff(u_grid)  # n-1 spacings (non-uniform OK)
    return du


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


@njit(cache=True)
def linterp_slice_2d_at_nodes(u_grid, slice2d, gl_u_nodes, n_grid, nqk):
    """Evaluate slice2d on a tensor grid of (gl_u_nodes, gl_u_nodes) via
    bilinear interpolation. Returns out[q_a, q_b]."""
    out = np.empty((nqk, nqk))
    for q_a in range(nqk):
        u_a = gl_u_nodes[q_a]
        # Locate interval in axis 0
        if u_a <= u_grid[0]: ia = 0; w_a = 0.0
        elif u_a >= u_grid[n_grid-1]: ia = n_grid-2; w_a = 1.0
        else:
            ia = _find_interval(u_grid, u_a, n_grid)
            w_a = (u_a - u_grid[ia]) / (u_grid[ia+1] - u_grid[ia])
        for q_b in range(nqk):
            u_b = gl_u_nodes[q_b]
            if u_b <= u_grid[0]: ib = 0; w_b = 0.0
            elif u_b >= u_grid[n_grid-1]: ib = n_grid-2; w_b = 1.0
            else:
                ib = _find_interval(u_grid, u_b, n_grid)
                w_b = (u_b - u_grid[ib]) / (u_grid[ib+1] - u_grid[ib])
            # Bilinear
            v = ((1-w_a)*(1-w_b)*slice2d[ia, ib]
                  + w_a*(1-w_b)*slice2d[ia+1, ib]
                  + (1-w_a)*w_b*slice2d[ia, ib+1]
                  + w_a*w_b*slice2d[ia+1, ib+1])
            out[q_a, q_b] = v
    return out


@njit(cache=True)
def build_mu_table_lin_kern(P_vals, u_grid, p_grid,
                              gl_u_nodes, gl_du_weights, tau,
                              n_grid, nqk, kernel_h):
    """mu(p, u_k) table on CDF-linear grid via kernel-band integral.
    gl_u_nodes: physical u-locations of GL quadrature
    gl_du_weights: full quadrature weight w_q * (du/dxi)_q
                   (precomputed since u-range is fixed)
    """
    G = n_grid
    G_p = p_grid.size
    inv_2h2 = 0.5 / (kernel_h * kernel_h)
    mu_table = np.empty((G_p, G))
    # Pre-compute f_v at GL nodes
    f0_gl = np.empty(nqk); f1_gl = np.empty(nqk)
    for q in range(nqk):
        u_q = gl_u_nodes[q]
        f0_gl[q] = f_signal_jit(u_q, 0, tau)
        f1_gl[q] = f_signal_jit(u_q, 1, tau)
    for k_node in range(G):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau)
        f1k = f_signal_jit(u_k, 1, tau)
        # Extract 2D slice = P[k_node, :, :]
        slice2d = np.empty((G, G))
        for jj in range(G):
            for kk in range(G):
                slice2d[jj, kk] = P_vals[k_node, jj, kk]
        # Evaluate slice on tensor GL nodes via bilinear interp
        P_at_nodes = linterp_slice_2d_at_nodes(u_grid, slice2d, gl_u_nodes,
                                                  G, nqk)
        for ip in range(G_p):
            p = p_grid[ip]
            A0 = 0.0; A1 = 0.0
            for q_a in range(nqk):
                wa = gl_du_weights[q_a]
                f0a = f0_gl[q_a]; f1a = f1_gl[q_a]
                for q_b in range(nqk):
                    wb = gl_du_weights[q_b]
                    diff = P_at_nodes[q_a, q_b] - p
                    w = math.exp(-diff * diff * inv_2h2)
                    f0b = f0_gl[q_b]; f1b = f1_gl[q_b]
                    A0 += wa * wb * w * f0a * f0b
                    A1 += wa * wb * w * f1a * f1b
            den = f0k * A0 + f1k * A1
            if den > 1e-300:
                mu_table[ip, k_node] = f1k * A1 / den
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
def phi_lin_kern_tab_jit(P_vals, u_grid, p_grid, gl_u_nodes, gl_du_weights,
                            tau, gamma, n_grid, nqk, kernel_h):
    mu_table = build_mu_table_lin_kern(P_vals, u_grid, p_grid,
                                          gl_u_nodes, gl_du_weights, tau,
                                          n_grid, nqk, kernel_h)
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
    """GL nodes mapped to [u_min, u_max]. Returns (nodes_u, du_weights).
    The full integration weight is w_q * (u_max-u_min)/2."""
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    u_nodes = 0.5*(u_max + u_min) + 0.5*(u_max - u_min)*n_xi
    du_w = 0.5*(u_max - u_min) * w_xi
    return u_nodes, du_w


def phi_lin_kern_tab(P_vals, u_grid, gamma=1.0, tau=1.0, G_p=121, NQK=16,
                       kernel_h=DEFAULT_H_KERN, p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    G = u_grid.size
    # Use slightly extended range for GL (cover the grid + a bit)
    u_min = u_grid[0]; u_max = u_grid[-1]
    gl_u, gl_du = make_gl_for_u(u_min, u_max, NQK)
    return phi_lin_kern_tab_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                                   tau, gamma, G, NQK, kernel_h)


if __name__ == '__main__':
    print('=== Linear-CDF kernel-band tabulated mu operator self-test ===\n')
    G = 7
    u_grid = make_cdf_uniform_grid(G)
    print(f'u_grid (G={G}): {u_grid}')
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = U1+U2+U3
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    # warmup
    _ = phi_lin_kern_tab(P_in, u_grid, kernel_h=0.3, G_p=121, NQK=16)
    print(f'\nTimings (G_p=121, NQK=16, h=0.3):')
    for trial in range(3):
        t0 = time.time()
        P_out = phi_lin_kern_tab(P_in, u_grid, kernel_h=0.3, G_p=121, NQK=16)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  trial {trial+1}: |F|={F:.3e}, t={dt*1000:.1f} ms')
