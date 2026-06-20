"""Kernel-band Chebyshev operator with mu(p, u_k) tabulation.

Replaces chebroots/bisect contour extraction with Gaussian kernel-band
co-area integral over the full 2D slice. Φ becomes analytic in P, so
Newton converges quadratically to machine epsilon.

Phase A (table build):
  For each Lobatto u_k:
    slice2(xi_a, xi_b) = P(xi_a, xi_b, xi_k) via Cheb 2D evaluation
    For each p in p_grid:
      A_v = ∫∫ K_h(P-p) f_v(u_a) f_v(u_b) * du_a/dxi_a * du_b/dxi_b dxi_a dxi_b
      via tensor GL quadrature (NQK x NQK nodes)
      mu_table[ip, k_node] = f_v(u_k) A_1 / (f0 A_0 + f_1 A_1)
Phase B (per-cube-cell):
  For each (i,j,k): lookup mu at (p_cell, i), (p_cell, j), (p_cell, k);
  run CRRA bisection to clear.
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, _T_basis, u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)

# Default kernel bandwidth -- tuneable
DEFAULT_H_KERN = 0.30


@njit(cache=True)
def cheb_eval_3d_at(coeffs, xi1, xi2, xi3, n_grid):
    """Evaluate Cheb 3D polynomial at (xi1, xi2, xi3)."""
    G = n_grid
    T1 = np.empty(G); T2 = np.empty(G); T3 = np.empty(G)
    _T_basis(xi1, G, T1); _T_basis(xi2, G, T2); _T_basis(xi3, G, T3)
    s = 0.0
    for ii in range(G):
        for jj in range(G):
            for kk in range(G):
                s += coeffs[ii, jj, kk] * T1[ii] * T2[jj] * T3[kk]
    return s


@njit(cache=True)
def cheb_eval_slice_axis0(coeffs, xi_fix, n_grid, gl_nodes_a, gl_nodes_b, nqk):
    """Evaluate slice2[q_a, q_b] = P(xi_fix, xi_a^(q_a), xi_b^(q_b))
    at all (NQK x NQK) GL quadrature pairs."""
    G = n_grid
    T_fix = np.empty(G); Ta = np.empty(G); Tb = np.empty(G)
    _T_basis(xi_fix, G, T_fix)
    slice2 = np.empty((nqk, nqk))
    for q_a in range(nqk):
        _T_basis(gl_nodes_a[q_a], G, Ta)
        for q_b in range(nqk):
            _T_basis(gl_nodes_b[q_b], G, Tb)
            s = 0.0
            for ii in range(G):
                tij_inner = 0.0
                for jj in range(G):
                    tij_kk = 0.0
                    for kk in range(G):
                        tij_kk += coeffs[ii, jj, kk] * Tb[kk]
                    tij_inner += Ta[jj] * tij_kk
                s += T_fix[ii] * tij_inner
            slice2[q_a, q_b] = s
    return slice2


@njit(cache=True)
def build_mu_table_kern(coeffs, lobatto, u_nodes, p_grid,
                          gl_nodes, gl_weights, tau, c_stretch,
                          n_grid, nqk, kernel_h):
    """Build mu(p, u_k) table via 2D Gaussian-kernel co-area integral."""
    G = n_grid
    G_p = p_grid.size
    inv_2h2 = 0.5 / (kernel_h * kernel_h)
    mu_table = np.empty((G_p, G))
    # Precompute u, dudxi, and signal densities at GL nodes
    u_gl = np.empty(nqk); dudxi_gl = np.empty(nqk)
    f0_gl = np.empty(nqk); f1_gl = np.empty(nqk)
    for q in range(nqk):
        u_gl[q] = u_of_xi_jit(gl_nodes[q], c_stretch)
        dudxi_gl[q] = dudxi_jit(gl_nodes[q], c_stretch)
        f0_gl[q] = f_signal_jit(u_gl[q], 0, tau)
        f1_gl[q] = f_signal_jit(u_gl[q], 1, tau)
    for k_node in range(G):
        xi_fix = lobatto[k_node]
        u_k = u_nodes[k_node]
        f0k = f_signal_jit(u_k, 0, tau)
        f1k = f_signal_jit(u_k, 1, tau)
        slice2 = cheb_eval_slice_axis0(coeffs, xi_fix, G, gl_nodes,
                                          gl_nodes, nqk)
        for ip in range(G_p):
            p = p_grid[ip]
            A0 = 0.0; A1 = 0.0
            for q_a in range(nqk):
                wa = gl_weights[q_a] * dudxi_gl[q_a]
                f0a = f0_gl[q_a]; f1a = f1_gl[q_a]
                for q_b in range(nqk):
                    wb = gl_weights[q_b] * dudxi_gl[q_b]
                    diff = slice2[q_a, q_b] - p
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
def phi_kern_tab_jit(P_vals, V_inv, lobatto, u_nodes, p_grid,
                       gl_nodes, gl_weights,
                       tau, gamma, c_stretch, n_grid, nqk, kernel_h):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid
    G_p = p_grid.size
    mu_table = build_mu_table_kern(coeffs, lobatto, u_nodes, p_grid,
                                     gl_nodes, gl_weights, tau, c_stretch,
                                     G, nqk, kernel_h)
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                eps_p = 1e-9
                if p_cell < eps_p: p_cell = eps_p
                elif p_cell > 1 - eps_p: p_cell = 1 - eps_p
                mu0 = _interp_mu(mu_table, p_cell, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, p_cell, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, p_cell, p_grid, k, G_p)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def make_p_grid(G_p=51, L=8.0):
    logit = np.linspace(-L, L, G_p)
    return 1.0 / (1.0 + np.exp(-logit))


def make_gl_nodes(nqk):
    """GL nodes and weights for nqk-point quadrature on [-1, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(nqk)
    return nodes, weights


def phi_kern_tab(P_vals, gamma=GAMMA, tau=TAU, G_p=51, NQK=NQ,
                  kernel_h=DEFAULT_H_KERN, p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    if NQK != NQ:
        gl_n, gl_w = make_gl_nodes(NQK)
    else:
        gl_n, gl_w = GL_NODES, GL_WEIGHTS
    return phi_kern_tab_jit(P_vals, V_INV, LOBATTO, U_NODES, p_grid,
                              gl_n, gl_w, tau, gamma, C_STRETCH,
                              N_GRID, NQK, kernel_h)


if __name__ == '__main__':
    print('=== Kernel-band Cheb-tab operator self-test ===\n')
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    def sg(x): return 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    # warmup
    _ = phi_kern_tab(P_in, kernel_h=0.30, G_p=51)
    print(f'Cheb-tab kernel operator, N={N_GRID-1}, G={N_GRID}')
    print(f'Cold start P_0 = sigmoid(0.5*T):')
    for h in [1.0, 0.5, 0.3, 0.2, 0.1]:
        t0 = time.time()
        P_new = phi_kern_tab(P_in, kernel_h=h, G_p=51)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_new - P_in)))
        print(f'  h={h:>4.2f}: |F|={F:.3e}, t={dt*1000:.1f} ms, '
              f'P range [{P_new.min():.3f}, {P_new.max():.3f}]')
