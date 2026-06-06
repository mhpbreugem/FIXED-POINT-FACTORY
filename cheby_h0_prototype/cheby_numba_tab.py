"""TRUE Chebyshev operator with mu(p, u_k) tabulation.

Key restructuring vs the cell-by-cell operator:

Original (chebroots cell-by-cell):
  for each cube cell (i, j, k):
      p_cell = P(cell)
      for each agent k_a:
          for each GL node q:
              chebroots → root_b → A_v contribution
          mu_k_a = f_1 A_1 / (f_0 A_0 + f_1 A_1)
      p_new = crra_clear(mu_0, mu_1, mu_2, gamma)
  -> G^3 × K × NQ chebroots calls

Tabulated:
  Build mu_table(p_grid, j) where j indexes Lobatto u_k values:
    for j = 0..G-1:
        slice2 = extract conjectured P at xi_k=lobatto[j]
        for each p in p_grid:
            (A_0, A_1) = co_area integral (with bisection on polynomial)
            mu_table[p, j] = f_1(u_k) A_1 / (f_0(u_k) A_0 + f_1(u_k) A_1)
  -> G_p × G × NQ chebroots calls (S3-symmetric P: SAME table works for all 3 agents)

  Then for each cube cell:
    p_cell = P(cell)
    mu_0 = interp(mu_table, p_cell, i)  (1D interp in p; integer in j)
    mu_1 = interp(mu_table, p_cell, j)
    mu_2 = interp(mu_table, p_cell, k)
    p_new = crra_clear(...)

Speedup at N=8: G_p × G × NQ = 21*9*12 = 2268 vs 729*3*12 = 26244 → 11.6x reduction
in contour calls. Combined with bisection-on-polynomial inside (~3-5x), total
chebroots-related speedup is ~30-50x. Per-Phi cost at N=8 drops from 1.23s
to ~25ms; Newton at N=8 from 11 min to ~15s.

The γ-dependence is recovered because the operator's μ_k tables depend on the
conjecture's shape (h ≠ 0 in lifted form), which IS what carries γ-info."""

import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, N, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit, f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)

from cheby_numba_bisect import cheb_bisect_single

# Tabulation grid in p: logit-uniform for better resolution at saturation
def make_p_grid(G_p=21, L=8.0):
    """logit-uniform p grid. p = σ(logit), logit ∈ [-L, L]."""
    logit = np.linspace(-L, L, G_p)
    return 1.0 / (1.0 + np.exp(-logit))

@njit(cache=True)
def co_area_one_target(slice2_coeffs, p_target, gl_nodes, gl_weights,
                         tau, c_stretch, n_grid, nq):
    """Single p_target co-area integral via bisection-on-polynomial."""
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas = np.empty(G)
    c1d_b = np.empty(G)
    der_buf = np.empty(G)
    for q in range(nq):
        xi_a = gl_nodes[q]
        w = gl_weights[q]
        _T_basis(xi_a, G, Tbas)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas[m]
            c1d_b[n] = s
        xi_b = cheb_bisect_single(c1d_b, G, p_target)
        if math.isnan(xi_b):
            continue
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        deg_der = chebder_jit(c1d_b, G, der_buf)
        dPdxi_b = chebval_jit(xi_b, der_buf, deg_der)
        if abs(dPdxi_b) < 1e-12:
            continue
        u_b = u_of_xi_jit(xi_b, c_stretch)
        dudxi_b = dudxi_jit(xi_b, c_stretch)
        f0b = f_signal_jit(u_b, 0, tau)
        f1b = f_signal_jit(u_b, 1, tau)
        dPdu_b = dPdxi_b / dudxi_b
        wt = w * dudxi_a * dudxi_b / abs(dPdu_b)
        A0 += wt * f0a * f0b
        A1 += wt * f1a * f1b
    return 0.5*A0, 0.5*A1


@njit(cache=True)
def build_mu_table(coeffs, lobatto, u_nodes, p_grid,
                     gl_nodes, gl_weights, tau, c_stretch, n_grid, nq):
    """Build μ(p, u_k) table for agent 0 (axis 0 fixed).
    By S_3 symmetry of conjecture, this table works for all 3 agents."""
    G = n_grid
    G_p = p_grid.shape[0]
    mu_table = np.empty((G_p, G))
    slice2 = np.empty((G, G))
    Tfix = np.empty(G)
    for j in range(G):
        xi_fix = lobatto[j]
        u_k = u_nodes[j]
        _T_basis(xi_fix, G, Tfix)
        # Extract slice fixing axis 0 (agent 0 perspective)
        for jj in range(G):
            for kk in range(G):
                s = 0.0
                for ii in range(G):
                    s += coeffs[ii, jj, kk] * Tfix[ii]
                slice2[jj, kk] = s
        f0 = f_signal_jit(u_k, 0, tau)
        f1 = f_signal_jit(u_k, 1, tau)
        for i in range(G_p):
            p = p_grid[i]
            A0, A1 = co_area_one_target(slice2, p, gl_nodes, gl_weights,
                                          tau, c_stretch, G, nq)
            den = f0*A0 + f1*A1
            if den > 1e-30:
                mu_table[i, j] = (f1*A1) / den
            else:
                mu_table[i, j] = 0.5
    return mu_table


@njit(cache=True, inline='always')
def interp_mu_1d(mu_table, p, p_grid, j_lobatto):
    """Linear interpolation of mu_table in p (j_lobatto is integer index)."""
    G_p = p_grid.shape[0]
    # Find bracketing p indices
    if p <= p_grid[0]: return mu_table[0, j_lobatto]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, j_lobatto]
    # Binary search
    lo = 0; hi = G_p - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_grid[mid] <= p:
            lo = mid
        else:
            hi = mid
    # Linear interpolation
    w = (p - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    return (1.0 - w) * mu_table[lo, j_lobatto] + w * mu_table[hi, j_lobatto]


@njit(cache=True)
def phi_one_cell_tab(P_vals_at_cell, i, j, k, mu_table, p_grid, gamma):
    """Use tabulated mu values; only do clearing per cell."""
    p_cell = P_vals_at_cell
    eps_p = 1e-9
    if p_cell < eps_p: p_cell = eps_p
    elif p_cell > 1 - eps_p: p_cell = 1 - eps_p
    mu0 = interp_mu_1d(mu_table, p_cell, p_grid, i)
    mu1 = interp_mu_1d(mu_table, p_cell, p_grid, j)
    mu2 = interp_mu_1d(mu_table, p_cell, p_grid, k)
    return crra_clear_jit(mu0, mu1, mu2, gamma)


@njit(cache=True)
def phi_jit_tab(P_vals, V_inv, lobatto, u_nodes, p_grid, gl_nodes, gl_weights,
                  tau, gamma, c_stretch, n_grid, nq):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    mu_table = build_mu_table(coeffs, lobatto, u_nodes, p_grid,
                                 gl_nodes, gl_weights, tau, c_stretch, n_grid, nq)
    G = n_grid
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = phi_one_cell_tab(P_vals[i, j, k], i, j, k,
                                                     mu_table, p_grid, gamma)
    return P_new


def phi_tab(P_vals, gamma=GAMMA, tau=TAU, G_p=21):
    p_grid = make_p_grid(G_p)
    return phi_jit_tab(P_vals, V_INV, LOBATTO, U_NODES, p_grid,
                          GL_NODES, GL_WEIGHTS, tau, gamma, C_STRETCH, N_GRID, NQ)


if __name__ == '__main__':
    print('=== Tabulated operator validation (N=6) ===\n')
    from cheby_numba import phi as phi_chebroots
    from cheby_numba_bisect import phi_bisect

    def sigmoid(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    P_in = sigmoid(0.5 * T)

    # Warmup
    _ = phi_chebroots(P_in)
    _ = phi_bisect(P_in)
    _ = phi_tab(P_in)

    print(f'  N=6 timings (5 trials):')
    print(f'  {"chebroots":>15} {"bisect":>15} {"tabulated":>15}')
    for trial in range(5):
        t0 = time.time(); P_cr = phi_chebroots(P_in); t_cr = time.time() - t0
        t0 = time.time(); P_bi = phi_bisect(P_in); t_bi = time.time() - t0
        t0 = time.time(); P_tb = phi_tab(P_in, G_p=21); t_tb = time.time() - t0
        diff_bi = float(np.max(np.abs(P_cr - P_bi)))
        diff_tb = float(np.max(np.abs(P_cr - P_tb)))
        print(f'  {t_cr:>15.4f} {t_bi:>15.4f} {t_tb:>15.4f}    '
              f'diff_bi={diff_bi:.3e} diff_tb={diff_tb:.3e}')

    # Try different G_p resolutions
    print(f'\n  Effect of G_p (number of p table points):')
    for G_p in [11, 21, 31, 51, 101]:
        t0 = time.time(); P_tb = phi_tab(P_in, G_p=G_p); t_tb = time.time() - t0
        diff_tb = float(np.max(np.abs(P_cr - P_tb)))
        print(f'  G_p={G_p:>3}: t={t_tb:.4f}s  max|P_tab - P_cr|={diff_tb:.3e}')
