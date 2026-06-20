"""Parameterised-N POU Cheb-tab operator."""
import math, time
import numpy as np
from numba import njit
from cheby_numba_pou_tab import (build_mu_table_pou_jit, _interp_mu,
                                    make_p_grid, all_roots_in_unit,
                                    N_SCAN_PER_DEGREE, MAX_ROOTS_PER_POLY)
from cheby_numba import (chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit, C_STRETCH, NQ)

def make_grid_N(N, c_stretch=C_STRETCH):
    G = N + 1
    lobatto = -np.cos(np.pi * np.arange(G) / N)
    u_nodes = c_stretch * np.arctanh(np.clip(lobatto, -0.9999, 0.9999))
    V = np.empty((G, G))
    for j in range(G):
        x = lobatto[j]
        V[j, 0] = 1.0
        if G > 1:
            V[j, 1] = x
            for k in range(1, G-1):
                V[j, k+1] = 2*x*V[j, k] - V[j, k-1]
    V_inv = np.linalg.inv(V)
    return G, lobatto, u_nodes, V_inv


@njit(cache=True)
def phi_pou_jit_N(P_vals, V_inv, lobatto, u_nodes, p_grid,
                     gl_nodes, gl_weights, tau, gamma, c_stretch,
                     n_grid, nq, n_scan):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid
    G_p = p_grid.size
    mu_table = build_mu_table_pou_jit(coeffs, lobatto, u_nodes, p_grid,
                                         gl_nodes, gl_weights, tau, c_stretch,
                                         G, nq, n_scan)
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


def phi_pou_N(P_vals, N, G_p=121, kernel_h=None, tau=1.0, gamma=1.0,
                c_stretch=C_STRETCH, n_scan=None, NQK=None):
    G, lob, u, V_inv = make_grid_N(N, c_stretch)
    if NQK is None: NQK = max(12, N+4)
    if n_scan is None: n_scan = max(6, 8*N)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQK)
    p_grid = make_p_grid(G_p)
    return phi_pou_jit_N(P_vals, V_inv, lob, u, p_grid, gl_n, gl_w,
                            tau, gamma, c_stretch, G, NQK, n_scan)


if __name__ == '__main__':
    print('=== POU Cheb-tab at various N ===\n')
    for N in [6, 8, 10]:
        G, lob, u, V_inv = make_grid_N(N)
        U1, U2, U3 = np.meshgrid(u, u, u, indexing='ij')
        T = 1.0*(U1+U2+U3)
        sg = lambda x: 1/(1+np.exp(-x))
        P_in = sg(0.5*T)
        _ = phi_pou_N(P_in, N, G_p=121)
        ts = []
        for _ in range(3):
            t0 = time.time(); P_out = phi_pou_N(P_in, N, G_p=121); ts.append(time.time()-t0)
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  N={N} G={G}: |F|={F:.3e}, median t={float(np.median(ts))*1000:.1f} ms')
