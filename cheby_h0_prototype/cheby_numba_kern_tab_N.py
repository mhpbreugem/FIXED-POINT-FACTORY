"""Parameterised-N version of the kernel-band Cheb-tab operator.
Lets us test N=8, N=10, ... without rebuilding the module constants.
"""
import math
import numpy as np
from numba import njit
from cheby_numba_kern_tab import (build_mu_table_kern, _interp_mu,
                                     phi_kern_tab_jit)
from cheby_numba import (chebval_jit, _T_basis, u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit, vals_to_coeffs_3d_jit,
                            C_STRETCH)

def make_grid_N(N, c_stretch=C_STRETCH):
    """Lobatto + atanh-stretched u nodes for arbitrary N."""
    G = N + 1
    lobatto = -np.cos(np.pi * np.arange(G) / N)
    u_nodes = c_stretch * np.arctanh(np.clip(lobatto, -0.9999, 0.9999))
    # Build Vandermonde matrix for vals→coeffs conversion
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


def make_p_grid(G_p=121, L=8.0):
    return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))


def phi_kern_tab_N(P_vals, N, kernel_h=0.30, G_p=121, NQK=None,
                     tau=1.0, gamma=1.0, c_stretch=C_STRETCH, p_grid=None):
    """Generic kernel-tab call for any N."""
    G, lobatto, u_nodes, V_inv = make_grid_N(N, c_stretch)
    if NQK is None:
        NQK = max(12, N + 4)
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQK)
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    return phi_kern_tab_jit(P_vals, V_inv, lobatto, u_nodes, p_grid,
                              gl_n, gl_w, tau, gamma, c_stretch,
                              G, NQK, kernel_h)


if __name__ == '__main__':
    import time
    print('Testing at N=6, 8, 10...')
    for N in [6, 8, 10]:
        G, lob, u, V_inv = make_grid_N(N)
        U1, U2, U3 = np.meshgrid(u, u, u, indexing='ij')
        T = u[0]*0 + 1.0*(U1+U2+U3)  # tau=1
        sg = lambda x: 1/(1+np.exp(-x))
        P_in = sg(0.5*T)
        # Warmup
        _ = phi_kern_tab_N(P_in, N, kernel_h=0.3, G_p=121)
        ts = []
        for _ in range(3):
            t0 = time.time()
            P_out = phi_kern_tab_N(P_in, N, kernel_h=0.3, G_p=121)
            ts.append(time.time() - t0)
        print(f'  N={N}: G={G}, cells={G**3}, median t={float(np.median(ts))*1000:.1f} ms')
