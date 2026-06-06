"""Richardson-extrapolated tabulated-mu operator (strict h=0 LIMIT,
smooth construction).

Build mu(p, u_k) at multiple bandwidths {h_1, h_2, ...}, then linearly
combine to cancel O(h^2) [and O(h^4) with 3 points] -- giving the
strict h=0 table value WITHOUT solving the strict-h=0 operator.

The operator is smooth (linear combination of smooth kernel-band
operators), Newton converges to machine eps. But the FP it finds is
unbiased w.r.t. the strict-h=0 FP -- no fake kernel-smoothing of the
1-R^2 deficit.

Architecture: Chebyshev tab + kernel band, same as cheby_numba_kern_tab,
just builds the table multiple times and linearly combines.
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, _T_basis, u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)
from cheby_numba_kern_tab import (cheb_eval_slice_axis0,
                                      build_mu_table_kern, _interp_mu)


def richardson_weights(hs, order=2):
    """Linear combination weights w_i such that
       sum_i w_i * mu_{h_i} = mu_0 + O(h^{2*(len(hs))}).

    For two hs and order 2: cancels h^2 term, leaves O(h^4).
    For three hs and order 4: cancels h^2 AND h^4, leaves O(h^6).
    General: solve Vandermonde-in-h^{2k} system.
    """
    n = len(hs)
    H = np.array(hs, dtype=float)
    # Matrix V[i, k] = H[i]^(2k) for k=0..n-1
    V = np.array([[H[i]**(2*k) for k in range(n)] for i in range(n)])
    # Solve V^T @ w = [1, 0, 0, ..., 0] (constant 1, higher orders 0)
    e0 = np.zeros(n); e0[0] = 1.0
    w = np.linalg.solve(V.T, e0)
    # Verify: sum(w_i * H[i]^0) = 1, sum(w_i * H[i]^2) = 0, ...
    return w


@njit(cache=True)
def phi_richardson_jit(P_vals, V_inv, lobatto, u_nodes, p_grid,
                          gl_nodes, gl_weights, tau, gamma, c_stretch,
                          n_grid, nqk, hs, weights):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid; G_p = p_grid.size
    n_h = hs.size
    # Build mu table at each h, accumulate linear combination
    mu_table_total = np.zeros((G_p, G))
    for h_idx in range(n_h):
        h = hs[h_idx]
        mu_h = build_mu_table_kern(coeffs, lobatto, u_nodes, p_grid,
                                       gl_nodes, gl_weights, tau, c_stretch,
                                       G, nqk, h)
        for ip in range(G_p):
            for k in range(G):
                mu_table_total[ip, k] += weights[h_idx] * mu_h[ip, k]
    # Per-cube CRRA clearing using extrapolated table
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                eps = 1e-9
                if p_cell < eps: p_cell = eps
                elif p_cell > 1 - eps: p_cell = 1 - eps
                mu0 = _interp_mu(mu_table_total, p_cell, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table_total, p_cell, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table_total, p_cell, p_grid, k, G_p)
                # Clip extrapolated mu (Richardson can overshoot occasionally)
                if mu0 < eps: mu0 = eps
                elif mu0 > 1 - eps: mu0 = 1 - eps
                if mu1 < eps: mu1 = eps
                elif mu1 > 1 - eps: mu1 = 1 - eps
                if mu2 < eps: mu2 = eps
                elif mu2 > 1 - eps: mu2 = 1 - eps
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def make_p_grid(G_p=121, L=8.0):
    return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))


def phi_richardson(P_vals, hs=(0.5, 0.3, 0.2), gamma=GAMMA, tau=TAU,
                      G_p=121, NQK=16):
    p_grid = make_p_grid(G_p)
    if NQK == NQ:
        gl_n, gl_w = GL_NODES, GL_WEIGHTS
    else:
        gl_n, gl_w = np.polynomial.legendre.leggauss(NQK)
    hs_arr = np.array(hs, dtype=float)
    w_arr = richardson_weights(hs)
    return phi_richardson_jit(P_vals, V_INV, LOBATTO, U_NODES, p_grid,
                                  gl_n, gl_w, tau, gamma, C_STRETCH,
                                  N_GRID, NQK, hs_arr, w_arr)


if __name__ == '__main__':
    print('=== Richardson-extrapolated tabulated-mu operator ===\n')
    G = N_GRID
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T_full = TAU*(U1+U2+U3)
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T_full)

    # Verify Richardson weights for several configurations
    print('Richardson weights for various h configs:')
    for hs in [(0.5, 0.3), (0.5, 0.3, 0.2), (0.7, 0.5, 0.3, 0.2)]:
        w = richardson_weights(hs)
        print(f'  hs={hs}: w={w}, sum={w.sum():.6f}, '
              f'sum w*h^2={np.sum(w * np.array(hs)**2):.3e}')

    # Time per Phi
    print('\nWarming up & timing:')
    for hs in [(0.5, 0.3), (0.5, 0.3, 0.2)]:
        _ = phi_richardson(P_in, hs=hs, G_p=121, NQK=16)
        ts = []
        for _ in range(3):
            t0 = time.time()
            P_out = phi_richardson(P_in, hs=hs, G_p=121, NQK=16)
            ts.append(time.time() - t0)
        print(f'  hs={hs}: t_median={float(np.median(ts))*1000:.1f} ms, '
              f'|F|={float(np.max(np.abs(P_out - P_in))):.3e}')
