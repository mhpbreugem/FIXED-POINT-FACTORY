"""Richardson extrapolation on the Linear-CDF kernel-band operator.

Same idea as cheby_richardson_tab but with Linear-CDF u-grid + bilinear
interp instead of Chebyshev. Should give strict-h=0 LIMIT lookup table
on the CDF grid, while remaining smooth (Newton-friendly).
"""
import time, math
import numpy as np
from numba import njit

from cheby_numba import f_signal_jit, crra_clear_jit
from lin_cdf_kern_tab import (build_mu_table_lin_kern, _interp_mu,
                                  make_cdf_uniform_grid, make_p_grid,
                                  make_gl_for_u)


def richardson_weights(hs):
    n = len(hs)
    H = np.array(hs, dtype=float)
    V = np.array([[H[i]**(2*k) for k in range(n)] for i in range(n)])
    e0 = np.zeros(n); e0[0] = 1.0
    return np.linalg.solve(V.T, e0)


@njit(cache=True)
def phi_lin_richardson_jit(P_vals, u_grid, p_grid, gl_u_nodes, gl_du_weights,
                                tau, gamma, n_grid, nqk, hs, weights):
    G = n_grid; G_p = p_grid.size; n_h = hs.size
    mu_total = np.zeros((G_p, G))
    for h_idx in range(n_h):
        h = hs[h_idx]
        mu_h = build_mu_table_lin_kern(P_vals, u_grid, p_grid,
                                            gl_u_nodes, gl_du_weights, tau,
                                            G, nqk, h)
        for ip in range(G_p):
            for k in range(G):
                mu_total[ip, k] += weights[h_idx] * mu_h[ip, k]
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                eps = 1e-9
                if p_cell < eps: p_cell = eps
                elif p_cell > 1 - eps: p_cell = 1 - eps
                mu0 = _interp_mu(mu_total, p_cell, p_grid, i, G_p)
                mu1 = _interp_mu(mu_total, p_cell, p_grid, j, G_p)
                mu2 = _interp_mu(mu_total, p_cell, p_grid, k, G_p)
                # Clip overshoot
                for mu in [mu0, mu1, mu2]: pass
                if mu0 < eps: mu0 = eps
                elif mu0 > 1-eps: mu0 = 1-eps
                if mu1 < eps: mu1 = eps
                elif mu1 > 1-eps: mu1 = 1-eps
                if mu2 < eps: mu2 = eps
                elif mu2 > 1-eps: mu2 = 1-eps
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def phi_lin_richardson(P_vals, u_grid, hs=(0.5, 0.3, 0.2), gamma=1.0,
                          tau=1.0, G_p=121, NQK=16, p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    G = u_grid.size
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    hs_arr = np.array(hs, dtype=float)
    w_arr = richardson_weights(hs)
    return phi_lin_richardson_jit(P_vals, u_grid, p_grid, gl_u, gl_du,
                                       tau, gamma, G, NQK, hs_arr, w_arr)


if __name__ == '__main__':
    print('=== Lin-CDF Richardson (3-pt) self-test ===\n')
    G = 7
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = U1+U2+U3
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    HS = (0.5, 0.3, 0.2)
    print(f'hs={HS}, weights={richardson_weights(HS)}')

    # JIT warmup
    _ = phi_lin_richardson(P_in, u_grid, hs=HS, gamma=1.0)
    ts = []
    for _ in range(3):
        t0 = time.time()
        P_out = phi_lin_richardson(P_in, u_grid, hs=HS, gamma=1.0)
        ts.append(time.time() - t0)
    print(f'Per-Phi: {float(np.median(ts))*1000:.1f} ms')
    print(f'|F| from cold (sigmoid): {float(np.max(np.abs(P_out - P_in))):.3e}')
    print(f'P_out range: [{P_out.min():.4f}, {P_out.max():.4f}]')
