"""POU Cheb-tab operator with numba chebroots (all real roots in [-1,1]
via companion-matrix eigenvalues)."""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)
from cheby_roots_numba import chebroots_companion

MAX_ROOTS_PER_POLY = 16


@njit(cache=True)
def co_area_pou_cr_one(slice2_coeffs, p_target, gl_nodes, gl_weights,
                         tau, c_stretch, n_grid, nq):
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas_a = np.empty(G); Tbas_b = np.empty(G)
    c1d_b = np.empty(G); c1d_b_shift = np.empty(G)
    c1d_a = np.empty(G); c1d_a_shift = np.empty(G)
    c1d_dap = np.empty(G); c1d_dbp = np.empty(G)
    der_buf = np.empty(G)
    col_buf = np.empty(G); row_buf = np.empty(G)
    roots_b = np.empty(MAX_ROOTS_PER_POLY)
    roots_a = np.empty(MAX_ROOTS_PER_POLY)

    # Term A^(b)
    for q in range(nq):
        xi_a = gl_nodes[q]; w_gl = gl_weights[q]
        u_a = u_of_xi_jit(xi_a, c_stretch); dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        _T_basis(xi_a, G, Tbas_a)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas_a[m]
            c1d_b[n] = s
        for n in range(G):
            c1d_b_shift[n] = c1d_b[n]
        c1d_b_shift[0] -= p_target
        nrb = chebroots_companion(c1d_b_shift, G, roots_b)
        if nrb == 0:
            continue
        for n in range(G):
            for m in range(G):
                col_buf[m] = slice2_coeffs[m, n]
            deg = chebder_jit(col_buf, G, der_buf)
            c1d_dap[n] = chebval_jit(xi_a, der_buf, deg)
        deg_db = chebder_jit(c1d_b, G, der_buf)
        for r in range(nrb):
            xi_b = roots_b[r]
            u_b = u_of_xi_jit(xi_b, c_stretch); dudxi_b = dudxi_jit(xi_b, c_stretch)
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            dPdxi_b = chebval_jit(xi_b, der_buf, deg_db)
            _T_basis(xi_b, G, Tbas_b)
            dPdxi_a = 0.0
            for n in range(G):
                dPdxi_a += c1d_dap[n] * Tbas_b[n]
            dPdu_a = dPdxi_a / dudxi_a; dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            wt = w_gl * dudxi_a * w_b / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a)
    for q in range(nq):
        xi_b = gl_nodes[q]; w_gl = gl_weights[q]
        u_b = u_of_xi_jit(xi_b, c_stretch); dudxi_b = dudxi_jit(xi_b, c_stretch)
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        _T_basis(xi_b, G, Tbas_b)
        for m in range(G):
            s = 0.0
            for n in range(G):
                s += slice2_coeffs[m, n] * Tbas_b[n]
            c1d_a[m] = s
        for m in range(G):
            c1d_a_shift[m] = c1d_a[m]
        c1d_a_shift[0] -= p_target
        nra = chebroots_companion(c1d_a_shift, G, roots_a)
        if nra == 0:
            continue
        for m in range(G):
            for n in range(G):
                row_buf[n] = slice2_coeffs[m, n]
            deg = chebder_jit(row_buf, G, der_buf)
            c1d_dbp[m] = chebval_jit(xi_b, der_buf, deg)
        deg_da = chebder_jit(c1d_a, G, der_buf)
        for r in range(nra):
            xi_a = roots_a[r]
            u_a = u_of_xi_jit(xi_a, c_stretch); dudxi_a = dudxi_jit(xi_a, c_stretch)
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            dPdxi_a = chebval_jit(xi_a, der_buf, deg_da)
            _T_basis(xi_a, G, Tbas_a)
            dPdxi_b = 0.0
            for m in range(G):
                dPdxi_b += c1d_dbp[m] * Tbas_a[m]
            dPdu_a = dPdxi_a / dudxi_a; dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            wt = w_gl * dudxi_b * w_a / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return A0, A1


@njit(cache=True)
def build_mu_table_pou_cr(coeffs, lobatto, u_nodes, p_grid,
                             gl_nodes, gl_weights, tau, c_stretch,
                             n_grid, nq):
    G = n_grid; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    slice2 = np.empty((G, G))
    T_fix = np.empty(G)
    for k_node in range(G):
        xi_fix = lobatto[k_node]; u_k = u_nodes[k_node]
        _T_basis(xi_fix, G, T_fix)
        for jj in range(G):
            for kk in range(G):
                s = 0.0
                for i in range(G):
                    s += coeffs[i, jj, kk] * T_fix[i]
                slice2[jj, kk] = s
        f0 = f_signal_jit(u_k, 0, tau); f1 = f_signal_jit(u_k, 1, tau)
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pou_cr_one(slice2, p, gl_nodes, gl_weights,
                                           tau, c_stretch, G, nq)
            den = f0*A0 + f1*A1
            if den > 1e-300:
                mu_table[ip, k_node] = f1*A1 / den
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
def phi_pou_cr_jit(P_vals, V_inv, lobatto, u_nodes, p_grid,
                      gl_nodes, gl_weights,
                      tau, gamma, c_stretch, n_grid, nq):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid; G_p = p_grid.size
    mu_table = build_mu_table_pou_cr(coeffs, lobatto, u_nodes, p_grid,
                                        gl_nodes, gl_weights, tau, c_stretch,
                                        G, nq)
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


def phi_pou_cr(P_vals, gamma=GAMMA, tau=TAU, G_p=121, p_grid=None):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    return phi_pou_cr_jit(P_vals, V_INV, LOBATTO, U_NODES, p_grid,
                             GL_NODES, GL_WEIGHTS, tau, gamma, C_STRETCH,
                             N_GRID, NQ)


if __name__ == '__main__':
    print('=== POU+chebroots (numba) self-test ===\n')
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)
    _ = phi_pou_cr(P_in, G_p=121)  # warmup
    print(f'N={N_GRID-1}, G={N_GRID}, G_p=121:')
    ts = []
    for _ in range(3):
        t0 = time.time(); P_out = phi_pou_cr(P_in, G_p=121); ts.append(time.time()-t0)
    print(f'  ms / Phi: median {float(np.median(ts))*1000:.1f}')
    F = float(np.max(np.abs(P_out - P_in)))
    print(f'  |F| from cold sigmoid: {F:.3e}')

    # Compare to scan-bisect
    from cheby_numba_pou_tab import phi_pou_tab
    _ = phi_pou_tab(P_in, G_p=121)
    P_scan = phi_pou_tab(P_in, G_p=121)
    diff = float(np.max(np.abs(P_out - P_scan)))
    print(f'  max|chebroots - scanbisect|: {diff:.3e}')
