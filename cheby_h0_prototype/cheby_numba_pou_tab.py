"""Strict h=0 Chebyshev co-area operator with partition-of-unity (POU).
ALL-NUMBA implementation.

NO kernel, NO bandwidth. Smoothness via:
  A_v(p) = A_v^(a)(p) + A_v^(b)(p)
  w_a = d_a²/(d_a²+d_b²),  w_b = d_b²/(d_a²+d_b²),  w_a + w_b = 1
Near a turning point (one derivative -> 0) the corresponding weight
-> 0, cancelling the 1/|d| blowup -> smooth everywhere except true
Morse-critical points (measure zero).

All-roots-in-[-1,1] finding via dense sign-change scan + bisection
(numba-friendly; finds all simple roots robustly).
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)


N_SCAN_PER_DEGREE = 6  # subintervals per polynomial degree for sign-change scan
MAX_ROOTS_PER_POLY = 16


@njit(cache=True, inline='always')
def _bisect_root(c, n, lo, hi, val_lo, val_hi, n_iters=80):
    """Bisection: find x in [lo, hi] with c(x) = 0, given sign change."""
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        val_mid = chebval_jit(mid, c, n)
        if val_lo * val_mid <= 0:
            hi = mid; val_hi = val_mid
        else:
            lo = mid; val_lo = val_mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


@njit(cache=True)
def all_roots_in_unit(c, n, roots_out, n_scan):
    """Find all real roots of Cheb-coeff polynomial c in [-1, 1].
    Stores up to MAX_ROOTS_PER_POLY roots in roots_out; returns count."""
    nr = 0
    dx = 2.0 / n_scan
    prev_x = -1.0
    prev_v = chebval_jit(prev_x, c, n)
    for k in range(1, n_scan + 1):
        x = -1.0 + k * dx
        v = chebval_jit(x, c, n)
        if prev_v == 0.0:
            if nr < MAX_ROOTS_PER_POLY:
                roots_out[nr] = prev_x; nr += 1
        elif prev_v * v < 0.0:
            r = _bisect_root(c, n, prev_x, x, prev_v, v)
            if nr < MAX_ROOTS_PER_POLY:
                roots_out[nr] = r; nr += 1
        prev_x = x; prev_v = v
    # last endpoint
    if prev_v == 0.0:
        if nr < MAX_ROOTS_PER_POLY:
            roots_out[nr] = 1.0; nr += 1
    return nr


@njit(cache=True)
def co_area_pou_one(slice2_coeffs, p_target,
                      gl_nodes, gl_weights,
                      tau, c_stretch, n_grid, nq, n_scan):
    """Partition-of-unity co-area at a single (p, slice) pair.
    Returns (A0, A1)."""
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

    # ===== Term A^(b): fix xi_a at GL node, find xi_b roots =====
    for q in range(nq):
        xi_a = gl_nodes[q]
        w_gl = gl_weights[q]
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        # 1D poly in xi_b: c1d_b[n] = sum_m slice2[m,n] T_m(xi_a)
        _T_basis(xi_a, G, Tbas_a)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas_a[m]
            c1d_b[n] = s
        # Shift by p_target
        for n in range(G):
            c1d_b_shift[n] = c1d_b[n]
        c1d_b_shift[0] -= p_target
        nrb = all_roots_in_unit(c1d_b_shift, G, roots_b, n_scan)
        if nrb == 0:
            continue
        # Build "d_a slice in xi_b" coeffs: c1d_dap[n] = T_m'(xi_a) * slice2[m, n]
        for n in range(G):
            for m in range(G):
                col_buf[m] = slice2_coeffs[m, n]
            deg = chebder_jit(col_buf, G, der_buf)
            c1d_dap[n] = chebval_jit(xi_a, der_buf, deg)
        # chebder of c1d_b for d_b along this slice
        deg_db = chebder_jit(c1d_b, G, der_buf)
        for r in range(nrb):
            xi_b = roots_b[r]
            u_b = u_of_xi_jit(xi_b, c_stretch)
            dudxi_b = dudxi_jit(xi_b, c_stretch)
            f0b = f_signal_jit(u_b, 0, tau)
            f1b = f_signal_jit(u_b, 1, tau)
            dPdxi_b = chebval_jit(xi_b, der_buf, deg_db)
            _T_basis(xi_b, G, Tbas_b)
            dPdxi_a = 0.0
            for n in range(G):
                dPdxi_a += c1d_dap[n] * Tbas_b[n]
            dPdu_a = dPdxi_a / dudxi_a
            dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300:
                continue
            w_b = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300:
                continue
            wt = w_gl * dudxi_a * w_b / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # ===== Term A^(a): fix xi_b at GL node, find xi_a roots =====
    # Need chebder of c1d_b for d_b along the slice (recomputed inside).
    # The chebder buffer is reused across q-loops.
    for q in range(nq):
        xi_b = gl_nodes[q]
        w_gl = gl_weights[q]
        u_b = u_of_xi_jit(xi_b, c_stretch)
        dudxi_b = dudxi_jit(xi_b, c_stretch)
        f0b = f_signal_jit(u_b, 0, tau)
        f1b = f_signal_jit(u_b, 1, tau)
        _T_basis(xi_b, G, Tbas_b)
        for m in range(G):
            s = 0.0
            for n in range(G):
                s += slice2_coeffs[m, n] * Tbas_b[n]
            c1d_a[m] = s
        for m in range(G):
            c1d_a_shift[m] = c1d_a[m]
        c1d_a_shift[0] -= p_target
        nra = all_roots_in_unit(c1d_a_shift, G, roots_a, n_scan)
        if nra == 0:
            continue
        # Build "d_b slice in xi_a" coeffs: c1d_dbp[m] = T_n'(xi_b) * slice2[m, n]
        for m in range(G):
            for n in range(G):
                row_buf[n] = slice2_coeffs[m, n]
            deg = chebder_jit(row_buf, G, der_buf)
            c1d_dbp[m] = chebval_jit(xi_b, der_buf, deg)
        # chebder of c1d_a for d_a along this slice
        deg_da = chebder_jit(c1d_a, G, der_buf)
        for r in range(nra):
            xi_a = roots_a[r]
            u_a = u_of_xi_jit(xi_a, c_stretch)
            dudxi_a = dudxi_jit(xi_a, c_stretch)
            f0a = f_signal_jit(u_a, 0, tau)
            f1a = f_signal_jit(u_a, 1, tau)
            dPdxi_a = chebval_jit(xi_a, der_buf, deg_da)
            _T_basis(xi_a, G, Tbas_a)
            dPdxi_b = 0.0
            for m in range(G):
                dPdxi_b += c1d_dbp[m] * Tbas_a[m]
            dPdu_a = dPdxi_a / dudxi_a
            dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300:
                continue
            w_a = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300:
                continue
            wt = w_gl * dudxi_b * w_a / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    return A0, A1


@njit(cache=True)
def build_mu_table_pou_jit(coeffs, lobatto, u_nodes, p_grid,
                              gl_nodes, gl_weights, tau, c_stretch,
                              n_grid, nq, n_scan):
    G = n_grid
    G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    slice2 = np.empty((G, G))
    T_fix = np.empty(G)
    for k_node in range(G):
        xi_fix = lobatto[k_node]
        u_k = u_nodes[k_node]
        _T_basis(xi_fix, G, T_fix)
        for jj in range(G):
            for kk in range(G):
                s = 0.0
                for i in range(G):
                    s += coeffs[i, jj, kk] * T_fix[i]
                slice2[jj, kk] = s
        f0 = f_signal_jit(u_k, 0, tau)
        f1 = f_signal_jit(u_k, 1, tau)
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pou_one(slice2, p, gl_nodes, gl_weights,
                                        tau, c_stretch, G, nq, n_scan)
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
def phi_pou_tab_jit(P_vals, V_inv, lobatto, u_nodes, p_grid,
                       gl_nodes, gl_weights,
                       tau, gamma, c_stretch, n_grid, nq, n_scan):
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


def make_p_grid(G_p=121, L=8.0):
    return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))


def phi_pou_tab(P_vals, gamma=GAMMA, tau=TAU, G_p=121, p_grid=None,
                  n_scan_per_degree=N_SCAN_PER_DEGREE):
    if p_grid is None:
        p_grid = make_p_grid(G_p)
    n_scan = n_scan_per_degree * (N_GRID - 1)
    if n_scan < 6: n_scan = 6
    return phi_pou_tab_jit(P_vals, V_INV, LOBATTO, U_NODES, p_grid,
                              GL_NODES, GL_WEIGHTS, tau, gamma, C_STRETCH,
                              N_GRID, NQ, n_scan)


if __name__ == '__main__':
    print('=== Strict h=0 POU Cheb-tab (numba) self-test ===\n')
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    def sg(x): return 1/(1+np.exp(-x))
    P_in = sg(0.5*T)
    # warmup
    _ = phi_pou_tab(P_in, G_p=121)
    print(f'N={N_GRID-1}, G={N_GRID}, G_p=121')
    print('Wall-time per Phi (3 trials):')
    for trial in range(3):
        t0 = time.time()
        P_out = phi_pou_tab(P_in, G_p=121)
        dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  trial {trial+1}: |F|={F:.3e}, t={dt*1000:.1f} ms')

    print(f'\nAlso test at G_p=51, 201:')
    for G_p in [51, 201]:
        _ = phi_pou_tab(P_in, G_p=G_p)
        t0 = time.time(); P_out = phi_pou_tab(P_in, G_p=G_p); dt = time.time() - t0
        F = float(np.max(np.abs(P_out - P_in)))
        print(f'  G_p={G_p}: |F|={F:.3e}, t={dt*1000:.1f} ms')
