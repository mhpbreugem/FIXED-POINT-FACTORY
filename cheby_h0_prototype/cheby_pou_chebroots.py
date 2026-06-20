"""POU Cheb-tab operator using numpy.polynomial.chebyshev.chebroots
(companion-matrix eigenvalues) instead of my scan-and-bisect, to test
whether root-extraction completeness is the cause of the 4e-3 floor.

Not numba (chebroots is scipy/numpy). Slower but cleaner."""
import os, time, math
import numpy as np
from numpy.polynomial import chebyshev as npc

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)


def all_roots_cheb(c1d, lo=-1.0, hi=1.0):
    """All real roots of Cheb-coeff polynomial c1d in [lo, hi] via chebroots."""
    if len(c1d) < 2:
        return np.array([])
    roots = npc.chebroots(c1d)
    real = roots[np.abs(roots.imag) < 1e-10].real
    return np.sort(real[(real >= lo) & (real <= hi)])


def co_area_pou_cr(slice2_coeffs, p_target, gl_nodes, gl_weights,
                     tau, c_stretch, n_grid, nq):
    """Same POU as cheby_numba_pou_tab but with chebroots."""
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas_a = np.empty(G); Tbas_b = np.empty(G)
    der_buf = np.empty(G)

    # Term A^(b): fix xi_a, find xi_b roots
    for q in range(nq):
        xi_a = gl_nodes[q]; w_gl = gl_weights[q]
        u_a = u_of_xi_jit(xi_a, c_stretch); dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        _T_basis(xi_a, G, Tbas_a)
        c1d_b = np.zeros(G)
        for n in range(G):
            for m in range(G):
                c1d_b[n] += slice2_coeffs[m, n] * Tbas_a[m]
        c1d_b_shift = c1d_b.copy(); c1d_b_shift[0] -= p_target
        roots = all_roots_cheb(c1d_b_shift)
        if len(roots) == 0:
            continue
        # d_a slice (in xi_b basis), via chebder along m-axis then eval at xi_a
        c1d_dap = np.zeros(G)
        for n in range(G):
            col = slice2_coeffs[:, n].copy()
            der = np.empty(G); deg = chebder_jit(col, G, der)
            c1d_dap[n] = chebval_jit(xi_a, der, deg)
        deg_db = chebder_jit(c1d_b, G, der_buf)
        for xi_b in roots:
            u_b = u_of_xi_jit(xi_b, c_stretch); dudxi_b = dudxi_jit(xi_b, c_stretch)
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            dPdxi_b = chebval_jit(xi_b, der_buf, deg_db)
            _T_basis(xi_b, G, Tbas_b)
            dPdxi_a = sum(c1d_dap[n] * Tbas_b[n] for n in range(G))
            dPdu_a = dPdxi_a / dudxi_a
            dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a**2 + dPdu_b**2
            if denom < 1e-300: continue
            w_b = dPdu_b**2 / denom
            if abs(dPdu_b) < 1e-300: continue
            wt = w_gl * dudxi_a * w_b / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a): fix xi_b, find xi_a roots
    for q in range(nq):
        xi_b = gl_nodes[q]; w_gl = gl_weights[q]
        u_b = u_of_xi_jit(xi_b, c_stretch); dudxi_b = dudxi_jit(xi_b, c_stretch)
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        _T_basis(xi_b, G, Tbas_b)
        c1d_a = np.zeros(G)
        for m in range(G):
            for n in range(G):
                c1d_a[m] += slice2_coeffs[m, n] * Tbas_b[n]
        c1d_a_shift = c1d_a.copy(); c1d_a_shift[0] -= p_target
        roots = all_roots_cheb(c1d_a_shift)
        if len(roots) == 0:
            continue
        c1d_dbp = np.zeros(G)
        for m in range(G):
            row = slice2_coeffs[m, :].copy()
            der = np.empty(G); deg = chebder_jit(row, G, der)
            c1d_dbp[m] = chebval_jit(xi_b, der, deg)
        deg_da = chebder_jit(c1d_a, G, der_buf)
        for xi_a in roots:
            u_a = u_of_xi_jit(xi_a, c_stretch); dudxi_a = dudxi_jit(xi_a, c_stretch)
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            dPdxi_a = chebval_jit(xi_a, der_buf, deg_da)
            _T_basis(xi_a, G, Tbas_a)
            dPdxi_b = sum(c1d_dbp[m] * Tbas_a[m] for m in range(G))
            dPdu_a = dPdxi_a / dudxi_a; dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a**2 + dPdu_b**2
            if denom < 1e-300: continue
            w_a = dPdu_a**2 / denom
            if abs(dPdu_a) < 1e-300: continue
            wt = w_gl * dudxi_b * w_a / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return A0, A1


def phi_pou_cr(P_vals, gamma=GAMMA, tau=TAU, G_p=121):
    p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, G_p)))
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_INV)
    G = N_GRID; G_p_n = p_grid.size
    mu_table = np.empty((G_p_n, G))
    slice2 = np.empty((G, G))
    T_fix = np.empty(G)
    for k in range(G):
        xi_fix = LOBATTO[k]; u_k = U_NODES[k]
        _T_basis(xi_fix, G, T_fix)
        for jj in range(G):
            for kk in range(G):
                s = 0.0
                for i in range(G):
                    s += coeffs[i, jj, kk] * T_fix[i]
                slice2[jj, kk] = s
        f0 = f_signal_jit(u_k, 0, tau); f1 = f_signal_jit(u_k, 1, tau)
        for ip in range(G_p_n):
            p = p_grid[ip]
            A0, A1 = co_area_pou_cr(slice2, p, GL_NODES, GL_WEIGHTS, tau,
                                       C_STRETCH, G, NQ)
            den = f0*A0 + f1*A1
            mu_table[ip, k] = f1*A1/den if den > 1e-300 else 0.5
    P_new = np.empty((G, G, G))
    def interp(p, k_idx):
        if p <= p_grid[0]: return mu_table[0, k_idx]
        if p >= p_grid[-1]: return mu_table[-1, k_idx]
        i = np.searchsorted(p_grid, p) - 1
        w = (p - p_grid[i]) / (p_grid[i+1] - p_grid[i])
        return (1-w)*mu_table[i, k_idx] + w*mu_table[i+1, k_idx]
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                p_cell = max(min(p_cell, 1-1e-9), 1e-9)
                P_new[i,j,k] = crra_clear_jit(interp(p_cell, i),
                                                 interp(p_cell, j),
                                                 interp(p_cell, k),
                                                 gamma)
    return P_new


if __name__ == '__main__':
    print('=== POU with chebroots (companion-matrix all-real-roots) ===\n')
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    from cheby_numba_pou_tab import phi_pou_tab
    _ = phi_pou_tab(P_in, G_p=121)  # warmup
    t0 = time.time(); P_bi = phi_pou_tab(P_in, G_p=121); t_bi = time.time()-t0
    t0 = time.time(); P_cr = phi_pou_cr(P_in, G_p=121); t_cr = time.time()-t0
    diff = float(np.max(np.abs(P_bi - P_cr)))
    print(f'  Scan-bisect: t={t_bi*1000:.1f} ms')
    print(f'  Chebroots:   t={t_cr:.2f} s')
    print(f'  max|scan - chebroots|: {diff:.3e}')
    F_bi = float(np.max(np.abs(P_bi - P_in)))
    F_cr = float(np.max(np.abs(P_cr - P_in)))
    print(f'  ||F_bi||_inf = {F_bi:.3e}, ||F_cr||_inf = {F_cr:.3e}')
