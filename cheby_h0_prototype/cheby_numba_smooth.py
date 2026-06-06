"""Smoothed co-area operator with explicit boundary corrections.

Diagnosis (see fp_analysis.tex): the chebroots / bisect operator floors at
||F||_inf ~ 3e-3 because A_v(p, u_k) has C^0 creases where contour roots
are born / die at the box edge xi_b = +/- 1. This file replaces the
"if NaN: continue" branch with a smooth boundary contribution:

  - Compute boundary values P_- = P(xi_a^(q), -1, u_k)
                            P_+ = P(xi_a^(q), +1, u_k)
  - Compute signed gap delta = max(0, P_min - p_target, p_target - P_max)
    where [P_min, P_max] is the polynomial's range on [-1, 1] approximated
    by min(P_-, P_+), max(P_-, P_+).
  - Apply smooth weight w(delta) = exp(-(delta / h_bw)^2)
  - When delta = 0 (root exists in interior), w = 1 and we use the actual
    root via bisection.
  - When delta > 0 (no interior root), we add a phantom contribution using
    the boundary value as a "ghost root" with weight w.

This makes A_v(p, u_k) a C^1 function of P, killing the creases.
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, N, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, V_MAT, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit, f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)

from cheby_numba_bisect import cheb_bisect_single


@njit(cache=True)
def co_area_smooth(slice2_coeffs, p_target, gl_nodes, gl_weights,
                    tau, c_stretch, n_grid, nq, h_bw):
    """Smoothed co-area integral with boundary corrections.
    h_bw is the smoothing bandwidth in p-space."""
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas = np.empty(G)
    c1d_b = np.empty(G)
    der_buf = np.empty(G)
    eps_box = 1e-12
    for q in range(nq):
        xi_a = gl_nodes[q]
        w_gl = gl_weights[q]
        _T_basis(xi_a, G, Tbas)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas[m]
            c1d_b[n] = s
        # Boundary values P(xi_a, +/-1)
        P_lo = chebval_jit(-1.0 + eps_box, c1d_b, G)
        P_hi = chebval_jit( 1.0 - eps_box, c1d_b, G)
        # Approximate polynomial range on [-1,1] by boundary values
        if P_lo <= P_hi:
            P_min = P_lo; P_max = P_hi
        else:
            P_min = P_hi; P_max = P_lo
        # Signed gap: 0 if inside range, positive if outside
        if p_target < P_min:
            delta = P_min - p_target
        elif p_target > P_max:
            delta = p_target - P_max
        else:
            delta = 0.0
        # Smooth weight
        w_smooth = math.exp(-(delta / h_bw) ** 2)

        # Common axis-a kinematics
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        deg_der = chebder_jit(c1d_b, G, der_buf)

        # Interior root contribution (if any)
        xi_b_root = cheb_bisect_single(c1d_b, G, p_target)
        if not math.isnan(xi_b_root):
            dPdxi_b = chebval_jit(xi_b_root, der_buf, deg_der)
            if abs(dPdxi_b) > 1e-12:
                u_b = u_of_xi_jit(xi_b_root, c_stretch)
                dudxi_b = dudxi_jit(xi_b_root, c_stretch)
                f0b = f_signal_jit(u_b, 0, tau)
                f1b = f_signal_jit(u_b, 1, tau)
                dPdu_b = dPdxi_b / dudxi_b
                wt = w_gl * dudxi_a * dudxi_b / abs(dPdu_b)
                # Interior contribution: full weight (delta=0 here so w_smooth~1)
                A0 += wt * f0a * f0b
                A1 += wt * f1a * f1b

        # Boundary "ghost root" contribution when delta > 0
        # (smoothly tapers contribution as the contour exits the box)
        if delta > 0.0 and w_smooth > 1e-14:
            # Pick the closer boundary
            if p_target < P_min:
                xi_b_bd = -1.0 + eps_box if P_lo < P_hi else 1.0 - eps_box
            else:
                xi_b_bd = 1.0 - eps_box if P_hi > P_lo else -1.0 + eps_box
            dPdxi_b = chebval_jit(xi_b_bd, der_buf, deg_der)
            if abs(dPdxi_b) > 1e-12:
                u_b = u_of_xi_jit(xi_b_bd, c_stretch)
                dudxi_b = dudxi_jit(xi_b_bd, c_stretch)
                f0b = f_signal_jit(u_b, 0, tau)
                f1b = f_signal_jit(u_b, 1, tau)
                dPdu_b = dPdxi_b / dudxi_b
                wt = w_gl * dudxi_a * dudxi_b / abs(dPdu_b)
                A0 += w_smooth * wt * f0a * f0b
                A1 += w_smooth * wt * f1a * f1b
    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def phi_one_cell_smooth(coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                         tau, gamma, c_stretch, n_grid, nq, h_bw):
    G = n_grid
    Ti = np.empty(G); Tj = np.empty(G); Tk = np.empty(G)
    _T_basis(lobatto[i], G, Ti)
    _T_basis(lobatto[j], G, Tj)
    _T_basis(lobatto[k], G, Tk)
    p_old = 0.0
    for ii in range(G):
        for jj in range(G):
            for kk in range(G):
                p_old += coeffs[ii, jj, kk] * Ti[ii] * Tj[jj] * Tk[kk]
    eps_p = 1e-9
    if p_old < eps_p: p_old = eps_p
    elif p_old > 1 - eps_p: p_old = 1 - eps_p

    slice2 = np.empty((G, G))
    Tfix = np.empty(G)
    mus = np.empty(3)
    for k_agent in range(3):
        if k_agent == 0: idx_fix = i
        elif k_agent == 1: idx_fix = j
        else: idx_fix = k
        xi_fix = lobatto[idx_fix]
        _T_basis(xi_fix, G, Tfix)
        if k_agent == 0:
            for jj in range(G):
                for kk in range(G):
                    s = 0.0
                    for ii in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[ii]
                    slice2[jj, kk] = s
        elif k_agent == 1:
            for ii in range(G):
                for kk in range(G):
                    s = 0.0
                    for jj in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[jj]
                    slice2[ii, kk] = s
        else:
            for ii in range(G):
                for jj in range(G):
                    s = 0.0
                    for kk in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[kk]
                    slice2[ii, jj] = s
        A0, A1 = co_area_smooth(slice2, p_old, gl_nodes, gl_weights,
                                  tau, c_stretch, G, nq, h_bw)
        u_own = u_of_xi_jit(xi_fix, c_stretch)
        f0 = f_signal_jit(u_own, 0, tau)
        f1 = f_signal_jit(u_own, 1, tau)
        den = f0 * A0 + f1 * A1
        if den > 1e-30:
            mus[k_agent] = (f1 * A1) / den
        else:
            mus[k_agent] = 0.5
    return crra_clear_jit(mus[0], mus[1], mus[2], gamma)


@njit(cache=True)
def phi_jit_smooth(P_vals, V_inv, lobatto, gl_nodes, gl_weights,
                    tau, gamma, c_stretch, n_grid, nq, h_bw):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = phi_one_cell_smooth(
                    coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                    tau, gamma, c_stretch, G, nq, h_bw)
    return P_new


def phi_smooth(P_vals, gamma=GAMMA, tau=TAU, h_bw=1e-3):
    return phi_jit_smooth(P_vals, V_INV, LOBATTO, GL_NODES, GL_WEIGHTS,
                            tau, gamma, C_STRETCH, N_GRID, NQ, h_bw)


if __name__ == '__main__':
    print('=== Smoothed operator: validate + sweep h_bw ===\n')
    from cheby_numba import phi as phi_chebroots
    from cheby_numba_bisect import phi_bisect

    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    P_in = sg(0.5 * T)

    # Warmup
    _ = phi_chebroots(P_in); _ = phi_bisect(P_in)
    _ = phi_smooth(P_in, h_bw=1e-3)

    print(f'N={N}, G={N_GRID}:')
    print(f'  Reference: |phi_bisect(P) - phi_chebroots(P)|_inf = '
          f'{float(np.max(np.abs(phi_bisect(P_in) - phi_chebroots(P_in)))):.3e}')
    print(f'  h_bw sweep (smooth vs chebroots, smooth vs bisect):')
    for h_bw in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-13]:
        P_sm = phi_smooth(P_in, h_bw=h_bw)
        d_cr = float(np.max(np.abs(P_sm - phi_chebroots(P_in))))
        d_bi = float(np.max(np.abs(P_sm - phi_bisect(P_in))))
        print(f'    h_bw={h_bw:>8.0e}: |sm-cr|={d_cr:.3e}, |sm-bi|={d_bi:.3e}')
