"""Drop-in operator using bisection-on-polynomial instead of chebroots.
Validates the actual speedup in the full Φ context, not just isolated chebroots.

The lifted form is monotone in xi_b (within each axis) when h is bounded.
We bisect the 1D Cheb polynomial P(xi_b) − p_target = 0 directly."""
import os, time, math
import numpy as np
from numba import njit

# Reuse all constants and helpers from cheby_numba
from cheby_numba import (TAU, GAMMA, C_STRETCH, N, NQ, N_GRID, LOBATTO, U_NODES,
                            V_INV, V_MAT, GL_NODES, GL_WEIGHTS,
                            chebval_jit, chebder_jit, _T_basis,
                            u_of_xi_jit, dudxi_jit, f_signal_jit, crra_clear_jit,
                            vals_to_coeffs_3d_jit)

@njit(cache=True)
def cheb_bisect_single(c, n, p_target):
    """Find one root of c.T - p_target in [-1+eps, 1-eps] via bisection."""
    lo = -1.0 + 1e-12
    hi = 1.0 - 1e-12
    val_lo = chebval_jit(lo, c, n) - p_target
    val_hi = chebval_jit(hi, c, n) - p_target
    if val_lo * val_hi > 0:
        return float('nan')
    if val_lo == 0.0: return lo
    if val_hi == 0.0: return hi
    # Standard bisection
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        val_mid = chebval_jit(mid, c, n) - p_target
        if val_lo * val_mid < 0:
            hi = mid
        else:
            lo = mid
            val_lo = val_mid
    return 0.5 * (lo + hi)

@njit(cache=True)
def co_area_evidence_bisect(slice2_coeffs, p_target, gl_nodes, gl_weights,
                              tau, c_stretch, n_grid, nq):
    """Replace chebroots inner loop with bisection. Assumes monotone polynomial."""
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas = np.empty(G)
    c1d_b = np.empty(G)
    c1d_b_shift = np.empty(G)
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
        for n in range(G):
            c1d_b_shift[n] = c1d_b[n]
        c1d_b_shift[0] -= p_target
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
    return 0.5 * A0, 0.5 * A1

@njit(cache=True)
def phi_one_cell_bisect(coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                          tau, gamma, c_stretch, n_grid, nq):
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
        A0, A1 = co_area_evidence_bisect(slice2, p_old, gl_nodes, gl_weights,
                                            tau, c_stretch, G, nq)
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
def phi_jit_bisect(P_vals, V_inv, lobatto, gl_nodes, gl_weights,
                     tau, gamma, c_stretch, n_grid, nq):
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = phi_one_cell_bisect(
                    coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                    tau, gamma, c_stretch, G, nq)
    return P_new


def phi_bisect(P_vals, gamma=GAMMA, tau=TAU):
    return phi_jit_bisect(P_vals, V_INV, LOBATTO, GL_NODES, GL_WEIGHTS,
                            tau, gamma, C_STRETCH, N_GRID, NQ)


if __name__ == '__main__':
    from cheby_numba import phi as phi_chebroots
    import numpy as np

    print('=== Validating bisection operator against chebroots operator ===\n')
    def sigmoid(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)

    # JIT warmup
    P_in = sigmoid(0.5*T)
    _ = phi_bisect(P_in); _ = phi_chebroots(P_in)

    print(f'N={N}, G={N_GRID}: 5 timing runs each')
    print(f'{"chebroots(s)":>15} {"bisection(s)":>15} {"speedup":>10} {"max diff":>14}')
    for trial in range(5):
        t0 = time.time(); P_cr = phi_chebroots(P_in); t_cr = time.time() - t0
        t0 = time.time(); P_bi = phi_bisect(P_in); t_bi = time.time() - t0
        diff = float(np.max(np.abs(P_cr - P_bi)))
        print(f'{t_cr:>15.4f} {t_bi:>15.4f} {t_cr/t_bi:>10.2f}x {diff:>14.3e}')

    # Now at different N (would need to refactor for general N — for now just N=6)
