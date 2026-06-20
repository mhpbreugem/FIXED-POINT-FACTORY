"""Cheby-native FP solver: at each cube cell, evaluate mu via analytic contour
at the exact p_cell. No bilinear interp, no kernel-band, no Richardson.
Test the hypothesis: is the slow Cheby convergence at high tau due to the
operator (Lin-CDF R4) or intrinsic to the equilibrium?
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          cheby_slice_2d, cheby_eval_1d_at_xi_a,
                                          find_real_roots_in_unit)
from cheby_numba import f_signal_jit, crra_clear_jit


def lookup_at(coefs_3d, U_MAX, p_target, u_k, tau, NQK=24):
    """Same as lookup_analytic_lobatto but with explicit NQK GL nodes."""
    coefs_2d = cheby_slice_2d(coefs_3d, U_MAX, u_k)
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    u_a_nodes = U_MAX * n_xi
    du_a_w = U_MAX * w_xi
    A0 = 0.0; A1 = 0.0
    for q in range(NQK):
        u_a = u_a_nodes[q]; xi_a = u_a / U_MAX
        coefs_b = cheby_eval_1d_at_xi_a(coefs_2d, xi_a).copy()
        coefs_b[0] -= p_target
        roots_xi = find_real_roots_in_unit(coefs_b)
        if len(roots_xi) == 0: continue
        coefs_b_deriv = cheb.chebder(coefs_b, 1)
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        for xi_b in roots_xi:
            u_b = xi_b * U_MAX
            dPdb_u = cheb.chebval(xi_b, coefs_b_deriv) / U_MAX
            if abs(dPdb_u) < 1e-300: continue
            wt = du_a_w[q] / abs(dPdb_u)
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
    den = f0k * A0 + f1k * A1
    if den > 1e-300:
        m = f1k * A1 / den
        return min(max(m, 1e-9), 1.0 - 1e-9)
    return 0.5


def phi_cheby_native(P, u_grid, U_MAX, tau, gamma, NQK=24):
    """One Phi call: fit Cheby to P, evaluate mu at each cube cell via analytic
    contour at the smooth p_cell (no interp)."""
    G = u_grid.size
    coefs = cheby_fit_lobatto_3d(P, U_MAX)
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                # p_cell from the smooth Cheby polynomial
                p_cell = cheb.chebval3d(u_grid[i]/U_MAX, u_grid[j]/U_MAX,
                                              u_grid[k]/U_MAX, coefs)
                p_cell = float(min(max(p_cell, 1e-9), 1.0 - 1e-9))
                mu0 = lookup_at(coefs, U_MAX, p_cell, u_grid[i], tau, NQK)
                mu1 = lookup_at(coefs, U_MAX, p_cell, u_grid[j], tau, NQK)
                mu2 = lookup_at(coefs, U_MAX, p_cell, u_grid[k], tau, NQK)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new, coefs


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def solve_cheby_native(gamma, tau, G, P_init=None, n_iter=30, target=1e-10,
                            NQK=24, verbose=True):
    """Anderson-accelerated Cheby-native FP solve."""
    u_grid, U_MAX = lobatto_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    if P_init is None: P_init = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    P = P_init.copy()
    Xh, Gh = [], []
    P_best = P.copy(); F_best = float("inf"); coefs_best = None
    for it in range(n_iter):
        t0 = time.time()
        P_new, coefs = phi_cheby_native(P, u_grid, U_MAX, tau, gamma, NQK=NQK)
        F = float(np.max(np.abs(P_new - P)))
        ec = edge_coef(coefs)
        wall = time.time() - t0
        if F < F_best: F_best = F; P_best = P.copy(); coefs_best = coefs.copy()
        if verbose:
            print(f"  it{it+1:2d}: |F|={F:.3e}  edge={ec:.3e}  ({wall:.1f}s)",
                  flush=True)
        if F < target: break
        # Damped Anderson
        gx_flat = P_new.ravel()
        Xh.append(P.ravel().copy()); Gh.append(gx_flat.copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: P_next = 0.4*P + 0.6*P_new
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x_anders = (Gh[k-1] + DG @ ga).reshape(G, G, G)
                P_next = 0.4*P + 0.6*x_anders
            except: P_next = 0.4*P + 0.6*P_new
        P = np.clip(P_next, 1e-9, 1-1e-9)
    return P_best, F_best, coefs_best, u_grid, U_MAX


if __name__ == "__main__":
    gamma, tau = 100.0, 1.0
    print(f"=== Cheby-native FP solve at gamma={gamma}, tau={tau} ===")
    for G in [11, 15, 21]:
        print(f"\n--- G={G} ---")
        P, F, coefs, u_grid, U_MAX = solve_cheby_native(gamma, tau, G,
                                                                  n_iter=20, target=1e-10,
                                                                  NQK=24)
        print(f"  final |F|={F:.3e}, edge_coef={edge_coef(coefs):.3e}")
