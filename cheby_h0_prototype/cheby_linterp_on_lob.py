"""Variant: linear interpolation BUT on Lobatto-atanh nodes (same as Cheb).
This isolates the effect of interpolation (linear vs Cheb-poly), keeping the
quadrature and node placement identical to the Cheb operator.

Phase A uses GL in xi (with atanh-stretched u_a, dudxi_a Jacobian, etc.)
but evaluates the slice via 2D linear interpolation on the Lobatto u-grid
(instead of via Cheb polynomial value).
"""
import os, time, math
import numpy as np
from numba import njit

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, N_GRID, LOBATTO, U_NODES,
                            GL_NODES, GL_WEIGHTS, u_of_xi_jit, dudxi_jit,
                            f_signal_jit, crra_clear_jit)
from cheby_linear_op import _find_interval, linear_1d, linear_1d_slope


@njit(cache=True)
def linterp_slice_eval(u_grid, P_slice2d, u_a, n_grid):
    """Same as slice_eval_2d but takes u_a directly."""
    out = np.empty(n_grid)
    if u_a <= u_grid[0]:
        for j in range(n_grid):
            out[j] = P_slice2d[0, j]
        return out
    if u_a >= u_grid[n_grid-1]:
        for j in range(n_grid):
            out[j] = P_slice2d[n_grid-1, j]
        return out
    i = _find_interval(u_grid, u_a, n_grid)
    w = (u_a - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    for j in range(n_grid):
        out[j] = (1.0 - w) * P_slice2d[i, j] + w * P_slice2d[i+1, j]
    return out


@njit(cache=True)
def find_root_linterp(u_grid, P_along_b, p_target, n_grid):
    """Find first u_b where piecewise-linear P_along_b crosses p_target."""
    for j in range(n_grid - 1):
        a = P_along_b[j] - p_target
        b = P_along_b[j+1] - p_target
        if a * b <= 0 and (a != 0 or b != 0):
            ya = P_along_b[j]; yb = P_along_b[j+1]
            xa = u_grid[j]; xb = u_grid[j+1]
            slope = (yb - ya) / (xb - xa)
            if slope == 0:
                continue
            u_root = xa + (p_target - ya) / slope
            return u_root, slope
    return float('nan'), float('nan')


@njit(cache=True)
def co_area_linterp_on_lob(slice2_lob, p_target, gl_nodes, gl_weights,
                              tau, c_stretch, u_lob_nodes, n_grid, nq):
    """Co-area integral using Lobatto-atanh quadrature in xi_a (same as Cheb),
    but slice eval and root via linear interp on u_lob_nodes."""
    A0 = 0.0; A1 = 0.0
    for q in range(nq):
        xi_a = gl_nodes[q]
        w = gl_weights[q]
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        P_along_b = linterp_slice_eval(u_lob_nodes, slice2_lob, u_a, n_grid)
        u_root, slope = find_root_linterp(u_lob_nodes, P_along_b, p_target, n_grid)
        if math.isnan(u_root):
            continue
        if abs(slope) < 1e-12:
            continue
        # slope is dP/du_b for linear interp; for derivative consistency
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        f0b = f_signal_jit(u_root, 0, tau)
        f1b = f_signal_jit(u_root, 1, tau)
        wt = w * dudxi_a / abs(slope)
        A0 += wt * f0a * f0b
        A1 += wt * f1a * f1b
    return 0.5 * A0, 0.5 * A1


@njit(cache=True)
def build_mu_table_linterp_lob(P_vals, u_lob_nodes, p_grid, gl_nodes, gl_weights,
                                  tau, c_stretch, n_grid, nq):
    """mu table using Lobatto quadrature but linear interp for slice/root."""
    G_p = p_grid.shape[0]
    mu_table = np.empty((G_p, n_grid))
    slice2 = np.empty((n_grid, n_grid))
    for k_node in range(n_grid):
        u_k = u_lob_nodes[k_node]
        f0 = f_signal_jit(u_k, 0, tau)
        f1 = f_signal_jit(u_k, 1, tau)
        for jj in range(n_grid):
            for kk in range(n_grid):
                slice2[jj, kk] = P_vals[k_node, jj, kk]
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_linterp_on_lob(slice2, p, gl_nodes, gl_weights,
                                              tau, c_stretch, u_lob_nodes,
                                              n_grid, nq)
            den = f0*A0 + f1*A1
            if den > 1e-30:
                mu_table[ip, k_node] = f1*A1 / den
            else:
                mu_table[ip, k_node] = 0.5
    return mu_table


@njit(cache=True, inline='always')
def interp_mu_1d_v2(mu_table, p, p_grid, k_idx, G_p):
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    i = _find_interval(p_grid, p, G_p)
    w = (p - p_grid[i]) / (p_grid[i+1] - p_grid[i])
    return (1.0 - w) * mu_table[i, k_idx] + w * mu_table[i+1, k_idx]


@njit(cache=True)
def phi_linterp_lob_jit(P_vals, u_lob_nodes, p_grid, gl_nodes, gl_weights,
                          tau, gamma, c_stretch, n_grid, nq):
    """Linear-interp-on-Lobatto operator."""
    mu_table = build_mu_table_linterp_lob(P_vals, u_lob_nodes, p_grid,
                                            gl_nodes, gl_weights, tau,
                                            c_stretch, n_grid, nq)
    G_p = p_grid.shape[0]
    P_new = np.empty((n_grid, n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                p_cell = P_vals[i, j, k]
                eps = 1e-9
                if p_cell < eps: p_cell = eps
                elif p_cell > 1 - eps: p_cell = 1 - eps
                mu0 = interp_mu_1d_v2(mu_table, p_cell, p_grid, i, G_p)
                mu1 = interp_mu_1d_v2(mu_table, p_cell, p_grid, j, G_p)
                mu2 = interp_mu_1d_v2(mu_table, p_cell, p_grid, k, G_p)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def phi_linterp_lob(P_vals, gamma=GAMMA, tau=TAU, G_p=51):
    p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8.0, 8.0, G_p)))
    return phi_linterp_lob_jit(P_vals, U_NODES, p_grid, GL_NODES, GL_WEIGHTS,
                                 tau, gamma, C_STRETCH, N_GRID, NQ)


if __name__ == '__main__':
    print('=== Linear-interp-on-Lobatto operator vs Cheb-tab ===\n')
    from cheby_numba_tab import phi_tab
    import numpy as np
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU*(U1+U2+U3)
    def sg(x): return 1/(1+np.exp(-x))
    P_in = sg(0.5*T)

    # Warmup
    _ = phi_tab(P_in); _ = phi_linterp_lob(P_in)

    print(f'N=6 (G={N_GRID}), tau={TAU}, gamma={GAMMA}, G_p=51:')
    for trial in range(3):
        t0 = time.time(); P_cheb = phi_tab(P_in, G_p=51); t_cheb = time.time()-t0
        t0 = time.time(); P_lint = phi_linterp_lob(P_in, G_p=51); t_lint = time.time()-t0
        diff = float(np.max(np.abs(P_lint - P_cheb)))
        print(f'  trial {trial+1}: Cheb={t_cheb*1000:.1f}ms, '
              f'Linterp(on lob)={t_lint*1000:.1f}ms, '
              f'max|lint-cheb|={diff:.3e}')

    # Iterate both
    def anderson(op, n_iter=60, m=8):
        x = sg(0.5*T).copy().reshape(-1)
        Xh, Gh = [], []; Fs = []
        sh = (N_GRID, N_GRID, N_GRID)
        for it in range(n_iter):
            gx = op(x.reshape(sh)).reshape(-1)
            F = gx - x; Fs.append(float(np.max(np.abs(F))))
            Xh.append(x.copy()); Gh.append(gx.copy())
            if len(Xh) > m: Xh.pop(0); Gh.pop(0)
            k = len(Xh)
            if k <= 1: x = gx
            else:
                DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
                R_k = Gh[k-1] - Xh[k-1]
                try:
                    A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                    ga = np.linalg.solve(A, -DR.T @ R_k)
                    DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                    x = Gh[k-1] + DG @ ga
                except: x = gx
        return Fs

    F_cheb = anderson(lambda P: phi_tab(P, G_p=51))
    F_lint = anderson(lambda P: phi_linterp_lob(P, G_p=51))
    print(f'\nAnderson floor (60 iters):')
    print(f'  Cheb tab:                {min(F_cheb):.3e}')
    print(f'  Linear-on-Lobatto:       {min(F_lint):.3e}')

    # Compute slopes
    P_cheb_conv = sg(0.5*T)
    for _ in range(60): P_cheb_conv = phi_tab(P_cheb_conv, G_p=51)
    P_lint_conv = sg(0.5*T)
    for _ in range(60): P_lint_conv = phi_linterp_lob(P_lint_conv, G_p=51)
    Lc = np.log(np.clip(P_cheb_conv, 1e-15, 1-1e-15)/(1-np.clip(P_cheb_conv, 1e-15, 1-1e-15))).ravel()
    sc = float(np.sum(Lc*T.ravel())/np.sum(T.ravel()**2))
    Ll = np.log(np.clip(P_lint_conv, 1e-15, 1-1e-15)/(1-np.clip(P_lint_conv, 1e-15, 1-1e-15))).ravel()
    sl = float(np.sum(Ll*T.ravel())/np.sum(T.ravel()**2))
    print(f'\nLogit-vs-T slope (after 60 iter, gamma=1):')
    print(f'  Cheb tab:    alpha* = {sc:.4f}')
    print(f'  Linterp lob: alpha* = {sl:.4f}')
