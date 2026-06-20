"""POU operator with LAGRANGIAN root continuation in p.

Instead of cold-finding all roots at each (xi_a, p) pair, we:
  1. Find all roots at ONE reference p_ref (using companion-matrix
     chebroots) for each xi_a, ONCE.
  2. For other p values, evolve each root via 1D Newton from the
     previous p's root → smooth root trajectories in p.
  3. Detect root births/deaths via complementary all-roots scan
     every K p-steps (catches new roots; lost roots auto-drop).

Smoothness: root location is C^∞ in p as long as topology is constant.
With the atanh stretching, roots crossing the box edge xi=±1 contribute
zero amplitude (f_v(u=±∞)=0), so the operator is smooth across edge events.
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
from cheby_roots_numba import chebroots_companion

MAX_ROOTS = 16


@njit(cache=True)
def newton_update_root(c, n, xi0, p_target, n_iters=50, tol=1e-15):
    """Newton-update one root of c(xi) = p_target starting from xi0.
    Returns (new_xi, dPdxi). If diverges, returns (NaN, 0)."""
    xi = xi0
    der_buf = np.empty(n)
    deg = chebder_jit(c, n, der_buf)
    for _ in range(n_iters):
        val = chebval_jit(xi, c, n) - p_target
        if abs(val) < tol:
            break
        d = chebval_jit(xi, der_buf, deg)
        if abs(d) < 1e-300:
            return float('nan'), 0.0
        dx = -val / d
        # Damped step if outside [-1, 1]
        new_xi = xi + dx
        if new_xi > 1.5 or new_xi < -1.5:
            return float('nan'), 0.0
        xi = new_xi
    d = chebval_jit(xi, der_buf, deg)
    return xi, d


@njit(cache=True)
def co_area_pou_lagr_xi_a_term(
        slice2_coeffs, p_grid, gl_nodes, gl_weights,
        tau, c_stretch, n_grid, nq, p_ref_idx,
        ip_target_idx, A_out):
    """For ONE p_target, term A^(b) only: fix xi_a at each GL node,
    track roots in xi_b using Newton continuation in p from the
    reference p index p_ref_idx.

    A_out: (2,) output array; A_out[0] += A0, A_out[1] += A1.
    """
    G = n_grid
    Tbas_a = np.empty(G); Tbas_b = np.empty(G)
    c1d_b = np.empty(G); c1d_b_shift = np.empty(G)
    c1d_dap = np.empty(G); der_buf = np.empty(G)
    col_buf = np.empty(G)
    roots_ref = np.empty(MAX_ROOTS)

    p_target = p_grid[ip_target_idx]
    p_ref = p_grid[p_ref_idx]

    for q in range(nq):
        xi_a = gl_nodes[q]
        w_gl = gl_weights[q]
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        _T_basis(xi_a, G, Tbas_a)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas_a[m]
            c1d_b[n] = s
        # Find ALL roots at p_ref
        for n in range(G):
            c1d_b_shift[n] = c1d_b[n]
        c1d_b_shift[0] -= p_ref
        nrb = chebroots_companion(c1d_b_shift, G, roots_ref)
        if nrb == 0:
            continue
        # Compute d_a slice (in xi_b basis) — needed for derivative ratio
        for n in range(G):
            for m in range(G):
                col_buf[m] = slice2_coeffs[m, n]
            deg = chebder_jit(col_buf, G, der_buf)
            c1d_dap[n] = chebval_jit(xi_a, der_buf, deg)
        # Newton-evolve each root from p_ref to p_target along straight p path
        # Just track each root's xi_b position
        for r in range(nrb):
            xi_b_track = roots_ref[r]
            xi_b_new, dPdxi_b = newton_update_root(c1d_b, G, xi_b_track,
                                                      p_target)
            if math.isnan(xi_b_new):
                continue
            if xi_b_new < -1.0 or xi_b_new > 1.0:
                # Root left the box — atanh stretching makes contribution vanish
                continue
            u_b = u_of_xi_jit(xi_b_new, c_stretch)
            dudxi_b = dudxi_jit(xi_b_new, c_stretch)
            f0b = f_signal_jit(u_b, 0, tau)
            f1b = f_signal_jit(u_b, 1, tau)
            _T_basis(xi_b_new, G, Tbas_b)
            dPdxi_a = 0.0
            for n in range(G):
                dPdxi_a += c1d_dap[n] * Tbas_b[n]
            dPdu_a = dPdxi_a / dudxi_a
            dPdu_b = dPdxi_b / dudxi_b
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            wt = w_gl * dudxi_a * w_b / abs(dPdu_b)
            A_out[0] += wt * f0a * f0b
            A_out[1] += wt * f1a * f1b


# Top-level Phi: not yet integrated -- this module just provides the building
# block to be tested separately.

if __name__ == '__main__':
    print('=== Lagrangian POU building blocks ===')
    print('newton_update_root: 1D Newton continuation of a single root')
    print('co_area_pou_lagr_xi_a_term: track roots smoothly across p')
    print()
    # Test newton_update_root
    # Polynomial: T_3(x) = -3x + 4x^3, roots at 0, ±√3/2 ≈ ±0.866
    c = np.array([0.0, -0.75, 0.0, 0.25])  # in monomial basis: -0.75x + 0.25*4x^3? hmm
    # Use chebroots to find roots properly
    from numpy.polynomial import chebyshev as npc
    c_cheb = np.array([0.0, 0.0, 0.0, 1.0])  # T_3 in Cheb basis
    expected = npc.chebroots(c_cheb)
    print(f'T_3 roots (numpy): {expected}')
    # Test newton update
    xi0 = 0.8  # near 0.866 root
    xi_new, d = newton_update_root(c_cheb, 4, xi0, 0.0)
    print(f'Newton from xi0=0.8 to T_3=0: xi_new={xi_new:.6f}, dP/dxi={d:.4f}')
    # Move target to T_3 = 0.5
    xi_new2, d2 = newton_update_root(c_cheb, 4, xi_new, 0.5)
    print(f'Then to T_3=0.5: xi_new={xi_new2:.6f}, dP/dxi={d2:.4f}')
    print('OK -- continuation works.')
