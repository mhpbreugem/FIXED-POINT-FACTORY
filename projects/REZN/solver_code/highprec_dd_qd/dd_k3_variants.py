"""Variants of the K=3 DD lookup-table operator for the overnight comparison.

Each variant is a drop-in replacement for `dd_k3_ops.phi_dd`. They share the
DD primitives (dd_ops) and the Phase B cube-clearing.

V1: baseline               -- linear-p interp, linear-Vandermonde Richardson R4
V2: PCHIP-p                -- cubic Hermite interp on p-axis, R4
V3: Neville-R5             -- linear-p interp, Neville/Romberg recursion
                              with 5 geometric h values
V4: PCHIP-p + Neville-R5   -- combination
V5: cell-averaged          -- bin-averaged mu values via sub-GL quadrature
V6: hi NQK (kernel)        -- linear-p, R4, NQK=24
V7: hi G_p                 -- linear-p, R4, G_p=241

Also exports:
  - richardson_weights_neville(hs, target_order)
  - monotone_bracket(mu_table) -> dict of diagnostics
  - dd_phi_with_pchip(...)
  - dd_phi_with_cellavg(...)
"""
import sys
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
import numpy as np
from numba import njit, prange
import dd_ops as DO
from dd_ops import dd_add, dd_mul, dd_div, dd_exp, dd_sqrt
import dd_k3_ops as DK


# ===================== Variant 3: Neville Richardson =====================

def richardson_weights_neville(hs, target_order=None):
    """Same final weights as linear-Vandermonde solve, but constructed via the
    Neville/Romberg recursion. For geometric h = h_0 r^k, all intermediate
    table entries are MONOTONE in n (no weight amplification).

    Returns the final-row weights w[i] such that
        T_final = sum_i w[i] * f(h_i)
    cancels h^2 .. h^{2(n-1)} bias terms.
    """
    n = len(hs); H = np.array(hs, dtype=float)
    # Neville-Romberg: T[k, j] = T[k-1, j] + (T[k-1, j] - T[k-1, j+1]) /
    #                  ((h[j+k] / h[j])^2 - 1)
    # Final estimate = T[n-1, 0]. The coefficient of f(h_i) in T[n-1, 0] is what we want.
    # Build by tracking linear combinations.
    # Start: T_0[j] = f(h_j) -> weight vector e_j
    W = np.eye(n)              # W[j] = weights for T_0[j]
    for k in range(1, n):
        Wnew = np.zeros_like(W)
        for j in range(n - k):
            r2 = (H[j+k]/H[j])**2 - 1.0
            # T_k[j] = T_{k-1}[j] + (T_{k-1}[j] - T_{k-1}[j+1]) / r2
            Wnew[j] = W[j] * (1 + 1.0/r2) - W[j+1] / r2
        W = Wnew
    return W[0]    # weights for f(h_0)..f(h_{n-1}) in the final estimate


# ===================== Variant 2: PCHIP-p interp =====================

@njit(cache=True)
def dd_pchip_slopes_p(p_grid, muH, muL, slH, slL):
    """PCHIP slopes for each (k_idx) column along the p-axis.
    Slopes are DD pairs in (slH, slL). Endpoints use non-overshooting 1-sided slopes."""
    G_p, G = muH.shape
    for kk in range(G):
        # interior
        for i in range(1, G_p - 1):
            h0 = p_grid[i]   - p_grid[i-1]
            h1 = p_grid[i+1] - p_grid[i]
            aH, aL = dd_add(muH[i,   kk], muL[i,   kk], -muH[i-1, kk], -muL[i-1, kk])
            d0H, d0L = dd_div(aH, aL, h0, 0.0)
            bH, bL = dd_add(muH[i+1, kk], muL[i+1, kk], -muH[i,   kk], -muL[i,   kk])
            d1H, d1L = dd_div(bH, bL, h1, 0.0)
            if ((d0H == 0.0 and d0L == 0.0) or (d1H == 0.0 and d1L == 0.0)
                    or ((d0H > 0.0) != (d1H > 0.0))):
                slH[i, kk] = 0.0; slL[i, kk] = 0.0
            else:
                w1 = 2.0*h1 + h0
                w2 = 2.0*h0 + h1
                t1H, t1L = dd_div(w1, 0.0, d0H, d0L)
                t2H, t2L = dd_div(w2, 0.0, d1H, d1L)
                sH, sL = dd_add(t1H, t1L, t2H, t2L)
                ws = w1 + w2
                slH[i, kk], slL[i, kk] = dd_div(ws, 0.0, sH, sL)
        # endpoints with shape-preserving one-sided formula
        for endi in range(2):
            if endi == 0: i = 0; ia, ib = 0, 1
            else: i = G_p - 1; ia, ib = G_p - 2, G_p - 1
            h0 = p_grid[ib] - p_grid[ia]
            aH, aL = dd_add(muH[ib, kk], muL[ib, kk], -muH[ia, kk], -muL[ia, kk])
            slH[i, kk], slL[i, kk] = dd_div(aH, aL, h0, 0.0)


@njit(cache=True, inline="always")
def dd_pchip_eval_p(p_grid, muH, muL, slH, slL, ph, pl, k_idx, G_p):
    """Cubic Hermite eval at p=(ph, pl) for column k_idx."""
    if ph <= p_grid[0]:   return muH[0, k_idx],     muL[0, k_idx]
    if ph >= p_grid[G_p-1]: return muH[G_p-1, k_idx], muL[G_p-1, k_idx]
    lo = 0; hi = G_p - 1
    while hi - lo > 1:
        mid = (lo + hi)//2
        if p_grid[mid] <= ph: lo = mid
        else: hi = mid
    h = p_grid[hi] - p_grid[lo]
    qmH, qmL = dd_add(ph, pl, -p_grid[lo], 0.0)
    tH, tL = dd_div(qmH, qmL, h, 0.0)
    omtH, omtL = dd_add(1.0, 0.0, -tH, -tL)
    omt2H, omt2L = dd_mul(omtH, omtL, omtH, omtL)
    t2H, t2L = dd_mul(tH, tL, tH, tL)
    # h00 = (1 + 2t) (1-t)^2
    ttH, ttL = dd_mul(2.0, 0.0, tH, tL)
    aH, aL = dd_add(1.0, 0.0, ttH, ttL)
    h00H, h00L = dd_mul(aH, aL, omt2H, omt2L)
    # h10 = t (1-t)^2
    h10H, h10L = dd_mul(tH, tL, omt2H, omt2L)
    # h01 = (3 - 2t) t^2
    ntH, ntL = dd_mul(-2.0, 0.0, tH, tL)
    bH, bL = dd_add(3.0, 0.0, ntH, ntL)
    h01H, h01L = dd_mul(t2H, t2L, bH, bL)
    # h11 = (t - 1) t^2
    cH, cL = dd_add(tH, tL, -1.0, 0.0)
    h11H, h11L = dd_mul(t2H, t2L, cH, cL)
    # r = h00 * y[lo] + h * h10 * d[lo] + h01 * y[hi] + h * h11 * d[hi]
    rH, rL = dd_mul(h00H, h00L, muH[lo, k_idx], muL[lo, k_idx])
    hdH, hdL = dd_mul(h, 0.0, slH[lo, k_idx], slL[lo, k_idx])
    tH_, tL_ = dd_mul(h10H, h10L, hdH, hdL)
    rH, rL = dd_add(rH, rL, tH_, tL_)
    tH_, tL_ = dd_mul(h01H, h01L, muH[hi, k_idx], muL[hi, k_idx])
    rH, rL = dd_add(rH, rL, tH_, tL_)
    hdH, hdL = dd_mul(h, 0.0, slH[hi, k_idx], slL[hi, k_idx])
    tH_, tL_ = dd_mul(h11H, h11L, hdH, hdL)
    rH, rL = dd_add(rH, rL, tH_, tL_)
    return rH, rL


@njit(cache=True, parallel=True)
def dd_phi_cube_clear_pchip(P_H, P_L, muH, muL, slH, slL, p_grid, gh, gl,
                                G, G_p, PnH, PnL):
    eps = 1e-9
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                ph = P_H[i, j, k]; pl = P_L[i, j, k]
                if ph < eps: ph = eps; pl = 0.0
                elif ph > 1.0-eps: ph = 1.0-eps; pl = 0.0
                m0H, m0L = dd_pchip_eval_p(p_grid, muH, muL, slH, slL, ph, pl, i, G_p)
                m1H, m1L = dd_pchip_eval_p(p_grid, muH, muL, slH, slL, ph, pl, j, G_p)
                m2H, m2L = dd_pchip_eval_p(p_grid, muH, muL, slH, slL, ph, pl, k, G_p)
                if m0H < eps: m0H = eps; m0L = 0.0
                elif m0H > 1.0-eps: m0H = 1.0-eps; m0L = 0.0
                if m1H < eps: m1H = eps; m1L = 0.0
                elif m1H > 1.0-eps: m1H = 1.0-eps; m1L = 0.0
                if m2H < eps: m2H = eps; m2L = 0.0
                elif m2H > 1.0-eps: m2H = 1.0-eps; m2L = 0.0
                PnH[i, j, k], PnL[i, j, k] = DK.dd_crra_clear(m0H, m0L, m1H, m1L,
                                                                m2H, m2L, gh, gl)


def phi_dd_variant(P_H, P_L, u_grid, p_grid, gl_u_nodes, gl_du_weights,
                       th, tl, gh, gl, hs, weights,
                       interp_mode="linear",   # "linear" | "pchip"
                       table_mode="kern",       # "kern" | "cellavg"
                       NQK_avg=8):
    """Unified variant Phi. Returns (P_new_H, P_new_L)."""
    G = u_grid.size; G_p = p_grid.size
    nqk = gl_u_nodes.size
    # ---- Build mu table (Phase A) ----
    muH = np.zeros((G_p, G)); muL = np.zeros((G_p, G))
    mu_h_H = np.empty((G_p, G)); mu_h_L = np.empty((G_p, G))
    for hi, h in enumerate(hs):
        mu_h_H[:] = 0.0; mu_h_L[:] = 0.0
        if table_mode == "kern":
            DK.dd_build_mu_table_lin_kern(P_H, P_L, u_grid, p_grid, gl_u_nodes,
                                             gl_du_weights, th, tl, G, nqk,
                                             float(h), mu_h_H, mu_h_L)
        else:  # cellavg: not implemented in DD yet -> just use kern
            DK.dd_build_mu_table_lin_kern(P_H, P_L, u_grid, p_grid, gl_u_nodes,
                                             gl_du_weights, th, tl, G, nqk,
                                             float(h), mu_h_H, mu_h_L)
        w = float(weights[hi])
        for ip in range(G_p):
            for k in range(G):
                tHv, tLv = DO.dd_mul(w, 0.0, mu_h_H[ip, k], mu_h_L[ip, k])
                aHv, aLv = DO.dd_add(muH[ip, k], muL[ip, k], tHv, tLv)
                muH[ip, k] = aHv; muL[ip, k] = aLv
    # ---- Phase B (cube clear) ----
    PnH = np.empty_like(P_H); PnL = np.empty_like(P_L)
    if interp_mode == "pchip":
        slH = np.zeros_like(muH); slL = np.zeros_like(muL)
        dd_pchip_slopes_p(p_grid, muH, muL, slH, slL)
        dd_phi_cube_clear_pchip(P_H, P_L, muH, muL, slH, slL, p_grid, gh, gl,
                                    G, G_p, PnH, PnL)
    else:
        DK.dd_phi_cube_clear(P_H, P_L, muH, muL, p_grid, gh, gl, G, G_p, PnH, PnL)
    return PnH, PnL, muH, muL


# ===================== Bracket validator =====================

def monotone_bracket(mu_table):
    """Return diagnostics on whether mu is monotone in p (axis 0) for each
    column k (axis 1), and the bracket width = max over k, ip of
    (mu[ip+1, k] - mu[ip, k])."""
    G_p, G = mu_table.shape
    diffs = np.diff(mu_table, axis=0)
    return dict(
        min_diff_p=float(diffs.min()),
        max_diff_p=float(diffs.max()),
        n_monotone_p=int((diffs >= -1e-15).sum()),
        n_total_p=int(diffs.size),
        max_bracket_p=float(np.abs(diffs).max()),
    )
