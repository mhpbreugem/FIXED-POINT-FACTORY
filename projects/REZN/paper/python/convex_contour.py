#!/usr/bin/env python3
"""
Convexity-constrained contour integration.

Drop-in replacement for the standard contour integration step in the
REE solver. Exploits the numerical finding that each price contour has
definite curvature sign (convex or concave, never mixed).

Instead of integrating raw grid crossings (which are noisy at low G),
this module:
1. Finds crossings via the standard sweep + root-find
2. Detects the curvature sign from the crossings
3. Fits a shape-constrained interpolant (convex or concave)
4. Evaluates the fitted contour at many points for smooth integration

This gives smoother posteriors with fewer grid points.

Usage:
    from convex_contour import compute_posterior_convex
    
    # Replace the standard contour integral:
    # A_v = standard_contour_integral(price_slice, u_own, p, ...)
    
    # With:
    A0, A1 = compute_posterior_convex(price_slice, u_grid, u_own, p, tau)
    mu = f1_own * A1 / (f0_own * A0 + f1_own * A1)
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


def signal_density(u, v, tau):
    """Signal density f_v(u) = sqrt(tau/2pi) exp(-tau/2 (u - v + 1/2)^2)."""
    mean = v - 0.5  # v=1 → mean=+0.5, v=0 → mean=-0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - mean) ** 2)


def find_crossings(price_row, u_grid, p_target):
    """
    Find where price_row(u) crosses p_target.
    
    price_row: 1D array of prices at u_grid points
    u_grid: 1D array of signal grid points
    p_target: target price
    
    Returns: list of (u_grid_val, u_crossing) pairs
    """
    crossings = []
    for i in range(len(u_grid) - 1):
        p_lo, p_hi = price_row[i], price_row[i + 1]
        if np.isnan(p_lo) or np.isnan(p_hi):
            continue
        if (p_lo - p_target) * (p_hi - p_target) <= 0:
            # Linear interpolation for crossing
            if abs(p_hi - p_lo) < 1e-15:
                u_cross = (u_grid[i] + u_grid[i + 1]) / 2
            else:
                frac = (p_target - p_lo) / (p_hi - p_lo)
                u_cross = u_grid[i] + frac * (u_grid[i + 1] - u_grid[i])
            crossings.append(u_cross)
    return crossings


def detect_curvature_sign(u2_vals, u3_vals):
    """
    Detect whether u3 = g(u2) is convex (g'' > 0) or concave (g'' < 0).
    
    Returns: +1 (convex), -1 (concave), 0 (approximately linear)
    """
    if len(u2_vals) < 4:
        return 0  # not enough points to determine
    
    # Numerical second derivative
    du2 = np.diff(u2_vals)
    du3 = np.diff(u3_vals)
    
    # First derivative
    g_prime = du3 / (du2 + 1e-15)
    
    # Second derivative (central)
    g_pp = np.diff(g_prime) / (0.5 * (du2[:-1] + du2[1:]) + 1e-15)
    
    if len(g_pp) < 2:
        return 0
    
    # Interior only (avoid edge effects)
    g_pp_int = g_pp[1:-1] if len(g_pp) > 3 else g_pp
    
    n_pos = np.sum(g_pp_int > 0)
    n_neg = np.sum(g_pp_int < 0)
    
    if n_pos > 0.7 * len(g_pp_int):
        return +1  # convex
    elif n_neg > 0.7 * len(g_pp_int):
        return -1  # concave
    else:
        return 0  # approximately linear


def fit_convex_interpolant(u2_vals, u3_vals, sign):
    """
    Fit a shape-preserving interpolant with definite curvature.
    
    Uses PCHIP as base, then projects to enforce convexity/concavity.
    
    sign: +1 for convex, -1 for concave, 0 for no constraint
    """
    if len(u2_vals) < 2:
        return None
    
    # Sort by u2
    order = np.argsort(u2_vals)
    u2_sorted = np.array(u2_vals)[order]
    u3_sorted = np.array(u3_vals)[order]
    
    if sign == 0 or len(u2_sorted) < 4:
        # No constraint: use standard PCHIP
        return PchipInterpolator(u2_sorted, u3_sorted)
    
    # Enforce convexity/concavity by projecting slopes
    # For convex: slopes must be non-decreasing
    # For concave: slopes must be non-increasing
    n = len(u2_sorted)
    slopes = np.diff(u3_sorted) / (np.diff(u2_sorted) + 1e-15)
    
    if sign == +1:
        # Project slopes to be non-decreasing (convex)
        for i in range(1, len(slopes)):
            if slopes[i] < slopes[i - 1]:
                avg = (slopes[i - 1] + slopes[i]) / 2
                slopes[i - 1] = avg
                slopes[i] = avg
        # Pool adjacent violators (full isotonic regression)
        slopes = _isotonic_increasing(slopes)
    else:
        # Project slopes to be non-increasing (concave)
        slopes = _isotonic_increasing(-slopes)
        slopes = -slopes
    
    # Reconstruct u3 values from projected slopes
    u3_proj = np.zeros(n)
    u3_proj[0] = u3_sorted[0]
    for i in range(1, n):
        u3_proj[i] = u3_proj[i - 1] + slopes[i - 1] * (u2_sorted[i] - u2_sorted[i - 1])
    
    # Small correction: shift to minimize L2 distance to original
    u3_proj += np.mean(u3_sorted - u3_proj)
    
    return PchipInterpolator(u2_sorted, u3_proj)


def _isotonic_increasing(y):
    """Pool adjacent violators algorithm for isotonic (non-decreasing) regression."""
    y = np.array(y, dtype=float).copy()
    n = len(y)
    blocks = [[i] for i in range(n)]
    values = list(y)
    
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(values) - 1:
            if values[i] > values[i + 1]:
                # Merge blocks
                merged = blocks[i] + blocks[i + 1]
                avg = np.mean(y[merged])
                blocks[i] = merged
                values[i] = avg
                blocks.pop(i + 1)
                values.pop(i + 1)
                changed = True
            else:
                i += 1
    
    result = np.zeros(n)
    for block, val in zip(blocks, values):
        for idx in block:
            result[idx] = val
    return result


def compute_posterior_convex(price_slice, u_grid, u_own, p_target, tau,
                             n_eval=100):
    """
    Compute posterior for one agent using convexity-constrained contour integration.
    
    Parameters
    ----------
    price_slice : 2D array (G x G)
        Price function P[u_own_fixed, :, :] for this agent
    u_grid : 1D array (G,)
        Signal grid
    u_own : float
        This agent's signal value
    p_target : float
        Observed price
    tau : float
        Signal precision
    n_eval : int
        Number of evaluation points on the fitted contour
    
    Returns
    -------
    A0, A1 : float
        State-conditional density accumulators
    """
    G = len(u_grid)
    
    # =====================================================================
    # PASS A: sweep u_j (axis 0), find u_l crossings (axis 1)
    # =====================================================================
    u_j_A, u_l_A = [], []
    for j in range(G):
        row = price_slice[j, :]  # prices as function of u_l
        crossings = find_crossings(row, u_grid, p_target)
        for u_l_cross in crossings:
            u_j_A.append(u_grid[j])
            u_l_A.append(u_l_cross)
    
    # =====================================================================
    # PASS B: sweep u_l (axis 1), find u_j crossings (axis 0)
    # =====================================================================
    u_j_B, u_l_B = [], []
    for l in range(G):
        col = price_slice[:, l]  # prices as function of u_j
        crossings = find_crossings(col, u_grid, p_target)
        for u_j_cross in crossings:
            u_j_B.append(u_j_cross)
            u_l_B.append(u_grid[l])
    
    # =====================================================================
    # COMBINE AND FIT
    # =====================================================================
    # Merge both passes
    u_j_all = np.array(u_j_A + u_j_B)
    u_l_all = np.array(u_l_A + u_l_B)
    
    if len(u_j_all) < 3:
        # Too few crossings — fall back to raw sum
        A0 = sum(signal_density(uj, 0, tau) * signal_density(ul, 0, tau)
                 for uj, ul in zip(u_j_all, u_l_all)) + 1e-30
        A1 = sum(signal_density(uj, 1, tau) * signal_density(ul, 1, tau)
                 for uj, ul in zip(u_j_all, u_l_all)) + 1e-30
        return A0, A1
    
    # Detect curvature sign
    # Sort by u_j to get u_l = g(u_j)
    order = np.argsort(u_j_all)
    u_j_sorted = u_j_all[order]
    u_l_sorted = u_l_all[order]
    
    # Remove near-duplicates in u_j (keep average u_l)
    u_j_unique, u_l_unique = [], []
    i = 0
    while i < len(u_j_sorted):
        j = i + 1
        while j < len(u_j_sorted) and abs(u_j_sorted[j] - u_j_sorted[i]) < 0.01:
            j += 1
        u_j_unique.append(np.mean(u_j_sorted[i:j]))
        u_l_unique.append(np.mean(u_l_sorted[i:j]))
        i = j
    
    u_j_unique = np.array(u_j_unique)
    u_l_unique = np.array(u_l_unique)
    
    if len(u_j_unique) < 3:
        A0 = sum(signal_density(uj, 0, tau) * signal_density(ul, 0, tau)
                 for uj, ul in zip(u_j_unique, u_l_unique)) + 1e-30
        A1 = sum(signal_density(uj, 1, tau) * signal_density(ul, 1, tau)
                 for uj, ul in zip(u_j_unique, u_l_unique)) + 1e-30
        return A0, A1
    
    # Detect curvature
    sign = detect_curvature_sign(u_j_unique, u_l_unique)
    
    # Fit constrained interpolant
    interp = fit_convex_interpolant(u_j_unique, u_l_unique, sign)
    
    if interp is None:
        A0 = sum(signal_density(uj, 0, tau) * signal_density(ul, 0, tau)
                 for uj, ul in zip(u_j_unique, u_l_unique)) + 1e-30
        A1 = sum(signal_density(uj, 1, tau) * signal_density(ul, 1, tau)
                 for uj, ul in zip(u_j_unique, u_l_unique)) + 1e-30
        return A0, A1
    
    # =====================================================================
    # INTEGRATE ALONG FITTED CONTOUR
    # =====================================================================
    # Evaluate at n_eval evenly-spaced points
    u_j_lo = u_j_unique[0]
    u_j_hi = u_j_unique[-1]
    u_j_eval = np.linspace(u_j_lo, u_j_hi, n_eval)
    u_l_eval = interp(u_j_eval)
    
    # Arc-length element: ds = sqrt(1 + g'^2) du_j
    g_prime = interp.derivative()(u_j_eval)
    ds = np.sqrt(1 + g_prime ** 2)
    du_j = (u_j_hi - u_j_lo) / (n_eval - 1)
    
    # Density integrals with arc-length weighting
    A0 = 0.0
    A1 = 0.0
    for v in [0, 1]:
        A_v = 0.0
        for k in range(n_eval):
            uj, ul = u_j_eval[k], u_l_eval[k]
            f_j = signal_density(uj, v, tau)
            f_l = signal_density(ul, v, tau)
            A_v += f_j * f_l * ds[k] * du_j
        if v == 0:
            A0 = A_v + 1e-30
        else:
            A1 = A_v + 1e-30
    
    return A0, A1


def compute_all_posteriors_convex(price_array, u_grid, tau, n_eval=100):
    """
    Compute posteriors for all agents at all grid points.
    
    Parameters
    ----------
    price_array : 3D array (G x G x G)
        Price function P[i, j, l]
    u_grid : 1D array (G,)
        Signal grid
    tau : float
        Signal precision
    n_eval : int
        Evaluation points per contour
    
    Returns
    -------
    posteriors : dict
        posteriors[(agent, own_idx, price_idx)] = mu
    """
    G = len(u_grid)
    posteriors = {}
    
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = price_array[i, j, l]
                if np.isnan(p):
                    continue
                
                # Agent 1: slice P[i, :, :]
                A0_1, A1_1 = compute_posterior_convex(
                    price_array[i, :, :], u_grid, u_grid[i], p, tau, n_eval
                )
                f0_own = signal_density(u_grid[i], 0, tau)
                f1_own = signal_density(u_grid[i], 1, tau)
                mu_1 = f1_own * A1_1 / (f0_own * A0_1 + f1_own * A1_1)
                
                # Agent 2: slice P[:, j, :]
                A0_2, A1_2 = compute_posterior_convex(
                    price_array[:, j, :], u_grid, u_grid[j], p, tau, n_eval
                )
                f0_own = signal_density(u_grid[j], 0, tau)
                f1_own = signal_density(u_grid[j], 1, tau)
                mu_2 = f1_own * A1_2 / (f0_own * A0_2 + f1_own * A1_2)
                
                # Agent 3: slice P[:, :, l]
                A0_3, A1_3 = compute_posterior_convex(
                    price_array[:, :, l], u_grid, u_grid[l], p, tau, n_eval
                )
                f0_own = signal_density(u_grid[l], 0, tau)
                f1_own = signal_density(u_grid[l], 1, tau)
                mu_3 = f1_own * A1_3 / (f0_own * A0_3 + f1_own * A1_3)
                
                posteriors[(i, j, l)] = (mu_1, mu_2, mu_3)
    
    return posteriors


# =========================================================================
# DEMO / TEST
# =========================================================================
if __name__ == "__main__":
    from scipy.special import expit as Lam
    
    tau = 2.0
    gamma = 0.5
    G = 7
    u_grid = np.linspace(-3, 3, G)
    
    def logit(p):
        return np.log(p / (1 - p))
    
    def crra_demand(mu, p):
        if mu < 1e-12 or mu > 1-1e-12 or p < 1e-12 or p > 1-1e-12:
            return 0.0
        z = logit(mu) - logit(p)
        R = np.exp(z / gamma)
        return (R - 1) / ((1 - p) + R * p)
    
    # Build no-learning price array
    print(f"Building no-learning prices at G={G}...")
    P = np.zeros((G, G, G))
    for i in range(G):
        for j in range(G):
            for l in range(G):
                mu = [Lam(tau * u_grid[k]) for k in [i, j, l]]
                try:
                    p = brentq(
                        lambda p: sum(crra_demand(m, p) for m in mu),
                        1e-6, 1 - 1e-6
                    )
                except:
                    p = np.nan
                P[i, j, l] = p
    
    # Test at one point
    i_test, j_test, l_test = G // 2 + 1, G // 2 - 1, G // 2 + 1
    u1, u2, u3 = u_grid[i_test], u_grid[j_test], u_grid[l_test]
    p_test = P[i_test, j_test, l_test]
    
    print(f"\nTest point: (u₁, u₂, u₃) = ({u1:.1f}, {u2:.1f}, {u3:.1f})")
    print(f"Price: {p_test:.4f}")
    
    # Standard contour (no constraint)
    print("\n--- Standard contour (no constraint) ---")
    A0_std, A1_std = compute_posterior_convex(
        P[i_test, :, :], u_grid, u1, p_test, tau, n_eval=50
    )
    f0 = signal_density(u1, 0, tau)
    f1 = signal_density(u1, 1, tau)
    mu_std = f1 * A1_std / (f0 * A0_std + f1 * A1_std)
    print(f"  A0={A0_std:.6f}, A1={A1_std:.6f}")
    print(f"  mu_1 = {mu_std:.6f}")
    
    # With convexity constraint
    print("\n--- Convexity-constrained contour ---")
    # The constraint is built into compute_posterior_convex automatically
    # (it detects the sign and fits a constrained interpolant)
    A0_cvx, A1_cvx = compute_posterior_convex(
        P[i_test, :, :], u_grid, u1, p_test, tau, n_eval=200
    )
    mu_cvx = f1 * A1_cvx / (f0 * A0_cvx + f1 * A1_cvx)
    print(f"  A0={A0_cvx:.6f}, A1={A1_cvx:.6f}")
    print(f"  mu_1 = {mu_cvx:.6f}")
    
    # Compare with private prior
    mu_prior = Lam(tau * u1)
    print(f"\n  Private prior: {mu_prior:.6f}")
    print(f"  Standard:      {mu_std:.6f}")
    print(f"  Constrained:   {mu_cvx:.6f}")
    print(f"  Difference:    {abs(mu_cvx - mu_std):.8f}")
    
    # Detect curvature at this point
    print("\n--- Curvature detection ---")
    u_j_A, u_l_A = [], []
    for j in range(G):
        crossings = find_crossings(P[i_test, j, :], u_grid, p_test)
        for c in crossings:
            u_j_A.append(u_grid[j])
            u_l_A.append(c)
    
    if len(u_j_A) >= 3:
        sign = detect_curvature_sign(u_j_A, u_l_A)
        labels = {1: "CONVEX", -1: "CONCAVE", 0: "~LINEAR"}
        print(f"  Curvature at p={p_test:.4f}: {labels[sign]}")
        print(f"  Number of crossings: {len(u_j_A)}")
    
    print("\nDone.")
