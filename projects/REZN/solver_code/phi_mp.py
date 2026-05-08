"""phi_mp.py — mpmath implementation of phi_K3_halo_smooth.

Mirrors the numba kernel in code/contour_K3_halo.py but uses arbitrary-
precision arithmetic via mpmath.  Intended for a final polishing phase:
warm-start from a float64 fixed-point (F~1e-11), then iterate in mpmath
until F < mp_picard_tol (e.g. 1e-50).

Public API
----------
phi_picard_mp(P_inner_np, halo_np, u_full_np, inner_lo, inner_hi,
              tau_vec_np, gamma_vec_np, W_vec_np, kernel_h,
              dps, tol, max_iters, reporter=None)
    -> (P_inner_final_np, F_inf_final, n_iters)
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# mpmath primitives (all receive mp as first arg so dps is locally set)
# ---------------------------------------------------------------------------

def _f_signal_mp(mp, u, v: int, tau):
    """Gaussian signal density f_v(u), v in {0, 1}."""
    mean = mp.mpf("0.5") if v == 1 else mp.mpf("-0.5")
    tau_ = mp.mpf(tau)
    coeff = mp.sqrt(tau_ / (2 * mp.pi))
    return coeff * mp.exp(-tau_ / 2 * (u - mean) ** 2)


def _logit_mp(mp, p):
    return mp.log(p) - mp.log(1 - p)


def _lam_mp(mp, z):
    return mp.mpf(1) / (1 + mp.exp(-z))


def _x_crra_mp(mp, mu, p, gamma, W):
    """CRRA demand x = W*(R-1)/((1-p)*e^{-z}+p), z=(logit mu - logit p)/gamma."""
    z = (_logit_mp(mp, mu) - _logit_mp(mp, p)) / gamma
    e = mp.exp(z)
    return W * (e - 1) / ((1 - p) + p * e)


def _clear_crra_mp(mp, mu_vec, gamma_vec, W_vec, eps=None):
    """Bisection for price p* in (eps, 1-eps) with sum x_k = 0."""
    if eps is None:
        eps = mp.mpf(10) ** (-int(mp.dps * 0.8))
    a = eps
    b = 1 - eps

    def excess(p):
        return sum(_x_crra_mp(mp, mu_vec[k], p, gamma_vec[k], W_vec[k])
                   for k in range(len(mu_vec)))

    fa = excess(a)
    if fa <= 0:
        return a
    fb = excess(b)
    if fb >= 0:
        return b

    # Number of bisection steps to reach tol = 10^{-dps}
    n_steps = int(mp.dps * 3.33) + 10  # log2(10^dps) ≈ 3.32 * dps
    for _ in range(n_steps):
        c = (a + b) / 2
        fc = excess(c)
        if fc >= 0:
            a = c
        else:
            b = c
    return (a + b) / 2


# ---------------------------------------------------------------------------
# Agent evidence (Gaussian-kernel Bayes integral over a 2-D slice)
# ---------------------------------------------------------------------------

def _agent_evidence_mp(mp, P_slice_mp, p_target, u_full_mp,
                       f0_u, f1_u,  # precomputed signal values per agent precision
                       tau_o0, tau_o1, kernel_h):
    """Return (A0, A1): kernel-weighted sums over G_full x G_full slice.

    P_slice_mp: 2-D list of mp.mpf, shape (G_full, G_full)
    f0_u, f1_u: lists of length G_full, precomputed f_v(u_full[i]) for
                the two off-axes (first index = axis-0 precision, second = axis-1).
    kernel_h: mp.mpf bandwidth
    """
    inv_2h2 = mp.mpf(1) / (2 * kernel_h * kernel_h)
    A0 = mp.mpf(0)
    A1 = mp.mpf(0)
    G = len(u_full_mp)
    f0_a_list, f0_b_list = f0_u
    f1_a_list, f1_b_list = f1_u
    for ia in range(G):
        f0a = f0_a_list[ia]
        f1a = f1_a_list[ia]
        row = P_slice_mp[ia]
        for ib in range(G):
            diff = row[ib] - p_target
            w = mp.exp(-diff * diff * inv_2h2)
            A0 += w * f0a * f0_b_list[ib]
            A1 += w * f1a * f1_b_list[ib]
    return A0, A1


# ---------------------------------------------------------------------------
# Full phi evaluation in mpmath
# ---------------------------------------------------------------------------

def phi_K3_smooth_mp(mp, P_full_mp, u_full_mp, inner_lo, inner_hi,
                     tau_vec_mp, gamma_vec_mp, W_vec_mp, kernel_h_mp):
    """Compute one application of phi_K3_halo_smooth in mpmath.

    P_full_mp : 3-D list of lists of lists of mp.mpf (G_full × G_full × G_full)
    Returns a new P_full_mp (same structure, halo unchanged).
    """
    G = len(u_full_mp)

    # Precompute signal densities for each tau and each u in the grid
    # f0[k][i] = f_0(u_full[i]; tau_k),  f1[k][i] = f_1(u_full[i]; tau_k)
    f0 = [[_f_signal_mp(mp, u_full_mp[i], 0, tau_vec_mp[k]) for i in range(G)]
          for k in range(3)]
    f1 = [[_f_signal_mp(mp, u_full_mp[i], 1, tau_vec_mp[k]) for i in range(G)]
          for k in range(3)]

    eps_p = mp.mpf(10) ** (-int(mp.dps * 0.8))

    # Copy halo unchanged; fill inner with new values
    P_new = [[[ P_full_mp[i][j][l] for l in range(G)] for j in range(G)]
             for i in range(G)]

    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full_mp[i][j][l]

                # Agent 0: slice P_full[i, :, :], tau_o0=tau[1], tau_o1=tau[2]
                slice0 = [P_full_mp[i][a] for a in range(G)]  # slice0[a][b]
                A0_0, A1_0 = _agent_evidence_mp(
                    mp, slice0, p, u_full_mp,
                    f0_u=(f0[1], f0[2]), f1_u=(f1[1], f1[2]),
                    tau_o0=tau_vec_mp[1], tau_o1=tau_vec_mp[2],
                    kernel_h=kernel_h_mp,
                )
                denom0 = f0[0][i] * A0_0 + f1[0][i] * A1_0
                mu0 = (f1[0][i] * A1_0 / denom0) if denom0 > 0 else mp.mpf("0.5")
                mu0 = max(eps_p, min(1 - eps_p, mu0))

                # Agent 1: slice P_full[:, j, :], tau_o0=tau[0], tau_o1=tau[2]
                slice1 = [[P_full_mp[a][j][b] for b in range(G)] for a in range(G)]
                A0_1, A1_1 = _agent_evidence_mp(
                    mp, slice1, p, u_full_mp,
                    f0_u=(f0[0], f0[2]), f1_u=(f1[0], f1[2]),
                    tau_o0=tau_vec_mp[0], tau_o1=tau_vec_mp[2],
                    kernel_h=kernel_h_mp,
                )
                denom1 = f0[1][j] * A0_1 + f1[1][j] * A1_1
                mu1 = (f1[1][j] * A1_1 / denom1) if denom1 > 0 else mp.mpf("0.5")
                mu1 = max(eps_p, min(1 - eps_p, mu1))

                # Agent 2: slice P_full[:, :, l], tau_o0=tau[0], tau_o1=tau[1]
                slice2 = [[P_full_mp[a][b][l] for b in range(G)] for a in range(G)]
                A0_2, A1_2 = _agent_evidence_mp(
                    mp, slice2, p, u_full_mp,
                    f0_u=(f0[0], f0[1]), f1_u=(f1[0], f1[1]),
                    tau_o0=tau_vec_mp[0], tau_o1=tau_vec_mp[1],
                    kernel_h=kernel_h_mp,
                )
                denom2 = f0[2][l] * A0_2 + f1[2][l] * A1_2
                mu2 = (f1[2][l] * A1_2 / denom2) if denom2 > 0 else mp.mpf("0.5")
                mu2 = max(eps_p, min(1 - eps_p, mu2))

                mu_vec = [mu0, mu1, mu2]
                p_star = _clear_crra_mp(mp, mu_vec, gamma_vec_mp, W_vec_mp, eps=eps_p)
                P_new[i][j][l] = p_star

    return P_new


# ---------------------------------------------------------------------------
# Helpers: convert between numpy and mpmath nested lists
# ---------------------------------------------------------------------------

def np_to_mp(mp, arr: np.ndarray):
    """Convert 3-D numpy array to nested list of mp.mpf."""
    if arr.ndim == 1:
        return [mp.mpf(str(x)) for x in arr]
    return [[[ mp.mpf(str(arr[i, j, l]))
               for l in range(arr.shape[2])]
             for j in range(arr.shape[1])]
            for i in range(arr.shape[0])]


def mp_to_np(P_mp) -> np.ndarray:
    """Convert 3-D nested list of mp.mpf back to float64 numpy."""
    G = len(P_mp)
    out = np.zeros((G, G, G), dtype=np.float64)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                out[i, j, l] = float(P_mp[i][j][l])
    return out


def extract_inner_mp(P_full_mp, inner_lo, inner_hi):
    """Extract the inner block as a flat list of mp.mpf (for F_inf computation)."""
    vals = []
    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                vals.append(P_full_mp[i][j][l])
    return vals


def f_inf_mp(mp, P_new_mp, P_old_mp, inner_lo, inner_hi):
    """||phi(P) - P||_inf over inner block."""
    F = mp.mpf(0)
    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                d = abs(P_new_mp[i][j][l] - P_old_mp[i][j][l])
                if d > F:
                    F = d
    return F


# ---------------------------------------------------------------------------
# Main polishing loop
# ---------------------------------------------------------------------------

def phi_picard_mp(
    P_inner_np: np.ndarray,
    halo_np: np.ndarray,
    u_full_np: np.ndarray,
    inner_lo: int,
    inner_hi: int,
    tau_vec_np: np.ndarray,
    gamma_vec_np: np.ndarray,
    W_vec_np: np.ndarray,
    kernel_h: float,
    dps: int = 100,
    tol_str: str = "1e-50",
    max_iters: int = 2000,
    alpha: float = 0.5,
    reporter: Any = None,
):
    """Run pure-Picard in mpmath starting from float64 warm-start.

    Parameters
    ----------
    P_inner_np : float64 inner block (G_inner^3)
    halo_np    : float64 full halo (G_full^3); inner cells are overwritten
    alpha      : damping coefficient (1.0 = no damping, 0.01 = heavy damping)
    tol_str    : string like "1e-50" for the mpmath tolerance
    reporter   : optional ProgressReporter to call .update(iter, ftol)

    Returns
    -------
    P_inner_final : float64 numpy array
    F_inf_final   : float (mpmath F_inf cast to float)
    n_iters       : int
    """
    try:
        import mpmath as _mp
    except ImportError:
        raise ImportError("mpmath is required for high-precision polishing. "
                          "Install with: pip install mpmath")

    _mp.mp.dps = dps + 10  # extra guard digits

    tol = _mp.mpf(tol_str)
    alpha_mp = _mp.mpf(str(alpha))
    one_minus_alpha = _mp.mpf(1) - alpha_mp

    # Build full P_full_mp from halo + P_inner
    G_full = halo_np.shape[0]
    P_full_np = halo_np.copy()
    P_full_np[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi] = P_inner_np

    print(f"[phi_mp] dps={dps} tol={tol_str} alpha={alpha} max_iters={max_iters}", flush=True)
    print(f"[phi_mp] G_full={G_full} inner=[{inner_lo},{inner_hi}] "
          f"inner_cells={(inner_hi-inner_lo)**3}", flush=True)

    P_full_mp = np_to_mp(_mp.mp, P_full_np)
    u_full_mp = [_mp.mpf(str(x)) for x in u_full_np]
    tau_mp    = [_mp.mpf(str(x)) for x in tau_vec_np]
    gamma_mp  = [_mp.mpf(str(x)) for x in gamma_vec_np]
    W_mp      = [_mp.mpf(str(x)) for x in W_vec_np]
    kernel_mp = _mp.mpf(str(kernel_h))

    t0 = time.perf_counter()
    F_inf = _mp.mpf("inf")
    n_iters = 0

    for it in range(max_iters):
        P_new_mp = phi_K3_smooth_mp(
            _mp.mp, P_full_mp, u_full_mp, inner_lo, inner_hi,
            tau_mp, gamma_mp, W_mp, kernel_mp,
        )
        F_inf = f_inf_mp(_mp.mp, P_new_mp, P_full_mp, inner_lo, inner_hi)
        n_iters = it + 1

        # Damped update (inner only; halo fixed)
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    P_full_mp[i][j][l] = (one_minus_alpha * P_full_mp[i][j][l]
                                          + alpha_mp * P_new_mp[i][j][l])

        F_float = float(F_inf)
        elapsed = time.perf_counter() - t0
        print(f"[phi_mp] iter={n_iters:5d}  F={F_float:.4e}  t={elapsed:.0f}s",
              flush=True)
        if reporter is not None:
            reporter.update(iter=n_iters, ftol=F_float)

        if F_inf < tol:
            print(f"[phi_mp] converged at iter={n_iters}  F={F_float:.4e}", flush=True)
            break

    # Extract final inner block as float64
    G_inner = inner_hi - inner_lo
    P_inner_final = np.zeros((G_inner, G_inner, G_inner), dtype=np.float64)
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_inner_final[i, j, l] = float(
                    P_full_mp[inner_lo + i][inner_lo + j][inner_lo + l]
                )

    return P_inner_final, float(F_inf), n_iters
