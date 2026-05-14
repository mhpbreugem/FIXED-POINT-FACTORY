"""
ode_sweep.py — Parameter continuation sweep over gamma.

Implements predictor-corrector continuation:
  - Predictor: linear extrapolation from last two solutions
  - Corrector: Anderson-accelerated Picard iteration (float64) then
    mpmath Newton polish to target precision

solve_sweep(phi_f64, phi_mp, mp, load_ckpt, gamma_grid, anchor_idx,
            P_anchor_full, u_full, inner_lo, inner_hi,
            tau_vec, gamma_scalar, W_vec, kernel_h,
            mp_dps, target_eps, max_iter, anderson_m, verbose)

Returns dict with keys: gamma_grid, P_outputs, F_outputs
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Anderson mixing (m-step, in-place on flat arrays)
# ---------------------------------------------------------------------------

def _anderson_step(F_hist: list, P_hist: list, m: int) -> np.ndarray:
    """Given history lists of residuals F=phi(P)-P and iterates P,
    return next Anderson iterate."""
    k = len(F_hist)
    if k == 0:
        raise ValueError("empty history")
    if k == 1:
        return P_hist[-1] + F_hist[-1]   # plain Picard

    mk = min(k, m)
    # Least-squares: min ||sum theta_i F_i|| s.t. sum theta_i = 1
    F_mat = np.column_stack([F_hist[-mk:]])   # n × mk
    # Normal equations via QR
    ones = np.ones(mk)
    try:
        c, _, _, _ = np.linalg.lstsq(F_mat.T @ F_mat + 1e-14 * np.eye(mk),
                                      F_mat.T @ F_mat @ ones / (ones @ ones),
                                      rcond=None)
        c = c / c.sum() if abs(c.sum()) > 1e-12 else ones / mk
    except Exception:
        c = ones / mk
    P_stack = np.column_stack([P_hist[-mk:]])
    return (P_stack + F_mat) @ c


def anderson_solve(phi_fn: Callable, P0: np.ndarray,
                   tol: float = 1e-8, max_iter: int = 300,
                   m: int = 5, verbose: bool = False) -> tuple[np.ndarray, float]:
    """Solve P = phi(P) via Anderson acceleration. Returns (P, residual)."""
    P = P0.copy().ravel()
    F_hist, P_hist = [], []
    best_P, best_res = P.copy(), float("inf")
    shape = P0.shape

    for it in range(max_iter):
        Phi = phi_fn(P.reshape(shape)).ravel()
        F = Phi - P
        res = float(np.max(np.abs(F)))
        if res < best_res:
            best_res, best_P = res, P.copy()
        if verbose and it % 20 == 0:
            print(f"    anderson it={it:4d}  ||F||={res:.3e}", flush=True)
        if res < tol:
            break
        F_hist.append(F.copy())
        P_hist.append(P.copy())
        P = _anderson_step(F_hist, P_hist, m)
        # Clip to (0,1)
        P = np.clip(P, 1e-9, 1 - 1e-9)

    return best_P.reshape(shape), best_res


# ---------------------------------------------------------------------------
# mp Newton polish (one step)
# ---------------------------------------------------------------------------

def _mp_residual(mp, phi_mp_fn, P_full_np: np.ndarray) -> float:
    """Evaluate ||phi_mp(P) - P||_inf at current mp.dps."""
    P_mp = [[[ mp.mpf(str(P_full_np[i, j, l]))
               for l in range(P_full_np.shape[2])]
             for j in range(P_full_np.shape[1])]
            for i in range(P_full_np.shape[0])]
    Phi_mp = phi_mp_fn(P_mp)
    G = P_full_np.shape[0]
    max_diff = mp.mpf(0)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                d = abs(Phi_mp[i][j][l] - P_mp[i][j][l])
                if d > max_diff:
                    max_diff = d
    return float(max_diff), Phi_mp, P_mp


def mp_newton_solve(mp, phi_mp_fn, P_full_np: np.ndarray,
                    inner_lo: int, inner_hi: int,
                    target_eps, max_iter: int = 30,
                    verbose: bool = False) -> tuple[np.ndarray, float]:
    """Newton polish: P ← P + (phi(P) - P) damped by step-halving."""
    G = P_full_np.shape[0]
    P_mp = [[[ mp.mpf(str(P_full_np[i, j, l]))
               for l in range(G)] for j in range(G)] for i in range(G)]

    best_res = mp.mpf("inf")
    best_P = P_full_np.copy()

    for it in range(max_iter):
        Phi_mp = phi_mp_fn(P_mp)
        # Compute residual and update
        max_diff = mp.mpf(0)
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    d = abs(Phi_mp[i][j][l] - P_mp[i][j][l])
                    if d > max_diff:
                        max_diff = d
        res = max_diff
        if verbose:
            print(f"    mp_newton it={it}  ||F||={float(res):.3e}", flush=True)
        if res < best_res:
            best_res = res
            best_P = np.array([[[float(P_mp[i][j][l]) for l in range(G)]
                                 for j in range(G)] for i in range(G)])
        if res <= target_eps:
            break
        # Update: P ← phi(P) (simple Picard step at mp precision)
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    P_mp[i][j][l] = Phi_mp[i][j][l]

    return best_P, float(best_res)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def solve_sweep(
    phi_f64_fn: Callable,    # phi_f64_fn(gamma_scalar) → phi: P_full → P_full
    phi_mp_fn_factory: Callable,  # phi_mp_fn_factory(gamma_scalar) → phi_mp: P_mp → P_mp
    mp,
    gamma_grid: list,
    anchor_idx: int,
    P_anchor_full: np.ndarray,
    inner_lo: int,
    inner_hi: int,
    mp_dps: int,
    target_eps,
    f64_tol: float = 5e-7,
    f64_max_iter: int = 400,
    anderson_m: int = 5,
    mp_max_iter: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Sweep gamma_grid using predictor-corrector continuation.
    anchor_idx: index into gamma_grid where P_anchor_full is the solution.

    Returns: {"gamma_grid": [...], "P_outputs": [...], "F_f64": [...], "F_mp": [...]}
    """
    mp.dps = mp_dps
    n = len(gamma_grid)
    P_outputs   = [None] * n
    F_f64_out   = [float("nan")] * n
    F_mp_out    = [float("nan")] * n

    P_outputs[anchor_idx] = P_anchor_full.copy()

    # Sweep rightward (anchor → end) then leftward (anchor-1 → 0)
    for direction, indices in [
        ("→", range(anchor_idx, n)),
        ("←", range(anchor_idx - 1, -1, -1)),
    ]:
        P_prev = P_anchor_full.copy()
        for idx in indices:
            if P_outputs[idx] is not None:
                P_prev = P_outputs[idx]
                continue
            gamma = float(gamma_grid[idx])
            t0 = time.time()
            if verbose:
                print(f"\n  {direction} gamma={gamma:.4f} (idx={idx})", flush=True)

            # --- float64 corrector ---
            phi_f64 = phi_f64_fn(gamma)
            P_f64, res_f64 = anderson_solve(
                phi_f64, P_prev,
                tol=f64_tol, max_iter=f64_max_iter, m=anderson_m,
                verbose=verbose,
            )
            F_f64_out[idx] = res_f64
            if verbose:
                print(f"    f64 done  ||F||={res_f64:.3e}  t={time.time()-t0:.0f}s", flush=True)

            # --- mp Newton polish ---
            phi_mp = phi_mp_fn_factory(gamma)
            P_mp_out, res_mp = mp_newton_solve(
                mp, phi_mp, P_f64,
                inner_lo, inner_hi,
                target_eps=target_eps,
                max_iter=mp_max_iter,
                verbose=verbose,
            )
            F_mp_out[idx] = res_mp
            if verbose:
                print(f"    mp  done  ||F||={res_mp:.3e}  t={time.time()-t0:.0f}s", flush=True)

            P_outputs[idx] = P_mp_out
            P_prev = P_mp_out

    return {
        "gamma_grid": list(gamma_grid),
        "P_outputs":  P_outputs,
        "F_f64":      F_f64_out,
        "F_mp":       F_mp_out,
    }
