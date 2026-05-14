"""
ode_sweep_rk4.py — ODE continuation sweep: RK4 predictor + JFNK + Anderson corrector.

THEORY
======
The fixed-point branch P*(γ) satisfies F(P, γ) = φ(P, γ) − P = 0 for all γ.
Differentiating implicitly w.r.t. γ gives the tangent ODE:

    (I − J) · v = ∂φ/∂γ        v = dP*/dγ,  J = ∂φ/∂P

Integrating this ODE with RK4 gives a 4th-order accurate predictor.
Each γ step then needs only a short corrector to snap back onto the manifold,
rather than a full fixed-point solve from scratch.

JACOBIAN-FREE KRYLOV (JFNK)
============================
Forming J costs n = G³ ≈ 4913 φ-evaluations. Instead, GMRES only needs
matrix-vector products (I−J)·w, approximated by a single finite difference:

    (I − J)·w  ≈  w − (φ(P + δw) − φ(P)) / δ

where δ = 1.5e-8 · ‖P‖ / ‖w‖  (Knoll–Keyes formula, ~sqrt(ε_mach)).

Each GMRES iteration costs exactly 1 φ-evaluation.
Typical convergence: 20–60 iterations → 80–250 φ-evals per RK4 step,
vs. 300–500 for a full Anderson solve.

RHS ∂φ/∂γ is a central finite difference (2 φ-evals).

RK4 PREDICTOR
=============
One step (γ → γ + h) requires 4 GMRES solves (one per stage k1…k4):

    k1 = solve(P,           γ      )
    k2 = solve(P + h/2·k1,  γ + h/2)
    k3 = solve(P + h/2·k2,  γ + h/2)
    k4 = solve(P + h·k3,    γ + h  )
    P_pred = P + (h/6)(k1 + 2k2 + 2k3 + k4)

CORRECTOR
=========
RK4 accumulates O(h⁵) drift off the manifold.  A short run of Anderson
acceleration (max_iter=80 by default) snaps P back to a tight fixed point.
The corrector starts from P_pred rather than P_prev, so it needs far fewer
iterations than a cold-start solve.

Setting mp_max_iter=0 skips the mpmath polish entirely.  This is appropriate
when RK4 continuation is used: the smooth branch tracking means there is no
risk of jumping to a different solution, and float64 machine precision
(~1e-14) is sufficient for quantities like 1-R².

PUBLIC API
==========
solve_sweep_rk4(phi_f64_fn, phi_mp_fn_factory, mp, gamma_grid, anchor_idx,
                P_anchor_full, inner_lo, inner_hi, mp_dps, target_eps, ...)

Same return format as ode_sweep.solve_sweep:
    {"gamma_grid": [...], "P_outputs": [...], "F_f64": [...], "F_mp": [...]}
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from ode_sweep import anderson_solve, mp_newton_solve


# ─────────────────────────────────────────────────────────────────────────────
# GMRES helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jfnk_matvec(
    phi_fn: Callable,
    P_flat: np.ndarray,
    phi_P_flat: np.ndarray,
    shape: tuple,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute (I − J)·w using one finite-difference φ-evaluation.

    δ is chosen by the Knoll-Keyes formula:
        δ = √ε_mach · (1 + ‖P‖) / ‖w‖   with √ε_mach ≈ 1.5e-8
    This ensures φ(P + δw) stays within the float64 linear regime.
    """
    norm_w = float(np.linalg.norm(w))
    if norm_w < 1e-15:
        return w.copy()
    delta = 1.5e-8 * (1.0 + float(np.linalg.norm(P_flat))) / norm_w
    P_pert = np.clip((P_flat + delta * w).reshape(shape), 1e-12, 1.0 - 1e-12)
    Jw = (phi_fn(P_pert).ravel() - phi_P_flat) / delta
    return w - Jw


def _fd_dgamma(
    phi_factory: Callable,
    P: np.ndarray,
    gamma: float,
    eps: float,
) -> np.ndarray:
    """
    Compute ∂φ/∂γ via central finite difference (2 φ-evals).
        b = (φ(P, γ+ε) − φ(P, γ−ε)) / (2ε)
    """
    return (
        phi_factory(gamma + eps)(P).ravel()
        - phi_factory(gamma - eps)(P).ravel()
    ) / (2.0 * eps)


def _solve_tangent(
    phi_factory: Callable,
    P: np.ndarray,
    gamma: float,
    eps_gamma: float,
    gmres_tol: float,
    gmres_restart: int,
    gmres_maxiter: int,
) -> tuple[np.ndarray, int]:
    """
    Solve (I − J)·v = ∂φ/∂γ  for the tangent v = dP*/dγ.

    Builds a LinearOperator for (I−J) using JFNK matvecs, then calls GMRES.

    Returns
    -------
    v     : np.ndarray, shape = P.shape
    info  : int   0 = converged, >0 = not converged (still used as best guess)
    """
    shape = P.shape
    n = P.size
    phi_fn = phi_factory(gamma)
    P_flat = P.ravel()
    phi_P_flat = phi_fn(P).ravel()

    A = LinearOperator(
        (n, n),
        matvec=lambda w: _jfnk_matvec(phi_fn, P_flat, phi_P_flat, shape, w),
        dtype=np.float64,
    )
    b = _fd_dgamma(phi_factory, P, gamma, eps_gamma)

    # scipy >= 1.12 uses rtol; older versions use tol.  Try both.
    try:
        v, info = gmres(A, b, rtol=gmres_tol, restart=gmres_restart,
                        maxiter=gmres_maxiter)
    except TypeError:
        v, info = gmres(A, b, tol=gmres_tol, restart=gmres_restart,
                        maxiter=gmres_maxiter)

    return v.reshape(shape), info


# ─────────────────────────────────────────────────────────────────────────────
# RK4 predictor
# ─────────────────────────────────────────────────────────────────────────────

def _rk_predict(
    phi_factory: Callable,
    P: np.ndarray,
    gamma: float,
    gamma_next: float,
    eps_gamma: float,
    gmres_tol: float,
    gmres_restart: int,
    gmres_maxiter: int,
    verbose: bool,
    order: int = 4,
) -> tuple[np.ndarray, float]:
    """
    Advance P from gamma to gamma_next with a single RK step of the tangent ODE.

    order in {1, 2, 4}:
        1 = explicit Euler    (1 GMRES solve)
        2 = explicit midpoint (2 GMRES solves)
        4 = classic RK4       (4 GMRES solves)

    Returns (P_pred, res_pred) where res_pred is ‖φ(P_pred, γ_next) − P_pred‖∞ —
    the manifold drift of the predictor.
    """
    h = gamma_next - gamma

    def tangent(P_eval: np.ndarray, g_eval: float) -> np.ndarray:
        v, info = _solve_tangent(
            phi_factory, P_eval, g_eval,
            eps_gamma, gmres_tol, gmres_restart, gmres_maxiter,
        )
        if verbose and info != 0:
            print(f"      GMRES unconverged (info={info}) at γ={g_eval:.4f}",
                  flush=True)
        return v

    clip = lambda X: np.clip(X, 1e-12, 1.0 - 1e-12)

    if order == 1:
        k1 = tangent(P, gamma)
        P_pred = P + h * k1
    elif order == 2:
        k1 = tangent(P, gamma)
        k2 = tangent(clip(P + 0.5*h*k1), gamma + 0.5*h)
        P_pred = P + h * k2
    elif order == 4:
        k1 = tangent(P,                       gamma)
        k2 = tangent(clip(P + 0.5*h*k1),      gamma + 0.5*h)
        k3 = tangent(clip(P + 0.5*h*k2),      gamma + 0.5*h)
        k4 = tangent(clip(P + h*k3),          gamma_next)
        P_pred = P + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    else:
        raise ValueError(f"order must be 1, 2, or 4 (got {order})")

    P_pred = clip(P_pred)

    # Measure drift off the manifold
    phi_check = phi_factory(gamma_next)
    res_pred = float(np.max(np.abs(phi_check(P_pred) - P_pred)))
    return P_pred, res_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def solve_sweep_rk4(
    phi_f64_fn: Callable,
    phi_mp_fn_factory: Callable,
    mp,
    gamma_grid: list,
    anchor_idx: int,
    P_anchor_full: np.ndarray,
    inner_lo: int,
    inner_hi: int,
    mp_dps: int,
    target_eps,
    # RK / GMRES
    order: int = 4,
    eps_gamma: float = 1e-5,
    gmres_tol: float = 1e-5,
    gmres_restart: int = 50,
    gmres_maxiter: int = 5,
    # Anderson corrector
    f64_tol: float = 5e-7,
    corrector_max_iter: int = 80,
    anderson_m: int = 5,
    # mpmath polish — set mp_max_iter=0 to skip (f64-only mode)
    mp_max_iter: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Sweep gamma_grid using RK4 ODE predictor + JFNK + Anderson corrector.

    Set mp_max_iter=0 to skip mpmath polish entirely and run in float64-only
    mode.  This is appropriate for RK4 continuation sweeps where the smooth
    branch tracking removes the risk of solution-jumping, and machine
    precision (~1e-14) is sufficient for 1-R² and similar diagnostics.

    Parameters
    ----------
    phi_f64_fn          phi_f64_fn(gamma) → phi_fn : P_full → P_full  (float64)
    phi_mp_fn_factory   phi_mp_fn_factory(gamma) → phi_mp_fn  (mpmath)
    mp                  mpmath module (mp.dps will be set to mp_dps)
    gamma_grid          list of γ values to sweep
    anchor_idx          index of the known solution in gamma_grid
    P_anchor_full       converged P at gamma_grid[anchor_idx]
    inner_lo, inner_hi  slice bounds for the inner grid
    mp_dps              mpmath decimal places (50 / 100 / 200)
    target_eps          mpmath convergence target (1e-40 / 1e-80 / 1e-150)
    eps_gamma           step size for ∂φ/∂γ finite difference  (default 1e-5)
    gmres_tol           GMRES relative tolerance                (default 1e-5)
    gmres_restart       GMRES restart (Krylov subspace dim)     (default 50)
    gmres_maxiter       GMRES outer restarts                    (default 5)
    corrector_max_iter  Anderson corrector max iterations       (default 80)
    anderson_m          Anderson history depth                  (default 5)
    f64_tol             corrector convergence target            (default 5e-7)
    mp_max_iter         mpmath Picard iterations; 0 = skip      (default 20)
    verbose             print progress                          (default True)

    Returns
    -------
    dict with keys: gamma_grid, P_outputs, F_f64, F_mp
    """
    mp.dps = mp_dps
    n = len(gamma_grid)
    P_outputs = [None] * n
    F_f64_out = [float("nan")] * n
    F_mp_out  = [float("nan")] * n

    P_outputs[anchor_idx] = P_anchor_full.copy()

    for direction, indices in [
        ("→", range(anchor_idx, n)),
        ("←", range(anchor_idx - 1, -1, -1)),
    ]:
        P_prev = P_anchor_full.copy()
        g_prev = float(gamma_grid[anchor_idx])

        for idx in indices:
            if P_outputs[idx] is not None:
                P_prev = P_outputs[idx]
                g_prev = float(gamma_grid[idx])
                continue

            gamma = float(gamma_grid[idx])
            t0 = time.time()
            if verbose:
                print(f"\n  {direction} γ={gamma:.4f} (idx={idx})", flush=True)

            # ── 1. RK predictor (order 1/2/4) ───────────────────────────────────────
            res_pred = float("nan")
            try:
                P_pred, res_pred = _rk_predict(
                    phi_f64_fn, P_prev, g_prev, gamma,
                    eps_gamma, gmres_tol, gmres_restart, gmres_maxiter,
                    verbose, order=order,
                )
                if verbose:
                    print(f"    RK{order} pred   ‖F‖={res_pred:.3e}  "
                          f"t={time.time()-t0:.0f}s", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"    RK{order} failed ({exc}), using P_prev", flush=True)
                P_pred = P_prev.copy()

            # ── 2. Anderson corrector (skip when corrector_max_iter=0) ──────────────
            if corrector_max_iter > 0:
                phi_fn = phi_f64_fn(gamma)
                P_corr, res_f64 = anderson_solve(
                    phi_fn, P_pred,
                    tol=f64_tol,
                    max_iter=corrector_max_iter,
                    m=anderson_m,
                    verbose=verbose,
                )
                F_f64_out[idx] = res_f64
                if verbose:
                    print(f"    corrector  ‖F‖={res_f64:.3e}  "
                          f"t={time.time()-t0:.0f}s", flush=True)
            else:
                # Pure ODE mode: accept RK4 prediction as-is
                P_corr = P_pred
                F_f64_out[idx] = res_pred
                if verbose:
                    print(f"    (corrector skipped — pure ODE mode)", flush=True)

            # ── 3. mpmath polish (skipped when mp_max_iter=0) ───────────────────────
            if mp_max_iter > 0:
                phi_mp = phi_mp_fn_factory(gamma)
                P_out, res_mp = mp_newton_solve(
                    mp, phi_mp, P_corr,
                    inner_lo, inner_hi,
                    target_eps=target_eps,
                    max_iter=mp_max_iter,
                    verbose=verbose,
                )
                F_mp_out[idx] = res_mp
                if verbose:
                    print(f"    mp polish  ‖F‖={res_mp:.3e}  "
                          f"t={time.time()-t0:.0f}s", flush=True)
            else:
                P_out = P_corr

            P_outputs[idx] = P_out
            P_prev = P_out
            g_prev = gamma

    return {
        "gamma_grid": list(gamma_grid),
        "P_outputs":  P_outputs,
        "F_f64":      F_f64_out,
        "F_mp":       F_mp_out,
    }
