"""Fixed-point solvers for P = Phi(P).

Two methods, both calling the same Phi closure:

  picard:    P_{n+1} = (1-alpha) P_n + alpha Phi(P_n)
  anderson:  type-II Anderson with history depth m

Both return (P_final, history) where history is a 1D float64 array of
||P - Phi(P)||_inf per iteration.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .config import DTYPE, SolverConfig
from .contour_K4 import residual_inf
from .demand import EPS_PRICE
from .symmetry import symmetrize


PhiCall = Callable[[np.ndarray], np.ndarray]


def _post_step(P_new: np.ndarray, do_symm: bool) -> np.ndarray:
    return symmetrize(P_new) if do_symm else P_new


def _clip_unit(P: np.ndarray) -> np.ndarray:
    """Clip into (EPS, 1-EPS); Anderson's LS update can overshoot."""
    return np.clip(P, EPS_PRICE, 1.0 - EPS_PRICE)


def picard(phi: PhiCall, P0: np.ndarray, cfg: SolverConfig,
           checkpoint: Optional[Callable[[int, np.ndarray], None]] = None
           ) -> tuple[np.ndarray, np.ndarray]:
    P = P0.astype(DTYPE, copy=True)
    hist = np.empty(cfg.max_iters, dtype=DTYPE)
    for n in range(cfg.max_iters):
        t0 = time.perf_counter()
        P_phi = _post_step(phi(P), cfg.symmetrize)
        r = residual_inf(P, P_phi)
        hist[n] = r
        P = (1.0 - cfg.damping) * P + cfg.damping * P_phi
        if cfg.verbose:
            print(f"[picard {n+1:3d}/{cfg.max_iters}] "
                  f"||P-Phi(P)||inf = {r:.3e}   "
                  f"({time.perf_counter() - t0:.2f}s)")
        if checkpoint is not None and cfg.checkpoint_every > 0 \
                and (n + 1) % cfg.checkpoint_every == 0:
            checkpoint(n + 1, P)
        if r < cfg.tol:
            return P, hist[: n + 1]
    return P, hist


def anderson(phi: PhiCall, P0: np.ndarray, cfg: SolverConfig,
             checkpoint: Optional[Callable[[int, np.ndarray], None]] = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """Type-II Anderson acceleration on the residual G(P) = Phi(P) - P.

    Standard formulation: maintain F = [Delta f_{n-m+1}, ..., Delta f_n]
    and X = [Delta x_{n-m+1}, ..., Delta x_n] (column-stacked flattened
    increments), solve gamma = F^+ f_n, update P = P + f - (X + F) gamma.
    """
    m = cfg.anderson_m
    P = P0.astype(DTYPE, copy=True).ravel()
    n_pts = P.size

    # ring buffers
    X_buf = np.zeros((n_pts, m), dtype=DTYPE)
    F_buf = np.zeros((n_pts, m), dtype=DTYPE)
    f_prev: Optional[np.ndarray] = None
    P_prev: Optional[np.ndarray] = None
    hist = np.empty(cfg.max_iters, dtype=DTYPE)

    for n in range(cfg.max_iters):
        t0 = time.perf_counter()
        P_arr = P.reshape(P0.shape)
        P_phi = _post_step(phi(P_arr), cfg.symmetrize).ravel()
        f = P_phi - P
        r = float(np.max(np.abs(f)))
        hist[n] = r
        if cfg.verbose:
            print(f"[anderson {n+1:3d}/{cfg.max_iters}] "
                  f"||P-Phi(P)||inf = {r:.3e}   "
                  f"({time.perf_counter() - t0:.2f}s)")
        if r < cfg.tol:
            if checkpoint is not None:
                checkpoint(n + 1, P.reshape(P0.shape))
            return P.reshape(P0.shape), hist[: n + 1]

        if f_prev is None:
            # first step: plain Picard
            P_next = P + f
        else:
            col = (n - 1) % m
            X_buf[:, col] = P - P_prev
            F_buf[:, col] = f - f_prev
            mk = min(n, m)
            X = X_buf[:, :mk]
            F = F_buf[:, :mk]
            # least squares F gamma = f
            gamma, *_ = np.linalg.lstsq(F, f, rcond=None)
            P_next = P + f - (X + F) @ gamma

        P_prev = P
        f_prev = f
        P = _clip_unit(P_next)

        if checkpoint is not None and cfg.checkpoint_every > 0 \
                and (n + 1) % cfg.checkpoint_every == 0:
            checkpoint(n + 1, P.reshape(P0.shape))

    return P.reshape(P0.shape), hist


def solve(phi: PhiCall, P0: np.ndarray, cfg: SolverConfig,
          checkpoint: Optional[Callable[[int, np.ndarray], None]] = None
          ) -> tuple[np.ndarray, np.ndarray]:
    if cfg.method == "picard":
        return picard(phi, P0, cfg, checkpoint)
    if cfg.method == "anderson":
        return anderson(phi, P0, cfg, checkpoint)
    raise ValueError(f"unknown solver method: {cfg.method}")
