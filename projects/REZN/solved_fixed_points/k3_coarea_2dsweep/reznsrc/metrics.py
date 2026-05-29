"""Diagnostics for a converged price array P.

Two main quantities used in the paper:

  revelation_deficit(P, ...)  = 1 - R^2 of logit(P) on T*, weighted by w
  trade_volume(P, ...)        = E[|x_k|] under the ex-ante signal density

Plus per-realisation posteriors as a byproduct of the regression weights.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import DTYPE
from .signals import t_star, weights
from .demand import x_crra, x_cara


def _safe_logit(p: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def revelation_deficit(P: np.ndarray, u_grid: np.ndarray,
                       tau_vec: np.ndarray, K: int,
                       eps: float = 1.0e-4) -> float:
    """Weighted regression 1 - R^2 of logit(P) on T*.

    Filter: only points with eps < P < 1-eps contribute (avoid logit blowup).
    Weight: w(u) = 1/2 (prod f_1 + prod f_0).
    """
    Ts = t_star(u_grid, tau_vec, K)
    w = weights(u_grid, tau_vec, K)
    L = _safe_logit(P)

    mask = (P > eps) & (P < 1.0 - eps)
    w = np.where(mask, w, 0.0)
    Wsum = w.sum()
    if Wsum <= 0.0:
        return float("nan")

    L_mean = (w * L).sum() / Wsum
    T_mean = (w * Ts).sum() / Wsum
    var_L = (w * (L - L_mean) ** 2).sum() / Wsum
    var_T = (w * (Ts - T_mean) ** 2).sum() / Wsum
    cov_LT = (w * (L - L_mean) * (Ts - T_mean)).sum() / Wsum
    if var_L <= 0.0 or var_T <= 0.0:
        return float("nan")
    r2 = (cov_LT * cov_LT) / (var_L * var_T)
    return float(1.0 - r2)


def trade_volume(P: np.ndarray, u_grid: np.ndarray, tau_vec: np.ndarray,
                 gamma_vec: np.ndarray, W_vec: np.ndarray,
                 K: int, cara: bool) -> np.ndarray:
    """E[|x_k|] for k=0..K-1.

    For the no-learning benchmark, posteriors are mu_k = Lambda(tau_k u_k).
    For the converged REE, callers should pass the converged posteriors
    via posterior_volume() (below) rather than this function.
    """
    from .signals import lam
    G = u_grid.size
    w = weights(u_grid, tau_vec, K)
    out = np.zeros(K, dtype=DTYPE)
    # iterate in float64; for K=4 this is G^4 = 160k at G=20 (cheap)
    it = np.ndindex(*((G,) * K))
    for idx in it:
        p = P[idx]
        weight = w[idx]
        if weight <= 0.0:
            continue
        for k in range(K):
            mu_k = lam(tau_vec[k] * u_grid[idx[k]])
            if cara:
                xk = (np.log(mu_k / (1 - mu_k)) - np.log(p / (1 - p))) / gamma_vec[k]
            else:
                xk = x_crra(mu_k, p, gamma_vec[k], W_vec[k])
            out[k] += weight * abs(xk)
    out /= w.sum()
    return out


def posterior_volume(P: np.ndarray, posteriors: np.ndarray,
                     u_grid: np.ndarray, tau_vec: np.ndarray,
                     gamma_vec: np.ndarray, W_vec: np.ndarray,
                     K: int, cara: bool) -> np.ndarray:
    """E[|x_k|] given the converged posterior array posteriors[K, G,...,G]."""
    G = u_grid.size
    w = weights(u_grid, tau_vec, K)
    out = np.zeros(K, dtype=DTYPE)
    it = np.ndindex(*((G,) * K))
    for idx in it:
        p = P[idx]
        weight = w[idx]
        if weight <= 0.0:
            continue
        for k in range(K):
            mu_k = posteriors[(k,) + idx]
            if cara:
                xk = (np.log(mu_k / (1 - mu_k)) - np.log(p / (1 - p))) / gamma_vec[k]
            else:
                xk = x_crra(mu_k, p, gamma_vec[k], W_vec[k])
            out[k] += weight * abs(xk)
    out /= w.sum()
    return out


def summary(P: np.ndarray, u_grid: np.ndarray, tau_vec: np.ndarray,
            K: int) -> Tuple[float, float, float, float]:
    """Return (1-R^2, p_min, p_max, p_mean) for quick diagnostics."""
    deficit = revelation_deficit(P, u_grid, tau_vec, K)
    return (deficit, float(P.min()), float(P.max()), float(P.mean()))
