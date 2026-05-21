"""Halo construction rules for the staggered scheme.

Two rules supported:

  no_learning_halo(u_full, tau_vec, gamma_vec, W_vec)
      Halo cells receive the analytic no-learning equilibrium price for
      that hypothetical realisation. Static across stages 1..end.

  extrapolate_inner_halo(P_full_prev, inner_lo, inner_hi)
      Halo cells receive a linear extrapolation in logit(p) of the
      previous stage's inner solution along each axis. The inner block
      itself is left intact.

The driver may also blend: halo = (1-w)*extrapolate + w*no_learning,
where w(u) is a smooth weight that goes to 1 deep in the tail and 0
near the inner boundary. We default to a hard "extrapolate near, no-
learning far" rule at this implementation stage.
"""

from __future__ import annotations

import numpy as np

from .config import DTYPE
from .contour_K4_halo import init_no_learning_halo
from .demand import EPS_PRICE


def no_learning_halo(u_full: np.ndarray, tau_vec: np.ndarray,
                     gamma_vec: np.ndarray, W_vec: np.ndarray
                     ) -> np.ndarray:
    """Full padded-grid P from the analytic no-learning equilibrium."""
    return init_no_learning_halo(u_full, tau_vec, gamma_vec, W_vec)


def replace_inner(P_full_halo: np.ndarray, P_inner: np.ndarray,
                  inner_lo: int, inner_hi: int) -> np.ndarray:
    """Return P_full = halo on outside, inner on inside.

    Works for any number of axes K = P_full_halo.ndim.
    """
    K = P_full_halo.ndim
    out = P_full_halo.copy()
    sl = (slice(inner_lo, inner_hi),) * K
    out[sl] = P_inner
    return out


def extract_inner(P_full: np.ndarray, inner_lo: int, inner_hi: int
                  ) -> np.ndarray:
    """Return the inner block. Works for any K = P_full.ndim."""
    K = P_full.ndim
    sl = (slice(inner_lo, inner_hi),) * K
    return P_full[sl].copy()


def _safe_logit(p: np.ndarray, eps: float = EPS_PRICE) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def extrapolated_halo(P_full_prev: np.ndarray, inner_lo: int,
                      inner_hi: int) -> np.ndarray:
    """Linear-in-logit extrapolation of the inner block into the halo,
    one axis at a time. Works for any K = P_full_prev.ndim.

    For each halo index `q` outside [inner_lo, inner_hi-1] in some axis,
    we replace P_full[..., q, ...] with the linear extension of
    logit(P_full[..., inner_lo .. inner_hi-1, ...]) along that axis.

    Cells in the corner (multiple halo coordinates) get the cumulative
    extrapolation: we apply the rule axis 0 first, then axis 1 on the
    intermediate result, etc.
    """
    K = P_full_prev.ndim
    out = P_full_prev.copy()
    L = _safe_logit(out)
    G_full = L.shape[0]

    for axis in range(K):
        lo_idx = inner_lo
        lo_next = inner_lo + 1
        hi_idx = inner_hi - 1
        hi_next = inner_hi - 2

        before = (slice(None),) * axis
        after = (slice(None),) * (K - 1 - axis)

        for q in range(0, inner_lo):
            slope_idx = before + (lo_idx,) + after
            slope_next = before + (lo_next,) + after
            target = before + (q,) + after
            slope = L[slope_idx] - L[slope_next]
            L[target] = L[slope_idx] + slope * (lo_idx - q)

        for q in range(inner_hi, G_full):
            slope_idx = before + (hi_idx,) + after
            slope_next = before + (hi_next,) + after
            target = before + (q,) + after
            slope = L[slope_idx] - L[slope_next]
            L[target] = L[slope_idx] + slope * (q - hi_idx)

    out = _logistic(L)
    return np.clip(out, EPS_PRICE, 1.0 - EPS_PRICE)


def blended_halo(extrapolated: np.ndarray, no_learning: np.ndarray,
                 u_full: np.ndarray, inner_lo: int, inner_hi: int,
                 transition_width: float = 1.0) -> np.ndarray:
    """Smooth blend in the halo region between extrapolation (near
    boundary) and no-learning (deep tail). Works for any K.
    """
    K = extrapolated.ndim
    G_full = u_full.size
    u_lo = u_full[inner_lo]
    u_hi = u_full[inner_hi - 1]

    d = np.zeros((G_full,), dtype=DTYPE)
    for q in range(G_full):
        u = u_full[q]
        if u < u_lo:
            d[q] = u_lo - u
        elif u > u_hi:
            d[q] = u - u_hi
        else:
            d[q] = 0.0

    D = np.zeros((G_full,) * K, dtype=DTYPE)
    for axis in range(K):
        shape = [1] * K
        shape[axis] = G_full
        D = np.maximum(D, d.reshape(shape))

    w = 1.0 / (1.0 + np.exp(-(D - transition_width) / (transition_width / 2.0)))
    return (1.0 - w) * extrapolated + w * no_learning
