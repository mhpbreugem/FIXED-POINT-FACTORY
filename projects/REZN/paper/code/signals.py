"""Information primitives.

Logistic Lambda, logit, signal densities f_v, sufficient statistic T*,
and the ex-ante weight w(u) = 1/2(prod f_1 + prod f_0) used by the
revelation regression and by ex-ante expectations.

All functions return float64 and are numba-compatible (no Python objects).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import DTYPE

LOG_2PI = math.log(2.0 * math.pi)


@njit(cache=True, fastmath=False)
def lam(z: float) -> float:
    """Logistic 1/(1+exp(-z)), numerically stable."""
    if z >= 0.0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


@njit(cache=True, fastmath=False)
def logit(p: float) -> float:
    """ln(p/(1-p)). Caller is responsible for p in (0,1)."""
    return math.log(p) - math.log(1.0 - p)


@njit(cache=True, fastmath=False)
def f_signal(u: float, v: int, tau: float) -> float:
    """Density f_v(u) of u_k = s_k - 1/2 under state v in {0,1}.

    Mean +1/2 if v=1, -1/2 if v=0; precision tau (variance 1/tau).
    """
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * d * d)


def t_star(u_grid: np.ndarray, tau_vec: np.ndarray, K: int) -> np.ndarray:
    """T* = sum_k tau_k u_k as a (G,)*K float64 array.

    Vectorised: builds the K-dimensional sum-of-axes table in one pass.
    """
    out = np.zeros((u_grid.size,) * K, dtype=DTYPE)
    shape_ones = [1] * K
    for k in range(K):
        sh = shape_ones.copy()
        sh[k] = u_grid.size
        out = out + (tau_vec[k] * u_grid).reshape(sh)
    return out


def weights(u_grid: np.ndarray, tau_vec: np.ndarray, K: int) -> np.ndarray:
    """Ex-ante signal weight w(u_1,...,u_K) = 1/2 (prod f_1 + prod f_0).

    Returned as a (G,)*K float64 array. Used by metrics and ex-ante
    expectations.
    """
    G = u_grid.size
    f1 = np.empty((K, G), dtype=DTYPE)
    f0 = np.empty((K, G), dtype=DTYPE)
    for k in range(K):
        s = math.sqrt(tau_vec[k] / (2.0 * math.pi))
        f1[k] = s * np.exp(-0.5 * tau_vec[k] * (u_grid - 0.5) ** 2)
        f0[k] = s * np.exp(-0.5 * tau_vec[k] * (u_grid + 0.5) ** 2)

    shape_ones = [1] * K
    prod1 = np.ones((G,) * K, dtype=DTYPE)
    prod0 = np.ones((G,) * K, dtype=DTYPE)
    for k in range(K):
        sh = shape_ones.copy()
        sh[k] = G
        prod1 = prod1 * f1[k].reshape(sh)
        prod0 = prod0 * f0[k].reshape(sh)
    return 0.5 * (prod1 + prod0)
