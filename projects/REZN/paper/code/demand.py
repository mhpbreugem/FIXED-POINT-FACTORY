"""Demand functions and per-realisation market clearing.

Market clearing solves Sum_k x_k(mu_k, p) = 0 in p in (0,1). Excess demand
is strictly decreasing in p under both CRRA and CARA, so a guarded
bisection on [eps, 1-eps] is the right tool: bullet-proof, jit-friendly,
and quick (60 iterations brings p to ~1e-18 precision).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .signals import lam, logit


# -- demand functions ----------------------------------------------------

@njit(cache=True, fastmath=False)
def x_crra(mu: float, p: float, gamma: float, W: float) -> float:
    """CRRA demand x = W (R-1) / ((1-p) + R p), R = exp((logit mu - logit p)/gamma)."""
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0.0:
        e = math.exp(-z)
        return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = math.exp(z)
    return W * (e - 1.0) / ((1.0 - p) + p * e)


@njit(cache=True, fastmath=False)
def x_cara(mu: float, p: float, alpha: float) -> float:
    """CARA demand x = (logit mu - logit p) / alpha."""
    return (logit(mu) - logit(p)) / alpha


# -- aggregate excess demand ---------------------------------------------

@njit(cache=True, fastmath=False)
def excess_crra(mu_vec: np.ndarray, gamma_vec: np.ndarray,
                W_vec: np.ndarray, p: float) -> float:
    s = 0.0
    for k in range(mu_vec.size):
        s += x_crra(mu_vec[k], p, gamma_vec[k], W_vec[k])
    return s


@njit(cache=True, fastmath=False)
def excess_cara(mu_vec: np.ndarray, alpha_vec: np.ndarray, p: float) -> float:
    s = 0.0
    for k in range(mu_vec.size):
        s += x_cara(mu_vec[k], p, alpha_vec[k])
    return s


# -- market clearing -----------------------------------------------------

EPS_PRICE = 1.0e-12


@njit(cache=True, fastmath=False)
def clear_crra(mu_vec: np.ndarray, gamma_vec: np.ndarray,
               W_vec: np.ndarray) -> float:
    """Bisection for the unique p in (eps, 1-eps) with sum x_k = 0."""
    a = EPS_PRICE
    b = 1.0 - EPS_PRICE
    fa = excess_crra(mu_vec, gamma_vec, W_vec, a)
    fb = excess_crra(mu_vec, gamma_vec, W_vec, b)
    # excess decreases in p: fa > 0 > fb (under any non-degenerate posteriors)
    if fa <= 0.0:
        return a
    if fb >= 0.0:
        return b
    for _ in range(60):
        c = 0.5 * (a + b)
        fc = excess_crra(mu_vec, gamma_vec, W_vec, c)
        if fc >= 0.0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if (b - a) < 1.0e-14:
            break
    return 0.5 * (a + b)


@njit(cache=True, fastmath=False)
def clear_cara(mu_vec: np.ndarray, alpha_vec: np.ndarray) -> float:
    """Closed form: logit p = (sum logit mu_k / alpha_k) / (sum 1/alpha_k)."""
    num = 0.0
    den = 0.0
    for k in range(mu_vec.size):
        w = 1.0 / alpha_vec[k]
        num += w * logit(mu_vec[k])
        den += w
    return lam(num / den)
