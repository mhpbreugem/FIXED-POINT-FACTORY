"""Hellwig closed-form: regress P on (theta, u), recover (a, b, c),
extract phi = (P - a)/b, then form the posterior mean

    E_bar(s, phi) = (tau_th * theta_bar + tau_eps * s + tau_phi * phi) / tau_1,
    v             = 1 / tau_1.

We do this analytically by leveraging the fact that for any LINEAR P
the regression is exact and we know (a, b, c) from the OLS.  This is the
gold-test step2 — it commutes with the analytic linear REE.
"""
from __future__ import annotations
import numpy as np

from ..config import Params
from ..grid.base import Grid


def _ols_planar(P: np.ndarray, theta: np.ndarray, u: np.ndarray):
    """Fit P_ij = a + b*theta_i + d*u_j by ordinary least squares.

    Returns (a, b, d) where d = -c in the analytic notation.
    """
    T, U = np.meshgrid(theta, u, indexing='ij')
    A = np.column_stack([np.ones(P.size), T.ravel(), U.ravel()])
    coef, *_ = np.linalg.lstsq(A, P.ravel(), rcond=None)
    a, b, d = coef
    return a, b, d


def learn_gaussian(P: np.ndarray, grid: Grid, params: Params) -> np.ndarray:
    """Return mu(theta, u) of shape P.shape."""
    theta = grid.u
    u = grid.u
    a, b, d = _ols_planar(P, theta, u)
    c = -d
    # Inferred price-precision: phi = (P - a)/b is informationally equivalent
    # to theta - (c/b) * u; the variance of (c/b)u under u ~ N(u_bar, 1/tau_u)
    # gives tau_phi = tau_u * (b/c)^2 — same as the analytic expression.
    tau_phi = params.tau_u * (b / c) ** 2 if c != 0 else 0.0
    tau_1   = params.tau_th + params.tau_eps + tau_phi

    T, U = np.meshgrid(theta, u, indexing='ij')
    phi = (P - a) / b if b != 0 else np.zeros_like(P)
    s = T  # in the continuum, each gridpoint represents a trader whose s = theta
    mu = (params.tau_th * params.theta_bar
            + params.tau_eps * s
            + tau_phi * phi) / tau_1
    return mu
