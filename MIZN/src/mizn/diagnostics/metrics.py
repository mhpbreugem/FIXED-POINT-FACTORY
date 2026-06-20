"""Standard metrics computed from a fixed-point P field."""
from __future__ import annotations
import numpy as np

from ..config import Params
from ..analytic import linear_REE


def compute_metrics(P: np.ndarray, params: Params) -> dict:
    G = params.G
    u = np.linspace(-params.umax, params.umax, G)
    T, U = np.meshgrid(u, u, indexing='ij')
    # slope of P on theta at the centre (informativeness)
    slope_T = float((P[G // 2 + 1, G // 2] - P[G // 2 - 1, G // 2])
                      / (u[G // 2 + 1] - u[G // 2 - 1]))
    # R^2 of P regressed on (theta, u)
    A = np.column_stack([np.ones(P.size), T.ravel(), U.ravel()])
    coef, *_ = np.linalg.lstsq(A, P.ravel(), rcond=None)
    Phat = A @ coef
    ss_res = float(np.sum((P.ravel() - Phat) ** 2))
    ss_tot = float(np.sum((P.ravel() - P.mean()) ** 2))
    deficit = 1.0 - (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    # d_FR — distance from full-revealing (FR price = theta)
    P_FR = T
    d_FR = float(np.sqrt(np.mean((P - P_FR) ** 2)))
    # check vs analytic if CARA
    if params.model == 'cara':
        r = linear_REE(params)
        P_ana = r.price(T, U)
        d_analytic = float(np.max(np.abs(P - P_ana)))
    else:
        d_analytic = float('nan')
    return dict(slope_T=slope_T, deficit=deficit, d_FR=d_FR,
                  d_analytic=d_analytic)
