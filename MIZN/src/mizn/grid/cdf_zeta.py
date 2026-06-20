"""CDF-based coordinate zeta = F_bar(u) where F_bar is the marginal CDF
of u (Gaussian with mean u_bar, precision tau_u).

Uniform nodes in zeta cluster physical u where the prior is concentrated
— good for Bayesian integrals.  Inverse F_bar^{-1} via scipy.norm.ppf.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from .base import Grid


def build_cdf_zeta(params) -> Grid:
    G = params.G
    # avoid endpoints exactly 0 and 1
    zeta_max = 0.9999
    zeta = np.linspace(1.0 - zeta_max, zeta_max, G)
    sigma_u = 1.0 / np.sqrt(params.tau_u)
    u = norm.ppf(zeta, loc=params.u_bar, scale=sigma_u)
    # clip u to params.umax box for compatibility with operators that
    # expect a finite chart
    u = np.clip(u, -params.umax, params.umax)
    # jacobian du/dzeta = 1 / phi(u) where phi is the pdf
    pdf = norm.pdf(u, loc=params.u_bar, scale=sigma_u)
    jacobian = 1.0 / np.maximum(pdf, 1e-300)
    dzeta = zeta[1] - zeta[0]
    w_z = np.full(G, dzeta, dtype=float)
    w_z[0] *= 0.5; w_z[-1] *= 0.5
    weights = w_z * jacobian
    return Grid(kind='zeta', nodes=zeta, weights=weights, u=u,
                 jacobian=jacobian, G=G)
