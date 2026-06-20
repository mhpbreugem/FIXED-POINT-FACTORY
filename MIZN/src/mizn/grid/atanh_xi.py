"""Compactified coordinate xi = tanh(u / T)  with  T = umax/atanh(0.9999).

Nodes are uniformly spaced in xi on (-0.9999, 0.9999); the physical u
nodes are u = T * arctanh(xi).  Jacobian du/dxi = T / (1 - xi^2).
"""
from __future__ import annotations
import numpy as np

from .base import Grid


def build_atanh_xi(params) -> Grid:
    G = params.G
    xi_max = 0.9999
    xi = np.linspace(-xi_max, xi_max, G)
    T = params.umax / np.arctanh(xi_max)
    u = T * np.arctanh(xi)
    jacobian = T / (1.0 - xi ** 2)
    # trapezoidal weights in xi multiplied by jacobian for integrals in u
    dxi = xi[1] - xi[0]
    w_xi = np.full(G, dxi, dtype=float)
    w_xi[0] *= 0.5; w_xi[-1] *= 0.5
    weights = w_xi * jacobian
    return Grid(kind='xi', nodes=xi, weights=weights, u=u,
                 jacobian=jacobian, G=G)
