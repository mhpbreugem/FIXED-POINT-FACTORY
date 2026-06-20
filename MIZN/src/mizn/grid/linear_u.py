"""Uniform grid in physical u ∈ [-umax, umax].

Trapezoidal weights; jacobian = 1 since nodes ARE u.
"""
from __future__ import annotations
import numpy as np

from .base import Grid


def build_linear_u(params) -> Grid:
    G = params.G
    u = np.linspace(-params.umax, params.umax, G)
    du = u[1] - u[0]
    w = np.full(G, du, dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    return Grid(kind='u', nodes=u.copy(), weights=w, u=u.copy(),
                 jacobian=np.ones_like(u), G=G)
