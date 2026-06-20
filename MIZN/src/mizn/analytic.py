"""Closed-form CARA-Hellwig linear REE.

Continuum economy: theta ~ N(theta_bar, 1/tau_th),
                   s_i = theta + eps_i with eps ~ N(0, 1/tau_eps),
                   supply u ~ N(u_bar, 1/tau_u),
                   CARA risk rho.

Equilibrium price has the linear form  P = a + b * theta - c * u  with
    tau_phi = tau_u * (tau_eps / rho) ** 2          # price informativeness
    tau_1   = tau_th + tau_eps + tau_phi
    a       = tau_th * theta_bar / tau_1
    b       = (tau_eps + tau_phi) / tau_1
    c       = rho * (tau_eps + tau_phi) / (tau_1 * tau_eps)

For default params (tau_th=1, tau_eps=2, tau_u=1, rho=2): b=0.75, c=0.75.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import Params


@dataclass(frozen=True)
class LinearREE:
    a: float
    b: float
    c: float
    tau_phi: float

    def price(self, theta: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.a + self.b * theta - self.c * u


def linear_REE(p: Params) -> LinearREE:
    tau_phi = p.tau_u * (p.tau_eps / p.rho) ** 2
    tau_1   = p.tau_th + p.tau_eps + tau_phi
    a = p.tau_th * p.theta_bar / tau_1
    b = (p.tau_eps + tau_phi) / tau_1
    c = p.rho * (p.tau_eps + tau_phi) / (tau_1 * p.tau_eps)
    return LinearREE(a=a, b=b, c=c, tau_phi=tau_phi)
