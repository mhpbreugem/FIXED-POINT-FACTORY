"""CARA market clearing.

Excess demand   E(theta, u) = Z(theta, u) - u(j).  At the FP, E ≡ 0.
The one-shot exact update for CARA is

    P_new = P + (Z - u) * rho * v
          = mu - rho * v * u                 (since x = (mu - P)/(rho v))

We return P_new and the max-abs residual ‖F‖_∞ = ‖Z - u‖_∞.
"""
from __future__ import annotations
import numpy as np

from ..config import Params


def clear_cara(Z: np.ndarray, P: np.ndarray, params: Params):
    G = P.shape[0]
    u = np.linspace(-params.umax, params.umax, G)  # tied to step1's chart
    # broadcast u along axis 1 (the "supply" axis)
    U = u[np.newaxis, :]
    tau_phi = params.tau_u * (params.tau_eps / params.rho) ** 2
    tau_1 = params.tau_th + params.tau_eps + tau_phi
    v = 1.0 / tau_1
    excess = Z - U                     # shape (G, G)
    P_new = P + excess * (params.rho * v)
    resid = float(np.max(np.abs(excess)))
    return P_new, resid
