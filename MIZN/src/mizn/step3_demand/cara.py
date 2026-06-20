"""CARA-Gaussian demand: x = (E[v|info] - P) / (rho * Var[v|info])."""
from __future__ import annotations
import numpy as np

from ..config import Params


def demand_cara(mu: np.ndarray, P: np.ndarray, params: Params) -> np.ndarray:
    # In the gaussian step2 we computed mu using tau_1; the conditional
    # variance is v = 1/tau_1.  Recompute tau_1 from params + the same
    # tau_phi derivation as the analytic solution (the step2 step already
    # used that v).
    tau_phi = params.tau_u * (params.tau_eps / params.rho) ** 2
    tau_1 = params.tau_th + params.tau_eps + tau_phi
    v = 1.0 / tau_1
    return (mu - P) / (params.rho * v)
