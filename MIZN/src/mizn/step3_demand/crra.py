"""Binary CRRA demand: x = (mu - P) / (gamma * mu * (1 - mu)).

`mu` here is interpreted as the posterior PROBABILITY of the high payoff
(value = 1), so it must live in [0, 1].  We clip to avoid division by
zero at the corners.
"""
from __future__ import annotations
import numpy as np

from ..config import Params


def demand_crra(mu: np.ndarray, P: np.ndarray, params: Params) -> np.ndarray:
    eps = 1e-9
    m = np.clip(mu, eps, 1.0 - eps)
    p = np.clip(P, eps, 1.0 - eps)
    return (m - p) / (params.gamma * m * (1.0 - m))
