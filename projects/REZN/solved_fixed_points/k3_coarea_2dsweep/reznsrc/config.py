"""Configuration dataclasses.

All floating-point arrays in this package are float64. The constant DTYPE
is the single source of truth; pass it to every np.zeros / np.empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

DTYPE = np.float64


@dataclass(frozen=True)
class Config:
    """Economic + grid parameters.

    Homogeneous-agent baseline. Heterogeneous (gamma_k, tau_k, alpha_k, W_k)
    is supported by the helpers below; pass per-agent vectors of length K.
    """

    K: int = 4
    G: int = 10
    gamma: float = 0.5          # CRRA risk aversion (gamma -> infinity is CARA)
    tau: float = 2.0            # signal precision
    W: float = 1.0              # initial wealth
    u_min: float = -4.0
    u_max: float = +4.0
    cara: bool = False          # if True, use CARA demand with alpha = gamma

    def u_grid(self) -> np.ndarray:
        return np.linspace(self.u_min, self.u_max, self.G, dtype=DTYPE)

    def gamma_vec(self) -> np.ndarray:
        return np.full(self.K, self.gamma, dtype=DTYPE)

    def tau_vec(self) -> np.ndarray:
        return np.full(self.K, self.tau, dtype=DTYPE)

    def W_vec(self) -> np.ndarray:
        return np.full(self.K, self.W, dtype=DTYPE)

    def shape(self) -> Tuple[int, ...]:
        return (self.G,) * self.K


@dataclass(frozen=True)
class SolverConfig:
    """Fixed-point solver parameters."""

    method: str = "anderson"    # {"picard", "anderson"}
    max_iters: int = 50
    tol: float = 1.0e-7
    damping: float = 0.3        # Picard relaxation alpha in [0,1]
    anderson_m: int = 8         # Anderson history depth
    symmetrize: bool = True     # average Phi(P) over S_K permutations each step
    checkpoint_every: int = 10  # iterations between npz checkpoints (0 = off)
    verbose: bool = True
