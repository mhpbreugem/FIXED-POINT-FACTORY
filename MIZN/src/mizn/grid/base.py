"""Grid dataclass — the only thing operators see."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    """1-D coordinate grid + quadrature weights + jacobian factors.

    The same `Grid` is used in each of the K axes (signals u_i); the loop
    builds K-fold tensor products on the fly.
    """
    kind:     str            # 'u' | 'xi' | 'zeta'
    nodes:    np.ndarray     # shape (G,) in NATIVE coords (e.g. xi, zeta)
    weights:  np.ndarray     # shape (G,) quadrature weights for integrals over the chart
    u:        np.ndarray     # shape (G,) physical u-coords (always provided)
    jacobian: np.ndarray     # shape (G,) du / dnative — useful for change-of-var integrals
    G:        int

    def __post_init__(self):
        assert self.nodes.shape == self.weights.shape == self.u.shape == self.jacobian.shape
        assert self.nodes.shape == (self.G,)
