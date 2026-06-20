"""Step 1: initial conjecture for the price field P(theta, u).

For CARA the natural warm-start IS the analytic linear REE — it's already
the fixed point, so the loop terminates in one Picard step.  Useful as a
gold test for the scaffolding.  For CRRA the same initial seed lands in
the PR basin (close-to-FR), which is good enough for warm-up.
"""
from __future__ import annotations
import numpy as np

from .config import Params
from .analytic import linear_REE
from .grid.base import Grid


def conjecture(grid: Grid, params: Params) -> np.ndarray:
    """Return P with shape (G, G).  Axis 0 = theta, axis 1 = u.

    We use u-grid as proxy for both theta and u axes (same physical chart
    in the Hellwig setup; agent signals live on theta-axis).  This matches
    the prior session's convention.
    """
    G = grid.G
    theta = grid.u
    u = grid.u
    T, U = np.meshgrid(theta, u, indexing='ij')
    r = linear_REE(params)
    return r.price(T, U)
