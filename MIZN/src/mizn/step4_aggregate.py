"""Step 4: aggregate demand Z(theta, u).

In the continuum-of-traders Hellwig setup, the cross-sectional average of
each trader's signal equals theta exactly (LLN), so the aggregate demand
function reduces to the demand of the "representative trader" at s=theta.
That makes Z(theta, u) = x(theta, u) as arrays — pass-through.
"""
from __future__ import annotations
import numpy as np


def aggregate(x: np.ndarray, grid, params) -> np.ndarray:
    return x
