"""Permutation symmetrisation of the K=4 price array.

Under homogeneous parameters the equilibrium price is a symmetric function
of the K signals. A single Phi step generally violates this symmetry by
small numerical errors; averaging over S_K = K! permutations restores it
and improves convergence.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from .config import DTYPE


def symmetrize(P: np.ndarray) -> np.ndarray:
    """Average P[i,j,l,m] over all S_K permutations of axes.

    Shape (G,)*K. Returns a new float64 array.
    """
    K = P.ndim
    out = np.zeros_like(P, dtype=DTYPE)
    perms = list(permutations(range(K)))
    for sigma in perms:
        out += np.transpose(P, sigma)
    out /= len(perms)
    return out


def is_symmetric(P: np.ndarray, atol: float = 1.0e-10) -> bool:
    """Return True if P is invariant under all axis permutations within atol."""
    K = P.ndim
    for sigma in permutations(range(K)):
        if not np.allclose(P, np.transpose(P, sigma), atol=atol):
            return False
    return True
