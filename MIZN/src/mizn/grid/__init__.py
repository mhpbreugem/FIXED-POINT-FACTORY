"""Swappable coordinate system.

Dispatch on `params.grid` ∈ {'u', 'xi', 'zeta'}.  Operators talk to the
grid only through the `Grid` dataclass — they don't care which coord chart
is underneath.
"""
from .base import Grid
from .linear_u import build_linear_u


def build(params) -> Grid:
    """Construct the grid the loop will iterate on."""
    kind = params.grid
    if kind == 'u':
        return build_linear_u(params)
    raise ValueError(f"grid kind {kind!r} not implemented yet")


__all__ = ['Grid', 'build']
