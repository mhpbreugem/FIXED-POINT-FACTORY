"""Swappable coordinate system.

Dispatch on `params.grid` ∈ {'u', 'xi', 'zeta'}.  Operators talk to the
grid only through the `Grid` dataclass — they don't care which coord chart
is underneath.
"""
from .base import Grid
from .linear_u import build_linear_u
from .atanh_xi import build_atanh_xi
from .cdf_zeta import build_cdf_zeta


def build(params) -> Grid:
    """Construct the grid the loop will iterate on."""
    kind = params.grid
    if kind == 'u':
        return build_linear_u(params)
    if kind == 'xi':
        return build_atanh_xi(params)
    if kind == 'zeta':
        return build_cdf_zeta(params)
    raise ValueError(f"grid kind {kind!r} not implemented yet")


__all__ = ['Grid', 'build']
