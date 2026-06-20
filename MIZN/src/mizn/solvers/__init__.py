"""Outer-loop fixed-point drivers."""
from .picard import update_picard


def update(P, P_new, params):
    if params.solver == 'picard':
        return update_picard(P, P_new, params)
    raise ValueError(f"solver {params.solver!r} not implemented yet")


__all__ = ['update']
