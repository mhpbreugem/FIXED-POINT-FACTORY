"""Outer-loop fixed-point drivers."""
from .picard import update_picard
from .anderson import update_anderson
from .newton_krylov import update_nk, solve_nk


def update(P, P_new, params):
    if params.solver == 'picard':
        return update_picard(P, P_new, params)
    if params.solver == 'anderson':
        return update_anderson(P, P_new, params)
    if params.solver == 'nk':
        return update_nk(P, P_new, params)
    raise ValueError(f"solver {params.solver!r} not implemented yet")


__all__ = ['update', 'solve_nk']
