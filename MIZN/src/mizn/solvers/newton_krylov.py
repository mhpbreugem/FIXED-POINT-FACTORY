"""scipy newton_krylov wrapper.

Wraps the 5-step loop's residual as a function of P (flattened) and
hands it to scipy.optimize.newton_krylov.  Used outside the main loop
(callers invoke solve_nk(params) directly); the in-loop `update(...)`
dispatch falls back to picard for solver='nk' so the iteration shell
remains the same — NK is one-shot.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from .picard import update_picard


def update_nk(P, P_new, params):
    """In-loop dispatch fallback: defer to picard so the 5-step loop
    keeps running; the NK driver below is for end-to-end solves."""
    return update_picard(P, P_new, params)


def solve_nk(params):
    """One-shot Newton-Krylov on the 5-step residual."""
    from .. import grid as grid_mod
    from .. import (step1_conjecture, step2_learning, step3_demand,
                      step4_aggregate, step5_clearing)
    g = grid_mod.build(params)
    P0 = step1_conjecture.conjecture(g, params)
    shape = P0.shape

    def residual(Pflat):
        P = Pflat.reshape(shape)
        mu = step2_learning.learn(P, g, params)
        x = step3_demand.demand(mu, P, params)
        Z = step4_aggregate.aggregate(x, g, params)
        P_new, _ = step5_clearing.clear_and_verify(Z, P, params)
        return (P_new - P).ravel()

    try:
        sol = newton_krylov(residual, P0.ravel(),
                              f_tol=params.tol, maxiter=params.maxit,
                              verbose=False)
        Finf = float(np.max(np.abs(residual(sol))))
        return sol.reshape(shape), np.array([Finf]), 1
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
        Finf = float(np.max(np.abs(residual(sol))))
        return sol.reshape(shape), np.array([Finf]), -1
