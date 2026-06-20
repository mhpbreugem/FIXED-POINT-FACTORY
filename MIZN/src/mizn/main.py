"""THE 5-step Hellwig loop.  ~30 lines, never changes."""
from __future__ import annotations
import numpy as np

from .config import Params
from . import grid as grid_mod
from . import step1_conjecture
from . import step2_learning
from . import step3_demand
from . import step4_aggregate
from . import step5_clearing
from . import solvers


def solve(params: Params):
    """Run the loop and return (P, Finf_history, iters)."""
    g = grid_mod.build(params)
    P = step1_conjecture.conjecture(g, params)
    hist = []
    for it in range(params.maxit):
        mu     = step2_learning.learn(P, g, params)
        x      = step3_demand.demand(mu, P, params)
        Z      = step4_aggregate.aggregate(x, g, params)
        P_new, resid = step5_clearing.clear_and_verify(Z, P, params)
        hist.append(resid)
        if resid < params.tol:
            P = P_new
            break
        P = solvers.update(P, P_new, params)
    return P, np.asarray(hist), it + 1


def main():
    p = Params()
    P, hist, iters = solve(p)
    print(f"CARA default: iters={iters}, Finf_final={hist[-1]:.3e}, "
          f"P[0,0]={P[0, 0]:.6f}")
    return P, hist, iters


if __name__ == '__main__':
    main()
