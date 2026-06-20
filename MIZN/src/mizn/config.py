"""Unified parameter container for MIZN.

A single dataclass that fully pins one execution: economic primitives,
grid choice, operator choice, solver choice, tolerances.  Swap models /
grids / solvers by flipping a string here; nothing else in main.py changes.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Literal


Model    = Literal['cara', 'crra']
GridKind = Literal['u', 'xi', 'zeta']
Step2    = Literal['gaussian', 'coarea_kernel', 'coarea_h0']
Solver   = Literal['picard', 'anderson', 'nk']


@dataclass(frozen=True)
class Params:
    # economic primitives
    tau_th:    float = 1.0     # precision of fundamental theta
    tau_eps:   float = 2.0     # precision of agent signal noise
    tau_u:     float = 1.0     # precision of supply noise (only used in CARA)
    rho:       float = 2.0     # CARA risk aversion
    gamma:     float = 2.0     # CRRA curvature (unused for cara)
    theta_bar: float = 0.0
    u_bar:     float = 0.0

    # numerics
    G:         int   = 9       # inner grid size (per axis)
    umax:      float = 4.0     # half-width in u
    model:     Model = 'cara'
    grid:      GridKind = 'u'
    step2:     Step2 = 'gaussian'
    solver:    Solver = 'picard'

    # solver controls
    tol:       float = 1e-12
    maxit:     int   = 200
    omega:     float = 0.5     # picard damping
    anderson_m: int  = 8

    def replace(self, **kw) -> 'Params':
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


CARA_DEFAULT = Params()  # tau_th=1, tau_eps=2, tau_u=1, rho=2 → b=0.75, c=0.75
