import numpy as np
from mizn import Params
from mizn.main import solve
from mizn.solvers import solve_nk
from mizn.solvers.anderson import reset_history


def test_picard_default():
    P, hist, iters = solve(Params(solver='picard'))
    assert hist[-1] < 1e-12


def test_anderson_default():
    reset_history()
    P, hist, iters = solve(Params(solver='anderson'))
    assert hist[-1] < 1e-10
    assert iters <= 50


def test_nk_one_shot():
    P, hist, iters = solve_nk(Params(G=7))
    assert hist[-1] < 1e-8


def test_nk_dispatch_via_loop():
    """solver='nk' inside the loop falls back to picard but converges."""
    P, hist, iters = solve(Params(solver='nk'))
    assert hist[-1] < 1e-10
