"""Gold test: with CARA defaults, the analytic linear REE is a fixed point
of the 5-step loop; starting from that warm-start should give Finf ≈ eps
in 1 iteration."""
import numpy as np
from mizn import Params, CARA_DEFAULT
from mizn.main import solve
from mizn.analytic import linear_REE


def test_cara_default_converges_machine_precision(cara_default):
    P, hist, iters = solve(cara_default)
    assert hist[-1] < 1e-12, f"hist[-1] = {hist[-1]:.3e}, iters={iters}"


def test_cara_small_grid_also_converges(cara_small):
    P, hist, iters = solve(cara_small)
    assert hist[-1] < cara_small.tol


def test_cara_default_P_matches_analytic_REE():
    """The fixed-point P returned by the loop equals the analytic price."""
    p = CARA_DEFAULT
    P, _, _ = solve(p)
    r = linear_REE(p)
    G = p.G
    u = np.linspace(-p.umax, p.umax, G)
    T, U = np.meshgrid(u, u, indexing='ij')
    P_ana = r.price(T, U)
    err = float(np.max(np.abs(P - P_ana)))
    assert err < 1e-10, f"max|P - P_analytic| = {err:.3e}"


def test_nontrivial_theta_bar():
    p = CARA_DEFAULT.replace(theta_bar=0.5)
    P, hist, iters = solve(p)
    assert hist[-1] < 1e-10
