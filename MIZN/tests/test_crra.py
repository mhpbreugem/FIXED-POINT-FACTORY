"""CRRA pipeline plumbing test. We don't yet have the strict-h=0 operator
ported, so we cannot reproduce the prior session's machine-precision PR
fixed point.  This test only verifies that:
  (i) crra demand+clearing are wired into the dispatchers, and
 (ii) they preserve P in [0,1].
"""
import numpy as np
from mizn import Params
from mizn.step3_demand import demand
from mizn.step5_clearing import clear_and_verify


def test_crra_demand_signs():
    p = Params(model='crra', gamma=2.0, G=5)
    mu = np.array([[0.6, 0.6], [0.3, 0.3]])
    P  = np.array([[0.4, 0.4], [0.4, 0.4]])
    x = demand(mu, P, p)
    assert x[0, 0] > 0      # mu > P → positive demand
    assert x[1, 0] < 0      # mu < P → negative demand


def test_crra_clearing_bounded():
    p = Params(model='crra', gamma=2.0, G=5, umax=4.0)
    P = np.full((5, 5), 0.5)
    Z = np.zeros_like(P)
    P_new, resid = clear_and_verify(Z, P, p)
    assert P_new.shape == P.shape
    assert (P_new > 0).all() and (P_new < 1).all()
    assert resid >= 0
