import numpy as np
from mizn.analytic import linear_REE
from mizn.config import Params, CARA_DEFAULT


def test_default_coefficients():
    r = linear_REE(CARA_DEFAULT)
    # tau_phi = 1*(2/2)^2 = 1, tau_1 = 1+2+1 = 4
    # a=0, b=3/4, c=2*3/(4*2) = 3/4
    assert abs(r.tau_phi - 1.0) < 1e-15
    assert abs(r.a - 0.0) < 1e-15
    assert abs(r.b - 0.75) < 1e-15
    assert abs(r.c - 0.75) < 1e-15


def test_price_linearity():
    r = linear_REE(CARA_DEFAULT)
    theta = np.linspace(-2, 2, 5)
    u     = np.linspace(-2, 2, 5)
    T, U = np.meshgrid(theta, u, indexing='ij')
    P = r.price(T, U)
    assert P.shape == (5, 5)
    # P(0,0) = a = 0; P(1,0) = b; P(0,1) = -c
    assert abs(P[2, 2] - 0.0) < 1e-15           # theta=0, u=0
    assert abs((P[3, 2] - P[2, 2]) - r.b) < 1e-12   # +1 in theta
    assert abs((P[2, 3] - P[2, 2]) - (-r.c)) < 1e-12  # +1 in u


def test_nonzero_theta_bar_shifts_intercept():
    p = CARA_DEFAULT.replace(theta_bar=2.0)
    r = linear_REE(p)
    # a = tau_th * theta_bar / tau_1 = 1*2/4 = 0.5
    assert abs(r.a - 0.5) < 1e-15
