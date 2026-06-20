import numpy as np
from mizn.signals import f_normal, sigmoid, logit, gl_nodes_weights


def test_normal_integrates_to_one():
    x = np.linspace(-10, 10, 4001)
    dx = x[1] - x[0]
    assert abs(np.trapezoid(f_normal(x, mu=0.5, tau=3.0), x) - 1.0) < 1e-6


def test_sigmoid_logit_inverse():
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.99])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-12)


def test_gl_integrates_polynomial():
    # GL with n nodes exact for poly up to degree 2n-1
    x, w = gl_nodes_weights(5, -2.0, 3.0)
    # integral of x^4 over [-2,3] = (3^5 - (-2)^5)/5 = (243 + 32)/5 = 55
    assert abs(np.sum(w * x ** 4) - 55.0) < 1e-10
