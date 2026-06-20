import numpy as np
from mizn.grid import build, Grid
from mizn.config import Params


def test_linear_u_shape_and_endpoints():
    g = build(Params(G=11, umax=4.0))
    assert isinstance(g, Grid)
    assert g.G == 11
    assert g.kind == 'u'
    assert g.nodes[0] == -4.0
    assert g.nodes[-1] == 4.0
    assert g.weights.shape == (11,)
    assert np.allclose(g.jacobian, 1.0)


def test_trapezoidal_integrates_constant():
    g = build(Params(G=21, umax=5.0))
    assert abs(np.sum(g.weights * np.ones_like(g.nodes)) - 10.0) < 1e-12


def test_trapezoidal_integrates_quadratic():
    # ∫_{-5}^5 u^2 du = 2 * 125/3 ≈ 83.333
    g = build(Params(G=201, umax=5.0))
    s = np.sum(g.weights * g.nodes ** 2)
    assert abs(s - 250.0 / 3) < 5e-2  # trapezoidal — coarse


def test_invalid_grid_kind_raises():
    import pytest
    with pytest.raises(ValueError, match="not implemented"):
        build(Params(grid='xi'))
