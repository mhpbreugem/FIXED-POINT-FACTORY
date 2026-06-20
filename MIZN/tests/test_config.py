from mizn.config import Params, CARA_DEFAULT


def test_default_is_frozen():
    import pytest
    with pytest.raises(Exception):
        CARA_DEFAULT.rho = 5.0  # frozen dataclass


def test_default_values(cara_default):
    assert cara_default.tau_th == 1.0
    assert cara_default.tau_eps == 2.0
    assert cara_default.tau_u == 1.0
    assert cara_default.rho == 2.0
    assert cara_default.model == 'cara'
    assert cara_default.grid == 'u'


def test_replace_returns_new_instance(cara_default):
    p2 = cara_default.replace(gamma=0.1, model='crra')
    assert p2.gamma == 0.1
    assert p2.model == 'crra'
    assert cara_default.gamma == 2.0   # original untouched


def test_to_dict_round_trip(cara_default):
    d = cara_default.to_dict()
    assert d['tau_th'] == 1.0
    assert Params(**d) == cara_default
