"""Shared fixtures."""
import pytest
from mizn.config import Params, CARA_DEFAULT


@pytest.fixture
def cara_default():
    return CARA_DEFAULT


@pytest.fixture
def cara_small():
    """Tiny grid for fast tests."""
    return Params(G=5, tol=1e-10, maxit=50)
