"""Gaussian primitives, logit / sigmoid, Gauss-Legendre nodes & weights."""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss


def f_normal(x: np.ndarray, mu: float = 0.0, tau: float = 1.0) -> np.ndarray:
    """N(mu, 1/tau) density evaluated pointwise."""
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (x - mu) ** 2)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def gl_nodes_weights(n: int, a: float = -1.0, b: float = 1.0):
    """Gauss-Legendre nodes/weights mapped to [a, b]."""
    x, w = leggauss(n)
    half = 0.5 * (b - a)
    mid  = 0.5 * (b + a)
    return mid + half * x, half * w
