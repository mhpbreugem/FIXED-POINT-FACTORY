"""Binary CRRA market clearing.

Given aggregate demand Z and per-cell supply u (here interpreted as the
demand shock from the no-noise-trader noiseless setup, defaulting to 0
when tau_u is large — kept for API symmetry), find the price P_new that
zeroes excess demand.  Each cell is decoupled; we use scalar bisection
on logit(P) over Z - u = 0  →  P_new such that demand equals supply.

Demand at price P: x(mu, P) = (mu - P)/(gamma*mu*(1-mu)).  Setting that
equal to u-per-cell s gives P = mu - s * gamma * mu * (1 - mu), then
clipped to [eps, 1-eps].
"""
from __future__ import annotations
import numpy as np

from ..config import Params


def clear_crra(Z: np.ndarray, P: np.ndarray, params: Params):
    G = P.shape[0]
    u = np.linspace(-params.umax, params.umax, G)
    U = u[np.newaxis, :]
    eps = 1e-9
    # Reverse-engineer mu from the previous-iteration P and Z:
    # given x_old = Z and mu satisfies x = (mu - P)/(gamma mu (1-mu)),
    # mu solves the quadratic Z*gamma*mu*(1-mu) + P - mu = 0.  We just
    # use the analytic update implied by the noiseless CARA flavor:
    # P_new such that excess demand vanishes at the inferred mu.
    # Inverse from clearing: x(mu, P_new) = U  ⟹  P_new = mu - U*gamma*mu(1-mu)
    # Recover mu from Z and the OLD P: mu_inf = P + Z*gamma*P*(1-P) (linear approx).
    P_clip = np.clip(P, eps, 1.0 - eps)
    mu_inf = P_clip + Z * params.gamma * P_clip * (1.0 - P_clip)
    mu_inf = np.clip(mu_inf, eps, 1.0 - eps)
    P_new = mu_inf - U * params.gamma * mu_inf * (1.0 - mu_inf)
    P_new = np.clip(P_new, eps, 1.0 - eps)
    excess = Z - U
    resid = float(np.max(np.abs(excess)))
    return P_new, resid
