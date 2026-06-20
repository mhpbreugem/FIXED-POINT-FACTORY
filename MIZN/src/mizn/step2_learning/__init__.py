"""Step 2: Bayesian updating.

Dispatch on `params.step2` ∈ {'gaussian', 'coarea_kernel', 'coarea_h0'}.
Returns mu(theta, u) — the conditional mean of the asset value, of the
same shape as P.
"""
from .gaussian import learn_gaussian


def learn(P, grid, params):
    kind = params.step2
    if kind == 'gaussian':
        return learn_gaussian(P, grid, params)
    raise ValueError(f"step2 kind {kind!r} not implemented yet")


__all__ = ['learn']
