"""Step 3: individual demand x(mu, P)."""
from .cara import demand_cara
from .crra import demand_crra


def demand(mu, P, params):
    model = params.model
    if model == 'cara':
        return demand_cara(mu, P, params)
    if model == 'crra':
        return demand_crra(mu, P, params)
    raise ValueError(f"model {model!r} not implemented yet")


__all__ = ['demand']
