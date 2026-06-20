"""Step 5: market clearing.  Returns (P_new, max-residual)."""
from .cara import clear_cara
from .crra import clear_crra


def clear_and_verify(Z, P, params):
    if params.model == 'cara':
        return clear_cara(Z, P, params)
    if params.model == 'crra':
        return clear_crra(Z, P, params)
    raise ValueError(f"model {params.model!r} not implemented yet")


__all__ = ['clear_and_verify']
