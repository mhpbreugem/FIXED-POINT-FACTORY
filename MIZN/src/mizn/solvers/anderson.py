"""Anderson acceleration (type-II) with history depth m.

State is held in a module-level dict keyed by id(params).  Pure-function
contract is preserved at the API level: callers just pass P, P_new, params
and get back the next iterate.  The history is implicit and reset on each
new Params id.
"""
from __future__ import annotations
import numpy as np

_HIST: dict = {}


def update_anderson(P, P_new, params):
    m = max(2, params.anderson_m)
    key = id(params)
    state = _HIST.setdefault(key, dict(X=[], F=[]))
    F = (P_new - P).ravel()
    X = P.ravel()
    state['X'].append(X.copy())
    state['F'].append(F.copy())
    if len(state['X']) > m + 1:
        state['X'].pop(0); state['F'].pop(0)
    if len(state['F']) < 2:
        return P + F.reshape(P.shape)         # first step = plain Picard
    dF = np.column_stack([state['F'][i + 1] - state['F'][i]
                            for i in range(len(state['F']) - 1)])
    dX = np.column_stack([state['X'][i + 1] - state['X'][i]
                            for i in range(len(state['X']) - 1)])
    try:
        gamma, *_ = np.linalg.lstsq(dF, F, rcond=None)
    except np.linalg.LinAlgError:
        return P + F.reshape(P.shape)
    upd = X + F - (dX + dF) @ gamma
    return upd.reshape(P.shape)


def reset_history():
    _HIST.clear()
