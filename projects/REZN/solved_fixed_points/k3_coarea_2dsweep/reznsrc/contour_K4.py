"""K=4 contour map Phi: P -> P_new.

For each grid point (i,j,l,m) and each agent k=0..3, the agent extracts
her own 3D slice of P (fixing her own index), traces the level set
{P = p} in the 3D space of the other agents' signals, and integrates
the joint signal density along that level set under each state v in {0,1}.

For K=4 the level set is a 2D surface in 3D; we approximate the surface
integral by sweeping each pair of axes on the grid and root-finding the
remaining axis off-grid (3 passes per agent), then averaging.

The kernel is serial @njit (caller is pinned to one core).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .signals import f_signal, lam, logit
from .demand import clear_crra, clear_cara, EPS_PRICE
from .config import DTYPE


# -- linear interpolation along one axis of P[i,:,:,:] -------------------

@njit(cache=True, fastmath=False, inline="always")
def _interp_axis(values: np.ndarray, idx: int, frac: float, axis: int,
                 j: int, l: int, m: int) -> float:
    """Linear interpolation along `axis` between idx and idx+1 with weight frac.

    `values` is the 3D slice (G,G,G); `axis` in {0,1,2} indicates which
    of (j,l,m) is the swept-off-grid coordinate.
    """
    if axis == 0:
        return (1.0 - frac) * values[idx, l, m] + frac * values[idx + 1, l, m]
    if axis == 1:
        return (1.0 - frac) * values[j, idx, m] + frac * values[j, idx + 1, m]
    return (1.0 - frac) * values[j, l, idx] + frac * values[j, l, idx + 1]


# -- root-find one off-grid coordinate at level p ------------------------
# Inputs: 3D slice, axis to scan, two grid indices for the other two axes.
# Returns: number of crossings found and (u_off, A1_inc, A0_inc) for each.
# We compress this to: caller passes accumulators by reference.

@njit(cache=True, fastmath=False)
def _scan_axis(P_slice: np.ndarray, p_target: float, axis: int,
               a_idx: int, b_idx: int, u_grid: np.ndarray, tau: float,
               acc: np.ndarray) -> None:
    """Scan one axis of P_slice for crossings of level p_target.

    The other two axes are fixed at grid indices (a_idx, b_idx). The role
    of (a_idx, b_idx) within (j,l,m) depends on `axis`:
        axis=0 -> sweep j off-grid, fix (l,m)=(a_idx,b_idx)
        axis=1 -> sweep l off-grid, fix (j,m)=(a_idx,b_idx)
        axis=2 -> sweep m off-grid, fix (j,l)=(a_idx,b_idx)
    Each crossing contributes f_v(u_a)*f_v(u_b)*f_v(u_off) to acc[v]
    (acc has length 2: acc[0] += under v=0, acc[1] += under v=1).
    """
    G = u_grid.size
    u_a = u_grid[a_idx]
    u_b = u_grid[b_idx]
    f0_a = f_signal(u_a, 0, tau)
    f1_a = f_signal(u_a, 1, tau)
    f0_b = f_signal(u_b, 0, tau)
    f1_b = f_signal(u_b, 1, tau)

    # Read P_slice along axis with the other two coords fixed at (a_idx, b_idx).
    # We need values v[0..G-1] = P_slice[..,..,..] in the right slot ordering.
    # axis=0: P_slice[i_axis, a_idx, b_idx]
    # axis=1: P_slice[a_idx, i_axis, b_idx]
    # axis=2: P_slice[a_idx, b_idx, i_axis]
    prev_v = (P_slice[0, a_idx, b_idx] if axis == 0
              else P_slice[a_idx, 0, b_idx] if axis == 1
              else P_slice[a_idx, b_idx, 0])
    for i in range(G - 1):
        next_v = (P_slice[i + 1, a_idx, b_idx] if axis == 0
                  else P_slice[a_idx, i + 1, b_idx] if axis == 1
                  else P_slice[a_idx, b_idx, i + 1])
        d_prev = prev_v - p_target
        d_next = next_v - p_target
        if (d_prev == 0.0 and d_next == 0.0):
            prev_v = next_v
            continue
        if d_prev * d_next <= 0.0:
            denom = next_v - prev_v
            if denom == 0.0:
                prev_v = next_v
                continue
            frac = (p_target - prev_v) / denom
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            u_off = (1.0 - frac) * u_grid[i] + frac * u_grid[i + 1]
            f0_off = f_signal(u_off, 0, tau)
            f1_off = f_signal(u_off, 1, tau)
            acc[0] += f0_a * f0_b * f0_off
            acc[1] += f1_a * f1_b * f1_off
        prev_v = next_v


# -- K=4 contour evidence for ONE agent at one realisation ---------------

@njit(cache=True, fastmath=False)
def _agent_evidence(P_slice: np.ndarray, p_target: float,
                    u_grid: np.ndarray, tau: float, acc: np.ndarray) -> None:
    """Compute (A0, A1) contour evidence for one agent.

    P_slice is the agent's (G,G,G) slice (own index fixed by caller).
    The level set {P_slice == p_target} is a 2D surface in 3D. We do
    three passes (one per choice of off-grid axis), summing crossings.
    Final acc[0]=A0, acc[1]=A1 are averaged over the three passes.
    """
    G = u_grid.size
    a0 = 0.0
    a1 = 0.0
    # pass A: scan axis 0 off-grid, sweep (axis 1, axis 2) on grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis(P_slice, p_target, 0, a_idx, b_idx, u_grid, tau, acc)
    a0 += acc[0]
    a1 += acc[1]
    # pass B: scan axis 1 off-grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis(P_slice, p_target, 1, a_idx, b_idx, u_grid, tau, acc)
    a0 += acc[0]
    a1 += acc[1]
    # pass C: scan axis 2 off-grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis(P_slice, p_target, 2, a_idx, b_idx, u_grid, tau, acc)
    a0 += acc[0]
    a1 += acc[1]
    acc[0] = a0 / 3.0
    acc[1] = a1 / 3.0


# -- agent posterior given own signal and contour evidence ---------------

@njit(cache=True, fastmath=False, inline="always")
def _bayes(u_own: float, tau_own: float, A0: float, A1: float) -> float:
    f0 = f_signal(u_own, 0, tau_own)
    f1 = f_signal(u_own, 1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > 1.0 - EPS_PRICE:
        return 1.0 - EPS_PRICE
    return mu


# -- the Phi map for K=4 -------------------------------------------------

@njit(cache=True, fastmath=False)
def phi_K4(P: np.ndarray, u_grid: np.ndarray,
           tau_vec: np.ndarray, gamma_vec: np.ndarray, W_vec: np.ndarray,
           cara: bool) -> np.ndarray:
    """One application of the contour map Phi: returns P_new with same shape."""
    G = u_grid.size
    P_new = np.empty((G, G, G, G), dtype=np.float64)
    mu_vec = np.empty(4, dtype=np.float64)
    acc = np.empty(2, dtype=np.float64)
    # pre-allocate per-agent slice buffers (we read from P, no allocation needed)

    for i in range(G):
        for j in range(G):
            for l in range(G):
                for m in range(G):
                    p = P[i, j, l, m]

                    # Agent 0 fixes own index i, slice P[i,:,:,:]
                    _agent_evidence(P[i, :, :, :], p, u_grid, tau_vec[0], acc)
                    mu_vec[0] = _bayes(u_grid[i], tau_vec[0], acc[0], acc[1])

                    # Agent 1 fixes index j, slice P[:,j,:,:]
                    _agent_evidence(P[:, j, :, :], p, u_grid, tau_vec[1], acc)
                    mu_vec[1] = _bayes(u_grid[j], tau_vec[1], acc[0], acc[1])

                    # Agent 2 fixes index l, slice P[:,:,l,:]
                    _agent_evidence(P[:, :, l, :], p, u_grid, tau_vec[2], acc)
                    mu_vec[2] = _bayes(u_grid[l], tau_vec[2], acc[0], acc[1])

                    # Agent 3 fixes index m, slice P[:,:,:,m]
                    _agent_evidence(P[:, :, :, m], p, u_grid, tau_vec[3], acc)
                    mu_vec[3] = _bayes(u_grid[m], tau_vec[3], acc[0], acc[1])

                    if cara:
                        P_new[i, j, l, m] = clear_cara(mu_vec, gamma_vec)
                    else:
                        P_new[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


# -- no-learning initialisation -----------------------------------------

@njit(cache=True, fastmath=False)
def init_no_learning(u_grid: np.ndarray, tau_vec: np.ndarray,
                     gamma_vec: np.ndarray, W_vec: np.ndarray,
                     cara: bool) -> np.ndarray:
    """Initial price P[i,j,l,m] from prior-only posteriors mu_k = Lambda(tau_k u_k)."""
    G = u_grid.size
    P = np.empty((G, G, G, G), dtype=np.float64)
    mu_vec = np.empty(4, dtype=np.float64)
    for i in range(G):
        m0 = lam(tau_vec[0] * u_grid[i])
        for j in range(G):
            m1 = lam(tau_vec[1] * u_grid[j])
            for l in range(G):
                m2 = lam(tau_vec[2] * u_grid[l])
                for m in range(G):
                    m3 = lam(tau_vec[3] * u_grid[m])
                    mu_vec[0] = m0
                    mu_vec[1] = m1
                    mu_vec[2] = m2
                    mu_vec[3] = m3
                    if cara:
                        P[i, j, l, m] = clear_cara(mu_vec, gamma_vec)
                    else:
                        P[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P


# -- residual measure ----------------------------------------------------

@njit(cache=True, fastmath=False)
def residual_inf(P: np.ndarray, P_new: np.ndarray) -> float:
    G = P.shape[0]
    r = 0.0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                for m in range(G):
                    d = P[i, j, l, m] - P_new[i, j, l, m]
                    if d < 0.0:
                        d = -d
                    if d > r:
                        r = d
    return r
