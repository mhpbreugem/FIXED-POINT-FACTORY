"""K=4 contour map Phi for heterogeneous (gamma_k, tau_k).

The homogeneous kernel in `contour_K4.py` passes a single tau to the
contour integration, which is correct only when all agents share the
same precision. With heterogeneous tau, each *other* agent's signal
density must use that agent's own tau.

Layout: agent k extracts her 3D slice (others' signals) and runs three
passes over the level surface {P_slice == p}. For each pass we sweep
two axes on grid and root-find the third off-grid. The three axes
correspond to *specific* other agents, identified by their original
index in the K=4 stack; the per-axis tau is looked up accordingly.

Agent k's slice axes (slice axes 0, 1, 2 in order):
    k = 0  -> others = (1, 2, 3)
    k = 1  -> others = (0, 2, 3)
    k = 2  -> others = (0, 1, 3)
    k = 3  -> others = (0, 1, 2)
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .signals import f_signal, lam
from .demand import clear_crra, EPS_PRICE
from .config import DTYPE


@njit(cache=True, fastmath=False)
def _scan_axis_het(P_slice: np.ndarray, p_target: float, axis: int,
                   a_idx: int, b_idx: int, u_grid: np.ndarray,
                   tau_a: float, tau_b: float, tau_off: float,
                   acc: np.ndarray) -> None:
    """Scan one axis of a 3D slice for crossings of level p_target.

    Parameters
    ----------
    axis : 0, 1, or 2 — which slice axis is the off-grid sweep.
    a_idx, b_idx : grid indices of the two on-grid axes.
    tau_a, tau_b, tau_off : signal precisions associated with axes
        (other_a, other_b, other_off). The mapping from slice axis
        to the original agent index is the caller's responsibility.

    Each crossing contributes f_v(u_a)*f_v(u_b)*f_v(u_off) to acc[v].
    """
    G = u_grid.size
    u_a = u_grid[a_idx]
    u_b = u_grid[b_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)
    f0_b = f_signal(u_b, 0, tau_b)
    f1_b = f_signal(u_b, 1, tau_b)

    prev_v = (P_slice[0, a_idx, b_idx] if axis == 0
              else P_slice[a_idx, 0, b_idx] if axis == 1
              else P_slice[a_idx, b_idx, 0])
    for i in range(G - 1):
        next_v = (P_slice[i + 1, a_idx, b_idx] if axis == 0
                  else P_slice[a_idx, i + 1, b_idx] if axis == 1
                  else P_slice[a_idx, b_idx, i + 1])
        d_prev = prev_v - p_target
        d_next = next_v - p_target
        if d_prev == 0.0 and d_next == 0.0:
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
            f0_off = f_signal(u_off, 0, tau_off)
            f1_off = f_signal(u_off, 1, tau_off)
            acc[0] += f0_a * f0_b * f0_off
            acc[1] += f1_a * f1_b * f1_off
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_het(P_slice: np.ndarray, p_target: float,
                        u_grid: np.ndarray,
                        tau_o0: float, tau_o1: float, tau_o2: float,
                        acc: np.ndarray) -> None:
    """Evidence (A_0, A_1) for one agent under heterogeneous taus.

    tau_o0, tau_o1, tau_o2 are the precisions of the OTHER agents
    in the order their signals appear along slice axes 0, 1, 2.
    """
    G = u_grid.size

    # pass A: axis 0 off-grid, sweep axes (1, 2) on grid
    a0 = 0.0
    a1 = 0.0
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis_het(P_slice, p_target, 0, a_idx, b_idx, u_grid,
                           tau_o1, tau_o2, tau_o0, acc)
    a0 += acc[0]
    a1 += acc[1]

    # pass B: axis 1 off-grid, sweep axes (0, 2) on grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis_het(P_slice, p_target, 1, a_idx, b_idx, u_grid,
                           tau_o0, tau_o2, tau_o1, acc)
    a0 += acc[0]
    a1 += acc[1]

    # pass C: axis 2 off-grid, sweep axes (0, 1) on grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        for b_idx in range(G):
            _scan_axis_het(P_slice, p_target, 2, a_idx, b_idx, u_grid,
                           tau_o0, tau_o1, tau_o2, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = a0 / 3.0
    acc[1] = a1 / 3.0


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


@njit(cache=True, fastmath=False)
def phi_K4_het(P: np.ndarray, u_grid: np.ndarray,
               tau_vec: np.ndarray, gamma_vec: np.ndarray,
               W_vec: np.ndarray) -> np.ndarray:
    """Heterogeneous-(gamma, tau) Phi map. CRRA only (no CARA branch)."""
    G = u_grid.size
    P_new = np.empty((G, G, G, G), dtype=np.float64)
    mu_vec = np.empty(4, dtype=np.float64)
    acc = np.empty(2, dtype=np.float64)

    for i in range(G):
        for j in range(G):
            for l in range(G):
                for m in range(G):
                    p = P[i, j, l, m]

                    # Agent 0 — slice axes (1, 2, 3); others' taus in order.
                    _agent_evidence_het(P[i, :, :, :], p, u_grid,
                                        tau_vec[1], tau_vec[2], tau_vec[3],
                                        acc)
                    mu_vec[0] = _bayes(u_grid[i], tau_vec[0],
                                       acc[0], acc[1])

                    # Agent 1 — slice axes (0, 2, 3).
                    _agent_evidence_het(P[:, j, :, :], p, u_grid,
                                        tau_vec[0], tau_vec[2], tau_vec[3],
                                        acc)
                    mu_vec[1] = _bayes(u_grid[j], tau_vec[1],
                                       acc[0], acc[1])

                    # Agent 2 — slice axes (0, 1, 3).
                    _agent_evidence_het(P[:, :, l, :], p, u_grid,
                                        tau_vec[0], tau_vec[1], tau_vec[3],
                                        acc)
                    mu_vec[2] = _bayes(u_grid[l], tau_vec[2],
                                       acc[0], acc[1])

                    # Agent 3 — slice axes (0, 1, 2).
                    _agent_evidence_het(P[:, :, :, m], p, u_grid,
                                        tau_vec[0], tau_vec[1], tau_vec[2],
                                        acc)
                    mu_vec[3] = _bayes(u_grid[m], tau_vec[3],
                                       acc[0], acc[1])

                    P_new[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


@njit(cache=True, fastmath=False)
def init_no_learning_het(u_grid: np.ndarray, tau_vec: np.ndarray,
                         gamma_vec: np.ndarray,
                         W_vec: np.ndarray) -> np.ndarray:
    """Heterogeneous no-learning P[i,j,l,m] from prior posteriors."""
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
                    P[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P
