"""K=4 contour map Phi on a padded grid: inner cells are unknowns,
halo cells are fixed boundary values.

Layout:
    u_full = linspace(-u_outer, +u_outer, G_full)
    inner_lo .. inner_hi-1 are the indices of the INNER unknowns
    everything else in u_full is a halo cell

The contour scan operates over the entire u_full grid, so a level set
{P_full == p} that escapes the inner region simply continues into
halo cells, where P_full has been pinned by the caller (no-learning
rule, extrapolation, or stage-(s-1) inner solution). This cures the
corner-cell residual blow-up of the bare-grid kernel.

Output: P_new agrees with P_full on halo cells and has updated values
on inner cells. Newton/Picard see a residual that is identically zero
outside the inner block.

Parallelism: the outer (i,j,l,m) loop is `prange`'d on the i-axis;
caller should set NUMBA_NUM_THREADS to the number of cores allocated.
Each prange iteration allocates its own scratch arrays (mu_vec, acc).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .signals import f_signal, lam
from .demand import clear_crra, EPS_PRICE


@njit(cache=True, fastmath=False)
def _scan_axis_halo(P_slice: np.ndarray, p_target: float, axis: int,
                    a_idx: int, b_idx: int, u_full: np.ndarray,
                    tau_a: float, tau_b: float, tau_off: float,
                    acc: np.ndarray) -> None:
    """Scan one axis of a (G_full, G_full, G_full) slice for crossings of
    p_target. Same as the het non-halo scan but consumes the entire
    padded grid u_full.
    """
    G_full = u_full.size
    u_a = u_full[a_idx]
    u_b = u_full[b_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)
    f0_b = f_signal(u_b, 0, tau_b)
    f1_b = f_signal(u_b, 1, tau_b)

    prev_v = (P_slice[0, a_idx, b_idx] if axis == 0
              else P_slice[a_idx, 0, b_idx] if axis == 1
              else P_slice[a_idx, b_idx, 0])
    for i in range(G_full - 1):
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
            u_off = (1.0 - frac) * u_full[i] + frac * u_full[i + 1]
            f0_off = f_signal(u_off, 0, tau_off)
            f1_off = f_signal(u_off, 1, tau_off)
            acc[0] += f0_a * f0_b * f0_off
            acc[1] += f1_a * f1_b * f1_off
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_halo(P_slice: np.ndarray, p_target: float,
                         u_full: np.ndarray,
                         tau_o0: float, tau_o1: float, tau_o2: float,
                         acc: np.ndarray) -> None:
    """Average the three off-grid sweeps for one agent's contour."""
    G_full = u_full.size
    a0 = 0.0
    a1 = 0.0

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            _scan_axis_halo(P_slice, p_target, 0, a_idx, b_idx, u_full,
                            tau_o1, tau_o2, tau_o0, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            _scan_axis_halo(P_slice, p_target, 1, a_idx, b_idx, u_full,
                            tau_o0, tau_o2, tau_o1, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            _scan_axis_halo(P_slice, p_target, 2, a_idx, b_idx, u_full,
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


@njit(cache=True, fastmath=False, parallel=True)
def phi_K4_halo(P_full: np.ndarray, u_full: np.ndarray,
                inner_lo: int, inner_hi: int,
                tau_vec: np.ndarray, gamma_vec: np.ndarray,
                W_vec: np.ndarray) -> np.ndarray:
    """Phi map: updates inner cells, leaves halo cells untouched.

    P_full has shape (G_full,) * 4. Inner cells live in
    [inner_lo:inner_hi]^4. The contour integration sees the entire
    P_full grid via the agent's slice.
    """
    P_new = P_full.copy()  # halo preserved by default

    for i in prange(inner_lo, inner_hi):
        # Per-thread scratch
        mu_vec = np.empty(4, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                for m in range(inner_lo, inner_hi):
                    p = P_full[i, j, l, m]

                    # Agent 0: slice axes (1, 2, 3)
                    _agent_evidence_halo(P_full[i, :, :, :], p, u_full,
                                         tau_vec[1], tau_vec[2], tau_vec[3],
                                         acc)
                    mu_vec[0] = _bayes(u_full[i], tau_vec[0],
                                       acc[0], acc[1])

                    # Agent 1: slice axes (0, 2, 3)
                    _agent_evidence_halo(P_full[:, j, :, :], p, u_full,
                                         tau_vec[0], tau_vec[2], tau_vec[3],
                                         acc)
                    mu_vec[1] = _bayes(u_full[j], tau_vec[1],
                                       acc[0], acc[1])

                    # Agent 2: slice axes (0, 1, 3)
                    _agent_evidence_halo(P_full[:, :, l, :], p, u_full,
                                         tau_vec[0], tau_vec[1], tau_vec[3],
                                         acc)
                    mu_vec[2] = _bayes(u_full[l], tau_vec[2],
                                       acc[0], acc[1])

                    # Agent 3: slice axes (0, 1, 2)
                    _agent_evidence_halo(P_full[:, :, :, m], p, u_full,
                                         tau_vec[0], tau_vec[1], tau_vec[2],
                                         acc)
                    mu_vec[3] = _bayes(u_full[m], tau_vec[3],
                                       acc[0], acc[1])

                    P_new[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


@njit(cache=True, fastmath=False, parallel=True)
def init_no_learning_halo(u_full: np.ndarray, tau_vec: np.ndarray,
                          gamma_vec: np.ndarray,
                          W_vec: np.ndarray) -> np.ndarray:
    """No-learning P_full[i,j,l,m] over the entire padded grid.

    Used both as the seed for the inner solve AND as the static halo.
    """
    G_full = u_full.size
    P_full = np.empty((G_full, G_full, G_full, G_full), dtype=np.float64)
    for i in prange(G_full):
        mu_vec = np.empty(4, dtype=np.float64)
        m0 = lam(tau_vec[0] * u_full[i])
        for j in range(G_full):
            m1 = lam(tau_vec[1] * u_full[j])
            for l in range(G_full):
                m2 = lam(tau_vec[2] * u_full[l])
                for m in range(G_full):
                    m3 = lam(tau_vec[3] * u_full[m])
                    mu_vec[0] = m0
                    mu_vec[1] = m1
                    mu_vec[2] = m2
                    mu_vec[3] = m3
                    P_full[i, j, l, m] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_full
