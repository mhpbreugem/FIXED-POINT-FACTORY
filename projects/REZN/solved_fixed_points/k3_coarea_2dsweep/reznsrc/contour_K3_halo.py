"""K=3 contour map Phi on a padded grid: inner cells are unknowns,
halo cells are fixed boundary values.

Three kernels:
    phi_K3_halo         linear-interp scan, original.
    phi_K3_halo_cubic   Hermite cubic root-find in the scan.
    phi_K3_halo_smooth  Gaussian-kernel-smoothed Bayes integral; no
                        root-find. Phi becomes an analytic function of
                        P, allowing Newton's quadratic convergence.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .signals import f_signal, lam
from .demand import clear_crra, EPS_PRICE


@njit(cache=True, fastmath=False)
def _scan_axis_K3(P_slice: np.ndarray, p_target: float, axis: int,
                  a_idx: int, u_full: np.ndarray,
                  tau_a: float, tau_off: float,
                  acc: np.ndarray) -> None:
    """Scan one axis of a (G_full, G_full) slice for crossings of p_target.

    `axis` in {0, 1}: which slice axis is the off-grid sweep.
    `a_idx`: grid index of the on-grid axis. Each crossing contributes
    f_v(u_a) * f_v(u_off) to acc[v].
    """
    G_full = u_full.size
    u_a = u_full[a_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)

    prev_v = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
    for i in range(G_full - 1):
        next_v = (P_slice[i + 1, a_idx] if axis == 0
                  else P_slice[a_idx, i + 1])
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
            acc[0] += f0_a * f0_off
            acc[1] += f1_a * f1_off
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_K3(P_slice: np.ndarray, p_target: float,
                       u_full: np.ndarray,
                       tau_o0: float, tau_o1: float,
                       acc: np.ndarray) -> None:
    """Average the two off-grid sweeps for one K=3 agent's contour.

    tau_o0, tau_o1 are the precisions of the two OTHER agents in slice
    axis order (0, 1).
    """
    G_full = u_full.size

    # pass A: scan axis 0 off-grid, sweep axis 1 on grid -> crossing at
    # (u_off, u_a). The on-grid axis is axis 1, so its tau is tau_o1.
    a0 = 0.0
    a1 = 0.0
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3(P_slice, p_target, 0, a_idx, u_full,
                      tau_o1, tau_o0, acc)
    a0 += acc[0]
    a1 += acc[1]

    # pass B: scan axis 1 off-grid, sweep axis 0 on grid
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3(P_slice, p_target, 1, a_idx, u_full,
                      tau_o0, tau_o1, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = a0 / 2.0
    acc[1] = a1 / 2.0


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
def phi_K3_halo(P_full: np.ndarray, u_full: np.ndarray,
                inner_lo: int, inner_hi: int,
                tau_vec: np.ndarray, gamma_vec: np.ndarray,
                W_vec: np.ndarray) -> np.ndarray:
    """Phi map for K=3 with halo. Updates inner cells, halo unchanged."""
    P_new = P_full.copy()

    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]

                # Agent 0: slice P_full[i, :, :], slice axes (1, 2)
                _agent_evidence_K3(P_full[i, :, :], p, u_full,
                                   tau_vec[1], tau_vec[2], acc)
                mu_vec[0] = _bayes(u_full[i], tau_vec[0],
                                   acc[0], acc[1])

                # Agent 1: slice P_full[:, j, :], slice axes (0, 2)
                _agent_evidence_K3(P_full[:, j, :], p, u_full,
                                   tau_vec[0], tau_vec[2], acc)
                mu_vec[1] = _bayes(u_full[j], tau_vec[1],
                                   acc[0], acc[1])

                # Agent 2: slice P_full[:, :, l], slice axes (0, 1)
                _agent_evidence_K3(P_full[:, :, l], p, u_full,
                                   tau_vec[0], tau_vec[1], acc)
                mu_vec[2] = _bayes(u_full[l], tau_vec[2],
                                   acc[0], acc[1])

                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


# =====================================================================
# Cubic-interpolation variant
# =====================================================================

@njit(cache=True, fastmath=False, inline="always")
def _hermite_cubic_root(p0: float, p1: float, p2: float, p3: float,
                        p_target: float, du: float) -> float:
    """Find t in [0, 1] such that the Hermite cubic through P=(p1, p2)
    with slopes m0=(p2-p0)/2, m1=(p3-p1)/2 equals p_target.

    p0..p3 are P at u_{i-1}, u_i, u_{i+1}, u_{i+2} (uniform spacing du).
    Newton iteration from t=0.5; falls back to linear secant if the
    cubic derivative is degenerate.
    """
    m0 = (p2 - p0) * 0.5
    m1 = (p3 - p1) * 0.5
    t = 0.5
    for _ in range(8):
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        Pt = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
        dh00 = 6.0 * t2 - 6.0 * t
        dh10 = 3.0 * t2 - 4.0 * t + 1.0
        dh01 = -6.0 * t2 + 6.0 * t
        dh11 = 3.0 * t2 - 2.0 * t
        dPt = dh00 * p1 + dh10 * m0 + dh01 * p2 + dh11 * m1
        if dPt == 0.0:
            break
        delta = (Pt - p_target) / dPt
        t -= delta
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        if delta < 0.0:
            delta = -delta
        if delta < 1.0e-10:
            break
    return t


@njit(cache=True, fastmath=False)
def _scan_axis_K3_cubic(P_slice: np.ndarray, p_target: float, axis: int,
                        a_idx: int, u_full: np.ndarray,
                        tau_a: float, tau_off: float,
                        acc: np.ndarray) -> None:
    """As _scan_axis_K3 but with Hermite cubic root-find.

    Cubic uses 4 surrounding cells; near the boundary we fall back to
    linear interpolation.
    """
    G_full = u_full.size
    u_a = u_full[a_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)

    prev_v = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
    for i in range(G_full - 1):
        next_v = (P_slice[i + 1, a_idx] if axis == 0
                  else P_slice[a_idx, i + 1])
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
            # cubic if interior, linear at boundary
            if i >= 1 and i <= G_full - 3:
                if axis == 0:
                    p0 = P_slice[i - 1, a_idx]
                    p3 = P_slice[i + 2, a_idx]
                else:
                    p0 = P_slice[a_idx, i - 1]
                    p3 = P_slice[a_idx, i + 2]
                du = u_full[i + 1] - u_full[i]
                t = _hermite_cubic_root(p0, prev_v, next_v, p3,
                                        p_target, du)
            else:
                t = (p_target - prev_v) / denom
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
            u_off = (1.0 - t) * u_full[i] + t * u_full[i + 1]
            f0_off = f_signal(u_off, 0, tau_off)
            f1_off = f_signal(u_off, 1, tau_off)
            acc[0] += f0_a * f0_off
            acc[1] += f1_a * f1_off
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_K3_cubic(P_slice: np.ndarray, p_target: float,
                             u_full: np.ndarray,
                             tau_o0: float, tau_o1: float,
                             acc: np.ndarray) -> None:
    G_full = u_full.size
    a0 = 0.0
    a1 = 0.0

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3_cubic(P_slice, p_target, 0, a_idx, u_full,
                            tau_o1, tau_o0, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3_cubic(P_slice, p_target, 1, a_idx, u_full,
                            tau_o0, tau_o1, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = a0 / 2.0
    acc[1] = a1 / 2.0


@njit(cache=True, fastmath=False, parallel=True)
def phi_K3_halo_cubic(P_full: np.ndarray, u_full: np.ndarray,
                      inner_lo: int, inner_hi: int,
                      tau_vec: np.ndarray, gamma_vec: np.ndarray,
                      W_vec: np.ndarray) -> np.ndarray:
    """Cubic-interpolation variant of phi_K3_halo."""
    P_new = P_full.copy()

    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]

                _agent_evidence_K3_cubic(P_full[i, :, :], p, u_full,
                                         tau_vec[1], tau_vec[2], acc)
                mu_vec[0] = _bayes(u_full[i], tau_vec[0],
                                   acc[0], acc[1])

                _agent_evidence_K3_cubic(P_full[:, j, :], p, u_full,
                                         tau_vec[0], tau_vec[2], acc)
                mu_vec[1] = _bayes(u_full[j], tau_vec[1],
                                   acc[0], acc[1])

                _agent_evidence_K3_cubic(P_full[:, :, l], p, u_full,
                                         tau_vec[0], tau_vec[1], acc)
                mu_vec[2] = _bayes(u_full[l], tau_vec[2],
                                   acc[0], acc[1])

                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


# =====================================================================
# Smooth (kernel-density) variant
# =====================================================================

@njit(cache=True, fastmath=False)
def _agent_evidence_K3_smooth(P_slice: np.ndarray, p_target: float,
                              u_full: np.ndarray,
                              tau_o0: float, tau_o1: float,
                              kernel_h: float,
                              acc: np.ndarray) -> None:
    """Kernel-smoothed Bayes integration: A_v = sum K_h(P-p) f_v(u_a) f_v(u_b)
    over the entire 2-D slice. K_h is Gaussian with bandwidth kernel_h.

    Cell axes are (axis 0, axis 1) of the slice; tau_o0 is the precision
    associated with axis 0, tau_o1 with axis 1.
    """
    G_full = u_full.size
    inv_2h2 = 0.5 / (kernel_h * kernel_h)
    acc[0] = 0.0
    acc[1] = 0.0
    for ia in range(G_full):
        u_a = u_full[ia]
        f0_a = f_signal(u_a, 0, tau_o0)
        f1_a = f_signal(u_a, 1, tau_o0)
        for ib in range(G_full):
            u_b = u_full[ib]
            diff = P_slice[ia, ib] - p_target
            w = np.exp(-diff * diff * inv_2h2)
            f0_b = f_signal(u_b, 0, tau_o1)
            f1_b = f_signal(u_b, 1, tau_o1)
            acc[0] += w * f0_a * f0_b
            acc[1] += w * f1_a * f1_b


@njit(cache=True, fastmath=False, parallel=True)
def phi_K3_halo_smooth(P_full: np.ndarray, u_full: np.ndarray,
                       inner_lo: int, inner_hi: int,
                       tau_vec: np.ndarray, gamma_vec: np.ndarray,
                       W_vec: np.ndarray,
                       kernel_h: float) -> np.ndarray:
    """Kernel-smoothed (no root-find) variant of phi_K3_halo.

    Phi is an analytic function of P_full (Gaussian kernel of differences),
    so Newton's quadratic convergence applies. Choose kernel_h small
    enough to localize on the level set but large enough that several
    grid cells contribute (rule of thumb: a few times the typical
    P-difference between adjacent cells).
    """
    P_new = P_full.copy()

    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]

                # Agent 0: slice axes (1, 2) -> tau_o0=tau[1], tau_o1=tau[2]
                _agent_evidence_K3_smooth(P_full[i, :, :], p, u_full,
                                          tau_vec[1], tau_vec[2],
                                          kernel_h, acc)
                mu_vec[0] = _bayes(u_full[i], tau_vec[0],
                                   acc[0], acc[1])

                # Agent 1: slice axes (0, 2)
                _agent_evidence_K3_smooth(P_full[:, j, :], p, u_full,
                                          tau_vec[0], tau_vec[2],
                                          kernel_h, acc)
                mu_vec[1] = _bayes(u_full[j], tau_vec[1],
                                   acc[0], acc[1])

                # Agent 2: slice axes (0, 1)
                _agent_evidence_K3_smooth(P_full[:, :, l], p, u_full,
                                          tau_vec[0], tau_vec[1],
                                          kernel_h, acc)
                mu_vec[2] = _bayes(u_full[l], tau_vec[2],
                                   acc[0], acc[1])

                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


@njit(cache=True, fastmath=False, parallel=True)
def init_no_learning_K3(u_full: np.ndarray, tau_vec: np.ndarray,
                        gamma_vec: np.ndarray,
                        W_vec: np.ndarray) -> np.ndarray:
    """No-learning K=3 P_full[i,j,l] over the entire padded grid."""
    G_full = u_full.size
    P_full = np.empty((G_full, G_full, G_full), dtype=np.float64)
    for i in prange(G_full):
        mu_vec = np.empty(3, dtype=np.float64)
        m0 = lam(tau_vec[0] * u_full[i])
        for j in range(G_full):
            m1 = lam(tau_vec[1] * u_full[j])
            for l in range(G_full):
                m2 = lam(tau_vec[2] * u_full[l])
                mu_vec[0] = m0
                mu_vec[1] = m1
                mu_vec[2] = m2
                P_full[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_full
