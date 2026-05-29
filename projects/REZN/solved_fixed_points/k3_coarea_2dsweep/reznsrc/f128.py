"""float128 (np.longdouble) versions of metrics and symmetrisation.

Numba @njit does not accept np.longdouble, so the hot Phi map stays
float64. We lift float128 in three places where it actually pays off:

    revelation_deficit_f128   regression of logit(P) on T*, computed in
                              longdouble; avoids the float64 cancellation
                              floor of `1 - cov^2/(varL*varT)` when
                              cov^2 ~= varL*varT.
    symmetrize_f128           24-permutation S_4 average accumulated in
                              longdouble before casting back to float64.
    weights_f128              ex-ante density weights in longdouble (used
                              by the regression).

A float128 reference Phi map is in `code/contour_K4_f128.py` (slow,
verification only).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

DTYPE_F128 = np.longdouble


def lam_f128(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=DTYPE_F128)
    return DTYPE_F128(1) / (DTYPE_F128(1) + np.exp(-z))


def f_signal_f128(u: np.ndarray, v: int, tau: float) -> np.ndarray:
    """Gaussian signal density f_v(u) = sqrt(tau/2pi) exp(-tau/2 (u - v + 0.5)^2)."""
    u = np.asarray(u, dtype=DTYPE_F128)
    tau = DTYPE_F128(tau)
    half = DTYPE_F128("0.5")
    pi = DTYPE_F128(np.pi)
    coeff = np.sqrt(tau / (DTYPE_F128(2) * pi))
    return coeff * np.exp(-tau / DTYPE_F128(2) * (u - DTYPE_F128(v) + half) ** 2)


def t_star_f128(u_grid: np.ndarray, tau_vec: np.ndarray, K: int) -> np.ndarray:
    """T* = sum_k tau_k u_k, broadcast to shape (G,)*K, in longdouble."""
    u = np.asarray(u_grid, dtype=DTYPE_F128)
    tau_vec = np.asarray(tau_vec, dtype=DTYPE_F128)
    G = u.size
    out = np.zeros((G,) * K, dtype=DTYPE_F128)
    for k in range(K):
        shape = [1] * K
        shape[k] = G
        out = out + tau_vec[k] * u.reshape(shape)
    return out


def weights_f128(u_grid: np.ndarray, tau_vec: np.ndarray,
                 K: int) -> np.ndarray:
    """w(u) = (1/2)(prod f_1(u_k) + prod f_0(u_k)) in longdouble."""
    u = np.asarray(u_grid, dtype=DTYPE_F128)
    tau_vec = np.asarray(tau_vec, dtype=DTYPE_F128)
    G = u.size
    prod1 = np.ones((G,) * K, dtype=DTYPE_F128)
    prod0 = np.ones((G,) * K, dtype=DTYPE_F128)
    for k in range(K):
        shape = [1] * K
        shape[k] = G
        f1k = f_signal_f128(u, 1, float(tau_vec[k])).reshape(shape)
        f0k = f_signal_f128(u, 0, float(tau_vec[k])).reshape(shape)
        prod1 = prod1 * f1k
        prod0 = prod0 * f0k
    return DTYPE_F128("0.5") * (prod1 + prod0)


def _safe_logit_f128(p: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    p = np.asarray(p, dtype=DTYPE_F128)
    eps128 = DTYPE_F128(eps)
    p_clip = np.clip(p, eps128, DTYPE_F128(1) - eps128)
    return np.log(p_clip) - np.log1p(-p_clip)


def revelation_deficit_f128(P: np.ndarray, u_grid: np.ndarray,
                            tau_vec: np.ndarray, K: int,
                            eps: float = 1.0e-4) -> float:
    """Weighted 1 - R^2 of logit(P) on T*, all moments in longdouble.

    Returns a Python float for storage convenience; intermediate
    arithmetic is np.longdouble.
    """
    Ts = t_star_f128(u_grid, tau_vec, K)
    w = weights_f128(u_grid, tau_vec, K)
    L = _safe_logit_f128(P)

    P128 = np.asarray(P, dtype=DTYPE_F128)
    eps128 = DTYPE_F128(eps)
    mask = (P128 > eps128) & (P128 < DTYPE_F128(1) - eps128)
    w = np.where(mask, w, DTYPE_F128(0))
    Wsum = w.sum(dtype=DTYPE_F128)
    if Wsum <= 0:
        return float("nan")
    L_mean = (w * L).sum(dtype=DTYPE_F128) / Wsum
    T_mean = (w * Ts).sum(dtype=DTYPE_F128) / Wsum
    var_L = (w * (L - L_mean) ** 2).sum(dtype=DTYPE_F128) / Wsum
    var_T = (w * (Ts - T_mean) ** 2).sum(dtype=DTYPE_F128) / Wsum
    cov_LT = (w * (L - L_mean) * (Ts - T_mean)).sum(dtype=DTYPE_F128) / Wsum
    if var_L <= 0 or var_T <= 0:
        return float("nan")
    r2 = (cov_LT * cov_LT) / (var_L * var_T)
    return float(DTYPE_F128(1) - r2)


def symmetrize_f128(P: np.ndarray) -> np.ndarray:
    """S_K average accumulated in longdouble; cast back to the input dtype."""
    K = P.ndim
    in_dtype = P.dtype
    P128 = np.asarray(P, dtype=DTYPE_F128)
    acc = np.zeros_like(P128, dtype=DTYPE_F128)
    perms = list(permutations(range(K)))
    for sigma in perms:
        acc += np.transpose(P128, sigma)
    acc /= DTYPE_F128(len(perms))
    return acc.astype(in_dtype, copy=False)
