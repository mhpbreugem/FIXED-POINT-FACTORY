"""Asymmetric Φ operator for the K-agent REE model.

Allows per-agent heterogeneity in:
  - signal precision τ_k  (each agent's signal u_k = v - 1/2 + ε_k, ε_k ~ N(0, 1/τ_k))
  - risk aversion γ_k     (CRRA demand x_k uses γ_k)
  - wealth W_k            (CRRA demand x_k scales by W_k)
  - prior q_k             (each agent's prior on v=1)

Storage is the full G^K tensor (no symmetric compression). For K=3, G=21
this is 9261 cells — fine.

Each Φ step at cell (i_1, ..., i_K):
  for k = 1..K:
    extract (K-1)-D slice of P fixing dim k at i_k
    contour-integrate using agent-j precisions {τ_j : j≠k}
    Bayes update agent-k's posterior using their own τ_k and prior q_k
  market-clear: Σ_k x(μ_k, p; γ_k, W_k) = 0
"""
from __future__ import annotations
import itertools
import math
import numpy as np

def _density(u: float, v: int, tau: float) -> float:
    mean = 0.5 if v == 1 else -0.5
    return math.sqrt(tau/(2*math.pi)) * math.exp(-0.5*tau*(u-mean)**2)

def _contour_integral_asym(
    P_slice: np.ndarray, p_target: float,
    u_grid: np.ndarray, taus_other: list[float],
) -> tuple[float, float]:
    """Contour integral for an asymmetric (K-1)-D slice.

    taus_other = [τ_j for j != k in some order matching P_slice's axes].
    """
    G = len(u_grid)
    K1 = P_slice.ndim
    A0_total = 0.0; A1_total = 0.0
    for scan_axis in range(K1):
        A0_pass = 0.0; A1_pass = 0.0
        n_other = K1 - 1
        tau_scan = taus_other[scan_axis]
        taus_grid = [taus_other[i] for i in range(K1) if i != scan_axis]
        if n_other == 0:
            prev = float(P_slice[0])
            for i in range(G-1):
                nxt = float(P_slice[i+1])
                dp, dn = prev - p_target, nxt - p_target
                if not (dp == 0.0 and dn == 0.0) and dp*dn <= 0.0:
                    denom = nxt - prev
                    if denom != 0.0:
                        frac = max(0.0, min(1.0, (p_target - prev)/denom))
                        u_off = (1.0-frac)*u_grid[i] + frac*u_grid[i+1]
                        A0_pass += _density(u_off, 0, tau_scan)
                        A1_pass += _density(u_off, 1, tau_scan)
                prev = nxt
        else:
            for idx_others in itertools.product(range(G), repeat=n_other):
                prod0 = 1.0; prod1 = 1.0
                for ig, ia in enumerate(idx_others):
                    prod0 *= _density(u_grid[ia], 0, taus_grid[ig])
                    prod1 *= _density(u_grid[ia], 1, taus_grid[ig])
                def _get(i_scan, _io=idx_others, _sa=scan_axis):
                    idx = list(_io); idx.insert(_sa, i_scan)
                    return float(P_slice[tuple(idx)])
                prev = _get(0)
                for i in range(G-1):
                    nxt = _get(i+1)
                    dp, dn = prev - p_target, nxt - p_target
                    if not (dp == 0.0 and dn == 0.0) and dp*dn <= 0.0:
                        denom = nxt - prev
                        if denom != 0.0:
                            frac = max(0.0, min(1.0, (p_target - prev)/denom))
                            u_off = (1.0-frac)*u_grid[i] + frac*u_grid[i+1]
                            f0_off = _density(u_off, 0, tau_scan)
                            f1_off = _density(u_off, 1, tau_scan)
                            A0_pass += prod0 * f0_off
                            A1_pass += prod1 * f1_off
                    prev = nxt
        A0_total += A0_pass
        A1_total += A1_pass
    return A0_total/K1, A1_total/K1


def _clear_crra_asym(mu_vec: list[float], gammas: list[float],
                     Ws: list[float]) -> float:
    """Bisection for Σ_k W_k x_k(μ_k, p; γ_k) = 0 with heterogeneous γ_k, W_k."""
    eps = 1e-12
    def excess(p):
        lp = math.log(p/(1-p))
        s = 0.0
        for mu, g, W in zip(mu_vec, gammas, Ws):
            lm = math.log(mu/(1-mu))
            z = (lm - lp)/g
            if z >= 0.0:
                e = math.exp(-z)
                s += W*(1.0-e)/((1.0-p)*e + p)
            else:
                e = math.exp(z)
                s += W*(e-1.0)/((1.0-p) + p*e)
        return s
    a, b = eps, 1.0-eps
    if excess(a) <= 0.0: return a
    if excess(b) >= 0.0: return b
    for _ in range(80):
        c = 0.5*(a+b)
        if excess(c) >= 0.0: a = c
        else: b = c
        if b - a < 1e-15: break
    return 0.5*(a+b)


def asym_phi(P_full: np.ndarray, u_grid: np.ndarray,
             taus: list[float], gammas: list[float], Ws: list[float],
             priors: list[float]) -> np.ndarray:
    """One iteration of the asymmetric Φ map. P_full shape = (G,)*K."""
    K = P_full.ndim
    G = P_full.shape[0]
    eps = 1e-12
    assert len(taus) == K and len(gammas) == K and len(Ws) == K and len(priors) == K
    new_P = np.empty_like(P_full)
    for cell in itertools.product(range(G), repeat=K):
        p = float(P_full[cell])
        mus = []
        for k in range(K):
            i_k = cell[k]
            u_k = float(u_grid[i_k])
            taus_other = [taus[j] for j in range(K) if j != k]
            P_slice = np.take(P_full, i_k, axis=k)
            A0, A1 = _contour_integral_asym(P_slice, p, u_grid, taus_other)
            f0k = _density(u_k, 0, taus[k]); f1k = _density(u_k, 1, taus[k])
            q = priors[k]
            num = q * f1k * A1
            den = (1-q) * f0k * A0 + num
            mu = num/den if den > 0.0 else q
            mus.append(max(eps, min(1.0-eps, mu)))
        new_P[cell] = _clear_crra_asym(mus, gammas, Ws)
    return new_P


def fr_logit_slope_asym(u_grid: np.ndarray, taus: list[float],
                        priors: list[float]) -> tuple[float, np.ndarray]:
    """For symmetric (q=0.5), the FR REE has logit P = Σ τ_k u_k.
    For heterogeneous priors, add log(q/(1-q)) shifts. Returns (constant, slopes).
    """
    bias = sum(math.log(q/(1-q)) for q in priors)
    return bias, np.array(taus, dtype=float)
