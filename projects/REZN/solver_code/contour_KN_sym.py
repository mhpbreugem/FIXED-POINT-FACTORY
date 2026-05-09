"""
contour_KN_sym.py — Symmetric-K REE solver for the homogeneous-CRRA model.

Exploits permutation symmetry of the homogeneous case: when all agents
share the same gamma and tau, the price function P(u_1,...,u_K) is
invariant under permutations of its arguments and can be stored on
the manifold of sorted index tuples.

Number of sorted K-tuples on a G-grid: C(G+K-1, K) (multiset coefficient).
At G=15:
    K=3:    680    cells
    K=4:  3,060
    K=5: 11,628
    K=6: 38,760
    K=7: 116,280
    K=8: 319,770

Public API:
    SymGrid(G, K)              — precomputed sorted-index machinery
    sym_phi(P_sorted, ...)      — Phi map operating on sorted storage
    sym_newton(...)             — Newton-Krylov fixed-point solver
    sym_to_full(P_sorted, sg)  — reconstruct full K-rank tensor (for plotting)
    sym_weighted_R2(...)       — weighted 1-R^2 with multiplicity counting

Validation: at K=3, this solver must reproduce the existing
contour_K3_halo results to ~1e-8 on any common (gamma, tau).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from math import comb
from typing import Tuple

import numpy as np


# =====================================================================
# Sorted-tuple indexing
# =====================================================================

def n_sorted(G: int, K: int) -> int:
    """Number of sorted K-tuples on a G-grid: C(G+K-1, K)."""
    return comb(G + K - 1, K)


def _build_sorted_tuples(G: int, K: int) -> np.ndarray:
    """Return a (n_sorted, K) int array of all sorted tuples in colex order."""
    out = np.zeros((n_sorted(G, K), K), dtype=np.int64)
    tup = [0] * K
    idx = 0
    while True:
        out[idx] = tup
        idx += 1
        k = K - 1
        while k >= 0 and tup[k] == G - 1:
            k -= 1
        if k < 0:
            break
        tup[k] += 1
        for j in range(k + 1, K):
            tup[j] = tup[k]
    return out


@dataclass
class SymGrid:
    """Sorted-tuple indexing precomputed for (G, K)."""
    G: int
    K: int
    n: int                       # = C(G+K-1, K)
    tuples: np.ndarray           # (n, K) int64 — sorted tuples in colex order
    lookup: dict                 # tuple -> flat sorted index

    @classmethod
    def build(cls, G: int, K: int) -> "SymGrid":
        tuples = _build_sorted_tuples(G, K)
        lookup = {tuple(t.tolist()): i for i, t in enumerate(tuples)}
        return cls(G=G, K=K, n=n_sorted(G, K), tuples=tuples, lookup=lookup)

    def index_of(self, unsorted_tuple: Tuple[int, ...]) -> int:
        return self.lookup[tuple(sorted(unsorted_tuple))]

    def multiplicity(self, sorted_idx: int) -> int:
        """Number of distinct permutations of the sorted tuple at this index."""
        t = self.tuples[sorted_idx]
        from collections import Counter
        c = Counter(t.tolist())
        denom = 1
        for v in c.values():
            denom *= math.factorial(v)
        return math.factorial(self.K) // denom


# =====================================================================
# Inflation: sorted -> full K-rank tensor
# =====================================================================

def sym_to_full(P_sorted: np.ndarray, sg: SymGrid) -> np.ndarray:
    """Reconstruct the full G^K array from a sorted-cell array of length sg.n."""
    G, K = sg.G, sg.K
    P_full = np.zeros((G,) * K, dtype=np.float64)
    for sorted_idx in range(sg.n):
        val = P_sorted[sorted_idx]
        sorted_tup = tuple(sg.tuples[sorted_idx].tolist())
        for perm in set(itertools.permutations(sorted_tup)):
            P_full[perm] = val
    return P_full


def full_to_sym(P_full: np.ndarray, sg: SymGrid) -> np.ndarray:
    """Project a full G^K array onto sorted storage by reading sorted entries."""
    out = np.empty(sg.n, dtype=np.float64)
    for sorted_idx in range(sg.n):
        out[sorted_idx] = P_full[tuple(sg.tuples[sorted_idx])]
    return out


# =====================================================================
# Internal helpers
# =====================================================================

def _signal_density_scalar(u: float, v: int, tau: float) -> float:
    mean = 0.5 if v == 1 else -0.5
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * (u - mean) ** 2)


def _contour_integral_kd(
    P_slice: np.ndarray,
    p_target: float,
    u_grid: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    """Contour integral for a (K-1)-dimensional slice at level p_target.

    Mirrors _agent_evidence_K3 from contour_K3_halo.py but generalised to
    arbitrary dimension K1 = P_slice.ndim.

    Method: scan each of the K1 axes in turn as the "off-grid" axis (linear
    interpolation for crossing location), average the K1 passes — exactly as
    the K=3 reference averages its two passes.

    Returns (A0, A1) where A_v = sum over contour of prod f_v(u_k).
    """
    G = len(u_grid)
    K1 = P_slice.ndim  # = K - 1

    A0_total = 0.0
    A1_total = 0.0

    for scan_axis in range(K1):
        A0_pass = 0.0
        A1_pass = 0.0
        n_other = K1 - 1

        if n_other == 0:
            # K=2 edge case: P_slice is 1-D, no on-grid axes
            prev = float(P_slice[0])
            for i in range(G - 1):
                nxt = float(P_slice[i + 1])
                dp, dn = prev - p_target, nxt - p_target
                if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                    denom = nxt - prev
                    if denom != 0.0:
                        frac = max(0.0, min(1.0, (p_target - prev) / denom))
                        u_off = (1.0 - frac) * u_grid[i] + frac * u_grid[i + 1]
                        A0_pass += _signal_density_scalar(u_off, 0, tau)
                        A1_pass += _signal_density_scalar(u_off, 1, tau)
                prev = nxt
        else:
            # General case: iterate over all G^n_other on-grid multi-indices
            for idx_others in itertools.product(range(G), repeat=n_other):
                prod0 = 1.0
                prod1 = 1.0
                for ia in idx_others:
                    prod0 *= float(f0[ia])
                    prod1 *= float(f1[ia])

                # Build full index by inserting scan_axis position
                def _get(i_scan: int, _io=idx_others, _sa=scan_axis) -> float:
                    idx: list[int] = list(_io)
                    idx.insert(_sa, i_scan)
                    return float(P_slice[tuple(idx)])

                prev = _get(0)
                for i in range(G - 1):
                    nxt = _get(i + 1)
                    dp, dn = prev - p_target, nxt - p_target
                    if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                        denom = nxt - prev
                        if denom != 0.0:
                            frac = max(0.0, min(1.0, (p_target - prev) / denom))
                            u_off = (1.0 - frac) * u_grid[i] + frac * u_grid[i + 1]
                            f0_off = _signal_density_scalar(u_off, 0, tau)
                            f1_off = _signal_density_scalar(u_off, 1, tau)
                            A0_pass += prod0 * f0_off
                            A1_pass += prod1 * f1_off
                    prev = nxt

        A0_total += A0_pass
        A1_total += A1_pass

    return A0_total / K1, A1_total / K1


def _clear_crra_sym(mu_vec: list[float], gamma: float, W: float) -> float:
    """Bisection for sum_k x_crra(mu_k, p, gamma, W) = 0."""
    eps = 1e-12

    def excess(p: float) -> float:
        lp = math.log(p / (1.0 - p))
        s = 0.0
        for mu in mu_vec:
            lm = math.log(mu / (1.0 - mu))
            z = (lm - lp) / gamma
            if z >= 0.0:
                e = math.exp(-z)
                s += W * (1.0 - e) / ((1.0 - p) * e + p)
            else:
                e = math.exp(z)
                s += W * (e - 1.0) / ((1.0 - p) + p * e)
        return s

    a, b = eps, 1.0 - eps
    if excess(a) <= 0.0:
        return a
    if excess(b) >= 0.0:
        return b
    for _ in range(60):
        c = 0.5 * (a + b)
        if excess(c) >= 0.0:
            a = c
        else:
            b = c
        if b - a < 1e-14:
            break
    return 0.5 * (a + b)


# =====================================================================
# Symmetric Phi map
# =====================================================================

def sym_phi(P_sorted: np.ndarray, sg: SymGrid, u_grid: np.ndarray,
            tau: float, gamma: float, W: float) -> np.ndarray:
    """One iteration of the Phi map on sorted storage.

    For each sorted cell (i_1,...,i_K):
    1. For each agent k, extract the (K-1)-D slice fixing dimension k at i_k.
       By symmetry of P (homogeneous model), np.take(P_full, i_k, axis=k)
       gives the correct contour slice for agent k.
    2. Compute agent k's contour integral A0, A1 via _contour_integral_kd,
       using linear interpolation exactly as _agent_evidence_K3 does.
    3. Bayes-update: mu_k = f1(u_{i_k}) * A1 / (f0 * A0 + f1 * A1).
    4. Market-clear all K posteriors: p_new = argzero sum_k x_crra(mu_k, p).

    At K=3, this exactly reproduces phi_K3_halo (same scan logic, same
    bisection) and must agree to ≤1e-6 on weighted 1-R².
    """
    G, K = sg.G, sg.K
    eps = 1e-12

    f0 = np.sqrt(tau / (2.0 * np.pi)) * np.exp(-0.5 * tau * (u_grid + 0.5) ** 2)
    f1 = np.sqrt(tau / (2.0 * np.pi)) * np.exp(-0.5 * tau * (u_grid - 0.5) ** 2)

    # Materialise full G^K tensor for slice extraction.
    # At K=8, G=15: 15^8 = 2.6 GB — too large; the GHA worker uses the
    # agent-0-fixed slice path below via _phi_k8_slice() instead.
    # For K<=7, G=15 (max 170 MB at K=7) this is fine.
    if K >= 8:
        return _sym_phi_large(P_sorted, sg, u_grid, tau, gamma, W, f0, f1)

    P_full = sym_to_full(P_sorted, sg)
    new_P_sorted = np.empty(sg.n, dtype=np.float64)

    for s in range(sg.n):
        i_tuple = sg.tuples[s]
        p = float(P_full[tuple(i_tuple)])

        mu_list: list[float] = []
        # Cache: if two agents have the same signal index their slice and
        # hence contour integral and posterior are identical.
        cache: dict[int, float] = {}
        for k in range(K):
            i_k = int(i_tuple[k])
            if i_k in cache:
                mu_list.append(cache[i_k])
                continue
            # (K-1)-D slice fixing dimension k at i_k
            P_slice = np.take(P_full, i_k, axis=k)
            A0, A1 = _contour_integral_kd(P_slice, p, u_grid, f0, f1, tau)
            f0k = float(f0[i_k])
            f1k = float(f1[i_k])
            num = f1k * A1
            den = f0k * A0 + num
            mu = (num / den) if den > 0.0 else 0.5
            mu = max(eps, min(1.0 - eps, mu))
            cache[i_k] = mu
            mu_list.append(mu)

        new_P_sorted[s] = _clear_crra_sym(mu_list, gamma, W)

    return new_P_sorted


def _sym_phi_large(P_sorted: np.ndarray, sg: SymGrid, u_grid: np.ndarray,
                   tau: float, gamma: float, W: float,
                   f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """K=8 memory-efficient variant: materialise one G^(K-1) slice at a time.

    For each unique first-index value i in 0..G-1, build the (K-1)-D slice
    P[i, :, :, ..., :] once and reuse it for all sorted cells whose agent
    with the matching index i needs that slice.  At K=8, G=15 each slice is
    15^7 ≈ 170 MB — fits comfortably.
    """
    G, K = sg.G, sg.K
    eps = 1e-12
    new_P_sorted = np.empty(sg.n, dtype=np.float64)

    # Build mapping: unique index value -> list of (cell_s, agent_k, cell_price)
    # We need per-cell prices, so we first need P values for all sorted cells.
    # Since K=8 G=15: n=319770 cells — small enough to store a price array.
    # We build P_diag = price at each sorted cell without full tensor.
    # P[sorted_tuple] = P_sorted[s] (by definition of sorted storage).
    # For _contour_integral_kd we need the SLICE, not the diagonal; we still
    # need sym_to_full for the slice.  So we materialise one slice at a time.

    # Collect, for each cell s, the unique signal indices needed
    # (one per unique value in the tuple).
    # For K=8: 319770 * 8 = ~2.5M lookups, manageable.

    # Step 1: for each sorted cell, record its current price.
    # We can read P_sorted directly since P_sorted[s] = P_full[sorted_tuple(s)].
    cell_prices = P_sorted.copy()  # indexed by s

    # Step 2: for each unique signal index i, build slice P[i, :, ..., :]
    # and compute the contour integral for every cell that needs it.
    # We then store (A0, A1) per (cell_s, agent_k).
    # But storing all (A0, A1) at once for K=8 requires 319770*8*2 floats ~ 40 MB: fine.
    A0_arr = np.zeros((sg.n, K), dtype=np.float64)
    A1_arr = np.zeros((sg.n, K), dtype=np.float64)

    for i_val in range(G):
        # Find all (cell_s, k) pairs where i_tuple[k] == i_val
        cells_needing = []
        for s in range(sg.n):
            for k in range(K):
                if sg.tuples[s, k] == i_val:
                    cells_needing.append((s, k))
        if not cells_needing:
            continue

        # Materialise slice: P[i_val, :, :, ..., :] of shape (G,)^{K-1}
        # We build this from P_sorted by iterating sorted (K-1)-tuples.
        sg_k1 = SymGrid.build(G, K - 1)
        P_slice_sorted = np.empty(sg_k1.n, dtype=np.float64)
        for s2 in range(sg_k1.n):
            t2 = tuple(sg_k1.tuples[s2].tolist())
            # Full (K)-tuple: insert i_val at position 0, then sort
            full_t = tuple(sorted((i_val,) + t2))
            s_full = sg.lookup.get(full_t)
            if s_full is not None:
                P_slice_sorted[s2] = P_sorted[s_full]
            else:
                P_slice_sorted[s2] = 0.5  # fallback

        P_slice_full = sym_to_full(P_slice_sorted, sg_k1)  # (G,)^{K-1}

        # For each (s, k) needing this slice, compute contour integral at cell's price
        done_at_price: dict[float, tuple[float, float]] = {}
        for s, k in cells_needing:
            p = float(cell_prices[s])
            if p in done_at_price:
                A0_arr[s, k], A1_arr[s, k] = done_at_price[p]
                continue
            A0, A1 = _contour_integral_kd(P_slice_full, p, u_grid, f0, f1, tau)
            done_at_price[p] = (A0, A1)
            A0_arr[s, k] = A0
            A1_arr[s, k] = A1

    # Step 3: assemble posteriors and market-clear
    for s in range(sg.n):
        mu_list: list[float] = []
        for k in range(K):
            i_k = int(sg.tuples[s, k])
            A0 = A0_arr[s, k]
            A1 = A1_arr[s, k]
            f0k = float(f0[i_k])
            f1k = float(f1[i_k])
            num = f1k * A1
            den = f0k * A0 + num
            mu = (num / den) if den > 0.0 else 0.5
            mu_list.append(max(eps, min(1.0 - eps, mu)))
        new_P_sorted[s] = _clear_crra_sym(mu_list, gamma, W)

    return new_P_sorted


# =====================================================================
# No-learning initialiser
# =====================================================================

def sym_init_no_learning(sg: SymGrid, u_grid: np.ndarray,
                         tau: float, gamma: float, W: float) -> np.ndarray:
    """No-learning equilibrium: p = sigma(tau * sum u_k), market-cleared."""
    eps = 1e-12
    P_sorted = np.empty(sg.n, dtype=np.float64)
    for s in range(sg.n):
        t = sg.tuples[s]
        T_star = tau * float(u_grid[t].sum())
        # No-learning: all agents share posterior mu = sigma(tau * u_k)
        mu_list = [max(eps, min(1.0 - eps,
                       math.exp(tau * float(u_grid[t[k]])) / (1.0 + math.exp(tau * float(u_grid[t[k]])))
                       if tau * float(u_grid[t[k]]) >= 0
                       else 1.0 / (1.0 + math.exp(-tau * float(u_grid[t[k]])))
                       ))
                   for k in range(sg.K)]
        P_sorted[s] = _clear_crra_sym(mu_list, gamma, W)
    return P_sorted


# =====================================================================
# Newton-Krylov solver
# =====================================================================

def sym_newton(sg: SymGrid, u_grid: np.ndarray,
               tau: float, gamma: float, W: float,
               P_init: np.ndarray | None = None,
               tol: float = 1e-7,
               max_iter: int = 100,
               verbose: bool = True,
               inner_m: int = 50) -> tuple[np.ndarray, list[float]]:
    """Newton-Krylov fixed-point solver on sorted storage.

    Solves F(P) = Phi(P) - P = 0 using scipy.optimize.newton_krylov
    (matrix-free Newton with LGMRES inner Krylov solver).
    Quadratic convergence; typically < 20 outer iterations.

    Returns (P_converged, residual_history).
    """
    from scipy.optimize import newton_krylov  # noqa: PLC0415

    P0 = sym_init_no_learning(sg, u_grid, tau, gamma, W) if P_init is None else P_init.copy()
    history: list[float] = []

    def residual(P_flat: np.ndarray) -> np.ndarray:
        phi_P = sym_phi(P_flat, sg, u_grid, tau, gamma, W)
        F = phi_P - P_flat
        history.append(float(np.max(np.abs(F))))
        if verbose and len(history) % 10 == 1:
            print(f"  newton eval {len(history):4d}  ||F||inf={history[-1]:.4e}", flush=True)
        return F

    P_sol = newton_krylov(
        residual, P0, f_tol=tol, maxiter=max_iter,
        method="lgmres", inner_m=inner_m, verbose=False,
    )
    if verbose:
        print(f"  newton done  evals={len(history)}  ||F||inf={history[-1]:.4e}", flush=True)
    return P_sol, history


# =====================================================================
# Weighted 1-R^2 with multiplicity
# =====================================================================

def sym_weighted_R2(P_sorted: np.ndarray, sg: SymGrid,
                    u_grid: np.ndarray, tau: float) -> dict:
    """Weighted 1-R^2 of logit(p) on T* = tau * sum(u_k)."""
    K = sg.K
    eps = 1e-12

    f0 = np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u_grid + 0.5) ** 2)
    f1 = np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u_grid - 0.5) ** 2)

    Tstar = np.empty(sg.n)
    logit_p = np.empty(sg.n)
    weights = np.empty(sg.n)

    for s in range(sg.n):
        t = sg.tuples[s]
        u_vals = u_grid[t]
        Tstar[s] = tau * u_vals.sum()
        p = float(P_sorted[s])
        p = min(max(p, eps), 1 - eps)
        logit_p[s] = math.log(p / (1 - p))
        prod0 = float(np.prod(f0[t]))
        prod1 = float(np.prod(f1[t]))
        mult = sg.multiplicity(s)
        weights[s] = mult * 0.5 * (prod0 + prod1)

    weights /= weights.sum()
    slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))
    pred = slope * Tstar + intercept
    mean_lp = float(np.average(logit_p, weights=weights))
    var_tot = float(np.average((logit_p - mean_lp) ** 2, weights=weights))
    var_res = float(np.average((logit_p - pred) ** 2, weights=weights))
    one_minus_r2 = var_res / var_tot if var_tot > 0 else float("nan")

    return {
        "1-R2": one_minus_r2,
        "slope": float(slope),
        "intercept": float(intercept),
        "n_cells": int(sg.n),
    }


# =====================================================================
# Smoke test
# =====================================================================

if __name__ == "__main__":
    print("=== Sorted cell counts at G=15 ===")
    for K in range(3, 9):
        sg = SymGrid.build(15, K)
        print(f"  K={K}: n={sg.n:>10,} cells")
    print("\n=== Multiplicity check (K=3, G=4) ===")
    sg = SymGrid.build(4, 3)
    total = 0
    for s in range(sg.n):
        m = sg.multiplicity(s)
        total += m
    print(f"  Sum of multiplicities = {total} (should be G^K = {sg.G ** sg.K})")
    assert total == sg.G ** sg.K, "multiplicity sum != G^K"
    print("  OK")
