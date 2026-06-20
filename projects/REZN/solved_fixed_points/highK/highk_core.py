"""Numba-accelerated symmetric-K REE solver (K = 3, 4, 5).

Semantics are an exact port of projects/REZN/solver_code/contour_KN_sym.py
(sym_phi: linear-interp contour scan averaged over the K-1 slice axes,
Bayes update, CRRA market clearing by bisection), but jitted with numba
and parallelised over sorted cells so that Newton-Krylov at K=4 (G=15,
3060 cells) and K=5 (G=11, 3003 cells) is affordable.

Permutation symmetry: storage is on sorted K-tuples (SymGrid from
contour_KN_sym).  Inflation sorted -> full uses a precomputed integer
map (full cell -> sorted index), so P_full = P_sorted[inv_map] is O(G^K)
numpy gather.  Slices for agent k are taken on axis 0 (P_full[i_k]),
valid because P is permutation-invariant.

Deficit metric matches the certified emin15/lowtau K=3 convention:
UNWEIGHTED 1-R^2 of logit(P) on T = sum_k u_k over the full cube
(= multiplicity-weighted on sorted storage).
"""
from __future__ import annotations

import math
import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code')
from contour_KN_sym import SymGrid  # noqa: E402

EPS = 1e-12


# =====================================================================
# Sorted-tuple machinery (vectorised inflation map)
# =====================================================================

def build_machinery(G: int, K: int):
    """Return (sg, inv_map, mult) where
    inv_map: (G^K,) int64 — full flat index -> sorted index
    mult:    (n,) float64 — number of permutations of each sorted cell."""
    sg = SymGrid.build(G, K)
    powers = G ** np.arange(K - 1, -1, -1)
    # encode each sorted tuple
    tup_code = sg.tuples @ powers
    code2sorted = np.full(G ** K, -1, dtype=np.int64)
    code2sorted[tup_code] = np.arange(sg.n)
    # all full cells
    idx = np.indices((G,) * K).reshape(K, -1).T          # (G^K, K)
    sidx = np.sort(idx, axis=1)
    inv_map = code2sorted[sidx @ powers]
    assert inv_map.min() >= 0
    mult = np.bincount(inv_map, minlength=sg.n).astype(np.float64)
    return sg, inv_map, mult


def inflate(P_sorted: np.ndarray, inv_map: np.ndarray, G: int, K: int) -> np.ndarray:
    return P_sorted[inv_map].reshape((G,) * K)


# =====================================================================
# numba primitives
# =====================================================================

@njit(cache=True, inline='always')
def _excess(mu_vec, p, gamma, W):
    lp = math.log(p / (1.0 - p))
    s = 0.0
    for k in range(mu_vec.shape[0]):
        mu = mu_vec[k]
        lm = math.log(mu / (1.0 - mu))
        z = (lm - lp) / gamma
        if z >= 0.0:
            e = math.exp(-z)
            s += W * (1.0 - e) / ((1.0 - p) * e + p)
        else:
            e = math.exp(z)
            s += W * (e - 1.0) / ((1.0 - p) + p * e)
    return s


@njit(cache=True)
def _clear(mu_vec, gamma, W):
    """Bisection for sum_k x_crra(mu_k, p) = 0 (port of _clear_crra_sym)."""
    a = EPS
    b = 1.0 - EPS
    if _excess(mu_vec, a, gamma, W) <= 0.0:
        return a
    if _excess(mu_vec, b, gamma, W) >= 0.0:
        return b
    for _ in range(60):
        c = 0.5 * (a + b)
        if _excess(mu_vec, c, gamma, W) >= 0.0:
            a = c
        else:
            b = c
        if b - a < 1e-14:
            break
    return 0.5 * (a + b)


# ---------------------------------------------------------------------
# Contour kernels: K1 = ndim of slice = K - 1.  Average over K1 passes.
# c1 = sqrt(tau/2pi), c2 = 0.5*tau
# ---------------------------------------------------------------------

@njit(cache=True, inline='always')
def _seg(prev, nxt, p, u_lo, u_hi, c1, c2, pf0, pf1, acc):
    dp = prev - p
    dn = nxt - p
    if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
        denom = nxt - prev
        if denom != 0.0:
            frac = (p - prev) / denom
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            uo = (1.0 - frac) * u_lo + frac * u_hi
            acc[0] += pf0 * c1 * math.exp(-c2 * (uo + 0.5) ** 2)
            acc[1] += pf1 * c1 * math.exp(-c2 * (uo - 0.5) ** 2)


@njit(cache=True)
def _contour2(S, p, u, f0, f1, c1, c2):
    """2-D slice (K=3)."""
    G = u.shape[0]
    A0 = 0.0
    A1 = 0.0
    acc = np.zeros(2)
    # pass: scan axis 0 (lines S[:, b])
    for b in range(G):
        pf0 = f0[b]
        pf1 = f1[b]
        prev = S[0, b]
        for i in range(G - 1):
            nxt = S[i + 1, b]
            _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
            prev = nxt
    # pass: scan axis 1 (lines S[a, :])
    for a in range(G):
        pf0 = f0[a]
        pf1 = f1[a]
        prev = S[a, 0]
        for i in range(G - 1):
            nxt = S[a, i + 1]
            _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
            prev = nxt
    A0 = acc[0]
    A1 = acc[1]
    return A0 / 2.0, A1 / 2.0


@njit(cache=True)
def _contour3(S, p, u, f0, f1, c1, c2):
    """3-D slice (K=4)."""
    G = u.shape[0]
    acc = np.zeros(2)
    # scan axis 0: lines S[:, b, c]
    for b in range(G):
        for c in range(G):
            pf0 = f0[b] * f0[c]
            pf1 = f1[b] * f1[c]
            prev = S[0, b, c]
            for i in range(G - 1):
                nxt = S[i + 1, b, c]
                _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                prev = nxt
    # scan axis 1: lines S[a, :, c]
    for a in range(G):
        for c in range(G):
            pf0 = f0[a] * f0[c]
            pf1 = f1[a] * f1[c]
            prev = S[a, 0, c]
            for i in range(G - 1):
                nxt = S[a, i + 1, c]
                _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                prev = nxt
    # scan axis 2: lines S[a, b, :]
    for a in range(G):
        for b in range(G):
            pf0 = f0[a] * f0[b]
            pf1 = f1[a] * f1[b]
            prev = S[a, b, 0]
            for i in range(G - 1):
                nxt = S[a, b, i + 1]
                _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                prev = nxt
    return acc[0] / 3.0, acc[1] / 3.0


@njit(cache=True)
def _contour4(S, p, u, f0, f1, c1, c2):
    """4-D slice (K=5)."""
    G = u.shape[0]
    acc = np.zeros(2)
    # scan axis 0
    for b in range(G):
        for c in range(G):
            for d in range(G):
                pf0 = f0[b] * f0[c] * f0[d]
                pf1 = f1[b] * f1[c] * f1[d]
                prev = S[0, b, c, d]
                for i in range(G - 1):
                    nxt = S[i + 1, b, c, d]
                    _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                    prev = nxt
    # scan axis 1
    for a in range(G):
        for c in range(G):
            for d in range(G):
                pf0 = f0[a] * f0[c] * f0[d]
                pf1 = f1[a] * f1[c] * f1[d]
                prev = S[a, 0, c, d]
                for i in range(G - 1):
                    nxt = S[a, i + 1, c, d]
                    _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                    prev = nxt
    # scan axis 2
    for a in range(G):
        for b in range(G):
            for d in range(G):
                pf0 = f0[a] * f0[b] * f0[d]
                pf1 = f1[a] * f1[b] * f1[d]
                prev = S[a, b, 0, d]
                for i in range(G - 1):
                    nxt = S[a, b, i + 1, d]
                    _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                    prev = nxt
    # scan axis 3
    for a in range(G):
        for b in range(G):
            for c in range(G):
                pf0 = f0[a] * f0[b] * f0[c]
                pf1 = f1[a] * f1[b] * f1[c]
                prev = S[a, b, c, 0]
                for i in range(G - 1):
                    nxt = S[a, b, c, i + 1]
                    _seg(prev, nxt, p, u[i], u[i + 1], c1, c2, pf0, pf1, acc)
                    prev = nxt
    return acc[0] / 4.0, acc[1] / 4.0


# ---------------------------------------------------------------------
# Phi maps (parallel over sorted cells)
# ---------------------------------------------------------------------

def _make_phi(K, contour):
    @njit(parallel=True)
    def phi(P_sorted, P_full, tuples, u, f0, f1, gamma, W, c1, c2):
        n = tuples.shape[0]
        out = np.empty(n)
        for s in prange(n):
            p = P_sorted[s]
            mu_vec = np.empty(K)
            cache_idx = np.full(K, -1, dtype=np.int64)
            cache_mu = np.empty(K)
            nc = 0
            for k in range(K):
                ik = tuples[s, k]
                found = -1
                for c in range(nc):
                    if cache_idx[c] == ik:
                        found = c
                        break
                if found >= 0:
                    mu_vec[k] = cache_mu[found]
                    continue
                A0, A1 = contour(P_full[ik], p, u, f0, f1, c1, c2)
                num = f1[ik] * A1
                den = f0[ik] * A0 + num
                mu = num / den if den > 0.0 else 0.5
                if mu < EPS:
                    mu = EPS
                elif mu > 1.0 - EPS:
                    mu = 1.0 - EPS
                cache_idx[nc] = ik
                cache_mu[nc] = mu
                nc += 1
                mu_vec[k] = mu
            out[s] = _clear(mu_vec, gamma, W)
        return out
    return phi


_phi_K3 = _make_phi(3, _contour2)
_phi_K4 = _make_phi(4, _contour3)
_phi_K5 = _make_phi(5, _contour4)
_PHI = {3: _phi_K3, 4: _phi_K4, 5: _phi_K5}


# =====================================================================
# Public model wrapper
# =====================================================================

class Model:
    def __init__(self, G: int, K: int, tau: float, gamma: float,
                 W: float = 1.0, u_max: float = 4.0):
        self.G, self.K, self.tau, self.gamma, self.W = G, K, tau, gamma, W
        self.u = np.linspace(-u_max, u_max, G)
        self.sg, self.inv_map, self.mult = build_machinery(G, K)
        self.c1 = math.sqrt(tau / (2.0 * math.pi))
        self.c2 = 0.5 * tau
        self.f0 = self.c1 * np.exp(-self.c2 * (self.u + 0.5) ** 2)
        self.f1 = self.c1 * np.exp(-self.c2 * (self.u - 0.5) ** 2)
        self.phi_fn = _PHI[K]
        self.n_evals = 0

    def phi(self, P_sorted: np.ndarray) -> np.ndarray:
        P_full = inflate(P_sorted, self.inv_map, self.G, self.K)
        self.n_evals += 1
        return self.phi_fn(P_sorted, P_full, self.sg.tuples, self.u,
                           self.f0, self.f1, self.gamma, self.W,
                           self.c1, self.c2)

    def init_no_learning(self) -> np.ndarray:
        out = np.empty(self.sg.n)
        for s in range(self.sg.n):
            t = self.sg.tuples[s]
            mu_vec = np.clip(1.0 / (1.0 + np.exp(-self.tau * self.u[t])),
                             EPS, 1 - EPS)
            out[s] = _clear(mu_vec, self.gamma, self.W)
        return out

    # ------------------------------------------------------------------
    def solve(self, P0=None, f_tol=1e-10, maxiter=60, picard_pre=5,
              verbose=False):
        """Newton-Krylov with short damped-Picard pre-smoothing.

        Returns (P, F_inf, n_evals, converged)."""
        from scipy.optimize import newton_krylov
        try:
            from scipy.optimize import NoConvergence
        except ImportError:
            from scipy.optimize._nonlin import NoConvergence

        P = self.init_no_learning() if P0 is None else P0.copy()
        for _ in range(picard_pre):
            P = 0.5 * P + 0.5 * self.phi(P)

        best = {'P': P, 'F': float(np.max(np.abs(self.phi(P) - P)))}

        def F(x):
            r = self.phi(x) - x
            f = float(np.max(np.abs(r)))
            if f < best['F']:
                best['F'] = f
                best['P'] = x.copy()
            return r

        converged = True
        try:
            P = newton_krylov(F, P, f_tol=f_tol, maxiter=maxiter,
                              method='lgmres', inner_m=40, verbose=verbose)
            Finf = float(np.max(np.abs(self.phi(P) - P)))
        except (NoConvergence, ValueError, Exception):
            converged = False
            P = best['P']
            Finf = best['F']
        if Finf > best['F']:
            P, Finf = best['P'], best['F']
        if Finf > f_tol:
            converged = False
        return P, Finf, self.n_evals, converged

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def deficit_slope(self, P_sorted: np.ndarray):
        """Unweighted (over the full cube) 1-R^2 of logit(P) on T = sum u_k.
        Matches emin15/lowtau certified K=3 convention."""
        Pc = np.clip(P_sorted, EPS, 1 - EPS)
        y = np.log(Pc / (1 - Pc))
        T = self.u[self.sg.tuples].sum(axis=1)
        w = self.mult
        sw = np.sqrt(w)
        a = np.polyfit(T, y, 1, w=sw)
        pred = a[0] * T + a[1]
        ybar = np.average(y, weights=w)
        var_res = np.average((y - pred) ** 2, weights=w)
        var_tot = np.average((y - ybar) ** 2, weights=w)
        return float(var_res / max(var_tot, 1e-30)), float(a[0])

    def econ_metrics(self, P_sorted: np.ndarray, n_bins: int = 200):
        """TV + value-of-information decomposition, port of
        lowtau_metrics.cell_metrics to sorted-K storage.

        Welfare under the TRUE measure mu_FR; posteriors by bin
        aggregation: price-only (price bin) and informed (own signal
        index x price bin)."""
        G, K, gamma, W = self.G, self.K, self.gamma, self.W
        sg = self.sg
        tup = sg.tuples
        mult = self.mult

        pf0 = np.prod(self.f0[tup], axis=1)
        pf1 = np.prod(self.f1[tup], axis=1)
        w_s = 0.5 * mult * (pf0 + pf1)          # cell weight incl. multiplicity
        mu_FR = pf1 / np.maximum(pf0 + pf1, 1e-30)

        # quantile bins over the FULL-cube price distribution
        P_rep = np.repeat(P_sorted, mult.astype(np.int64))
        edges = np.quantile(P_rep, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        bin_idx = np.clip(np.searchsorted(edges, P_sorted, side='right') - 1,
                          0, n_bins - 1)

        # price-only posterior per bin (weights with multiplicity)
        b1 = np.bincount(bin_idx, weights=mult * pf1, minlength=n_bins)
        b0 = np.bincount(bin_idx, weights=mult * pf0, minlength=n_bins)
        mu_PR_bin = np.clip(b1 / np.maximum(b0 + b1, 1e-30), EPS, 1 - EPS)

        # informed posterior per (own index, bin): each slot k of cell s
        # contributes weight mult/K (per-agent share of the permutations)
        pair = (tup * n_bins + bin_idx[:, None])           # (n, K)
        i1 = np.bincount(pair.ravel(), weights=np.repeat(mult / K * pf1, K),
                         minlength=G * n_bins)
        i0 = np.bincount(pair.ravel(), weights=np.repeat(mult / K * pf0, K),
                         minlength=G * n_bins)
        mu_inf_pair = np.clip(i1 / np.maximum(i0 + i1, 1e-30), EPS, 1 - EPS)

        tv = eu_inf = eu_pr = eu_fr = eu_prior = 0.0
        wsum = 0.0
        for s in range(sg.n):
            p_eq = float(np.clip(P_sorted[s], EPS, 1 - EPS))
            mfr = float(mu_FR[s])
            ws = float(w_s[s])
            wsum += ws

            def trade_eu(mu):
                x = _x_crra_py(mu, p_eq, gamma, W)
                W1 = W + x * (1 - p_eq)
                W0 = W - x * p_eq
                return x, mfr * _crra_u_py(W1, gamma) + (1 - mfr) * _crra_u_py(W0, gamma)

            # informed: average over the K agent slots
            eu_i_cell = 0.0
            tv_cell = 0.0
            for k in range(K):
                mu_i = float(mu_inf_pair[pair[s, k]])
                x_i, eu_i = trade_eu(mu_i)
                eu_i_cell += eu_i / K
                tv_cell += abs(x_i) / K
            mu_p = float(mu_PR_bin[bin_idx[s]])
            _, eu_p = trade_eu(mu_p)
            _, eu_f = trade_eu(mfr)
            _, eu_0 = trade_eu(0.5)

            tv += ws * tv_cell
            eu_inf += ws * eu_i_cell
            eu_pr += ws * eu_p
            eu_fr += ws * eu_f
            eu_prior += ws * eu_0

        CE_informed = _crra_inv_py(eu_inf / wsum, gamma)
        CE_priceonly = _crra_inv_py(eu_pr / wsum, gamma)
        CE_FR = _crra_inv_py(eu_fr / wsum, gamma)
        CE_prior = _crra_inv_py(eu_prior / wsum, gamma)
        return dict(
            TV=float(tv / wsum),
            CE_informed=float(CE_informed), CE_priceonly=float(CE_priceonly),
            CE_FR=float(CE_FR), CE_prior=float(CE_prior),
            Vi_private=float(CE_informed - CE_priceonly),
            Vi_public=float(CE_priceonly - CE_prior),
            Vi_total=float(CE_informed - CE_prior),
            Vi_FR_gap=float(CE_FR - CE_informed),
        )


# =====================================================================
# Smooth-kernel halo operator (the certified emin15/lowtau family),
# generalised to symmetric K.
#
#   A_v = sum_{slice cells} exp(-(P_slice - p)^2 / (2 h^2)) * prod f_v(u)
#   h = C * sqrt(du),  C = 0.45  (joint-limit schedule)
#   pad = 2 halo fixed at the no-learning price.
#
# The slice for the agent with signal index i is P_full[i] (axis 0, by
# permutation symmetry) and is itself S_{K-1}-symmetric, so the kernel
# sum runs over sorted (K-1)-tuples with multiplicity weights:
# n_{K-1} = C(G_full+K-2, K-1) cells instead of G_full^{K-1}.
# =====================================================================

@njit(parallel=True, cache=True)
def _phi_smooth_kernel(x, P_sorted, inner_tuples, slice_map, wf0, wf1,
                       f0, f1, gamma, W, h):
    n = inner_tuples.shape[0]
    K = inner_tuples.shape[1]
    m = wf0.shape[0]
    inv2h2 = 0.5 / (h * h)
    out = np.empty(n)
    for s in prange(n):
        p = x[s]
        mu_vec = np.empty(K)
        cache_idx = np.full(K, -1, dtype=np.int64)
        cache_mu = np.empty(K)
        nc = 0
        for k in range(K):
            ik = inner_tuples[s, k]
            found = -1
            for c in range(nc):
                if cache_idx[c] == ik:
                    found = c
                    break
            if found >= 0:
                mu_vec[k] = cache_mu[found]
                continue
            A0 = 0.0
            A1 = 0.0
            for j in range(m):
                d = P_sorted[slice_map[ik, j]] - p
                w = math.exp(-d * d * inv2h2)
                A0 += w * wf0[j]
                A1 += w * wf1[j]
            num = f1[ik] * A1
            den = f0[ik] * A0 + num
            mu = num / den if den > 0.0 else 0.5
            if mu < EPS:
                mu = EPS
            elif mu > 1.0 - EPS:
                mu = 1.0 - EPS
            cache_idx[nc] = ik
            cache_mu[nc] = mu
            nc += 1
            mu_vec[k] = mu
        out[s] = _clear(mu_vec, gamma, W)
    return out


class SmoothModel:
    """Symmetric-K port of the emin15/lowtau certified operator:
    phi_K*_halo_smooth with h = C*sqrt(du), pad-2 no-learning halo,
    inner cells as unknowns."""

    PAD = 2
    C = 0.45
    _MACH_CACHE: dict = {}

    def __init__(self, G_inner: int, K: int, tau: float, gamma: float,
                 W: float = 1.0, u_max: float = 4.0):
        self.Gi, self.K, self.tau, self.gamma, self.W = G_inner, K, tau, gamma, W
        pad = self.PAD
        du = 2.0 * u_max / (G_inner - 1)
        self.du = du
        self.h = self.C * math.sqrt(du)
        Gf = G_inner + 2 * pad
        self.Gf = Gf
        self.lo, self.hi = pad, pad + G_inner
        self.u_full = np.array([-u_max + (q - pad) * du for q in range(Gf)])

        c1 = math.sqrt(tau / (2.0 * math.pi))
        c2 = 0.5 * tau
        self.f0 = c1 * np.exp(-c2 * (self.u_full + 0.5) ** 2)
        self.f1 = c1 * np.exp(-c2 * (self.u_full - 0.5) ** 2)

        # full sorted grid over Gf, and inner subset (cached across cells)
        ck = (Gf, K, self.lo, self.hi)
        if ck not in SmoothModel._MACH_CACHE:
            sg = SymGrid.build(Gf, K)
            tup = sg.tuples
            inner_mask = np.all((tup >= self.lo) & (tup < self.hi), axis=1)
            inner_ids = np.nonzero(inner_mask)[0]
            inner_tuples = tup[inner_ids].copy()
            mult_inner = np.array(
                [sg.multiplicity(int(i)) for i in inner_ids], dtype=np.float64)
            sg_km1 = SymGrid.build(Gf, K - 1)
            tup1 = sg_km1.tuples
            mult1 = np.array([sg_km1.multiplicity(s)
                              for s in range(sg_km1.n)], dtype=np.float64)
            powers = Gf ** np.arange(K - 1, -1, -1)
            code2sorted = np.full(Gf ** K, -1, dtype=np.int64)
            code2sorted[tup @ powers] = np.arange(sg.n)
            sm = np.empty((Gf, sg_km1.n), dtype=np.int64)
            for i in range(Gf):
                ext = np.concatenate(
                    [np.full((sg_km1.n, 1), i, dtype=np.int64), tup1], axis=1)
                ext.sort(axis=1)
                sm[i] = code2sorted[ext @ powers]
            assert sm.min() >= 0
            SmoothModel._MACH_CACHE[ck] = (
                sg, inner_ids, inner_tuples, mult_inner, sg_km1, mult1, sm)
        (self.sg, self.inner_ids, self.inner_tuples, self.mult_inner,
         self.sg_km1, mult1, self.slice_map) = SmoothModel._MACH_CACHE[ck]
        tup = self.sg.tuples
        tup1 = self.sg_km1.tuples
        self.n_inner = len(self.inner_ids)

        self.wf0 = mult1 * np.prod(self.f0[tup1], axis=1)
        self.wf1 = mult1 * np.prod(self.f1[tup1], axis=1)

        # fixed base: no-learning price at every sorted cell
        self.P_base = np.empty(self.sg.n)
        for s in range(self.sg.n):
            mu_vec = np.clip(
                1.0 / (1.0 + np.exp(-tau * self.u_full[tup[s]])), EPS, 1 - EPS)
            self.P_base[s] = _clear(mu_vec, gamma, W)

        self.n_evals = 0

    # ------------------------------------------------------------------
    def phi(self, x: np.ndarray) -> np.ndarray:
        P_sorted = self.P_base.copy()
        P_sorted[self.inner_ids] = x
        self.n_evals += 1
        return _phi_smooth_kernel(x, P_sorted, self.inner_tuples,
                                  self.slice_map, self.wf0, self.wf1,
                                  self.f0, self.f1, self.gamma, self.W,
                                  self.h)

    def init_no_learning(self) -> np.ndarray:
        return self.P_base[self.inner_ids].copy()

    def solve(self, P0=None, f_tol=1e-10, maxiter=60, picard_pre=3,
              verbose=False):
        from scipy.optimize import newton_krylov
        try:
            from scipy.optimize import NoConvergence
        except ImportError:
            from scipy.optimize._nonlin import NoConvergence

        x = self.init_no_learning() if P0 is None else P0.copy()
        for _ in range(picard_pre):
            x = 0.5 * x + 0.5 * self.phi(x)
        best = {'P': x, 'F': float(np.max(np.abs(self.phi(x) - x)))}

        def F(z):
            r = self.phi(z) - z
            f = float(np.max(np.abs(r)))
            if f < best['F']:
                best['F'] = f
                best['P'] = z.copy()
            return r

        converged = True
        try:
            x = newton_krylov(F, x, f_tol=f_tol, maxiter=maxiter,
                              method='lgmres', verbose=verbose)
            Finf = float(np.max(np.abs(self.phi(x) - x)))
        except Exception:
            converged = False
            x, Finf = best['P'], best['F']
        if Finf > best['F']:
            x, Finf = best['P'], best['F']
        if Finf > f_tol:
            converged = False
        return x, Finf, self.n_evals, converged

    # ------------------------------------------------------------------
    def deficit_slope(self, x: np.ndarray):
        """emin15 metric: unweighted (full inner cube) 1-R^2 of logit(P)
        on T = sum u_k."""
        Pc = np.clip(x, EPS, 1 - EPS)
        y = np.log(Pc / (1 - Pc))
        T = self.u_full[self.inner_tuples].sum(axis=1)
        w = self.mult_inner
        a = np.polyfit(T, y, 1, w=np.sqrt(w))
        pred = a[0] * T + a[1]
        ybar = np.average(y, weights=w)
        var_res = np.average((y - pred) ** 2, weights=w)
        var_tot = np.average((y - ybar) ** 2, weights=w)
        return float(var_res / max(var_tot, 1e-30)), float(a[0])

    def econ_metrics(self, x: np.ndarray, n_bins: int = 200):
        """Port of lowtau_metrics.cell_metrics (bin-aggregated posteriors,
        welfare under mu_FR) to sorted-K inner storage."""
        K, gamma, W = self.K, self.gamma, self.W
        tup = self.inner_tuples
        mult = self.mult_inner
        Gf = self.Gf

        pf0 = np.prod(self.f0[tup], axis=1)
        pf1 = np.prod(self.f1[tup], axis=1)
        w_s = 0.5 * mult * (pf0 + pf1)
        mu_FR = pf1 / np.maximum(pf0 + pf1, 1e-30)

        P_rep = np.repeat(x, mult.astype(np.int64))
        edges = np.quantile(P_rep, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-12
        edges[-1] += 1e-12
        bin_idx = np.clip(np.searchsorted(edges, x, side='right') - 1,
                          0, n_bins - 1)

        b1 = np.bincount(bin_idx, weights=mult * pf1, minlength=n_bins)
        b0 = np.bincount(bin_idx, weights=mult * pf0, minlength=n_bins)
        mu_PR_bin = np.clip(b1 / np.maximum(b0 + b1, 1e-30), EPS, 1 - EPS)

        pair = tup * n_bins + bin_idx[:, None]
        i1 = np.bincount(pair.ravel(), weights=np.repeat(mult / K * pf1, K),
                         minlength=Gf * n_bins)
        i0 = np.bincount(pair.ravel(), weights=np.repeat(mult / K * pf0, K),
                         minlength=Gf * n_bins)
        mu_inf_pair = np.clip(i1 / np.maximum(i0 + i1, 1e-30), EPS, 1 - EPS)

        tv = eu_inf = eu_pr = eu_fr = eu_prior = wsum = 0.0
        for s in range(self.n_inner):
            p_eq = float(np.clip(x[s], EPS, 1 - EPS))
            mfr = float(mu_FR[s])
            ws = float(w_s[s])
            wsum += ws

            def trade_eu(mu):
                xx = _x_crra_py(mu, p_eq, gamma, W)
                W1 = W + xx * (1 - p_eq)
                W0 = W - xx * p_eq
                return xx, (mfr * _crra_u_py(W1, gamma)
                            + (1 - mfr) * _crra_u_py(W0, gamma))

            eu_i_cell = tv_cell = 0.0
            for k in range(K):
                mu_i = float(mu_inf_pair[pair[s, k]])
                x_i, eu_i = trade_eu(mu_i)
                eu_i_cell += eu_i / K
                tv_cell += abs(x_i) / K
            _, eu_p = trade_eu(float(mu_PR_bin[bin_idx[s]]))
            _, eu_f = trade_eu(mfr)
            _, eu_0 = trade_eu(0.5)

            tv += ws * tv_cell
            eu_inf += ws * eu_i_cell
            eu_pr += ws * eu_p
            eu_fr += ws * eu_f
            eu_prior += ws * eu_0

        CE_informed = _crra_inv_py(eu_inf / wsum, gamma)
        CE_priceonly = _crra_inv_py(eu_pr / wsum, gamma)
        CE_FR = _crra_inv_py(eu_fr / wsum, gamma)
        CE_prior = _crra_inv_py(eu_prior / wsum, gamma)
        return dict(
            TV=float(tv / wsum),
            CE_informed=float(CE_informed), CE_priceonly=float(CE_priceonly),
            CE_FR=float(CE_FR), CE_prior=float(CE_prior),
            Vi_private=float(CE_informed - CE_priceonly),
            Vi_public=float(CE_priceonly - CE_prior),
            Vi_total=float(CE_informed - CE_prior),
            Vi_FR_gap=float(CE_FR - CE_informed),
        )


def _x_crra_py(mu, p, gamma, W):
    mu = max(EPS, min(1 - EPS, mu))
    p = max(EPS, min(1 - EPS, p))
    z = (math.log(mu / (1 - mu)) - math.log(p / (1 - p))) / gamma
    if z >= 0:
        e = math.exp(-z)
        return W * (1 - e) / ((1 - p) * e + p)
    e = math.exp(z)
    return W * (e - 1) / ((1 - p) + p * e)


def _crra_u_py(w, gamma):
    if w <= 0:
        return -1e18
    if abs(gamma - 1) < 1e-9:
        return math.log(w)
    return w ** (1 - gamma) / (1 - gamma)


def _crra_inv_py(eu, gamma):
    if abs(gamma - 1) < 1e-9:
        return math.exp(eu)
    val = eu * (1 - gamma)
    if val <= 0:
        return 1e-18
    return val ** (1 / (1 - gamma))
