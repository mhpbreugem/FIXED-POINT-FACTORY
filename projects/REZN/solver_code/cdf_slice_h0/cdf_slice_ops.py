"""STRICT h=0 fixed-point operator for the K=3 CRRA REE model via the
CDF-slice construction.

Replaces the kernel-smoothed evidence A_v(p) = sum K_h(P-p) w_v of
phi_K3_halo_smooth by the EXACT (h identically zero) co-area density

    A_v(p) = d/dp G_v(p),
    G_v(p) = sum_cells [cell-centered weight w_v * cell area]
             * [area fraction of the cell where the interpolated
                surface S <= p],

with each rectangular cell split into two linear triangles.  For a
linear triangle the sub-level area fraction is a closed-form piecewise
quadratic in p (C^1), so G_v is exactly computable, monotone, and C^1.
A_v is obtained two ways:

  hat   : the exact analytic derivative of G_v (piecewise linear,
          continuous "hat" sums) -- used for unit tests / diagnostics.
  cheb  : per-segment Chebyshev fits of G_v between detected knots
          (slice-critical values of S), differentiated ANALYTICALLY
          (chebder).  This is the operator's evidence: h = 0 with the
          dense non-critical C^1 micro-kinks smoothed by the fit, and
          genuine knots (topology changes of the level set) respected.

Bayes posterior and CRRA market clearing are imported unchanged from
the reference package (reznsrc), so the operator differs ONLY in the
evidence computation.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.polynomial import chebyshev as Cheb

REZNSRC_PARENT = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                  'solved_fixed_points/k3_coarea_2dsweep')
if REZNSRC_PARENT not in sys.path:
    sys.path.insert(0, REZNSRC_PARENT)

from numba import njit                                    # noqa: E402
from reznsrc.signals import f_signal, lam                 # noqa: E402,F401
from reznsrc.demand import clear_crra, EPS_PRICE          # noqa: E402
from reznsrc.contour_K3_halo import (init_no_learning_K3,  # noqa: E402,F401
                                     phi_K3_halo_smooth)

TINY = 1.0e-300


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------

def build_grid(Gi: int, UMAX: float = 4.0, pad: int = 2):
    """Return (du, u_full, inner_lo, inner_hi) with the halo convention
    of the reference operator: inner grid linspace(-UMAX, UMAX, Gi),
    halo of `pad` cells with the same spacing on each side."""
    du = 2.0 * UMAX / (Gi - 1)
    Gf = Gi + 2 * pad
    u_full = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
    return du, u_full, pad, pad + Gi


# ----------------------------------------------------------------------
# triangulation of one 2-D slice
# ----------------------------------------------------------------------

def make_tris(S: np.ndarray) -> np.ndarray:
    """Split each grid cell into two linear triangles along the
    (+1,+1) diagonal; return sorted vertex values, shape (Ntri, 3).

    Order: first all lower triangles (c00, c10, c11) in cell raveled
    order, then all upper triangles (c00, c01, c11)."""
    c00 = S[:-1, :-1]
    c10 = S[1:, :-1]
    c01 = S[:-1, 1:]
    c11 = S[1:, 1:]
    t1 = np.stack([c00, c10, c11], axis=-1).reshape(-1, 3)
    t2 = np.stack([c00, c01, c11], axis=-1).reshape(-1, 3)
    tri = np.concatenate([t1, t2], axis=0)
    tri.sort(axis=1)
    return tri


def tri_weights(u_full: np.ndarray, tau_a: float, tau_b: float,
                du: float) -> np.ndarray:
    """Cell-centered weights w_v(center) * cell area, distributed half
    to each triangle, ordered to match make_tris.  Returns (Ntri, 2)
    with columns (v=0, v=1)."""
    uc = 0.5 * (u_full[:-1] + u_full[1:])
    out = []
    for v in (0, 1):
        m = v - 0.5
        fa = np.sqrt(tau_a / (2 * np.pi)) * np.exp(-0.5 * tau_a * (uc - m) ** 2)
        fb = np.sqrt(tau_b / (2 * np.pi)) * np.exp(-0.5 * tau_b * (uc - m) ** 2)
        cw = np.outer(fa, fb).ravel() * du * du * 0.5
        out.append(np.concatenate([cw, cw]))
    return np.stack(out, axis=1)


# ----------------------------------------------------------------------
# exact weighted CDF G_v(p) and its exact "hat" derivative
# ----------------------------------------------------------------------

def cdf_eval(tri: np.ndarray, Wt: np.ndarray, p: np.ndarray,
             chunk: int = 2_000_000) -> np.ndarray:
    """Exact weighted CDF G(p) = sum_tri Wt * frac(p), vectorized.

    tri : (Ntri, 3) sorted vertex values
    Wt  : (Ntri, nw) per-triangle weights (columns = weight fields)
    p   : (m,) evaluation points
    returns (m, nw)."""
    Ntri = tri.shape[0]
    m = p.size
    out = np.zeros((m, Wt.shape[1]))
    step = max(1, chunk // max(m, 1))
    for a in range(0, Ntri, step):
        b = min(Ntri, a + step)
        s1 = tri[a:b, 0:1]
        s2 = tri[a:b, 1:2]
        s3 = tri[a:b, 2:3]
        P = p[None, :]
        d31 = s3 - s1
        den_lo = np.maximum((s2 - s1) * d31, TINY)
        den_hi = np.maximum((s3 - s2) * d31, TINY)
        below = P <= s1
        above = P >= s3
        mid_lo = (P > s1) & (P < s2)
        frac = np.where(
            below, 0.0,
            np.where(above, 1.0,
                     np.where(mid_lo,
                              (P - s1) ** 2 / den_lo,
                              1.0 - (s3 - P) ** 2 / den_hi)))
        out += frac.T @ Wt[a:b]
    return out


def hat_eval(tri: np.ndarray, Wt: np.ndarray, p: np.ndarray,
             chunk: int = 2_000_000) -> np.ndarray:
    """Exact analytic derivative A(p) = dG/dp (piecewise-linear hat
    sums).  Degenerate (flat) triangles contribute 0 (their exact
    contribution is an atom; see module docstring).  Returns (m, nw)."""
    Ntri = tri.shape[0]
    m = p.size
    out = np.zeros((m, Wt.shape[1]))
    step = max(1, chunk // max(m, 1))
    for a in range(0, Ntri, step):
        b = min(Ntri, a + step)
        s1 = tri[a:b, 0:1]
        s2 = tri[a:b, 1:2]
        s3 = tri[a:b, 2:3]
        P = p[None, :]
        d31 = s3 - s1
        den_lo = np.maximum((s2 - s1) * d31, TINY)
        den_hi = np.maximum((s3 - s2) * d31, TINY)
        mid_lo = (P > s1) & (P < s2)
        mid_hi = (P >= s2) & (P < s3)
        g = np.where(mid_lo, 2.0 * (P - s1) / den_lo,
                     np.where(mid_hi, 2.0 * (s3 - P) / den_hi, 0.0))
        out += g.T @ Wt[a:b]
    return out


# ----------------------------------------------------------------------
# knot detection (slice-critical values of the triangulated field)
# ----------------------------------------------------------------------

# 6-neighborhood of the (+1,+1)-diagonal triangulation, cyclic order.
_SHIFTS = ((1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1))


def detect_knots(S: np.ndarray) -> np.ndarray:
    """Values of S at vertices where the level-set topology can change:
    local minima / maxima / saddles of S on the triangulated grid graph
    (interior: cyclic sign-change count of the 6 neighbor differences
    != 2), plus boundary local extrema and the 4 corners."""
    Gf, Hf = S.shape
    C = S[1:-1, 1:-1]
    b = np.stack([S[1 + da:Gf - 1 + da, 1 + db:Hf - 1 + db] > C
                  for (da, db) in _SHIFTS])
    changes = np.sum(b != np.roll(b, -1, axis=0), axis=0)
    vals = [C[changes != 2].ravel()]

    # boundary vertices: local extremum among available 6-stencil
    # neighbors (or corner).
    bvals = []
    for (a0, b0) in _boundary_indices(Gf, Hf):
        diffs = []
        for (da, db) in _SHIFTS:
            a1, b1 = a0 + da, b0 + db
            if 0 <= a1 < Gf and 0 <= b1 < Hf:
                diffs.append(S[a1, b1] - S[a0, b0])
        diffs = np.array(diffs)
        corner = (a0 in (0, Gf - 1)) and (b0 in (0, Hf - 1))
        if corner or np.all(diffs >= 0.0) or np.all(diffs <= 0.0):
            bvals.append(S[a0, b0])
    if bvals:
        vals.append(np.array(bvals))
    return np.concatenate(vals)


def _boundary_indices(Gf: int, Hf: int):
    for b0 in range(Hf):
        yield 0, b0
        yield Gf - 1, b0
    for a0 in range(1, Gf - 1):
        yield a0, 0
        yield a0, Hf - 1


def build_edges(knot_vals: np.ndarray, smin: float, smax: float,
                min_seg: float = 1.0e-9, lmax: float | None = None,
                nsub: int = 24) -> np.ndarray:
    """Segment edges: sorted knots in (smin, smax), short segments
    (< min_seg) merged, long segments subdivided so no segment exceeds
    lmax (default (smax-smin)/nsub) -- keeps the per-segment Chebyshev
    fits local."""
    rng = smax - smin
    if lmax is None:
        lmax = rng / nsub
    k = np.unique(knot_vals)
    k = k[(k > smin + min_seg) & (k < smax - min_seg)]
    edges = [smin]
    for v in k:
        if v - edges[-1] >= min_seg:
            edges.append(v)
    if smax - edges[-1] < min_seg:
        edges[-1] = smax
    else:
        edges.append(smax)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, int(np.ceil((b - a) / lmax)))
        out.extend(np.linspace(a, b, n + 1)[:-1])
    out.append(smax)
    return np.asarray(out)


# ----------------------------------------------------------------------
# per-segment Chebyshev evidence
# ----------------------------------------------------------------------

def refine_edges(edges0: np.ndarray, cdf_fn, lmax: float,
                 min_len: float, mtol: float,
                 maxdepth: int = 60) -> np.ndarray:
    """Mass-adaptive bisection of the knot intervals.

    A segment is split while it is longer than lmax, or while it
    carries more than mtol of the total CDF mass of EITHER weight
    column and is longer than 2*min_len.  This resolves near-atomic
    value concentrations (e.g. demand-cap plateaus of the price
    surface) that have no critical vertices, down to width min_len."""
    G0 = cdf_fn(edges0)
    Gtot = np.maximum(G0[-1] - G0[0], TINY)
    cur = [(edges0[i], edges0[i + 1], G0[i], G0[i + 1])
           for i in range(len(edges0) - 1)]
    final = []
    for _ in range(maxdepth):
        split = []
        for seg in cur:
            a, b, Ga, Gb = seg
            L = b - a
            need = (L > lmax) or (L > 2.0 * min_len
                                  and float(np.max((Gb - Ga) / Gtot)) > mtol)
            (split if need else final).append(seg)
        if not split:
            cur = []
            break
        mids = np.array([0.5 * (s[0] + s[1]) for s in split])
        Gm = cdf_fn(mids)
        cur = []
        for s, m, gm in zip(split, mids, Gm):
            cur.append((s[0], m, s[2], gm))
            cur.append((m, s[1], gm, s[3]))
    final.extend(cur)                       # depth-capped leftovers
    final.sort(key=lambda s: s[0])
    return np.array([s[0] for s in final] + [final[-1][1]])


PCLIP = 1.0e-13          # clip for the logit transform (price slices)
_FIT_CACHE: dict = {}


def _fit_mats(deg: int, oversample: int):
    """Precomputed (sample nodes xi in [0,1], least-squares operator
    PINV mapping samples -> Chebyshev coefficients on [-1,1], and the
    Chebyshev differentiation matrix D)."""
    key = (deg, oversample)
    if key not in _FIT_CACHE:
        npts = oversample * deg + 1
        xi = 0.5 * (1.0 - np.cos(np.pi * np.arange(npts) / (npts - 1)))
        tcheb = 2.0 * xi - 1.0
        V = Cheb.chebvander(tcheb, deg)              # (npts, deg+1)
        PINV = np.linalg.pinv(V)                     # (deg+1, npts)
        D = np.zeros((deg, deg + 1))
        for k in range(deg + 1):
            e = np.zeros(deg + 1)
            e[k] = 1.0
            D[:, k] = Cheb.chebder(e)
        _FIT_CACHE[key] = (xi, PINV, D)
    return _FIT_CACHE[key]


def slice_evidence(S: np.ndarray, p_targets: np.ndarray, Wt: np.ndarray,
                   deg: int = 12, nsub: int = 24, oversample: int = 4,
                   min_seg: float = 1.0e-9, knot_tol: float = 1.0e-13,
                   transform: str = 'identity', qseg: float = 1.0,
                   mtol: float = 0.03,
                   stats: dict | None = None) -> np.ndarray:
    """Evidence A_v(p_targets) = dG_v/dp for one 2-D slice, h == 0.

    One exact-CDF build + per-segment Chebyshev fit (analytic
    derivative) serves all targets.  With transform='logit' the CDF is
    fitted as a function of q = logit(p) (price surfaces concentrate
    their value distribution near 0/1; in q the structure is well
    spread); the exact Jacobian dq/dp is applied afterwards (it cancels
    in the Bayes ratio anyway).  Targets exactly at a segment edge
    (within knot_tol) get the average of the left/right segment
    derivatives.  Returns (m, nw)."""
    nw = Wt.shape[1]
    m = p_targets.size
    if transform == 'logit':
        Sc = np.clip(S, PCLIP, 1.0 - PCLIP)
        Sw = np.log(Sc) - np.log1p(-Sc)
        pc = np.clip(p_targets, PCLIP, 1.0 - PCLIP)
        pw = np.log(pc) - np.log1p(-pc)
        jac = 1.0 / (pc * (1.0 - pc))
        lmax = qseg
    else:
        Sw = S
        pw = p_targets
        jac = None
        lmax = None

    tri = make_tris(Sw)
    smin = float(Sw.min())
    smax = float(Sw.max())
    if smax - smin < 1.0e-12:
        # entirely flat slice (cannot happen for inner slices): the
        # pushforward is an atom; return zero density.
        return np.zeros((m, nw))

    knots = detect_knots(Sw)
    edges0 = build_edges(knots, smin, smax, min_seg=min_seg, lmax=np.inf)
    lmax_val = lmax if lmax is not None else (smax - smin) / nsub
    edges = refine_edges(edges0, lambda x: cdf_eval(tri, Wt, np.asarray(x)),
                         lmax_val, min_seg, mtol)
    nseg = edges.size - 1

    # sample exact CDF at Chebyshev-Lobatto points of every segment
    xi, PINV, D = _fit_mats(deg, oversample)
    seg_len = edges[1:] - edges[:-1]
    Xs = edges[:-1, None] + seg_len[:, None] * xi[None, :]
    Gv = cdf_eval(tri, Wt, Xs.ravel()).reshape(nseg, xi.size, nw)

    # batch least-squares fit + analytic differentiation
    C = np.einsum('dn,snw->sdw', PINV, Gv)           # (nseg, deg+1, nw)
    dcoefs = np.einsum('ek,skw->sew', D, C)          # (nseg, deg, nw)
    dcoefs *= (2.0 / seg_len)[:, None, None]

    # minimum-length segments still carrying > mtol of the mass are
    # unresolvable near-atoms (value ties below the 1e-9 scale): use
    # the atom-average density (centered finite difference of the
    # EXACT CDF across the segment).  Documented epsilon ~ min_seg.
    segmass = Gv[:, -1, :] - Gv[:, 0, :]
    Gtot = np.maximum(Gv[-1, -1, :] - Gv[0, 0, :], TINY)
    atomic = ((seg_len <= 2.05 * min_seg)
              & ((segmass / Gtot).max(axis=1) > mtol))
    n_atomic = int(np.sum(atomic))
    if n_atomic:
        dcoefs[atomic] = 0.0
        dcoefs[atomic, 0, :] = segmass[atomic] / seg_len[atomic, None]

    # evaluate targets
    p = np.clip(pw, smin, smax)
    sidx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, nseg - 1)
    A = np.empty((m, nw))
    for s in np.unique(sidx):
        msk = sidx == s
        t = 2.0 * (p[msk] - edges[s]) / seg_len[s] - 1.0
        A[msk] = Cheb.chebval(t, dcoefs[s]).T

    # targets at interior segment edges: average left/right derivative
    j = np.searchsorted(edges, p)
    n_at_knot = 0
    for q in range(m):
        for e in (j[q] - 1, j[q]):
            if 1 <= e <= nseg - 1 and abs(p[q] - edges[e]) <= knot_tol:
                left = Cheb.chebval(1.0, dcoefs[e - 1])
                right = Cheb.chebval(-1.0, dcoefs[e])
                A[q] = 0.5 * (left + right)
                n_at_knot += 1
                break

    # --- exact-hat fallback -------------------------------------------
    # (a) targets at the slice min/max: the true density vanishes there
    #     (generic extremum), so Bayes would hit 0/0.  The correct h->0
    #     posterior uses the one-sided LIMIT ratio, obtained exactly
    #     from the piecewise-linear hat density evaluated a relative
    #     1e-9 inside the support (the hat is linear there, so the
    #     ratio is the exact limit).
    # (b) targets where the fitted density is <= 0 in some column
    #     (fit wiggle in near-zero-density regions): replace both
    #     columns by the exact hat density.
    rng = smax - smin
    dlt = 1.0e-9 * rng
    is_lo = p <= smin + knot_tol
    is_hi = p >= smax - knot_tol
    need = (A <= 0.0).any(axis=1) | is_lo | is_hi
    n_end = int(np.sum(is_lo | is_hi))
    n_fix = int(np.sum(need)) - n_end
    if np.any(need):
        idx = np.nonzero(need)[0]
        q_eval = p[idx].copy()
        q_eval[is_lo[idx]] = smin + dlt
        q_eval[is_hi[idx]] = smax - dlt
        Ah = hat_eval(tri, Wt, q_eval)
        zero = ~(Ah > 0.0).any(axis=1)
        if np.any(zero):       # genuine zero-density point: two-sided
            qz = q_eval[zero]
            Ah[zero] = 0.5 * (hat_eval(tri, Wt, qz - dlt)
                              + hat_eval(tri, Wt, qz + dlt))
        A[idx] = Ah

    if jac is not None:
        A *= jac[:, None]
    if stats is not None:
        stats['nseg'] = stats.get('nseg', 0) + nseg
        stats['nslice'] = stats.get('nslice', 0) + 1
        stats['n_at_knot'] = stats.get('n_at_knot', 0) + n_at_knot
        stats['n_endpoint'] = stats.get('n_endpoint', 0) + n_end
        stats['n_hatfix'] = stats.get('n_hatfix', 0) + n_fix
        stats['n_atomic'] = stats.get('n_atomic', 0) + n_atomic
        stats['max_nseg'] = max(stats.get('max_nseg', 0), nseg)
    return A


# ----------------------------------------------------------------------
# Bayes + clearing (identical to reference)
# ----------------------------------------------------------------------

def bayes_vec(f0: float, f1: float, A0: np.ndarray,
              A1: np.ndarray) -> np.ndarray:
    """Vectorized replica of reznsrc.contour_K3_halo._bayes."""
    num = f1 * A1
    den = f0 * A0 + num
    mu = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.5)
    return np.clip(mu, EPS_PRICE, 1.0 - EPS_PRICE)


@njit(cache=True, fastmath=False)
def _clear_grid(mu_arr, gamma_vec, W_vec):
    Gi = mu_arr.shape[0]
    out = np.empty((Gi, Gi, Gi), dtype=np.float64)
    mu_vec = np.empty(3, dtype=np.float64)
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                mu_vec[0] = mu_arr[i, j, l, 0]
                mu_vec[1] = mu_arr[i, j, l, 1]
                mu_vec[2] = mu_arr[i, j, l, 2]
                out[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return out


# ----------------------------------------------------------------------
# the operator
# ----------------------------------------------------------------------

class CDFSliceOperator:
    """Phi_cdf(P_full) -> P_new with the same halo convention as
    phi_K3_halo_smooth: inner cells updated, halo untouched."""

    def __init__(self, Gi: int, tau_vec, gamma_vec, W_vec,
                 UMAX: float = 4.0, pad: int = 2,
                 deg: int = 12, nsub: int = 24, oversample: int = 4,
                 transform: str = 'logit', qseg: float = 1.0):
        self.Gi = Gi
        self.transform = transform
        self.qseg = qseg
        self.tau_vec = np.asarray(tau_vec, dtype=np.float64)
        self.gamma_vec = np.asarray(gamma_vec, dtype=np.float64)
        self.W_vec = np.asarray(W_vec, dtype=np.float64)
        self.deg = deg
        self.nsub = nsub
        self.oversample = oversample
        self.du, self.u_full, self.lo, self.hi = build_grid(Gi, UMAX, pad)
        self.halo = init_no_learning_K3(self.u_full, self.tau_vec,
                                        self.gamma_vec, self.W_vec)
        # per-agent triangle weights for the two slice axes
        ax = [(1, 2), (0, 2), (0, 1)]
        self.Wt = [tri_weights(self.u_full, self.tau_vec[a],
                               self.tau_vec[b], self.du) for a, b in ax]
        # own-signal densities on the full grid
        self.f_own = []
        for k in range(3):
            f0 = np.array([f_signal(u, 0, self.tau_vec[k])
                           for u in self.u_full])
            f1 = np.array([f_signal(u, 1, self.tau_vec[k])
                           for u in self.u_full])
            self.f_own.append((f0, f1))
        self.last_stats: dict = {}

    # -- full operator -------------------------------------------------

    def phi(self, P_full: np.ndarray) -> np.ndarray:
        Gi, lo, hi = self.Gi, self.lo, self.hi
        inner = slice(lo, hi)
        mu = np.empty((Gi, Gi, Gi, 3))
        st: dict = {}
        kw = dict(deg=self.deg, nsub=self.nsub,
                  oversample=self.oversample, stats=st,
                  transform=self.transform, qseg=self.qseg)

        for i in range(lo, hi):                      # agent 0
            A = slice_evidence(P_full[i], P_full[i, inner, inner].ravel(),
                               self.Wt[0], **kw)
            f0, f1 = self.f_own[0]
            mu[i - lo, :, :, 0] = bayes_vec(f0[i], f1[i], A[:, 0],
                                            A[:, 1]).reshape(Gi, Gi)
        for j in range(lo, hi):                      # agent 1
            A = slice_evidence(P_full[:, j, :],
                               P_full[inner, j, inner].ravel(),
                               self.Wt[1], **kw)
            f0, f1 = self.f_own[1]
            mu[:, j - lo, :, 1] = bayes_vec(f0[j], f1[j], A[:, 0],
                                            A[:, 1]).reshape(Gi, Gi)
        for l in range(lo, hi):                      # agent 2
            A = slice_evidence(P_full[:, :, l],
                               P_full[inner, inner, l].ravel(),
                               self.Wt[2], **kw)
            f0, f1 = self.f_own[2]
            mu[:, :, l - lo, 2] = bayes_vec(f0[l], f1[l], A[:, 0],
                                            A[:, 1]).reshape(Gi, Gi)

        P_new = P_full.copy()
        P_new[inner, inner, inner] = _clear_grid(mu, self.gamma_vec,
                                                 self.W_vec)
        self.last_stats = st
        return P_new

    # -- helpers for solving -------------------------------------------

    def embed(self, x: np.ndarray) -> np.ndarray:
        Pf = self.halo.copy()
        s = (slice(self.lo, self.hi),) * 3
        Pf[s] = x.reshape((self.Gi,) * 3)
        return Pf

    def residual(self, x: np.ndarray) -> np.ndarray:
        Pf = self.embed(x)
        s = (slice(self.lo, self.hi),) * 3
        return (self.phi(Pf) - Pf)[s].ravel()
