"""Zero-h FP solver for the K=3 CRRA REE.

Hypothesis (Finding 10 follow-up)
=================================
Per-segment Chebyshev (between detected critical-p_c knots) achieves machine
epsilon in-support. If we use that zero-h pipeline as the *inner lookup*
inside the Picard/Anderson FP iteration, the operator should become
Lipschitz-smooth (Cheby polynomials are smooth between knots), so the FP
iteration should converge cleanly -- breaking the |F| ~ 0.05 stall observed
in the kernel-band Lin-CDF R4 solver at high tau.

Operator definition
-------------------
Given current P_cube (shape (G,G,G)):

  1. mu_raw[ip, k] = build_mu_table_lin_strict(P_cube, ...)   # strict-h=0 samples
  2. p_c_values     = find_critical_points(P_cube, u_grid)    # cusp knots
  3. For each u_k slice mu_raw[:, k]:
         build piecewise Cheby on [p_min_sup, p_max_sup] segments + tails
         (per the zero_h_pipeline)
  4. mu_smooth[ip, k] = evaluate piecewise Cheby at p_grid[ip]
  5. For each cube cell (i, j, k):
         p_old   = P_cube[i, j, k]
         mu_k    = mu_smooth_eval(p_old, u_k)   for k in {i, j, k}
         P_new[i, j, k] = CRRA-clear(mu0, mu1, mu2, gamma)

The CRRA clearing is the same as in dd_k3_optB_numba.

Outer wrapper: Anderson + NK fallback (same pattern as dd_k3_strict_solve.py).

Failure-mode handling
---------------------
If critical points drift between iterations (the cusp positions move when P
moves), the lookup itself becomes a non-Lipschitz function of P. To control
this we offer:

  - mode="recompute"   : recompute p_c every iteration (most faithful, may
                         not converge if cusps wander).
  - mode="frozen"      : after `freeze_after` iterations, lock p_c to whatever
                         was detected at that point.
  - mode="pchip_across": use a single PCHIP across all knots per slice rather
                         than per-segment Cheby (less prone to ringing under
                         knot perturbation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

import numpy as np
import numpy.polynomial.chebyshev as cheb

sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")

os.environ.setdefault("NUMBA_NUM_THREADS", "6")

from lin_cdf_strict import (
    build_mu_table_lin_strict,
    make_cdf_uniform_grid,
    make_gl_for_u,
    make_p_grid,
)
from cheby_numba import crra_clear_jit
from dd_k3_critpts import find_critical_points, critical_p_values

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = "/home/user/FIXED-POINT-FACTORY"
FP_DIR = f"{ROOT}/projects/REZN/solved_fixed_points/dd_k3_overnight"
OUT_DIR = f"{FP_DIR}/zero_h_solver"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/figs", exist_ok=True)

G = 11
G_P = 121
NQ_STRICT = 16
DEG_IN = 8           # in-support Cheby degree per segment (deg=8, mss=2
                     # reaches machine eps in-support per Finding 10)
MIN_SAMPLES_PER_SEG = 2
TAIL_EPS = 1e-9
SUPPORT_LO = 1e-6    # lower threshold to call mu "in support"
SUPPORT_HI = 1 - 1e-6

# ---------------------------------------------------------------------------
# Per-segment Chebyshev fit (vector-friendly)
# ---------------------------------------------------------------------------
def _detect_support(p_grid, mu_slice,
                    threshold_low=SUPPORT_LO, threshold_high=SUPPORT_HI,
                    fallback_value=0.5, fallback_tol=1e-14):
    """Locate the in-support segment of mu_slice.

    Strict-h=0 build_mu_table_lin_strict emits exactly fallback_value (=0.5)
    where the co-area integrand has no roots in the cube. Those points
    carry zero information and we must NOT include them in the support.
    The true support is the contiguous block of p where mu is strictly
    between (threshold_low, threshold_high) AND not equal to fallback_value.
    """
    is_fb = np.abs(mu_slice - fallback_value) < fallback_tol
    mask = (mu_slice > threshold_low) & (mu_slice < threshold_high) & (~is_fb)
    if not mask.any():
        return None, None
    idx = np.where(mask)[0]
    # take the LARGEST contiguous run (skips spurious isolated false-positives)
    runs = []
    cur_start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((cur_start, prev))
            cur_start = i
            prev = i
    runs.append((cur_start, prev))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return p_grid[a], p_grid[b]


def _fit_segment_cheby(p_seg, mu_seg, deg):
    """LSQ Cheby fit on segment with auto-limited degree.

    Domain is (min, max) of p_seg.
    """
    n = len(p_seg)
    if n == 0:
        return None
    if n == 1:
        return ("const", float(mu_seg[0]))
    p_lo, p_hi = float(p_seg[0]), float(p_seg[-1])
    if p_hi <= p_lo:
        return ("const", float(mu_seg.mean()))
    deg_eff = min(deg, n - 1)
    xi = 2.0 * (p_seg - p_lo) / (p_hi - p_lo) - 1.0
    V = cheb.chebvander(xi, deg_eff)
    coefs, *_ = np.linalg.lstsq(V, mu_seg, rcond=None)
    return ("cheby", p_lo, p_hi, coefs)


def _fit_segment_cheby_bounds(p_seg, mu_seg, p_lo, p_hi, deg):
    """LSQ Cheby fit with the domain FIXED to (p_lo, p_hi) regardless of
    where the samples actually lie. Used for the borrow-neighbours case
    where the sample p's may straddle (a, b).
    """
    n = len(p_seg)
    if n == 0 or p_hi <= p_lo:
        return None
    if n == 1:
        return ("const", float(mu_seg[0]))
    deg_eff = min(deg, n - 1)
    xi = 2.0 * (p_seg - p_lo) / (p_hi - p_lo) - 1.0
    # No clip -- we want Cheb to fit borrowed (out-of-(a,b)) samples too.
    V = cheb.chebvander(xi, deg_eff)
    coefs, *_ = np.linalg.lstsq(V, mu_seg, rcond=None)
    return ("cheby", float(p_lo), float(p_hi), coefs)


def _fit_tail_cheby_v2(p_tail_min, p_min_support, mu_min_support,
                       dmu_min_support, side="lower"):
    """Quadratic Cheby with BCs:
        mu(p_tail) = 0 (lower) or 1 (upper)
        mu(p_min_sup) = mu_min_support
        mu'(p_min_sup) = dmu_min_support   (continuity of slope)
    Returns ('cheby', p_lo, p_hi, coefs) or None.
    """
    if p_tail_min >= p_min_support:
        return None
    p_lo, p_hi = float(p_tail_min), float(p_min_support)
    L = p_hi - p_lo
    bc_left = 0.0 if side == "lower" else 1.0
    # 3 BCs on a deg-2 (3-coef) Cheby in xi = 2*(p - p_lo)/L - 1:
    # T_n(-1)=(-1)^n, T_n(+1)=1, dT_n/dxi |+1 = n^2
    # dcheb/dp |xi=+1 = sum_n c_n * n^2 * (2/L)
    A = np.array([
        [1.0, -1.0, 1.0],
        [1.0, +1.0, 1.0],
        [0.0,  2.0 / L,  8.0 / L],   # n=0,1,2 -> 0, 1*(2/L), 4*(2/L)
    ])
    b = np.array([bc_left, float(mu_min_support), float(dmu_min_support)])
    try:
        coefs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return ("cheby", p_lo, p_hi, coefs)


def _eval_segment_vec(seg, p_arr):
    """Vector evaluation of a segment at p_arr (no support check; caller masks)."""
    if seg is None:
        return np.full_like(p_arr, 0.5)
    if seg[0] == "const":
        return np.full_like(p_arr, seg[1])
    _, p_lo, p_hi, coefs = seg
    xi = 2.0 * (p_arr - p_lo) / (p_hi - p_lo) - 1.0
    return cheb.chebval(xi, coefs)


def build_zero_h_lookup_slice(p_grid, mu_slice, p_cs, deg_in=DEG_IN,
                              tail_eps=TAIL_EPS,
                              min_samples_per_seg=MIN_SAMPLES_PER_SEG):
    """Build the per-slice (u_k fixed) zero-h pipeline lookup.

    Strategy
    --------
    1. Restrict to in-support samples (drop the 0.5 fallback plateau).
    2. Filter p_cs to those inside (p_min_sup, p_max_sup).
    3. MERGE adjacent p_c knots that produce segments with < min_samples_per_seg
       interior raw samples -- merging guarantees each fit has enough data.
       The largest knot of a merged group is kept as the segment boundary.
    4. Per-segment LSQ Cheby with degree min(deg_in, n_samples - 1).
    5. Lower & upper tail Cheby with BCs (mu->0 / mu->1).

    Returns
    -------
    info : dict
        keys = ('knots', 'segs', 'lower', 'upper', 'p_min_sup', 'p_max_sup')
    """
    p_min_sup, p_max_sup = _detect_support(p_grid, mu_slice)
    if p_min_sup is None:
        return None
    # restrict to in-support samples
    in_sup = (p_grid >= p_min_sup) & (p_grid <= p_max_sup) & \
                (np.abs(mu_slice - 0.5) > 1e-14)
    p_sup = p_grid[in_sup]
    mu_sup = mu_slice[in_sup]
    if len(p_sup) < 2:
        return None
    # candidate knots (only those strictly inside)
    p_cs_used = sorted(pc for pc in p_cs if p_min_sup < pc < p_max_sup)
    candidates = [p_min_sup] + p_cs_used + [p_max_sup]
    # MERGE: walk through candidates, dropping inner knots so that every
    # resulting segment has >= min_samples_per_seg interior samples.
    merged = [candidates[0]]
    for c in candidates[1:]:
        # how many samples between merged[-1] and c?
        n_in = int(np.sum((p_sup >= merged[-1]) & (p_sup <= c)))
        if n_in >= min_samples_per_seg or c == candidates[-1]:
            merged.append(c)
        # else: drop this knot, accumulate into next
    # If the LAST segment has too few samples, drop the second-to-last knot
    while len(merged) >= 3:
        a, b = merged[-2], merged[-1]
        n_in = int(np.sum((p_sup >= a) & (p_sup <= b)))
        if n_in < min_samples_per_seg:
            del merged[-2]
        else:
            break
    knots = np.array(merged)
    segs = []
    for k in range(len(knots) - 1):
        a, b = knots[k], knots[k + 1]
        mask = (p_sup >= a) & (p_sup <= b)
        p_seg = p_sup[mask]
        mu_seg = mu_sup[mask]
        if len(p_seg) < 2:
            segs.append(None)
            continue
        # Fit Cheby with domain = (a, b) so neighbouring segments meet
        # exactly at the knot. (LSQ in the local Cheb basis; degree
        # auto-limited by n_samples - 1, capped at deg_in.)
        seg = _fit_segment_cheby_bounds(p_seg, mu_seg, a, b, deg_in)
        segs.append(seg)
    # Tails
    lower = None
    if p_min_sup > tail_eps:
        idx_lo = int(np.searchsorted(p_grid, p_min_sup))
        if idx_lo + 1 < len(p_grid):
            mu_at_min = float(mu_slice[idx_lo])
            mu_next = float(mu_slice[idx_lo + 1])
            dmu_at_min = (mu_next - mu_at_min) / (p_grid[idx_lo + 1] - p_grid[idx_lo])
            lower = _fit_tail_cheby_v2(tail_eps, p_min_sup, mu_at_min,
                                        dmu_at_min, side="lower")
    upper = None
    if p_max_sup < 1.0 - tail_eps:
        idx_hi = int(np.searchsorted(p_grid, p_max_sup)) - 1
        if idx_hi - 1 >= 0:
            mu_at_max = float(mu_slice[idx_hi])
            mu_prev = float(mu_slice[idx_hi - 1])
            dmu_at_max = (mu_at_max - mu_prev) / (p_grid[idx_hi] - p_grid[idx_hi - 1])
            upper = _fit_tail_cheby_v2(p_max_sup, 1.0 - tail_eps, mu_at_max,
                                        dmu_at_max, side="upper")
    return dict(knots=knots, segs=segs, lower=lower, upper=upper,
                p_min_sup=p_min_sup, p_max_sup=p_max_sup)


def eval_lookup_vec(info, p_arr):
    """Evaluate the per-slice lookup at p_arr (1-D)."""
    out = np.empty_like(p_arr)
    if info is None:
        return np.full_like(p_arr, 0.5)
    knots = info["knots"]
    placed = np.zeros_like(p_arr, dtype=bool)
    # Order matters: interior segments FIRST so a point exactly at a knot
    # gets claimed by the segment on either side (no fall-through to tail).
    for k in range(len(knots) - 1):
        a, b = knots[k], knots[k + 1]
        seg = info["segs"][k]
        if seg is None:
            continue
        mask = (p_arr >= a) & (p_arr <= b) & (~placed)
        if not np.any(mask):
            continue
        out[mask] = _eval_segment_vec(seg, p_arr[mask])
        placed |= mask
    # tails (only for points strictly outside the interior knot range)
    if info["lower"] is not None:
        a, b = info["lower"][1], info["lower"][2]
        # strictly below knots[0]
        mask = (p_arr < knots[0]) & (p_arr >= a) & (~placed)
        if np.any(mask):
            out[mask] = _eval_segment_vec(info["lower"], p_arr[mask])
            placed |= mask
    if info["upper"] is not None:
        a, b = info["upper"][1], info["upper"][2]
        mask = (p_arr > knots[-1]) & (p_arr <= b) & (~placed)
        if np.any(mask):
            out[mask] = _eval_segment_vec(info["upper"], p_arr[mask])
            placed |= mask
    # very-far asymptotes (p < tail_eps or p > 1 - tail_eps)
    if not placed.all():
        leftover = ~placed
        below = leftover & (p_arr < knots[0])
        above = leftover & (p_arr > knots[-1])
        out[below] = 0.0
        out[above] = 1.0
        other = leftover & (~below) & (~above)
        if np.any(other):
            out[other] = 0.5
    # Clamp to (eps, 1-eps) -- but with a LARGER eps so we don't crush
    # legitimate ~0.99 values (CRRA clear needs strict (0,1)).
    eps = 1e-9
    np.clip(out, eps, 1.0 - eps, out=out)
    return out


# ---------------------------------------------------------------------------
# Build a smoothed mu_table from raw strict samples + critical points
# ---------------------------------------------------------------------------
def build_smoothed_mu_table(mu_raw, p_grid, p_cs, deg_in=DEG_IN):
    """Apply the zero-h pipeline to each u_k slice of mu_raw.

    Returns
    -------
    mu_smooth : (G_p, G) np.ndarray
    infos     : list of per-slice info dicts (for downstream eval)
    """
    G_p, G_ = mu_raw.shape
    mu_smooth = np.empty_like(mu_raw)
    infos = []
    for k in range(G_):
        info = build_zero_h_lookup_slice(p_grid, mu_raw[:, k], p_cs,
                                          deg_in=deg_in)
        infos.append(info)
        mu_smooth[:, k] = eval_lookup_vec(info, p_grid)
    return mu_smooth, infos


def crra_cube_clear(P_cube, mu_smooth_at_p, p_grid, gamma):
    """Given mu_smooth_at_p (shape G_p, G) and P_cube (G,G,G), compute the
    cube-clearing P_new[i,j,k] = crra_clear(mu(P[i,j,k], i),
                                              mu(P[i,j,k], j),
                                              mu(P[i,j,k], k)).
    Linear interpolation along p_grid for the mu lookup.
    """
    G_ = P_cube.shape[0]
    G_p = p_grid.size
    P_new = np.empty_like(P_cube)
    # vectorize the linear interp for all cube cells
    pflat = P_cube.ravel()
    eps_clip = 1e-9
    pflat_c = np.clip(pflat, eps_clip, 1.0 - eps_clip)
    # bracket indices
    lo = np.searchsorted(p_grid, pflat_c, side="right") - 1
    np.clip(lo, 0, G_p - 2, out=lo)
    hi = lo + 1
    w = (pflat_c - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    # per-slice (cube axis -> u-index axis mapping)
    idx = np.arange(G_ ** 3)
    i_idx = idx // (G_ ** 2)
    j_idx = (idx // G_) % G_
    k_idx = idx % G_
    # gather mu(p_old, u_i), mu(p_old, u_j), mu(p_old, u_k)
    def gather(k_axis):
        mu_lo = mu_smooth_at_p[lo, k_axis]
        mu_hi = mu_smooth_at_p[hi, k_axis]
        return (1.0 - w) * mu_lo + w * mu_hi
    mu0 = gather(i_idx)
    mu1 = gather(j_idx)
    mu2 = gather(k_idx)
    # clip again, then CRRA-clear
    np.clip(mu0, eps_clip, 1.0 - eps_clip, out=mu0)
    np.clip(mu1, eps_clip, 1.0 - eps_clip, out=mu1)
    np.clip(mu2, eps_clip, 1.0 - eps_clip, out=mu2)
    P_new_flat = np.empty(G_ ** 3)
    for n in range(G_ ** 3):
        P_new_flat[n] = crra_clear_jit(float(mu0[n]), float(mu1[n]),
                                          float(mu2[n]), float(gamma))
    return P_new_flat.reshape(G_, G_, G_)


# ---------------------------------------------------------------------------
# The zero-h Phi operator
# ---------------------------------------------------------------------------
class ZeroHOperator:
    """Encapsulate state so we can reuse JIT-compiled helpers + frozen critpts.

    Modes:
      mode="recompute" : detect critical points every Phi call.
      mode="frozen"    : after `freeze_after` calls, lock p_cs.
      mode="pchip"     : single PCHIP across all knots per slice.
    """

    def __init__(self, u_grid, p_grid, gl_u, gl_du, tau, gamma,
                 mode="recompute", freeze_after=10, deg_in=DEG_IN,
                 verbose=False):
        self.u_grid = u_grid
        self.p_grid = p_grid
        self.gl_u = gl_u
        self.gl_du = gl_du
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.G = u_grid.size
        self.G_p = p_grid.size
        self.NQ = NQ_STRICT
        self.mode = mode
        self.freeze_after = int(freeze_after)
        self.deg_in = int(deg_in)
        self.verbose = verbose
        self.n_calls = 0
        self.frozen_p_cs = None   # set when freezing
        self.last_info = None     # store last lookup infos
        self.diag = []            # per-call diagnostics

    def detect_critpts(self, P_cube):
        cps = find_critical_points(P_cube, self.u_grid, tol=1e-8)
        pvals = critical_p_values(cps, p_lo=1e-4, p_hi=1.0 - 1e-4)
        return pvals

    def __call__(self, P_cube):
        """One Phi call. Returns P_new (same shape as P_cube)."""
        t0 = time.time()
        # 1) Raw strict samples
        mu_raw = build_mu_table_lin_strict(P_cube, self.u_grid, self.p_grid,
                                             self.gl_u, self.gl_du, self.tau,
                                             self.G, self.NQ)
        t1 = time.time()
        # 2) Critical points (or frozen)
        if self.mode == "frozen" and self.frozen_p_cs is not None:
            p_cs = self.frozen_p_cs
        else:
            try:
                p_cs = self.detect_critpts(P_cube)
            except Exception as e:
                p_cs = np.array([])
                if self.verbose:
                    print(f"    [warn] find_critical_points failed: {e}",
                          flush=True)
        t2 = time.time()
        # 3) Smoothed mu_table via zero-h pipeline
        if self.mode == "pchip":
            mu_smooth = _smooth_pchip_table(mu_raw, self.p_grid, p_cs)
            infos = None
        else:
            mu_smooth, infos = build_smoothed_mu_table(mu_raw, self.p_grid,
                                                         p_cs, deg_in=self.deg_in)
        self.last_info = (mu_raw, mu_smooth, p_cs, infos)
        t3 = time.time()
        # 4) Cube clearing using smoothed mu
        P_new = crra_cube_clear(P_cube, mu_smooth, self.p_grid, self.gamma)
        t4 = time.time()
        self.n_calls += 1
        # 5) Maybe freeze
        if self.mode == "frozen" and self.frozen_p_cs is None and \
                self.n_calls >= self.freeze_after:
            self.frozen_p_cs = p_cs.copy() if len(p_cs) else np.array([])
            if self.verbose:
                print(f"    [info] froze p_cs at call {self.n_calls}: "
                      f"{len(self.frozen_p_cs)} critpts", flush=True)
        # diagnostics
        smooth_err = float(np.max(np.abs(mu_smooth - mu_raw)))
        self.diag.append(dict(
            call=self.n_calls,
            n_pcs=int(len(p_cs)),
            t_mu_raw=t1 - t0, t_critpts=t2 - t1,
            t_smooth=t3 - t2, t_clear=t4 - t3, t_total=t4 - t0,
            smooth_vs_raw_inf=smooth_err,
        ))
        return P_new


def _smooth_pchip_table(mu_raw, p_grid, p_cs):
    """Alternative smoothing: PCHIP across all knots (cusps + in-support
    sample points). Less prone to ringing if knots drift between iterations.
    """
    from scipy.interpolate import PchipInterpolator
    G_p, G_ = mu_raw.shape
    out = np.empty_like(mu_raw)
    for k in range(G_):
        # Use the raw samples as knots; PCHIP is monotone-aware so it
        # smooths gracefully. (We could insert p_cs knots too but they
        # might fall between samples without providing new info.)
        try:
            pch = PchipInterpolator(p_grid, mu_raw[:, k], extrapolate=True)
            out[:, k] = pch(p_grid)
        except Exception:
            out[:, k] = mu_raw[:, k]
    return out


# ---------------------------------------------------------------------------
# Anderson + best-iterate semantics (Cooper from dd_k3_strict_solve.py)
# ---------------------------------------------------------------------------
def anderson_solve(P0, op: ZeroHOperator, target=1e-10, n_iter=80, m_mem=10,
                   verbose=True, store_history=False):
    """Anderson(m_mem) accelerated Picard on op."""
    G_ = P0.shape[0]
    x = P0.ravel().copy()
    Xh, Gh = [], []
    Fs = []
    x_best = x.copy()
    f_best = float("inf")
    history = []
    for it in range(n_iter):
        gx_arr = op(x.reshape(G_, G_, G_)).ravel()
        Fv = gx_arr - x
        f = float(np.max(np.abs(Fv)))
        Fs.append(f)
        if f < f_best:
            f_best = f
            x_best = x.copy()
        if verbose:
            print(f"    iter {it:3d}: |F|={f:.3e}  best={f_best:.3e}  "
                  f"n_pcs={op.diag[-1]['n_pcs']:3d}", flush=True)
        if store_history:
            history.append({"iter": it, "F_inf": f, "f_best": f_best,
                             "n_pcs": int(op.diag[-1]["n_pcs"])})
        if f < target:
            break
        Xh.append(x.copy())
        Gh.append(gx_arr.copy())
        if len(Xh) > m_mem:
            Xh.pop(0)
            Gh.pop(0)
        k = len(Xh)
        if k <= 1:
            x = gx_arr
        else:
            try:
                DR = np.column_stack([(Gh[i] - Xh[i]) - (Gh[k - 1] - Xh[k - 1])
                                       for i in range(k - 1)])
                R_k = Gh[k - 1] - Xh[k - 1]
                A = DR.T @ DR + 1e-12 * np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i] - Gh[k - 1] for i in range(k - 1)])
                x = Gh[k - 1] + DG @ ga
            except Exception:
                x = gx_arr
    return x_best.reshape(G_, G_, G_), f_best, Fs, history


def nk_polish(P_in, op: ZeroHOperator, target=1e-10, maxiter=20):
    """Newton-Krylov fallback."""
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize._nonlin import NoConvergence
    G_ = P_in.shape[0]

    def F(xflat):
        Pn = op(xflat.reshape(G_, G_, G_))
        return (Pn - xflat.reshape(G_, G_, G_)).ravel()
    try:
        xnk = newton_krylov(F, P_in.ravel(), f_tol=target, maxiter=maxiter,
                             verbose=False)
        fnk = float(np.max(np.abs(F(xnk))))
        return xnk.reshape(G_, G_, G_), fnk
    except NoConvergence as e:
        xnk = e.args[0]
        fnk = float(np.max(np.abs(F(xnk))))
        return xnk.reshape(G_, G_, G_), fnk
    except Exception:
        return P_in, float("inf")


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------
def solve_cell(gamma, tau, mode="recompute", n_iter=60, m_mem=10,
                deg_in=DEG_IN, verbose=True, do_nk=False,
                target=1e-10, freeze_after=10):
    """Solve one (gamma, tau) cell using the zero-h FP solver."""
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_P)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    # Warm start
    warm_path = f"{FP_DIR}/dd_k3_strict_fp_g{int(gamma)}_t{tau:.4f}.npz"
    if os.path.exists(warm_path):
        d = np.load(warm_path)
        P0 = d["P_strict"].astype(np.float64)
        mu_strict_ws = d["mu_strict"].astype(np.float64)
    else:
        P0 = np.full((G, G, G), 0.5)
        mu_strict_ws = None

    op = ZeroHOperator(u_grid, p_grid, gl_u, gl_du, tau, gamma,
                        mode=mode, freeze_after=freeze_after,
                        deg_in=deg_in, verbose=verbose)
    t0 = time.time()
    P_star, F_star, Fs, history = anderson_solve(P0, op, target=target,
                                                    n_iter=n_iter, m_mem=m_mem,
                                                    verbose=verbose,
                                                    store_history=True)
    wall_aa = time.time() - t0
    if do_nk and F_star > target:
        if verbose:
            print(f"  Anderson stalled at {F_star:.3e}, trying NK...",
                  flush=True)
        P_nk, F_nk = nk_polish(P_star, op, target=target, maxiter=20)
        if F_nk < F_star:
            P_star, F_star = P_nk, F_nk
    wall = time.time() - t0
    # Final diagnostics: compare to warm-start
    bias_P = (float(np.max(np.abs(P_star - P0))) if P0.shape == P_star.shape
              else float("nan"))
    # Build the final smoothed mu (last call's state)
    mu_raw_final, mu_smooth_final, p_cs_final, _ = op.last_info
    mu_vs_warm = float(np.max(np.abs(mu_smooth_final - mu_strict_ws))) \
                    if mu_strict_ws is not None else float("nan")
    res = dict(
        gamma=float(gamma), tau=float(tau),
        mode=mode, deg_in=int(deg_in), m_mem=int(m_mem),
        n_iter_aa=len(Fs), F_history=Fs,
        F_final=float(F_star),
        wall_aa=wall_aa, wall_total=wall,
        n_pcs_final=int(len(p_cs_final)),
        bias_P_vs_warm=bias_P,
        mu_vs_warm=mu_vs_warm,
        smooth_vs_raw_final=op.diag[-1]["smooth_vs_raw_inf"],
        history=history,
    )
    out_npz = f"{OUT_DIR}/zero_h_fp_g{int(gamma)}_t{tau:.4f}_{mode}.npz"
    np.savez(out_npz, P_star=P_star, mu_smooth=mu_smooth_final,
              mu_raw=mu_raw_final, p_cs=p_cs_final,
              F_history=np.asarray(Fs),
              P0=P0, mu_strict_ws=mu_strict_ws if mu_strict_ws is not None
                   else np.array([]))
    res["out_npz"] = out_npz
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Cells as 'g100_t1.0' etc.")
    parser.add_argument("--mode", default="recompute",
                        choices=["recompute", "frozen", "pchip"])
    parser.add_argument("--n_iter", type=int, default=60)
    parser.add_argument("--m_mem", type=int, default=10)
    parser.add_argument("--deg_in", type=int, default=DEG_IN)
    parser.add_argument("--freeze_after", type=int, default=10)
    parser.add_argument("--target", type=float, default=1e-10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_nk", action="store_true",
                        help="Skip Newton-Krylov polish (NK is slow due to "
                             "4.5s/Phi from per-call find_critical_points).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.cells is None:
        cells = [(100, 1.0), (100, 0.2)]
    else:
        cells = []
        for cs in args.cells:
            # 'g100_t1.0' -> (100, 1.0)
            g_part, t_part = cs.split("_")
            cells.append((int(g_part[1:]), float(t_part[1:])))

    results = []
    for gamma, tau in cells:
        print(f"\n{'='*70}\n=== g={gamma}, tau={tau}, mode={args.mode} ==="
              f"\n{'='*70}", flush=True)
        try:
            r = solve_cell(gamma, tau, mode=args.mode, n_iter=args.n_iter,
                            m_mem=args.m_mem, deg_in=args.deg_in,
                            verbose=not args.quiet,
                            do_nk=(not args.no_nk),
                            target=args.target,
                            freeze_after=args.freeze_after)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}", flush=True)
            results.append(dict(gamma=float(gamma), tau=float(tau),
                                  mode=args.mode, error=str(e)))
            continue
        print(f"\n  RESULT g={gamma}_t{tau}: "
              f"F_final={r['F_final']:.3e}  "
              f"iters={r['n_iter_aa']}  "
              f"n_pcs={r['n_pcs_final']}  "
              f"bias|P-warm|={r['bias_P_vs_warm']:.3e}  "
              f"|mu_smooth-mu_warm|={r['mu_vs_warm']:.3e}  "
              f"wall={r['wall_total']:.1f}s",
              flush=True)
        results.append(r)
    out_json = args.out or f"{OUT_DIR}/results_{args.mode}.json"
    json.dump(results, open(out_json, "w"), indent=2, default=str)
    print(f"\nJSON -> {out_json}", flush=True)
    return results


if __name__ == "__main__":
    main()
