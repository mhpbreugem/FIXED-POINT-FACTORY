"""Tail-aware OptB+ lookup builders with explicit boundary conditions on mu(p,u_k).

Motivation. The Option B+ baseline (bilinear-refined empirical CDF -> PCHIP ->
density -> Bayes ratio) is unreliable at p near 0 and p near 1 because PCHIP
density extrapolation does not enforce the natural BC
    mu(0, u_k) = 0,   mu(1, u_k) = 1.
Finding 8 of the K=3 overnight summary: this is the dominant source of
max-error (~0.5 in mu) for the multi-element OptB+ test.

Variants tested:
  A) BASELINE OptB+ (no BC; PCHIP through raw empirical CDF). For reference.
  B) BC-AUGMENTED PCHIP: prepend (eps, 0) and append (1-eps, total_mass_v)
     to the CDF before fitting PCHIP. Pins the CDF at the boundary so the
     PCHIP derivative respects mass conservation.
  C) LOGIT-SPACE PCHIP: fit PCHIP to L_v(p) = logit(F_v(p)/total_mass_v).
     The logit map [0,1] -> [-inf, inf] regularizes density at the boundary
     since the derivative of a finite slope in logit space corresponds to
     density tending to zero in CDF space.
  D) BETA-PRIOR REGULARIZATION: parametric Beta(alpha_v, beta_v) CDF with
     shape parameters fit from data moments (method of moments); enforces
     correct boundary behavior exactly. May lose bulk fidelity.

All four are tested against mu_strict at the 4 saved strict FPs
(g{100,1000}, t{0.2,1.0}).

Implementation note. The baseline numba version uses an internal PCHIP; for
clarity, fairness across variants, and because this is a research evaluation
(not an inner-loop solver) we use scipy.interpolate.PchipInterpolator. All four
variants share the same CDF assembly path so any speed/accuracy differences
trace to the boundary treatment, not the interpolator.
"""
import os, sys, json, time, glob, math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from scipy.special import logit, expit

# Try to add cheby_h0_prototype to path so we can reuse make_cdf_uniform_grid
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "cheby_h0_prototype"))
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")


def make_cdf_uniform_grid(G, eps_q=0.01):
    """CDF-uniform u-grid (standard normal quantiles)."""
    qs = np.linspace(eps_q, 1.0 - eps_q, G)
    return norm.ppf(qs)


def make_p_grid(G_p=121, L=8.0):
    """Logit-uniform p-grid."""
    return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))


def make_helpers(u_grid, tau):
    """Trapezoid weights on u_grid plus Gaussian densities f_v on u_grid."""
    G = u_grid.size
    du = np.diff(u_grid)
    w_trap = np.empty(G)
    w_trap[0] = 0.5 * du[0]
    w_trap[-1] = 0.5 * du[-1]
    w_trap[1:-1] = 0.5 * (du[:-1] + du[1:])
    sigma = 1.0 / math.sqrt(tau)
    f0_u = norm.pdf(u_grid, loc=-0.5, scale=sigma)
    f1_u = norm.pdf(u_grid, loc= 0.5, scale=sigma)
    return w_trap, f0_u, f1_u


def build_refined_grid(u_grid, n_sub, tau):
    """Bilinear refinement: refined u-grid + trap weights + f_v values."""
    G = u_grid.size
    Gr = (G - 1) * n_sub + 1
    refined_u = np.empty(Gr)
    idx = 0
    for i in range(G - 1):
        for k in range(n_sub):
            t = k / n_sub
            refined_u[idx] = (1.0 - t) * u_grid[i] + t * u_grid[i + 1]
            idx += 1
    refined_u[idx] = u_grid[-1]
    du = np.diff(refined_u)
    w_r = np.empty(Gr)
    w_r[0] = 0.5 * du[0]
    w_r[-1] = 0.5 * du[-1]
    w_r[1:-1] = 0.5 * (du[:-1] + du[1:])
    sigma = 1.0 / math.sqrt(tau)
    f0_r = norm.pdf(refined_u, loc=-0.5, scale=sigma)
    f1_r = norm.pdf(refined_u, loc= 0.5, scale=sigma)
    return refined_u, w_r, f0_r, f1_r


def bilin_refine_slice(P_slice, u_grid, refined_u):
    """Bilinear interpolation of P_slice (G x G) -> (Gr x Gr) at refined_u^2."""
    G = u_grid.size
    Gr = refined_u.size
    # find segment indices and weights for refined_u in u_grid
    idx = np.searchsorted(u_grid, refined_u, side="right") - 1
    idx = np.clip(idx, 0, G - 2)
    w = (refined_u - u_grid[idx]) / (u_grid[idx + 1] - u_grid[idx])
    # endpoints sanity
    w[refined_u <= u_grid[0]] = 0.0
    idx[refined_u <= u_grid[0]] = 0
    w[refined_u >= u_grid[-1]] = 1.0
    idx[refined_u >= u_grid[-1]] = G - 2
    # 2D
    out = np.empty((Gr, Gr))
    # P_slice[i,j]; bilinear along (a, b)
    # row interp first along axis 1 (b), then axis 0 (a)
    # Build row interp matrix once
    # row_interp[a_i, jb] = (1-w_b)*P_slice[a_i, idx_b] + w_b*P_slice[a_i, idx_b+1]
    P_row = (1 - w)[None, :] * P_slice[:, idx] + w[None, :] * P_slice[:, idx + 1]
    # Then interp along axis 0 (a)
    out = (1 - w)[:, None] * P_row[idx, :] + w[:, None] * P_row[idx + 1, :]
    return out


def assemble_slice_samples(P_vals, k_node, u_grid, refined_u, w_r, f0_r, f1_r):
    """Returns sorted unique-P arrays (P, F0, F1) for slice k_node.

    F_v(p) = sum_{cells with P_cell <= p}  w * f_v(u_a) * f_v(u_b)
    where the sum is over the bilinear-refined cube cells in this u_k slice.
    """
    Gr = refined_u.size
    slice_P = P_vals[k_node]  # (G, G)
    refined_P = bilin_refine_slice(slice_P, u_grid, refined_u)
    # full weight grids (outer)
    W = np.outer(w_r, w_r)
    F0w = np.outer(f0_r, f0_r) * W
    F1w = np.outer(f1_r, f1_r) * W
    P_flat = refined_P.ravel()
    F0_flat = F0w.ravel()
    F1_flat = F1w.ravel()
    order = np.argsort(P_flat)
    P_sorted = P_flat[order]
    F0_cum = np.cumsum(F0_flat[order])
    F1_cum = np.cumsum(F1_flat[order])
    # dedupe ties: keep last cumulative value at each unique P (more correct)
    keep = np.concatenate([np.diff(P_sorted) > 1e-15, [True]])
    return P_sorted[keep], F0_cum[keep], F1_cum[keep]


# ============================================================================
#  Variant A:  BASELINE  (PCHIP through raw empirical CDF; current OptB+)
# ============================================================================
def build_mu_baseline(P_vals, u_grid, p_grid, tau, n_sub=8):
    G = u_grid.size
    G_p = p_grid.size
    _, f0_u, f1_u = make_helpers(u_grid, tau)
    refined_u, w_r, f0_r, f1_r = build_refined_grid(u_grid, n_sub, tau)
    mu = np.empty((G_p, G))
    for k in range(G):
        P_s, F0c, F1c = assemble_slice_samples(P_vals, k, u_grid, refined_u, w_r, f0_r, f1_r)
        if P_s.size < 4:
            mu[:, k] = 0.5
            continue
        f0_pchip = PchipInterpolator(P_s, F0c, extrapolate=False)
        f1_pchip = PchipInterpolator(P_s, F1c, extrapolate=False)
        d0 = f0_pchip.derivative()(p_grid)
        d1 = f1_pchip.derivative()(p_grid)
        # nan from extrapolation -> 0
        d0 = np.where(np.isfinite(d0), d0, 0.0)
        d1 = np.where(np.isfinite(d1), d1, 0.0)
        den = f0_u[k] * d0 + f1_u[k] * d1
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 1e-300, f1_u[k] * d1 / den, 0.5)
        m = np.clip(m, 1e-9, 1.0 - 1e-9)
        mu[:, k] = m
    return mu


# ============================================================================
#  Variant B:  BC-AUGMENTED PCHIP
#  prepend (eps, 0) and append (1-eps, total_mass_v)
# ============================================================================
def build_mu_bc_augmented(P_vals, u_grid, p_grid, tau, n_sub=8, eps=1e-12):
    """BC-augmented PCHIP.

    Pin the CDF at the boundary by prepending (eps, 0) and appending
    (1-eps, total_mass_v). For p outside the empirical support (where the
    PCHIP density is zero in both numerator AND denominator, so the Bayes
    ratio is 0/0), enforce the BC explicitly: mu -> 0 as p -> 0, mu -> 1
    as p -> 1, with a smooth blend across the support boundary.
    """
    G = u_grid.size
    G_p = p_grid.size
    _, f0_u, f1_u = make_helpers(u_grid, tau)
    refined_u, w_r, f0_r, f1_r = build_refined_grid(u_grid, n_sub, tau)
    mu = np.empty((G_p, G))
    for k in range(G):
        P_s, F0c, F1c = assemble_slice_samples(P_vals, k, u_grid, refined_u, w_r, f0_r, f1_r)
        if P_s.size < 4:
            mu[:, k] = 0.5
            continue
        tot0 = F0c[-1]
        tot1 = F1c[-1]
        P_lo = float(P_s[0])
        P_hi = float(P_s[-1])
        # Augment with boundary knots
        knots_P = [eps]; knots_F0 = [0.0]; knots_F1 = [0.0]
        keep = P_s > eps + 1e-15
        knots_P.extend(P_s[keep].tolist())
        knots_F0.extend(F0c[keep].tolist())
        knots_F1.extend(F1c[keep].tolist())
        if knots_P[-1] < 1.0 - eps - 1e-15:
            knots_P.append(1.0 - eps); knots_F0.append(tot0); knots_F1.append(tot1)
        else:
            knots_F0[-1] = tot0; knots_F1[-1] = tot1
        Pk = np.asarray(knots_P, dtype=np.float64)
        F0k = np.asarray(knots_F0, dtype=np.float64)
        F1k = np.asarray(knots_F1, dtype=np.float64)
        if Pk.size < 4:
            mu[:, k] = 0.5
            continue
        f0_pchip = PchipInterpolator(Pk, F0k, extrapolate=False)
        f1_pchip = PchipInterpolator(Pk, F1k, extrapolate=False)
        d0 = f0_pchip.derivative()(p_grid)
        d1 = f1_pchip.derivative()(p_grid)
        d0 = np.where(np.isfinite(d0), d0, 0.0)
        d1 = np.where(np.isfinite(d1), d1, 0.0)
        den = f0_u[k] * d0 + f1_u[k] * d1
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 1e-300, f1_u[k] * d1 / den, np.nan)
        # Outside the empirical support both densities are zero (0/0). Fill
        # with the explicit BC: mu -> 0 for p < P_lo, mu -> 1 for p > P_hi.
        below = (p_grid < P_lo) & ~np.isfinite(m)
        above = (p_grid > P_hi) & ~np.isfinite(m)
        m = np.where(below, 1e-9, m)
        m = np.where(above, 1.0 - 1e-9, m)
        # Any remaining NaN -> 0.5 sentinel
        m = np.where(np.isfinite(m), m, 0.5)
        m = np.clip(m, 1e-9, 1.0 - 1e-9)
        mu[:, k] = m
    return mu


# ============================================================================
#  Variant C:  LOGIT-SPACE PCHIP
#  L_v(p) := logit(F_v(p) / total_mass_v); fit PCHIP through (p, L_v)
#  Density: a_v(p) = d F_v / dp = total_mass_v * dL/dp * sigmoid(L)*(1-sigmoid(L))
# ============================================================================
def build_mu_logit(P_vals, u_grid, p_grid, tau, n_sub=8, eps=1e-10, eps_F=1e-10):
    G = u_grid.size
    G_p = p_grid.size
    _, f0_u, f1_u = make_helpers(u_grid, tau)
    refined_u, w_r, f0_r, f1_r = build_refined_grid(u_grid, n_sub, tau)
    mu = np.empty((G_p, G))
    for k in range(G):
        P_s, F0c, F1c = assemble_slice_samples(P_vals, k, u_grid, refined_u, w_r, f0_r, f1_r)
        if P_s.size < 4:
            mu[:, k] = 0.5
            continue
        tot0 = F0c[-1]
        tot1 = F1c[-1]
        if tot0 < 1e-300 or tot1 < 1e-300:
            mu[:, k] = 0.5
            continue
        # Augment with boundary knots (P=eps, F=0 ; P=1-eps, F=total) so the
        # logit transform is well-defined just inside the boundary.
        keep = (P_s > eps + 1e-15) & (P_s < 1.0 - eps - 1e-15)
        Pk = np.concatenate(([eps], P_s[keep], [1.0 - eps]))
        F0k = np.concatenate(([0.0], F0c[keep], [tot0]))
        F1k = np.concatenate(([0.0], F1c[keep], [tot1]))
        if Pk.size < 4:
            mu[:, k] = 0.5
            continue
        # Normalize and clip into (eps_F, 1-eps_F) for logit
        f0n = np.clip(F0k / tot0, eps_F, 1.0 - eps_F)
        f1n = np.clip(F1k / tot1, eps_F, 1.0 - eps_F)
        L0 = logit(f0n)
        L1 = logit(f1n)
        # PCHIP requires strictly increasing y? No - x must be strictly inc, y monotone OK.
        # logit of a CDF is monotone increasing -> good.
        # Guard against numerical non-monotonicity:
        for i in range(1, L0.size):
            if L0[i] <= L0[i-1]: L0[i] = L0[i-1] + 1e-12
            if L1[i] <= L1[i-1]: L1[i] = L1[i-1] + 1e-12
        p0_pchip = PchipInterpolator(Pk, L0, extrapolate=False)
        p1_pchip = PchipInterpolator(Pk, L1, extrapolate=False)
        L0_eval = p0_pchip(p_grid)
        L1_eval = p1_pchip(p_grid)
        dL0 = p0_pchip.derivative()(p_grid)
        dL1 = p1_pchip.derivative()(p_grid)
        # Replace NaN (out-of-knot-range) by boundary values
        # eps is much smaller than any p_grid[0] in practice, but to be safe:
        mask0 = ~np.isfinite(L0_eval)
        L0_eval = np.where(mask0, 0.0, L0_eval)
        L1_eval = np.where(mask0, 0.0, L1_eval)
        dL0 = np.where(np.isfinite(dL0), dL0, 0.0)
        dL1 = np.where(np.isfinite(dL1), dL1, 0.0)
        s0 = expit(L0_eval); s1 = expit(L1_eval)
        # a_v(p) = total_v * dL/dp * s*(1-s)
        a0 = tot0 * dL0 * s0 * (1.0 - s0)
        a1 = tot1 * dL1 * s1 * (1.0 - s1)
        den = f0_u[k] * a0 + f1_u[k] * a1
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 1e-300, f1_u[k] * a1 / den, 0.5)
        m = np.clip(m, 1e-9, 1.0 - 1e-9)
        mu[:, k] = m
    return mu


# ============================================================================
#  Variant D:  BETA-PRIOR REGULARIZATION
#  Fit Beta(alpha_v, beta_v) by method of moments to the empirical
#  P-distribution weighted by w * f_v. Density is Beta pdf (exact BC).
# ============================================================================
def _beta_mom_from_samples(P, w):
    """Method-of-moments Beta(alpha, beta) fit to weighted samples on [0,1]."""
    W = w.sum()
    if W <= 0:
        return 1.0, 1.0
    mu = float(np.sum(P * w) / W)
    var = float(np.sum((P - mu) ** 2 * w) / W)
    mu = min(max(mu, 1e-6), 1.0 - 1e-6)
    if var <= 0:
        var = 1e-6
    # variance of Beta: mu(1-mu)/(alpha+beta+1)
    nu = mu * (1.0 - mu) / var - 1.0
    if nu <= 0:
        nu = 0.1  # over-dispersed; clamp to a wide-but-valid Beta
    alpha = mu * nu
    beta = (1.0 - mu) * nu
    alpha = max(alpha, 0.01)
    beta = max(beta, 0.01)
    return alpha, beta


def _beta_pdf(p, alpha, beta):
    """Beta(alpha,beta) PDF at p in (0,1); 0 outside."""
    p = np.asarray(p)
    out = np.zeros_like(p, dtype=np.float64)
    mask = (p > 0.0) & (p < 1.0)
    if not mask.any():
        return out
    pp = p[mask]
    # log pdf: (a-1)*ln p + (b-1)*ln(1-p) - logB(a,b)
    from scipy.special import betaln
    lpdf = (alpha - 1.0) * np.log(pp) + (beta - 1.0) * np.log1p(-pp) - betaln(alpha, beta)
    out[mask] = np.exp(lpdf)
    return out


def build_mu_beta(P_vals, u_grid, p_grid, tau, n_sub=8):
    G = u_grid.size
    G_p = p_grid.size
    _, f0_u, f1_u = make_helpers(u_grid, tau)
    refined_u, w_r, f0_r, f1_r = build_refined_grid(u_grid, n_sub, tau)
    mu = np.empty((G_p, G))
    for k in range(G):
        # Reassemble per-cell weights (we need the per-cell P and per-cell w,
        # not the cumulative)
        slice_P = bilin_refine_slice(P_vals[k], u_grid, refined_u)
        W = np.outer(w_r, w_r)
        F0w = np.outer(f0_r, f0_r) * W
        F1w = np.outer(f1_r, f1_r) * W
        P_flat = slice_P.ravel()
        w0_flat = F0w.ravel()
        w1_flat = F1w.ravel()
        tot0 = float(w0_flat.sum()); tot1 = float(w1_flat.sum())
        if tot0 < 1e-300 or tot1 < 1e-300:
            mu[:, k] = 0.5
            continue
        # Method-of-moments Beta fit per signal
        a0, b0 = _beta_mom_from_samples(P_flat, w0_flat)
        a1, b1 = _beta_mom_from_samples(P_flat, w1_flat)
        # Density at p (total mass times pdf since these are unnormalized CDFs)
        d0 = tot0 * _beta_pdf(p_grid, a0, b0)
        d1 = tot1 * _beta_pdf(p_grid, a1, b1)
        den = f0_u[k] * d0 + f1_u[k] * d1
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(den > 1e-300, f1_u[k] * d1 / den, 0.5)
        m = np.clip(m, 1e-9, 1.0 - 1e-9)
        mu[:, k] = m
    return mu


# ============================================================================
#  Test harness
# ============================================================================
VARIANTS = {
    "baseline":      build_mu_baseline,
    "bc_augmented":  build_mu_bc_augmented,
    "logit":         build_mu_logit,
    "beta":          build_mu_beta,
}


def reconstruct_P_from_FP(d):
    """The strict FP NPZ stores P_strict directly. Old files may use mu_hi/mu_lo."""
    if "P_strict" in d.files:
        return d["P_strict"].astype(np.float64)
    if "P" in d.files:
        return d["P"].astype(np.float64)
    if "mu_hi" in d.files and "mu_lo" in d.files:
        return ((d["mu_hi"] + d["mu_lo"]) * 0.5).astype(np.float64)
    raise ValueError(f"Cannot reconstruct P from NPZ; files={d.files}")


def error_stats(mu_test, mu_ref, p_grid, u_grid, restrict_nontrivial=True):
    """Compute max, median, RMS errors and top-5 error locations."""
    diff = np.abs(mu_test - mu_ref)
    # Only evaluate on the non-trivial region of mu_ref (where strict is not 0.5
    # fallback), so the boundary 0.5 stuff doesn't dominate.
    if restrict_nontrivial:
        mask = np.abs(mu_ref - 0.5) > 1e-12  # mu_ref non-trivial
    else:
        mask = np.ones_like(mu_ref, dtype=bool)
    if not mask.any():
        return dict(max=0.0, med=0.0, rms=0.0, top5=[])
    d_sub = diff[mask]
    out = dict(
        max=float(d_sub.max()),
        med=float(np.median(d_sub)),
        rms=float(np.sqrt(np.mean(d_sub ** 2))),
        n_eval=int(mask.sum()),
    )
    # Top-5 error (p_idx, u_idx, p, u, err) over the masked region
    idx = np.argsort(-d_sub)[:5]
    # map back to 2D
    p_idx_all, u_idx_all = np.where(mask)
    top5 = []
    for r in idx:
        pi = int(p_idx_all[r]); ui = int(u_idx_all[r])
        top5.append(dict(
            p_idx=pi, u_idx=ui,
            p=float(p_grid[pi]), u=float(u_grid[ui]),
            err=float(diff[pi, ui]),
            mu_ref=float(mu_ref[pi, ui]),
            mu_test=float(mu_test[pi, ui]),
        ))
    out["top5"] = top5
    return out


def error_stats_all(mu_test, mu_ref):
    """Errors over ALL (p, u) including the strict-default 0.5 boundary.

    Caveat: mu_strict is 0.5 at p outside the empirical support of the strict
    integrator (no-root fallback), NOT the true mu. So large errors here for
    BC-aware variants are actually a feature (correct BC) not a bug.
    """
    diff = np.abs(mu_test - mu_ref)
    return dict(
        max_all=float(diff.max()),
        med_all=float(np.median(diff)),
        rms_all=float(np.sqrt(np.mean(diff ** 2))),
    )


def bc_compliance(mu_test, p_grid):
    """How well does mu_test honor the true BC mu(0)=0, mu(1)=1?

    Reports max |mu - 0| for p < 0.01 and max |mu - 1| for p > 0.99 across
    all u_k. Lower is better.
    """
    G_p, G = mu_test.shape
    mask_lo = p_grid < 0.01
    mask_hi = p_grid > 0.99
    return dict(
        lo_max=float(np.max(mu_test[mask_lo, :])) if mask_lo.any() else float("nan"),
        lo_mean=float(np.mean(mu_test[mask_lo, :])) if mask_lo.any() else float("nan"),
        hi_min=float(np.min(mu_test[mask_hi, :])) if mask_hi.any() else float("nan"),
        hi_mean=float(np.mean(mu_test[mask_hi, :])) if mask_hi.any() else float("nan"),
        # Score: how far is mu from the true BC at the tails?
        bc_err_max=float(max(
            np.max(mu_test[mask_lo, :]) if mask_lo.any() else 0.0,
            np.max(1.0 - mu_test[mask_hi, :]) if mask_hi.any() else 0.0,
        )),
    )


def monotonicity_stats(mu_test):
    """Check monotone-in-p property."""
    diffs = np.diff(mu_test, axis=0)
    return dict(
        viols=int(np.sum(diffs < -1e-9)),
        n=int(diffs.size),
        min_dmu=float(diffs.min()),
    )


# ============================================================================
#  Main
# ============================================================================
def main():
    out_dir = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/optB_tail"
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)
    fp_root = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
    # The 4 strict FPs we evaluate against
    fp_specs = [
        ("g100_t0.2",  100.0, 0.2,  f"{fp_root}/dd_k3_strict_fp_g100_t0.2000.npz"),
        ("g100_t1.0",  100.0, 1.0,  f"{fp_root}/dd_k3_strict_fp_g100_t1.0000.npz"),
        ("g1000_t0.2", 1000.0, 0.2, f"{fp_root}/dd_k3_strict_fp_g1000_t0.2000.npz"),
        ("g1000_t1.0", 1000.0, 1.0, f"{fp_root}/dd_k3_strict_fp_g1000_t1.0000.npz"),
    ]
    n_sub = 8
    G = 11
    G_p = 121
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)

    results = {"meta": dict(G=G, G_p=G_p, n_sub=n_sub,
                             u_grid=u_grid.tolist(),
                             p_grid=p_grid.tolist())}
    print(f"\n=== Tail-aware OptB+ test ===  G={G}, G_p={G_p}, n_sub={n_sub}", flush=True)
    print(f"Variants: {list(VARIANTS.keys())}", flush=True)

    mu_store = {}   # mu_store[fp_label][variant] = array
    for label, gamma, tau, path in fp_specs:
        if not os.path.exists(path):
            print(f"MISSING {path}, skipping {label}", flush=True)
            continue
        d = np.load(path)
        P = reconstruct_P_from_FP(d)
        mu_strict = d["mu_strict"].astype(np.float64)
        assert P.shape == (G, G, G), f"P shape {P.shape} mismatch G={G}"
        assert mu_strict.shape == (G_p, G), f"mu_strict shape mismatch"
        print(f"\n--- {label}  (gamma={gamma}, tau={tau}) ---", flush=True)
        results[label] = dict(gamma=gamma, tau=tau, file=path, variants={})
        mu_store[label] = {"strict": mu_strict}
        for vname, vfunc in VARIANTS.items():
            t0 = time.time()
            mu_test = vfunc(P, u_grid, p_grid, tau, n_sub=n_sub)
            wall = time.time() - t0
            stats_nt = error_stats(mu_test, mu_strict, p_grid, u_grid,
                                          restrict_nontrivial=True)
            stats_all = error_stats_all(mu_test, mu_strict)
            bc = bc_compliance(mu_test, p_grid)
            mono = monotonicity_stats(mu_test)
            results[label]["variants"][vname] = {
                **stats_nt, **stats_all, "bc": bc, "mono": mono, "wall_s": wall,
            }
            mu_store[label][vname] = mu_test
            print(f"  {vname:14s}  max(nt)={stats_nt['max']:.3e}  "
                  f"med(nt)={stats_nt['med']:.3e}  rms(nt)={stats_nt['rms']:.3e}  "
                  f"BC_err={bc['bc_err_max']:.3e}  "
                  f"mono_viol={mono['viols']:4d}/{mono['n']}  "
                  f"wall={wall*1000:.0f}ms", flush=True)

    # Save JSON
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {json_path}", flush=True)

    # Summary table across variants
    print(f"\n=== SUMMARY (max error over non-trivial region) ===", flush=True)
    hdr = f"{'FP':14s} " + "  ".join(f"{v:>14s}" for v in VARIANTS.keys())
    print(hdr, flush=True)
    for label, _, _, _ in fp_specs:
        if label not in results:
            continue
        row = f"{label:14s} " + "  ".join(
            f"{results[label]['variants'][v]['max']:14.3e}" for v in VARIANTS.keys()
        )
        print(row, flush=True)
    print(f"\n=== SUMMARY (max error over ALL p; note: strict has 0.5 fallback at boundary) ===", flush=True)
    print(hdr, flush=True)
    for label, _, _, _ in fp_specs:
        if label not in results:
            continue
        row = f"{label:14s} " + "  ".join(
            f"{results[label]['variants'][v]['max_all']:14.3e}" for v in VARIANTS.keys()
        )
        print(row, flush=True)

    print(f"\n=== SUMMARY (BC compliance: max{{mu(p<.01), 1-mu(p>.99)}} -- LOWER is better) ===", flush=True)
    print(hdr, flush=True)
    for label, _, _, _ in fp_specs:
        if label not in results:
            continue
        row = f"{label:14s} " + "  ".join(
            f"{results[label]['variants'][v]['bc']['bc_err_max']:14.3e}" for v in VARIANTS.keys()
        )
        print(row, flush=True)

    print(f"\n=== SUMMARY (monotone-in-p violations / 1320) ===", flush=True)
    print(hdr, flush=True)
    for label, _, _, _ in fp_specs:
        if label not in results:
            continue
        row = f"{label:14s} " + "  ".join(
            f"{results[label]['variants'][v]['mono']['viols']:14d}" for v in VARIANTS.keys()
        )
        print(row, flush=True)

    # ---------------- Figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}", flush=True)
        return results

    # Fig 1: mu(p) curves at a few u_k values for each variant, per FP
    for label, _, _, _ in fp_specs:
        if label not in mu_store:
            continue
        ms = mu_store[label]
        ks_to_plot = [0, 3, 5, 7, 10]
        fig, axes = plt.subplots(1, len(ks_to_plot), figsize=(15, 3.2), sharey=True)
        for ax, k in zip(axes, ks_to_plot):
            ax.plot(p_grid, ms["strict"][:, k], "k-", lw=2, label="strict")
            for vname, ls in zip(VARIANTS.keys(), ["--", "-.", ":", "-"]):
                ax.plot(p_grid, ms[vname][:, k], ls, lw=1.2, label=vname)
            ax.set_xscale("logit")
            ax.set_xlim(1e-3, 1 - 1e-3)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"u_k={u_grid[k]:.2f}")
            ax.set_xlabel("p")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(r"$\mu(p, u_k)$")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{label}: mu(p) at selected u_k (n_sub={n_sub})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figs", f"mu_curves_{label}.png"), dpi=110)
        plt.close(fig)
        # Error heatmap
        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(16, 3.5), sharey=True)
        for ax, vname in zip(axes, VARIANTS.keys()):
            err = np.abs(ms[vname] - ms["strict"])
            im = ax.imshow(err, aspect="auto", origin="lower",
                            extent=[u_grid[0], u_grid[-1], 0, G_p-1],
                            cmap="viridis", vmin=0, vmax=min(0.5, err.max()))
            ax.set_xlabel("u_k"); ax.set_title(f"{vname} max={err.max():.2e}")
            plt.colorbar(im, ax=ax, fraction=0.05)
        axes[0].set_ylabel("p_idx")
        fig.suptitle(f"{label}: |mu_test - mu_strict|")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figs", f"err_heat_{label}.png"), dpi=110)
        plt.close(fig)

    # Fig: summary bar chart of max errors across variants & FPs
    labels = [s[0] for s in fp_specs if s[0] in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(labels))
    width = 0.18
    for i, vname in enumerate(VARIANTS.keys()):
        vals_nt = [results[l]['variants'][vname]['max'] for l in labels]
        ax.bar(xs + (i - 1.5) * width, vals_nt, width, label=vname)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("max |mu - mu_strict| (non-trivial region)")
    ax.set_title("Tail-aware OptB+ variants: max error on non-trivial region")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.axhline(0.05, color="r", linestyle="--", lw=1, label="success target")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "summary_max_nt.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, vname in enumerate(VARIANTS.keys()):
        vals_all = [results[l]['variants'][vname]['max_all'] for l in labels]
        ax.bar(xs + (i - 1.5) * width, vals_all, width, label=vname)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("max |mu - mu_strict| (ALL p)")
    ax.set_title("Tail-aware OptB+ variants: max error including boundary")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figs", "summary_max_all.png"), dpi=110)
    plt.close(fig)

    print(f"\nSaved figs in {out_dir}/figs/", flush=True)
    return results


if __name__ == "__main__":
    main()
