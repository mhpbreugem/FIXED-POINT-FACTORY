"""V4: monotone tail extrapolation via PCHIP.

V3 used degree-2 Cheby for tails -> non-monotone in 4-11/11 slices.
Fix: use PCHIP through 4 points (BC, intermediate at 25%, intermediate
at 75%, match) -- shape-preserving by construction.
"""
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_optB_pieceCheby import detect_critical_points_from_mu
from dd_k3_zero_h_pipeline_v2 import (detect_support_per_slice, fit_segment_cheby,
                                              eval_segment)
from dd_k3_zero_h_pipeline_v3 import deriv_at_segment_endpoint


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_pipeline_v4"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def build_monotone_tail_pchip(p_bc, p_match, mu_bc, mu_match, dmu_match=None,
                                  n_extra=2, side="lower"):
    """PCHIP interpolant through 2+n_extra knots in [p_bc, p_match] with:
    - mu(p_bc) = mu_bc (BC)
    - mu(p_match) = mu_match (continuity)
    The intermediate points are placed linearly in p to give PCHIP enough
    knots for a smooth approximation; PCHIP handles monotonicity by
    construction (shape-preserving)."""
    if p_match <= p_bc: return None
    # Linear interpolation as intermediate
    p_knots = np.linspace(p_bc, p_match, 2 + n_extra)
    mu_knots = np.linspace(mu_bc, mu_match, 2 + n_extra)
    try:
        pchip = PchipInterpolator(p_knots, mu_knots, extrapolate=False)
        return ("pchip", p_bc, p_match, pchip)
    except Exception:
        return None


def eval_seg_v4(seg, p):
    if seg is None: return None
    if seg[0] == "const": return seg[1]
    if seg[0] == "cheby":
        from numpy.polynomial import chebyshev as cheb
        _, p_lo, p_hi, coefs = seg
        xi = 2*(p - p_lo)/(p_hi - p_lo) - 1
        return float(cheb.chebval(xi, coefs))
    if seg[0] == "pchip":
        _, p_lo, p_hi, pchip = seg
        if p < p_lo: p = p_lo
        if p > p_hi: p = p_hi
        return float(pchip(p))
    return 0.5


def build_full_v4(p_grid, mu_slice, p_cs_slice, deg=6, p_eps=1e-6,
                       mu_bc_lo=0.0, mu_bc_hi=1.0):
    i_lo, i_hi = detect_support_per_slice(p_grid, mu_slice)
    if i_lo is None: return None, None
    p_sup_lo, p_sup_hi = float(p_grid[i_lo]), float(p_grid[i_hi])
    p_cs_in = [pc for pc in p_cs_slice if p_sup_lo < pc < p_sup_hi]
    insup_knots = sorted([p_sup_lo] + p_cs_in + [p_sup_hi])
    insup_segs = []
    for k in range(len(insup_knots) - 1):
        a, b = insup_knots[k], insup_knots[k+1]
        mask = (p_grid >= a) & (p_grid <= b)
        if not mask.any():
            insup_segs.append(None); continue
        insup_segs.append(fit_segment_cheby(p_grid[mask], mu_slice[mask], deg))
    all_knots = list(insup_knots); all_segs = list(insup_segs)
    # Lower tail PCHIP
    if p_sup_lo > p_eps:
        first_seg = insup_segs[0] if insup_segs else None
        if first_seg is not None:
            mu_at_lo = eval_segment(first_seg, p_sup_lo)
            lower = build_monotone_tail_pchip(p_eps, p_sup_lo, mu_bc_lo, mu_at_lo, side="lower")
            if lower is not None:
                all_knots = [p_eps] + all_knots
                all_segs = [lower] + all_segs
    # Upper tail PCHIP
    if p_sup_hi < 1.0 - p_eps:
        last_seg = insup_segs[-1] if insup_segs else None
        if last_seg is not None:
            mu_at_hi = eval_segment(last_seg, p_sup_hi)
            upper = build_monotone_tail_pchip(p_sup_hi, 1.0 - p_eps, mu_at_hi, mu_bc_hi, side="upper")
            if upper is not None:
                all_knots = all_knots + [1.0 - p_eps]
                all_segs = all_segs + [upper]
    return np.array(all_knots), all_segs


def evaluate_full_v4(p, knots, segments, mu_bc_lo=0.0, mu_bc_hi=1.0):
    if knots is None: return 0.5
    if p < knots[0]: return mu_bc_lo
    if p > knots[-1]: return mu_bc_hi
    for k in range(len(knots) - 1):
        if knots[k] <= p <= knots[k+1]:
            v = eval_seg_v4(segments[k], p)
            return v if v is not None else 0.5
    return 0.5


def test_cell(gamma, tau, deg=6):
    print(f"\n=== g={gamma}, tau={tau}, deg={deg} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    mu_strict = fp["mu_strict"].astype(np.float64)
    G_p, G = mu_strict.shape
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)
    p_cs = detect_critical_points_from_mu(mu_strict, p_grid)

    mu_pipe = np.full((G_p, G), 0.5)
    in_support_mask = np.zeros((G_p, G), dtype=bool)
    for k in range(G):
        knots, segs = build_full_v4(p_grid, mu_strict[:, k], p_cs, deg=deg)
        if knots is None: continue
        i_lo, i_hi = detect_support_per_slice(p_grid, mu_strict[:, k])
        p_sup_lo, p_sup_hi = p_grid[i_lo], p_grid[i_hi]
        for ip in range(G_p):
            mu_pipe[ip, k] = evaluate_full_v4(p_grid[ip], knots, segs)
            if p_sup_lo <= p_grid[ip] <= p_sup_hi:
                in_support_mask[ip, k] = True

    err = np.abs(mu_pipe - mu_strict)
    in_err = err[in_support_mask]
    n_bc_lo_ok = sum(1 for k in range(G) if mu_pipe[0, k] < 1e-3)
    n_bc_hi_ok = sum(1 for k in range(G) if mu_pipe[-1, k] > 1.0 - 1e-3)
    n_mono = sum(1 for k in range(G) if np.all(np.diff(mu_pipe[:, k]) >= -1e-9))

    print(f"  cusps={len(p_cs)}, in-support max={in_err.max():.3e}, "
          f"median={np.median(in_err):.3e}")
    print(f"  BC mu(0)~0: {n_bc_lo_ok}/{G}, BC mu(1)~1: {n_bc_hi_ok}/{G}, "
          f"monotone: {n_mono}/{G}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    k0 = G // 2
    axes[0].plot(p_grid, mu_strict[:, k0], 'k-', lw=2, alpha=0.5, label='strict (artifact in tails)')
    axes[0].plot(p_grid, mu_pipe[:, k0], 'r-', label='pipeline v4 (monotone tails)', lw=1.5)
    for pc in p_cs: axes[0].axvline(pc, color='gray', alpha=0.2, lw=0.5)
    axes[0].set_xlabel('p'); axes[0].set_ylabel(f'mu(p, u_k={u_grid[k0]:.2f})')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'g={gamma}, tau={tau}, deg={deg}')
    cmap = plt.cm.viridis
    for k in range(G):
        c = cmap(k/G)
        axes[1].plot(p_grid, mu_pipe[:, k], '-', color=c, alpha=0.6, lw=0.5)
    axes[1].set_xlabel('p'); axes[1].set_ylabel('mu (all u_k slices)')
    axes[1].grid(alpha=0.3); axes[1].set_title('All u_k slices')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}_deg{deg}.png", dpi=120)
    plt.close()

    return dict(gamma=gamma, tau=tau, deg=deg, n_cusps=len(p_cs),
                in_support_max=float(in_err.max()),
                in_support_median=float(np.median(in_err)),
                bc_lo_ok=n_bc_lo_ok, bc_hi_ok=n_bc_hi_ok, mono_ok=n_mono)


def main():
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    results = []
    for g, t in cells:
        for deg in [6, 8]:
            try: results.append(test_cell(g, t, deg=deg))
            except Exception as e:
                import traceback; traceback.print_exc()
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"{'cell':>15} {'deg':>4} {'cusps':>6} {'IS-max':>10} {'IS-med':>10} {'BC-lo':>6} {'BC-hi':>6} {'Mono':>6}")
    for r in results:
        print(f"g{r['gamma']:>4g}_t{r['tau']:.2f}".rjust(15) +
              f" {r['deg']:>4d} {r['n_cusps']:>6d} {r['in_support_max']:>10.2e} "
              f"{r['in_support_median']:>10.2e} {r['bc_lo_ok']:>6d} {r['bc_hi_ok']:>6d} {r['mono_ok']:>6d}")


if __name__ == "__main__":
    main()
