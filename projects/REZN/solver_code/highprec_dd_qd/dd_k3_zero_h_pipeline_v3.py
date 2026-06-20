"""V3 zero-h pipeline: in-support + tail extrapolation.

V2 achieved machine eps in-support but the out-of-support region (where
mu_strict has the no-roots-fallback = 0.5) was left at the fallback.
For production use, we need μ(p) for ALL p in [eps, 1-eps], including
where the cube CDF has no support.

Tail extrapolation strategy:
- Lower tail [eps, p_min_support]: Cheby segment matched to:
    mu(eps) = 0  (Bayesian BC)
    mu(p_min_support) = mu_in_support[0]  (continuity)
    mu'(p_min_support) = derivative of in-support Cheby at p_min  (C^1)
- Upper tail [p_max_support, 1-eps]: similarly with mu(1-eps) = 1.

The TRUE μ(p, u_k) at out-of-support p is mathematically defined (the
posterior conditional on a price that's outside the observed cube range
but still meaningful given the asymptotic behavior). The 0.5 fallback in
mu_strict is an artifact; our extrapolation is more accurate by construction
because it uses Bayesian BCs.

We can't easily verify accuracy out-of-support against ground truth (mu_strict
gives wrong values there). What we CAN verify:
- Continuity at p_min, p_max (no jumps).
- Monotonicity over the full range.
- Correct BCs at p=eps and p=1-eps.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_optB_pieceCheby import detect_critical_points_from_mu
from dd_k3_zero_h_pipeline_v2 import (detect_support_per_slice, fit_segment_cheby,
                                              eval_segment)


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_pipeline_v3"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def fit_tail_cheby_bc(p_lo, p_hi, mu_bc_left, mu_match, dmu_match, side="lower"):
    """Fit degree-2 Cheby on [p_lo, p_hi] with 3 BCs:
    - mu(p_lo) = mu_bc_left  (0 for lower tail, 1 for upper)
    - mu(p_hi) = mu_match  (match with in-support segment)
    - dmu/dp(p_hi) = dmu_match  (match derivative)
    """
    if p_hi <= p_lo: return None
    L = p_hi - p_lo
    # Chebyshev coefs c0, c1, c2 with basis T_0=1, T_1=xi, T_2=2xi^2-1
    # at xi = 2(p-p_lo)/L - 1
    # xi=-1: T_0=1, T_1=-1, T_2=1
    # xi=+1: T_0=1, T_1=1, T_2=1
    # dT_n/dxi at xi=+1: 0, 1, 4
    # dmu/dp = (dmu/dxi)*(2/L)
    A = np.array([
        [1, -1, 1],        # mu(p_lo)
        [1, +1, 1],        # mu(p_hi)
        [0, 2/L, 8/L],     # dmu/dp at p_hi
    ])
    b = np.array([mu_bc_left, mu_match, dmu_match])
    try:
        coefs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return ("cheby", p_lo, p_hi, coefs)


def deriv_at_segment_endpoint(seg, at_high=True):
    """dmu/dp at the high (or low) endpoint of a Cheby segment."""
    if seg is None or seg[0] != "cheby": return 0.0
    _, p_lo, p_hi, coefs = seg
    L = p_hi - p_lo
    # dT_n/dxi at xi=+1 is n^2; at xi=-1 is (-1)^(n+1) * n^2
    n_coefs = len(coefs)
    if at_high:
        dmu_dxi = sum(coefs[n] * n*n for n in range(n_coefs))
    else:
        dmu_dxi = sum(coefs[n] * ((-1)**(n+1)) * n*n for n in range(n_coefs))
    return dmu_dxi * (2 / L)


def build_full_lookup_slice(p_grid, mu_slice, p_cs_slice, deg=6,
                                 p_eps=1e-6, mu_bc_lo=0.0, mu_bc_hi=1.0):
    """Build per-segment Cheby PLUS tail extrapolation. Returns:
    (knots, segments)  including extra tail segments at both ends."""
    i_lo, i_hi = detect_support_per_slice(p_grid, mu_slice)
    if i_lo is None: return None, None
    p_sup_lo, p_sup_hi = float(p_grid[i_lo]), float(p_grid[i_hi])
    # In-support knots
    p_cs_in = [pc for pc in p_cs_slice if p_sup_lo < pc < p_sup_hi]
    insup_knots = sorted([p_sup_lo] + p_cs_in + [p_sup_hi])
    insup_segs = []
    for k in range(len(insup_knots) - 1):
        a, b = insup_knots[k], insup_knots[k+1]
        mask = (p_grid >= a) & (p_grid <= b)
        if not mask.any():
            insup_segs.append(None); continue
        insup_segs.append(fit_segment_cheby(p_grid[mask], mu_slice[mask], deg))
    # Tail extrapolation
    all_knots = list(insup_knots); all_segs = list(insup_segs)
    if p_sup_lo > p_eps:
        # Lower tail
        first_seg = insup_segs[0] if insup_segs else None
        if first_seg is not None:
            mu_at_lo = eval_segment(first_seg, p_sup_lo)
            dmu_at_lo = deriv_at_segment_endpoint(first_seg, at_high=False)
            lower = fit_tail_cheby_bc(p_eps, p_sup_lo, mu_bc_lo, mu_at_lo, dmu_at_lo, side="lower")
            if lower is not None:
                all_knots = [p_eps] + all_knots
                all_segs = [lower] + all_segs
    if p_sup_hi < 1.0 - p_eps:
        # Upper tail
        last_seg = insup_segs[-1] if insup_segs else None
        if last_seg is not None:
            mu_at_hi = eval_segment(last_seg, p_sup_hi)
            dmu_at_hi = deriv_at_segment_endpoint(last_seg, at_high=True)
            upper = fit_tail_cheby_bc(p_sup_hi, 1.0 - p_eps, mu_at_hi, mu_bc_hi, dmu_at_hi, side="upper")
            # Wait, BC reversed for upper: mu_bc_hi is the BC at upper end (1-eps)
            # match value is mu_at_hi at lower end (p_sup_hi)
            # Let me re-derive: fit on [p_sup_hi, 1-eps]:
            #   mu(p_sup_hi) = mu_at_hi (match)
            #   mu(1-eps) = 1 (BC)
            #   dmu/dp(p_sup_hi) = dmu_at_hi (match deriv)
            L_t = 1.0 - p_eps - p_sup_hi
            A = np.array([
                [1, -1, 1],          # mu at p_lo = p_sup_hi
                [1, +1, 1],          # mu at p_hi = 1-eps
                [0, 2/L_t, -8/L_t],  # dmu/dp at p_sup_hi (xi=-1): sum c_n*(-1)^(n+1)*n^2 * (2/L)
            ])
            b = np.array([mu_at_hi, mu_bc_hi, dmu_at_hi])
            try:
                coefs = np.linalg.solve(A, b)
                upper = ("cheby", p_sup_hi, 1.0 - p_eps, coefs)
                all_knots = all_knots + [1.0 - p_eps]
                all_segs = all_segs + [upper]
            except:
                pass
    return np.array(all_knots), all_segs


def evaluate_full(p, knots, segments, mu_bc_lo=0.0, mu_bc_hi=1.0):
    if knots is None: return 0.5
    if p < knots[0]: return mu_bc_lo
    if p > knots[-1]: return mu_bc_hi
    for k in range(len(knots) - 1):
        if knots[k] <= p <= knots[k+1]:
            seg = segments[k]
            if seg is None: return 0.5
            v = eval_segment(seg, p)
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
    print(f"  {len(p_cs)} cusps from internal detector")

    mu_pipe = np.full((G_p, G), 0.5)
    mu_pipe_full = np.full((G_p, G), 0.5)  # full including tail extrapolation
    in_support_mask = np.zeros((G_p, G), dtype=bool)
    for k in range(G):
        knots, segs = build_full_lookup_slice(p_grid, mu_strict[:, k], p_cs, deg=deg)
        if knots is None: continue
        i_lo, i_hi = detect_support_per_slice(p_grid, mu_strict[:, k])
        p_sup_lo, p_sup_hi = p_grid[i_lo], p_grid[i_hi]
        for ip in range(G_p):
            mu_pipe_full[ip, k] = evaluate_full(p_grid[ip], knots, segs)
            if p_sup_lo <= p_grid[ip] <= p_sup_hi:
                mu_pipe[ip, k] = mu_pipe_full[ip, k]
                in_support_mask[ip, k] = True

    err = np.abs(mu_pipe - mu_strict)
    in_err = err[in_support_mask]
    print(f"  in-support points: {in_support_mask.sum()}/{mu_strict.size}")
    print(f"  in-support max = {in_err.max():.3e}, median = {np.median(in_err):.3e}")

    # Sanity: BC and monotonicity check on full pipeline
    n_bc_lo_ok = 0; n_bc_hi_ok = 0; n_mono_ok = 0
    for k in range(G):
        if mu_pipe_full[0, k] < 1e-3: n_bc_lo_ok += 1
        if mu_pipe_full[-1, k] > 1.0 - 1e-3: n_bc_hi_ok += 1
        # Monotone check
        if np.all(np.diff(mu_pipe_full[:, k]) >= -1e-6): n_mono_ok += 1
    print(f"  BC mu(0)~0 at {n_bc_lo_ok}/{G} u_k slices")
    print(f"  BC mu(1)~1 at {n_bc_hi_ok}/{G} u_k slices")
    print(f"  Monotone at {n_mono_ok}/{G} u_k slices")

    # Plot one slice
    k0 = G // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(p_grid, mu_strict[:, k0], 'k-', label='strict (with fallback artifact)', lw=2, alpha=0.6)
    axes[0].plot(p_grid, mu_pipe_full[:, k0], 'r--', label='pipeline v3 (with tail extrap)', alpha=0.8)
    axes[0].plot(p_grid, mu_pipe[:, k0], 'b:', label='pipeline v3 (in-support only)', alpha=0.8)
    for pc in p_cs: axes[0].axvline(pc, color='gray', alpha=0.2)
    axes[0].set_xlabel('p'); axes[0].set_ylabel(f'mu(p, u_k={u_grid[k0]:.2f})')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f'g={gamma}, tau={tau}, deg={deg}')

    # All slices overlay
    cmap = plt.cm.viridis
    for k in range(G):
        c = cmap(k/G)
        axes[1].plot(p_grid, mu_pipe_full[:, k], '-', color=c, alpha=0.6, lw=0.5)
    axes[1].set_xlabel('p'); axes[1].set_ylabel('mu (all u_k)')
    axes[1].set_title('Full pipeline output across u_k slices')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}_deg{deg}.png", dpi=120)
    plt.close()

    return dict(gamma=gamma, tau=tau, deg=deg, n_cusps=len(p_cs),
                in_support_max=float(in_err.max()),
                in_support_median=float(np.median(in_err)),
                bc_lo_ok=n_bc_lo_ok, bc_hi_ok=n_bc_hi_ok, mono_ok=n_mono_ok)


def main():
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    results = []
    for g, t in cells:
        for deg in [6, 8]:
            try:
                r = test_cell(g, t, deg=deg)
                results.append(r)
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
