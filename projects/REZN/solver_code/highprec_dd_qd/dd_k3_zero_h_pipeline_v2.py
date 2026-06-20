"""V2 zero-h pipeline using the CORRECTED critical-point detector.

Key fix: use the mu-based internal detector (detect cusps from mu_strict
itself per-slice) instead of the 3D ∇P=0 detector. The cusps of 1D
mu(p, u_k) come from critical values of the SLICE map at fixed u_k, not
the 3D map.

Pipeline:
1. Load strict-h=0 mu_strict.
2. For each u_k slice, detect cusps via |mu''| + first-diff jumps.
3. Segment p axis at those knots; degree-6 Cheby per segment.
4. Outside support, extrapolate via Cheby with BC mu->0 (lower) / 1 (upper).
5. Evaluate at p_grid; report max/median error vs mu_strict, restricted
   to the in-support region (avoiding the no-roots-fallback artifact).
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


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_pipeline_v2"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def detect_support_per_slice(p_grid, mu_slice, fallback=0.5, thresh=1e-9):
    """Detect contiguous in-support region: longest run of indices where
    |mu - fallback| > thresh (avoiding the no-roots-fallback artifact)."""
    diff = np.abs(mu_slice - fallback) > thresh
    if not diff.any():
        return None, None
    runs = []; in_run = False; start = 0
    for i, b in enumerate(diff):
        if b and not in_run:
            start = i; in_run = True
        elif not b and in_run:
            runs.append((start, i-1)); in_run = False
    if in_run:
        runs.append((start, len(diff)-1))
    if not runs: return None, None
    longest = max(runs, key=lambda r: r[1] - r[0])
    return longest[0], longest[1]


def fit_segment_cheby(p_seg, mu_seg, deg):
    n = len(p_seg)
    if n == 0: return None
    deg_eff = min(deg, n - 1)
    if deg_eff < 0: return None
    if n == 1: return ("const", float(mu_seg[0]))
    p_lo, p_hi = float(p_seg[0]), float(p_seg[-1])
    if p_hi <= p_lo: return ("const", float(mu_seg.mean()))
    xi = 2 * (p_seg - p_lo) / (p_hi - p_lo) - 1
    V = cheb.chebvander(xi, deg_eff)
    coefs, *_ = np.linalg.lstsq(V, mu_seg, rcond=None)
    return ("cheby", p_lo, p_hi, coefs)


def eval_segment(seg, p):
    if seg is None: return None
    if seg[0] == "const": return seg[1]
    _, p_lo, p_hi, coefs = seg
    xi = 2 * (p - p_lo) / (p_hi - p_lo) - 1
    return float(cheb.chebval(xi, coefs))


def build_zero_h_slice(p_grid, mu_slice, p_cs_slice, deg=6):
    """Build per-segment Cheby for a single u_k slice."""
    i_lo, i_hi = detect_support_per_slice(p_grid, mu_slice)
    if i_lo is None: return None, None, None
    p_sup_lo, p_sup_hi = float(p_grid[i_lo]), float(p_grid[i_hi])
    # Use only cusps within the support
    p_cs_in = [pc for pc in p_cs_slice if p_sup_lo < pc < p_sup_hi]
    knots = sorted([p_sup_lo] + p_cs_in + [p_sup_hi])
    segments = []
    for k in range(len(knots) - 1):
        a, b = knots[k], knots[k+1]
        mask = (p_grid >= a) & (p_grid <= b)
        if not mask.any():
            segments.append(None); continue
        seg = fit_segment_cheby(p_grid[mask], mu_slice[mask], deg)
        segments.append(seg)
    return np.array(knots), segments, (i_lo, i_hi)


def evaluate_slice(p, knots, segments, support_idx, fallback=0.5):
    """Evaluate at scalar p. Returns fallback outside support."""
    if p < knots[0] or p > knots[-1]: return fallback
    for k in range(len(knots) - 1):
        if knots[k] <= p <= knots[k+1]:
            return eval_segment(segments[k], p) or fallback
    return fallback


def test_cell(gamma, tau, deg=6):
    print(f"\n=== g={gamma}, tau={tau}, deg={deg} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    mu_strict = fp["mu_strict"].astype(np.float64)
    G_p, G = mu_strict.shape
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)

    # Use INTERNAL mu-based detector (pooled across u_k)
    p_cs = detect_critical_points_from_mu(mu_strict, p_grid)
    print(f"  internal detector: {len(p_cs)} cusps")

    # Per-slice build
    mu_pipe = np.full((G_p, G), 0.5)
    in_support_mask = np.zeros((G_p, G), dtype=bool)
    n_segs_per_slice = []
    for k in range(G):
        knots, segs, sup = build_zero_h_slice(p_grid, mu_strict[:, k], p_cs, deg=deg)
        if knots is None: n_segs_per_slice.append(0); continue
        n_segs_per_slice.append(len(segs))
        for ip in range(G_p):
            if p_grid[ip] >= knots[0] and p_grid[ip] <= knots[-1]:
                mu_pipe[ip, k] = evaluate_slice(p_grid[ip], knots, segs, sup)
                in_support_mask[ip, k] = True

    # In-support error metrics
    err = np.abs(mu_pipe - mu_strict)
    in_err = err[in_support_mask]
    print(f"  in-support points: {in_support_mask.sum()}/{mu_strict.size}")
    print(f"  in-support max = {in_err.max():.3e}")
    print(f"  in-support median = {np.median(in_err):.3e}")
    print(f"  in-support RMS = {np.sqrt(np.mean(in_err**2)):.3e}")
    # Out-of-support: both pipeline and mu_strict at fallback 0.5; should be zero err
    out_err = err[~in_support_mask]
    print(f"  out-of-support max = {out_err.max():.3e} (should be ~0)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    log_err_masked = np.where(in_support_mask, np.log10(err + 1e-20), np.nan)
    im = axes[0, 0].imshow(log_err_masked, aspect="auto", origin="lower",
                              extent=[u_grid[0], u_grid[-1], p_grid[0], p_grid[-1]],
                              cmap='viridis')
    axes[0, 0].set_xlabel("u_k"); axes[0, 0].set_ylabel("p")
    axes[0, 0].set_title("log10|err| (in-support only, NaN out)")
    plt.colorbar(im, ax=axes[0, 0])

    axes[0, 1].hist(np.log10(in_err + 1e-20), bins=50)
    axes[0, 1].axvline(np.log10(in_err.max()), color='r', label=f'max={in_err.max():.2e}')
    axes[0, 1].set_xlabel('log10 |error| (in-support)')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    k0 = G // 2
    axes[1, 0].plot(p_grid, mu_strict[:, k0], 'k-', label='strict', lw=2)
    axes[1, 0].plot(p_grid, mu_pipe[:, k0], 'r--', label='pipeline', alpha=0.7)
    for pc in p_cs:
        axes[1, 0].axvline(pc, color='gray', alpha=0.2, lw=0.5)
    axes[1, 0].set_xlabel('p'); axes[1, 0].set_ylabel(f'mu(p, u_k=0)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(u_grid, n_segs_per_slice, '-o')
    axes[1, 1].set_xlabel('u_k'); axes[1, 1].set_ylabel('n segments')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}_deg{deg}.png", dpi=120)
    plt.close()

    return dict(gamma=gamma, tau=tau, deg=deg, n_cusps=len(p_cs),
                in_support_max=float(in_err.max()),
                in_support_median=float(np.median(in_err)),
                in_support_rms=float(np.sqrt(np.mean(in_err**2))),
                out_max=float(out_err.max()),
                n_in_support=int(in_support_mask.sum()),
                n_total=int(mu_strict.size))


def main():
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    results = []
    for g, t in cells:
        for deg in [4, 6, 8]:
            try:
                r = test_cell(g, t, deg=deg)
                results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL: {e}")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"{'cell':>15} {'deg':>4} {'cusps':>6} {'IS-max':>12} {'IS-med':>12} {'OS-max':>12}")
    for r in results:
        print(f"g{r['gamma']:>4g}_t{r['tau']:.2f}".rjust(15) +
              f" {r['deg']:>4d} {r['n_cusps']:>6d} {r['in_support_max']:>12.3e} "
              f"{r['in_support_median']:>12.3e} {r['out_max']:>12.3e}")


if __name__ == "__main__":
    main()
