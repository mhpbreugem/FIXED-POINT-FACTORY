"""Production zero-h lookup pipeline.

Combines (1) per-segment Chebyshev fit within support (machine eps from
Finding 10) with (2) tail-asymptotic extrapolation outside support
(mu -> 0 as p -> 0, mu -> 1 as p -> 1, with derivative matching).

Pipeline:
  1. Load strict-h=0 P and mu_strict (best-iterate FP).
  2. Detect critical p_c via dd_k3_critpts.find_critical_points.
  3. Segment [p_min_support, p_max_support] at p_c knots; degree-6
     Cheby per segment via LSQ.
  4. Tail extrapolation: for p in [eps, p_min_support], use a Cheby
     that matches mu(p_min) value and dmu(p_min) and pins mu(eps) = 0.
     Same for upper tail.
  5. Evaluate at p_grid; report max error vs mu_strict globally.

Goal: machine eps globally.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_critpts import find_critical_points


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_pipeline"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def detect_support(p_grid, mu_strict_slice, threshold_low=1e-6, threshold_high=1-1e-6):
    """Find p_min, p_max where mu_strict has 'meaningful' info.
    Below p_min, mu ~ 0; above p_max, mu ~ 1."""
    # Find first/last p where mu_strict is between threshold and 1-threshold
    mask = (mu_strict_slice > threshold_low) & (mu_strict_slice < threshold_high)
    if not mask.any(): return None, None
    indices = np.where(mask)[0]
    return p_grid[indices[0]], p_grid[indices[-1]]


def fit_segment_cheby(p_seg, mu_seg, deg):
    """Fit degree-deg Cheby to (p_seg, mu_seg). Auto-limit degree
    if too few samples."""
    n = len(p_seg)
    if n == 0: return None
    deg_eff = min(deg, n - 1)
    if deg_eff < 0: return None
    if n == 1: return ("const", float(mu_seg[0]))
    p_lo, p_hi = float(p_seg[0]), float(p_seg[-1])
    if p_hi <= p_lo: return ("const", float(mu_seg.mean()))
    # Map p to [-1, 1]
    xi = 2 * (p_seg - p_lo) / (p_hi - p_lo) - 1
    V = cheb.chebvander(xi, deg_eff)
    coefs, *_ = np.linalg.lstsq(V, mu_seg, rcond=None)
    return ("cheby", p_lo, p_hi, coefs)


def eval_segment(seg, p):
    """Evaluate fitted segment at p (scalar)."""
    if seg is None: return 0.5
    if seg[0] == "const": return seg[1]
    _, p_lo, p_hi, coefs = seg
    xi = 2 * (p - p_lo) / (p_hi - p_lo) - 1
    return float(cheb.chebval(xi, coefs))


def fit_tail_cheby(p_tail_min, p_min_support, mu_min_support, dmu_min_support,
                       deg=4, side="lower"):
    """Fit Cheby on [p_tail_min, p_min_support] (lower tail) with BCs:
      mu(p_tail_min) = 0  (lower) or 1 (upper)
      mu(p_min_support) = mu_min_support
      mu'(p_min_support) = dmu_min_support
    Then add smoothness to fill the rest.
    """
    if p_tail_min >= p_min_support: return None
    # Use BCs as constraints; degree 3 needed for 4 BCs.
    # Map to [-1, 1] on [p_tail_min, p_min_support]
    p_lo, p_hi = p_tail_min, p_min_support
    L = p_hi - p_lo
    # 4 BC constraints, choose deg=3 (4 coefs):
    # cheb(-1) = bc_left (=0 or 1)
    # cheb(+1) = mu_min_support
    # dcheb/dp (+1) = dmu_min_support
    # cheb(0) = something smooth  -- for deg 3 only 3 conditions; add monotonicity by choosing midpoint
    bc_left = 0.0 if side == "lower" else 1.0
    # Build 4 constraints as linear system on 4 cheby coefs c_0, c_1, c_2, c_3:
    # cheb(-1) = c_0 - c_1 + c_2 - c_3  = bc_left
    # cheb(+1) = c_0 + c_1 + c_2 + c_3  = mu_min_support
    # dcheb/dp at xi=+1: dxi/dp = 2/L; dT_n/dxi at xi=+1 = n^2 (well-known)
    # So sum_n c_n * n^2 * (2/L) = dmu_min_support
    # 4th constraint: smoothness at xi=-1: dcheb/dp = ? we don't have BC, use deg 2 (3 coefs)
    # Use deg=2: 3 coefs, 3 constraints
    # T_0(-1)=1, T_1(-1)=-1, T_2(-1)=1
    # T_0(+1)=1, T_1(+1)=1, T_2(+1)=1
    # d T_n /d xi at +1 = n^2
    A = np.array([
        [1, -1, 1],
        [1, +1, 1],
        [0, (2/L), (2/L)*4],  # n^2 * (2/L), but T_n derivative at xi=+1 = n^2
    ])
    # Wait, derivative of T_n at xi=+1 is n^2 (correct), so dcheb/dxi |+1 = sum c_n * n^2
    # dcheb/dp = dcheb/dxi * (2/L)
    A_corr = np.array([
        [1, -1, 1],
        [1, +1, 1],
        [0, 1*(2/L), 4*(2/L)],
    ])
    b = np.array([bc_left, mu_min_support, dmu_min_support])
    try:
        coefs = np.linalg.solve(A_corr, b)
    except np.linalg.LinAlgError:
        return None
    return ("cheby", p_lo, p_hi, coefs)


def build_zero_h_lookup(p_grid, mu_strict_slice, p_cs_in_support, deg_in=6, deg_tail=2,
                            tail_eps=1e-9):
    """Full pipeline for one u_k slice."""
    # Detect support
    p_min_sup, p_max_sup = detect_support(p_grid, mu_strict_slice)
    if p_min_sup is None or p_max_sup is None:
        return None, None, None, None
    # Segment in-support region at critical p_c knots (keep only those inside support)
    p_cs_used = [pc for pc in p_cs_in_support if p_min_sup < pc < p_max_sup]
    knots = np.array(sorted([p_min_sup] + p_cs_used + [p_max_sup]))
    segments = []
    for k in range(len(knots) - 1):
        a, b = knots[k], knots[k+1]
        mask = (p_grid >= a) & (p_grid <= b)
        p_seg = p_grid[mask]
        mu_seg = mu_strict_slice[mask]
        if len(p_seg) < 2:
            segments.append(None)
            continue
        segments.append(fit_segment_cheby(p_seg, mu_seg, deg_in))
    # Outermost segment values + slopes for tail extrapolation
    # Lower tail: from tail_eps to p_min_sup
    lower_seg = None
    if p_min_sup > tail_eps:
        first_in_sup_idx = np.searchsorted(p_grid, p_min_sup)
        if first_in_sup_idx + 2 < len(p_grid):
            mu_at_min = float(mu_strict_slice[first_in_sup_idx])
            mu_next = float(mu_strict_slice[first_in_sup_idx + 1])
            dmu_at_min = (mu_next - mu_at_min) / (p_grid[first_in_sup_idx + 1] - p_grid[first_in_sup_idx])
            lower_seg = fit_tail_cheby(tail_eps, p_min_sup, mu_at_min, dmu_at_min, side="lower")
    upper_seg = None
    if p_max_sup < 1 - tail_eps:
        last_in_sup_idx = np.searchsorted(p_grid, p_max_sup) - 1
        if last_in_sup_idx - 1 >= 0:
            mu_at_max = float(mu_strict_slice[last_in_sup_idx])
            mu_prev = float(mu_strict_slice[last_in_sup_idx - 1])
            dmu_at_max = (mu_at_max - mu_prev) / (p_grid[last_in_sup_idx] - p_grid[last_in_sup_idx - 1])
            upper_seg = fit_tail_cheby(p_max_sup, 1 - tail_eps, mu_at_max, dmu_at_max, side="upper")
    return knots, segments, lower_seg, upper_seg


def evaluate_pipeline(p, knots, segments, lower_seg, upper_seg, mu_default=0.5):
    """Evaluate the pipeline at scalar p."""
    if p < knots[0]:
        if lower_seg is not None: return eval_segment(lower_seg, p)
        return 0.0
    if p > knots[-1]:
        if upper_seg is not None: return eval_segment(upper_seg, p)
        return 1.0
    # find segment
    for k in range(len(knots) - 1):
        if knots[k] <= p <= knots[k+1]:
            seg = segments[k]
            if seg is None: return mu_default
            return eval_segment(seg, p)
    return mu_default


def test_cell(gamma, tau, deg_in=6):
    print(f"\n=== g={gamma}, tau={tau}, deg_in={deg_in} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    P = fp["P_strict"].astype(np.float64)
    mu_strict = fp["mu_strict"].astype(np.float64)
    G_p, G = mu_strict.shape
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)

    # critical p_c from saved JSON
    cp = json.load(open(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/critpts/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.json"))
    p_cs_all = sorted(cp["p_c_values"])
    print(f"  {len(p_cs_all)} critical p_c values")

    # Per u_k slice
    mu_pipe = np.empty((G_p, G))
    n_seg_total = 0
    for k_node in range(G):
        knots, segs, lo_seg, up_seg = build_zero_h_lookup(
            p_grid, mu_strict[:, k_node], p_cs_all, deg_in=deg_in)
        if segs is None:
            mu_pipe[:, k_node] = 0.5; continue
        n_seg_total += len(segs)
        for ip, p in enumerate(p_grid):
            mu_pipe[ip, k_node] = evaluate_pipeline(p, knots, segs, lo_seg, up_seg)

    err = np.abs(mu_pipe - mu_strict)
    max_e = float(np.max(err)); med_e = float(np.median(err)); rms_e = float(np.sqrt(np.mean(err**2)))
    print(f"  pipeline vs mu_strict: max={max_e:.3e}, median={med_e:.3e}, RMS={rms_e:.3e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # Heatmap of error
    im = axes[0, 0].imshow(np.log10(err + 1e-20), aspect="auto", origin="lower",
                              extent=[u_grid[0], u_grid[-1], p_grid[0], p_grid[-1]])
    axes[0, 0].set_xlabel("u_k"); axes[0, 0].set_ylabel("p")
    axes[0, 0].set_title(f"log10|err| pipeline vs strict")
    plt.colorbar(im, ax=axes[0, 0])

    # Histogram
    axes[0, 1].hist(np.log10(err.ravel() + 1e-20), bins=60)
    axes[0, 1].axvline(np.log10(max_e), color='r', label=f'max={max_e:.2e}')
    axes[0, 1].axvline(np.log10(med_e+1e-20), color='g', label=f'med={med_e:.2e}')
    axes[0, 1].set_xlabel('log10 |error|'); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # mu(p, u_k=0) curves
    k0 = G // 2
    axes[1, 0].plot(p_grid, mu_strict[:, k0], 'k-', label='strict', lw=2)
    axes[1, 0].plot(p_grid, mu_pipe[:, k0], 'r--', label='pipeline', alpha=0.7)
    axes[1, 0].set_xlabel('p'); axes[1, 0].set_ylabel(f'mu(p, u_k=0)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('mu curves')

    # error per u_k
    e_per_uk = err.max(axis=0)
    axes[1, 1].semilogy(u_grid, e_per_uk, '-o')
    axes[1, 1].set_xlabel('u_k'); axes[1, 1].set_ylabel('max |error|')
    axes[1, 1].set_title('max error vs u_k')
    axes[1, 1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}_deg{deg_in}.png", dpi=120)
    plt.close()

    return dict(gamma=gamma, tau=tau, deg_in=deg_in,
                n_critpts=len(p_cs_all), n_seg_total=n_seg_total,
                max_err=max_e, median_err=med_e, rms_err=rms_e)


def main():
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    results = []
    for g, t in cells:
        for deg in [4, 6, 8]:
            try:
                r = test_cell(g, t, deg_in=deg)
                results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAIL: {e}")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(f"{'cell':>15} {'deg':>4} {'n_pc':>5} {'max':>10} {'med':>10} {'rms':>10}")
    for r in results:
        print(f"g{r['gamma']:>4g}_t{r['tau']:.2f}".rjust(15) +
              f" {r['deg_in']:>4d} {r['n_critpts']:>5d} {r['max_err']:>10.3e} {r['median_err']:>10.3e} {r['rms_err']:>10.3e}")


if __name__ == "__main__":
    main()
