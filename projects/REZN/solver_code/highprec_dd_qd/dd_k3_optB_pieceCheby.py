"""Per-segment Chebyshev lookup representation for mu(p, u_k).

Concept (parallel next-step B from the K=3 study summary)
=========================================================
The lookup function mu(p, u_k=fixed) is conjectured to have C^k cusps at a set
of critical-value p_c's. Between adjacent cusps, mu is C^infty smooth. We test
whether a piecewise Chebyshev polynomial representation -- with the segment
knots placed at the detected p_c's -- converges spectrally (machine eps for
modest degree d).

For each cell (gamma, tau) and each u_k slice:
  1. Detect critical p_c values from the strict-h=0 ground-truth mu_strict
     using a second-derivative magnitude detector (Finding-8 style: locate
     indices where |mu''(p)| spikes well above its background).
  2. Build segment knots = sorted([eps] + p_c_list + [1 - eps]).
  3. On each segment [pL, pR], fit a degree-d Chebyshev polynomial:
       - If we have >= d+1 mu_strict samples strictly inside, do an
         unconstrained LSQ Cheby fit (cheb.chebfit).
       - If fewer, drop d down to (n_samples - 1), capped at 0.
  4. Evaluate the piecewise representation back on the 121-pt p-grid and
     compute max/median error vs mu_strict.
  5. Sweep d = 2..8 and record per-segment effective degree, errors.

The key question: does per-segment Cheby reach machine eps for moderate d?
If yes -> cusps are clean C^k discontinuities, well-localized.
If max error plateaus around 1e-2 -> cusps are smeared / non-local.

CAVEAT: many segments will have only 1-3 mu_strict samples (segments narrow
where p_c values cluster). For those, low-degree Cheby (deg 0 or 1) is forced.
We report the fraction of segments hitting the sample-count constraint.
"""
import os
import sys
import json
import time
import numpy as np
import numpy.polynomial.chebyshev as cheb

sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
from lin_cdf_strict import make_p_grid  # logit-uniform 121-pt grid


# ----------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------
ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
FP_DIR = ROOT
CRITPT_DIR = os.path.join(ROOT, "critpts")
OUT_DIR = os.path.join(ROOT, "optB_pieceCheby")
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CRITPT_DIR, exist_ok=True)

G = 11
G_P = 121
EPS_P = 1e-4                # left/right end-cap offset for segments
DEG_SWEEP = list(range(2, 9))   # 2..8
MAX_DEG_HARD = 8                # cap for very long segments


# ----------------------------------------------------------------------
# Critical-point detector (Finding-8 style)
# ----------------------------------------------------------------------
def detect_critical_points_from_mu(mu_strict, p_grid,
                                       thresh_factor=5.0,
                                       min_sep_logit=0.04):
    """Detect cusp p_c values from a (G_p, G) strict mu_strict table.

    For each u_k slice, compute |mu''| via finite differences on the
    logit-uniform p-grid; mark indices where |mu''| exceeds
    thresh_factor * (median |mu''| of that slice). Pool across u_k
    (union of all detected points, then deduplicate by logit-distance).

    Returns
    -------
    np.ndarray of detected p_c values, strictly inside (0, 1), sorted.
    """
    G_p, G_ = mu_strict.shape
    pooled = []
    for k in range(G_):
        y = mu_strict[:, k]
        # second-difference (un-normalised by dp -- we just want the spike
        # pattern; the logit-uniform spacing varies near the boundary).
        d1 = np.diff(y) / np.diff(p_grid)
        d2 = np.diff(d1) / np.diff(p_grid)[:-1]
        a = np.abs(d2)
        if not np.any(a > 0):
            continue
        # background = median of the strictly-positive |d2| (excluding
        # plateau regions where the slice is constant).
        nz = a[a > 1e-12]
        if nz.size == 0:
            continue
        bg = np.median(nz)
        thresh = thresh_factor * bg
        # spike indices in the d2 array correspond to p_grid[1:-1]
        spike = np.where(a > thresh)[0]
        # also include the first-difference jump-detector: where d1 itself
        # jumps in magnitude (catches cusps that flank a non-zero plateau,
        # which the 2nd-difference can miss if the cusp is sandwiched
        # between identical jumps).
        ad1 = np.abs(np.diff(d1))
        if ad1.size > 0 and np.max(ad1) > 0:
            bg1 = np.median(ad1[ad1 > 1e-12]) if np.any(ad1 > 1e-12) else 0.0
            spike1 = np.where(ad1 > thresh_factor * max(bg1, 1e-12))[0]
            spike = np.union1d(spike, spike1)
        for s in spike:
            pooled.append(float(p_grid[1 + s]))
    if not pooled:
        return np.array([])
    pooled.sort()
    # merge close-by detections (use logit-distance for fair merging)
    def logit(p):
        pc = max(min(p, 1.0 - 1e-12), 1e-12)
        return np.log(pc / (1 - pc))
    merged = [pooled[0]]
    for p in pooled[1:]:
        if logit(p) - logit(merged[-1]) > min_sep_logit:
            merged.append(p)
    return np.asarray(merged)


def _merge_logit(pcs, min_sep=0.04):
    if len(pcs) == 0:
        return np.array([])
    pcs = np.sort(np.asarray(pcs, dtype=float))
    def logit(p):
        pc = max(min(p, 1.0 - 1e-12), 1e-12)
        return np.log(pc / (1.0 - pc))
    out = [pcs[0]]
    for p in pcs[1:]:
        if logit(p) - logit(out[-1]) > min_sep:
            out.append(p)
    return np.asarray(out)


def load_or_detect_critpts(key, mu_strict, p_grid, mode="union"):
    """Acquire critical p_c values for one cell.

    mode = "union"    -> Finding-8 JSON + internal mu-based detector,
                          deduplicated by logit-distance (BEST: covers cusps
                          missed by either detector alone).
    mode = "f8_only"  -> use only the Finding-8 critpts JSON if it exists.
    mode = "internal" -> use only the internal mu-based detector.

    Always caches the *internal* detector output to CRITPT_DIR if no
    Finding-8 JSON was found there (so downstream consumers see something).
    """
    f8_path = os.path.join(CRITPT_DIR, f"dd_k3_strict_fp_{key}.json")
    pc_f8 = None
    if os.path.exists(f8_path) and os.path.getsize(f8_path) > 10:
        try:
            j = json.load(open(f8_path))
            pc_f8 = np.asarray(j["p_c_values"], dtype=float)
        except (json.JSONDecodeError, KeyError):
            pc_f8 = None
    pc_internal = detect_critical_points_from_mu(mu_strict, p_grid)
    if pc_f8 is None:
        # cache our detection so the file is non-empty for later consumers
        json.dump({"file": f"dd_k3_strict_fp_{key}.npz",
                   "G": int(mu_strict.shape[1]),
                   "p_c_values": pc_internal.tolist(),
                   "detector": "internal_second_diff_pool_fallback"},
                  open(f8_path, "w"), indent=2)
        return pc_internal, "internal_fallback"
    if mode == "f8_only":
        return pc_f8, "f8_only"
    if mode == "internal":
        return pc_internal, "internal_only"
    # union (default)
    pc_union = _merge_logit(np.concatenate([pc_f8, pc_internal]),
                                  min_sep=0.04)
    return pc_union, f"union_f8={len(pc_f8)}_int={len(pc_internal)}_u={len(pc_union)}"


# ----------------------------------------------------------------------
# Per-segment Chebyshev fit
# ----------------------------------------------------------------------
def fit_segment_cheby(p_seg_samples, y_seg_samples, pL, pR, d_target):
    """Fit a degree-d Cheb polynomial on segment [pL, pR] to the samples.

    Uses LSQ in the Cheb basis on the rescaled coordinate
        x(p) = 2*(p - pL)/(pR - pL) - 1  in [-1, 1].
    Drops degree if too few samples.

    Returns (coefs, deg_eff). coefs is a length-(deg_eff+1) Cheb-coef array.
    """
    n = len(p_seg_samples)
    if n == 0:
        # No interior samples -> constant at midpoint guess (mean of endpoints
        # is unknown; just emit 0.5 as a neutral fallback). We never expect
        # this since the segment is bounded by p_c's of the lookup table.
        return np.array([0.5]), 0
    deg_eff = min(d_target, n - 1, MAX_DEG_HARD)
    if deg_eff <= 0:
        # piecewise-constant fit = mean of samples
        return np.array([float(np.mean(y_seg_samples))]), 0
    # map p -> x in [-1, 1]
    width = pR - pL
    if width <= 0:
        return np.array([float(np.mean(y_seg_samples))]), 0
    x = 2.0 * (np.asarray(p_seg_samples) - pL) / width - 1.0
    # chebfit returns coeffs in the standard Cheb basis (T_0..T_deg)
    coefs = cheb.chebfit(x, np.asarray(y_seg_samples), deg_eff)
    return coefs, deg_eff


def build_piecewise_cheby(mu_slice, p_grid, knots, d_target):
    """Build a piecewise Cheb representation for one u_k slice.

    Parameters
    ----------
    mu_slice : (G_p,) array of mu_strict[:, k]
    p_grid   : (G_p,) array of p-grid points
    knots    : sorted (n_knot,) array including the two endpoints
    d_target : target Cheb degree per segment

    Returns
    -------
    list of dicts: [{pL, pR, coefs, deg_eff, n_samples, n_inside}, ...]
    """
    segs = []
    for j in range(len(knots) - 1):
        pL = float(knots[j]); pR = float(knots[j + 1])
        # samples strictly within [pL, pR], inclusive on the closer side
        # to make sure boundary samples are used by at least one segment.
        mask = (p_grid >= pL) & (p_grid <= pR)
        p_in = p_grid[mask]
        y_in = mu_slice[mask]
        coefs, deg_eff = fit_segment_cheby(p_in, y_in, pL, pR, d_target)
        segs.append(dict(pL=pL, pR=pR, coefs=coefs, deg_eff=int(deg_eff),
                          n_samples=int(len(p_in))))
    return segs


def eval_piecewise_cheby(p_eval, segs):
    """Evaluate the piecewise Cheb at a set of p values (vectorized per
    segment)."""
    out = np.empty_like(p_eval, dtype=float)
    # for points outside all segments (shouldn't happen if [eps, 1-eps] covers
    # the grid), fall back to nearest segment.
    placed = np.zeros_like(p_eval, dtype=bool)
    for seg in segs:
        pL = seg["pL"]; pR = seg["pR"]; coefs = seg["coefs"]
        width = pR - pL
        m = (p_eval >= pL) & (p_eval <= pR) & (~placed)
        if not np.any(m):
            continue
        x = 2.0 * (p_eval[m] - pL) / width - 1.0
        out[m] = cheb.chebval(x, coefs)
        placed[m] = True
    if not np.all(placed):
        # snap any leftover to the nearest segment
        idx_left = np.where(~placed)[0]
        for i in idx_left:
            best = min(segs,
                       key=lambda s: min(abs(p_eval[i] - s["pL"]),
                                          abs(p_eval[i] - s["pR"])))
            width = best["pR"] - best["pL"]
            x = 2.0 * (p_eval[i] - best["pL"]) / width - 1.0
            x = max(-1.0, min(1.0, x))
            out[i] = cheb.chebval(x, best["coefs"])
    return out


# ----------------------------------------------------------------------
# Per-cell driver: sweep degree, record per-segment / aggregate errors
# ----------------------------------------------------------------------
def process_cell(key, gamma, tau, mode="union"):
    fp_path = os.path.join(FP_DIR, f"dd_k3_strict_fp_{key}.npz")
    d = np.load(fp_path)
    mu_strict = d["mu_strict"]                     # (G_p=121, G=11)
    assert mu_strict.shape == (G_P, G), \
        f"unexpected mu_strict shape {mu_strict.shape}"
    p_grid = make_p_grid(G_P)
    # 1) critical points (load Finding-8 JSON or detect)
    pc, src = load_or_detect_critpts(key, mu_strict, p_grid, mode=mode)
    # build knots
    knots = np.concatenate([[EPS_P], np.sort(pc), [1.0 - EPS_P]])
    # drop near-duplicates and clamp to the working window
    knots = knots[(knots >= EPS_P) & (knots <= 1.0 - EPS_P)]
    if knots[0] > EPS_P:
        knots = np.concatenate([[EPS_P], knots])
    if knots[-1] < 1.0 - EPS_P:
        knots = np.concatenate([knots, [1.0 - EPS_P]])
    # deduplicate
    knots = np.unique(np.round(knots, 12))
    n_seg = len(knots) - 1
    # 2) per-degree sweep, per u_k slice
    # results[d_target] = dict with arrays
    cell_results = {
        "key": key, "gamma": gamma, "tau": tau,
        "n_critpts": int(len(pc)),
        "critpts_source": src,
        "n_segments": int(n_seg),
        "knots": knots.tolist(),
        "p_c_values": pc.tolist(),
        "deg_results": {},
    }
    # samples-per-segment statistics
    samples_per_seg = []
    for j in range(n_seg):
        pL = knots[j]; pR = knots[j + 1]
        n_in = int(np.sum((p_grid >= pL) & (p_grid <= pR)))
        samples_per_seg.append(n_in)
    cell_results["samples_per_segment"] = samples_per_seg
    cell_results["frac_seg_le_2_samples"] = float(
        np.mean(np.asarray(samples_per_seg) <= 2))
    cell_results["frac_seg_le_3_samples"] = float(
        np.mean(np.asarray(samples_per_seg) <= 3))
    # sweep degrees
    for d_target in DEG_SWEEP:
        # per-slice errors at p_grid
        err_grid = np.zeros((G_P, G))
        # per-slice/per-segment effective degrees
        deg_eff_arr = np.zeros((G, n_seg), dtype=int)
        for k in range(G):
            mu_slice = mu_strict[:, k]
            segs = build_piecewise_cheby(mu_slice, p_grid, knots, d_target)
            for j, s in enumerate(segs):
                deg_eff_arr[k, j] = s["deg_eff"]
            mu_eval = eval_piecewise_cheby(p_grid, segs)
            err_grid[:, k] = mu_eval - mu_slice
        max_err = float(np.max(np.abs(err_grid)))
        med_err = float(np.median(np.abs(err_grid)))
        per_uk_max = np.max(np.abs(err_grid), axis=0).tolist()
        cell_results["deg_results"][str(d_target)] = {
            "max_err": max_err,
            "median_err": med_err,
            "per_uk_max_err": per_uk_max,
            "deg_eff_mean": float(deg_eff_arr.mean()),
            "deg_eff_min": int(deg_eff_arr.min()),
            "deg_eff_max": int(deg_eff_arr.max()),
            "deg_eff_per_uk_seg": deg_eff_arr.tolist(),
        }
        print(f"    d={d_target}: max={max_err:.3e}  med={med_err:.3e}  "
              f"deg_eff in [{deg_eff_arr.min()},{deg_eff_arr.max()}] "
              f"(mean {deg_eff_arr.mean():.2f})", flush=True)
    return cell_results


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def make_figures(all_results, p_grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, R in all_results.items():
        # ---- (1) convergence: max & median err vs degree ----
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
        ds = sorted(int(d) for d in R["deg_results"])
        max_errs = [R["deg_results"][str(d)]["max_err"] for d in ds]
        med_errs = [R["deg_results"][str(d)]["median_err"] for d in ds]
        ax.semilogy(ds, max_errs, "o-", label="max |err|")
        ax.semilogy(ds, med_errs, "s--", label="median |err|")
        ax.axhline(1e-12, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("target Cheb degree per segment")
        ax.set_ylabel("piecewise-Cheby vs mu_strict")
        ax.set_title(f"{key} ($\\gamma={R['gamma']}, \\tau={R['tau']}$): "
                     f"{R['n_segments']} segments, "
                     f"{R['n_critpts']} critpts")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"converge_{key}.png"), dpi=130)
        plt.close(fig)

        # ---- (2) per-segment effective degree heatmap at d_target = 8 ----
        deg_eff = np.asarray(R["deg_results"]["8"]["deg_eff_per_uk_seg"])
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.5))
        im = ax.imshow(deg_eff, aspect="auto", cmap="viridis",
                            vmin=0, vmax=8, origin="lower")
        ax.set_xlabel("segment index (between p_c's)")
        ax.set_ylabel("u_k slice")
        ax.set_title(f"{key}: effective Cheb degree per segment "
                     f"(target 8). Mean = {deg_eff.mean():.2f}")
        fig.colorbar(im, ax=ax, label="deg_eff")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"deg_heat_{key}.png"), dpi=130)
        plt.close(fig)

        # ---- (3) sample mu slice with reconstruction & knots ----
        # Pick u_k slice with largest mu range for visual.
        mu_strict = np.load(os.path.join(FP_DIR,
                                f"dd_k3_strict_fp_{key}.npz"))["mu_strict"]
        ranges = mu_strict.max(0) - mu_strict.min(0)
        k_show = int(np.argmax(ranges))
        knots = np.asarray(R["knots"])
        # rebuild a d=8 piecewise rep on the fly for the plot
        segs = build_piecewise_cheby(mu_strict[:, k_show], p_grid, knots, 8)
        # dense plotting grid
        p_dense = np.linspace(p_grid[0], p_grid[-1], 2000)
        mu_eval = eval_piecewise_cheby(p_dense, segs)
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))
        ax.plot(p_grid, mu_strict[:, k_show], "ko", ms=3,
                  label=f"mu_strict (u_k={k_show})")
        ax.plot(p_dense, mu_eval, "b-", lw=1.0, label="piecewise-Cheb d=8")
        for kn in knots[1:-1]:
            ax.axvline(kn, color="r", lw=0.4, alpha=0.5)
        ax.set_xlabel("p"); ax.set_ylabel(r"$\mu(p, u_k)$")
        ax.set_title(f"{key} u_k={k_show}: "
                     f"{len(segs)} Cheb segments. Red lines = p_c")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"slice_{key}.png"), dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p_grid = make_p_grid(G_P)
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    modes = ["f8_only", "internal", "union"]
    all_results = {m: {} for m in modes}
    t0 = time.time()
    for mode in modes:
        print(f"\n############## MODE = {mode} ##############", flush=True)
        for gamma, tau in cells:
            key = f"g{gamma}_t{tau:.4f}"
            print(f"\n=== {key} [{mode}] ===", flush=True)
            try:
                R = process_cell(key, gamma, tau, mode=mode)
            except FileNotFoundError as e:
                print(f"  SKIP -- {e}", flush=True)
                continue
            all_results[mode][key] = R
            print(f"  n_critpts={R['n_critpts']}  n_seg={R['n_segments']}  "
                  f"frac<=2-samples={R['frac_seg_le_2_samples']:.2f}",
                  flush=True)
    # save JSON
    out_json = os.path.join(OUT_DIR, "pieceCheby_results.json")
    json.dump(all_results, open(out_json, "w"), indent=2)
    print(f"\nJSON -> {out_json}", flush=True)
    # figures: build for the "union" mode (the headline result)
    print("Building figures (union mode)...", flush=True)
    make_figures(all_results["union"], p_grid)
    # summary
    print(f"\n--- summary: max|err| at d=8, per mode ---", flush=True)
    print(f"{'cell':<22} {'f8_only':>12} {'internal':>12} {'union':>12}",
          flush=True)
    for key in [f"g{g}_t{t:.4f}" for g, t in cells]:
        row = [f"{key:<22}"]
        for mode in modes:
            if key in all_results[mode]:
                me = all_results[mode][key]["deg_results"]["8"]["max_err"]
                row.append(f"{me:>12.3e}")
            else:
                row.append(f"{'--':>12}")
        print(" ".join(row), flush=True)
    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)
    return all_results


if __name__ == "__main__":
    main()
