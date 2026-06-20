"""V5 final: pipeline applied to FIXED mu_strict (no fallback artifacts).

Combines:
- fix_fallback (PCHIP through in-support + Bayesian BCs in tails) → clean mu_strict
- pipeline v4 (per-segment Cheby + PCHIP tails) on the clean input.

This should be the COMPLETE solution: machine eps + monotone globally.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_optB_pieceCheby import detect_critical_points_from_mu
from dd_k3_zero_h_pipeline_v4 import build_full_v4, evaluate_full_v4
from dd_k3_strict_fixed_fallback import fix_fallback


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_pipeline_v5"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def test_cell(gamma, tau, deg=6):
    print(f"\n=== g={gamma}, tau={tau}, deg={deg} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    mu_raw = fp["mu_strict"].astype(np.float64)
    G_p, G = mu_raw.shape
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)

    # Step 1: clean mu_strict
    mu_clean = fix_fallback(mu_raw, p_grid)
    # Step 2: detect cusps in cleaned version
    p_cs = detect_critical_points_from_mu(mu_clean, p_grid)
    print(f"  cusps in clean mu: {len(p_cs)}")
    # Step 3: pipeline v4 on cleaned mu (treats whole [eps, 1-eps] as in-support since clean)
    # Detect support of clean mu
    mu_pipe = np.full((G_p, G), 0.5)
    for k in range(G):
        knots, segs = build_full_v4(p_grid, mu_clean[:, k], p_cs, deg=deg)
        if knots is None:
            mu_pipe[:, k] = mu_clean[:, k]; continue
        for ip in range(G_p):
            mu_pipe[ip, k] = evaluate_full_v4(p_grid[ip], knots, segs)

    err = np.abs(mu_pipe - mu_clean)
    print(f"  pipeline vs cleaned: max={err.max():.3e}, "
          f"median={np.median(err):.3e}")
    # Global monotonicity check on pipeline output (dense)
    p_dense = np.linspace(1e-6, 1 - 1e-6, 1001)
    n_mono = 0
    min_diff = float("inf")
    for k in range(G):
        # Rebuild for dense eval
        knots, segs = build_full_v4(p_grid, mu_clean[:, k], p_cs, deg=deg)
        if knots is None: continue
        mu_dense = np.array([evaluate_full_v4(p, knots, segs) for p in p_dense])
        m = float(np.min(np.diff(mu_dense)))
        min_diff = min(min_diff, m)
        if np.all(np.diff(mu_dense) >= -1e-9): n_mono += 1
    print(f"  pipeline monotone (dense): {n_mono}/{G}, min diff: {min_diff:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    cmap = plt.cm.viridis
    for k in range(G):
        c = cmap(k/G)
        axes[0].plot(p_grid, mu_raw[:, k], '-', color=c, alpha=0.5, lw=0.5)
        axes[1].plot(p_grid, mu_clean[:, k], '-', color=c, alpha=0.5, lw=0.5)
        axes[2].plot(p_grid, mu_pipe[:, k], '-', color=c, alpha=0.5, lw=0.5)
    axes[0].set_title('raw mu_strict (with fallback)'); axes[0].grid(alpha=0.3)
    axes[1].set_title('cleaned (fallback fixed)'); axes[1].grid(alpha=0.3)
    axes[2].set_title('pipeline v5 (deg=6 Cheby + PCHIP tail)'); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel('p'); ax.set_ylabel('mu')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}_deg{deg}.png", dpi=120)
    plt.close()

    np.savez(f"{OUT}/mu_g{gamma}_t{tau:.4f}.npz",
             mu_raw=mu_raw, mu_clean=mu_clean, mu_pipe=mu_pipe, p_grid=p_grid)
    return dict(gamma=gamma, tau=tau, deg=deg, n_cusps=len(p_cs),
                max_err_vs_clean=float(err.max()),
                median_err_vs_clean=float(np.median(err)),
                n_mono=n_mono, min_diff=float(min_diff))


def main():
    results = []
    for g, t in [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]:
        for deg in [6, 8]:
            try: results.append(test_cell(g, t, deg=deg))
            except Exception as e:
                import traceback; traceback.print_exc()
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"{'cell':>15} {'deg':>4} {'cusps':>6} {'max':>10} {'med':>10} {'mono':>6} {'min_d':>10}")
    for r in results:
        print(f"g{r['gamma']:>4g}_t{r['tau']:.2f}".rjust(15) +
              f" {r['deg']:>4d} {r['n_cusps']:>6d} {r['max_err_vs_clean']:>10.2e} "
              f"{r['median_err_vs_clean']:>10.2e} {r['n_mono']:>6d} {r['min_diff']:>10.2e}")


if __name__ == "__main__":
    main()
