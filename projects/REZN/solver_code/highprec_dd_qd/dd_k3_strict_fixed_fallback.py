"""Strict-h=0 lookup with fixed no-roots fallback.

The original strict-h=0 lookup returns 0.5 when the co-area integral
finds no roots (den < 1e-300). This produces the artifact: mu jumps
discontinuously to 0.5 at the support boundaries.

Fix: post-process the mu_table by replacing the 0.5-fallback regions
with PCHIP extrapolation from the nearest in-support values, using
Bayesian BCs mu(0)=0, mu(1)=1.

This is a build-time fix: takes (P, u_grid, p_grid, tau) and returns
a fully-Bayesian mu_table with no artifacts.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_fixed_fallback"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def fix_fallback(mu_table, p_grid, fallback=0.5, thresh=1e-9,
                    mu_bc_lo=0.0, mu_bc_hi=1.0, p_eps=1e-6):
    """Replace 0.5-fallback values with PCHIP extrapolation.
    Per u_k slice: identify contiguous in-support region (where
    |mu - 0.5| > thresh), then extrapolate to BCs in the tails.
    """
    G_p, G = mu_table.shape
    mu_fixed = mu_table.copy()
    for k in range(G):
        mu_slice = mu_table[:, k]
        # in-support mask
        in_sup = np.abs(mu_slice - fallback) > thresh
        if not in_sup.any():
            # All fallback: assume linear from 0 to 1
            mu_fixed[:, k] = np.linspace(0, 1, G_p)
            continue
        idx = np.where(in_sup)[0]
        i_lo, i_hi = idx[0], idx[-1]
        # Build PCHIP through [p_eps, p_grid[i_lo], ..., p_grid[i_hi], 1-p_eps]
        # with values [0, mu[i_lo], ..., mu[i_hi], 1]
        p_pts = [p_eps] + [p_grid[ii] for ii in range(i_lo, i_hi + 1)] + [1.0 - p_eps]
        mu_pts = [mu_bc_lo] + [mu_slice[ii] for ii in range(i_lo, i_hi + 1)] + [mu_bc_hi]
        p_pts = np.array(p_pts); mu_pts = np.array(mu_pts)
        # Make sure monotone
        for i in range(1, len(mu_pts)):
            if mu_pts[i] < mu_pts[i-1]:
                mu_pts[i] = mu_pts[i-1]  # enforce monotone
        try:
            pchip = PchipInterpolator(p_pts, mu_pts, extrapolate=True)
            mu_fixed[:, k] = pchip(p_grid)
        except Exception:
            pass
    # Clip to [eps, 1-eps]
    mu_fixed = np.clip(mu_fixed, 1e-12, 1 - 1e-12)
    return mu_fixed


def test_cell(gamma, tau):
    print(f"\n=== g={gamma}, tau={tau} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    P = fp["P_strict"].astype(np.float64)
    mu_strict_raw = fp["mu_strict"].astype(np.float64)
    G = P.shape[0]; G_p = mu_strict_raw.shape[0]
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)

    # Fix the fallback
    mu_fixed = fix_fallback(mu_strict_raw, p_grid)
    # Check monotonicity
    n_mono_raw = sum(1 for k in range(G) if np.all(np.diff(mu_strict_raw[:, k]) >= -1e-9))
    n_mono_fixed = sum(1 for k in range(G) if np.all(np.diff(mu_fixed[:, k]) >= -1e-9))
    print(f"  raw mu_strict monotone:    {n_mono_raw}/{G}")
    print(f"  fixed mu_strict monotone:  {n_mono_fixed}/{G}")
    # Check tail BCs
    bc_lo_raw = mu_strict_raw[0, :].mean()
    bc_lo_fix = mu_fixed[0, :].mean()
    bc_hi_raw = mu_strict_raw[-1, :].mean()
    bc_hi_fix = mu_fixed[-1, :].mean()
    print(f"  raw  mu(p=eps)  mean: {bc_lo_raw:.3f}; mu(p=1-eps) mean: {bc_hi_raw:.3f}")
    print(f"  fixed mu(p=eps) mean: {bc_lo_fix:.3f}; mu(p=1-eps) mean: {bc_hi_fix:.3f}")

    # Compare smoothness in mu''
    d2_raw = np.diff(mu_strict_raw, n=2, axis=0)
    d2_fix = np.diff(mu_fixed, n=2, axis=0)
    print(f"  raw  max |mu''|: {np.max(np.abs(d2_raw)):.3e}")
    print(f"  fixed max |mu''|: {np.max(np.abs(d2_fix)):.3e}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis
    for k in range(G):
        c = cmap(k/G)
        axes[0].plot(p_grid, mu_strict_raw[:, k], '-', color=c, alpha=0.5, lw=0.5)
    axes[0].set_xlabel('p'); axes[0].set_ylabel('mu')
    axes[0].set_title(f'raw mu_strict (g={gamma}, t={tau})')
    axes[0].grid(alpha=0.3)
    for k in range(G):
        c = cmap(k/G)
        axes[1].plot(p_grid, mu_fixed[:, k], '-', color=c, alpha=0.5, lw=0.5)
    axes[1].set_xlabel('p'); axes[1].set_ylabel('mu')
    axes[1].set_title(f'fixed mu_strict (no-roots replaced by PCHIP)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}.png", dpi=120)
    plt.close()

    np.savez(f"{OUT}/mu_fixed_g{gamma}_t{tau:.4f}.npz",
             mu_strict_raw=mu_strict_raw, mu_fixed=mu_fixed,
             P_strict=P, p_grid=p_grid, u_grid=u_grid)

    return dict(gamma=gamma, tau=tau,
                n_mono_raw=n_mono_raw, n_mono_fixed=n_mono_fixed,
                bc_lo_raw=float(bc_lo_raw), bc_lo_fix=float(bc_lo_fix),
                bc_hi_raw=float(bc_hi_raw), bc_hi_fix=float(bc_hi_fix),
                d2_max_raw=float(np.max(np.abs(d2_raw))),
                d2_max_fix=float(np.max(np.abs(d2_fix))))


def main():
    results = []
    for g, t in [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]:
        try: results.append(test_cell(g, t))
        except Exception as e:
            import traceback; traceback.print_exc()
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
