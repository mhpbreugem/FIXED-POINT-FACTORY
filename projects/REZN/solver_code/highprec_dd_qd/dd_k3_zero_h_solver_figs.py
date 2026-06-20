"""Figures for the zero-h FP solver report.

Generates:
  fig1_convergence.png   -- |F|_inf per iteration (all 6 runs).
  fig2_mu_smoothing.png  -- mu_smooth vs mu_raw for tau=1.0 at warm-start.
  fig3_P_diff.png        -- P_star - P0 heatmap (best-iterate vs warm).
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
from lin_cdf_strict import make_cdf_uniform_grid, make_p_grid

ROOT = "/home/user/FIXED-POINT-FACTORY"
OUT_DIR = f"{ROOT}/projects/REZN/solved_fixed_points/dd_k3_overnight/zero_h_solver"
FIG_DIR = f"{OUT_DIR}/figs"
os.makedirs(FIG_DIR, exist_ok=True)

# Pre-recorded |F| histories (from logs); also reload from saved JSON if available.
HISTORY = {
    # (gamma, tau, mode) -> list of F_inf per iter
    (100, 1.0, "recompute"): [5.768e-2, 4.818e-2, 4.230e-2, 5.252e-2,
                              2.482e-2, 1.911e-2, 1.689e-2, 2.578e-2,
                              1.486e-2, 2.236e-2, 1.489e-2, 2.041e-2,
                              3.426e-2, 3.052e-2, 3.199e-2],
    (100, 1.0, "frozen"):    [5.768e-2, 4.818e-2, 4.230e-2, 5.252e-2,
                              2.484e-2, 1.912e-2, 1.717e-2, 2.573e-2,
                              1.446e-2, 2.170e-2, 1.473e-2, 2.848e-2,
                              3.551e-2, 2.154e-2, 1.201e-2, 2.038e-2,
                              1.479e-2, 1.635e-2, 1.836e-2, 1.163e-2,
                              2.070e-2, 1.899e-2, 1.371e-2, 1.119e-2,
                              1.352e-2],
    (100, 0.2, "recompute"): [7.238e-2, 1.438e-1, 1.942e-1, 1.204e-1,
                              1.254e-1, 6.864e-2, 1.148e-1, 1.246e-1,
                              1.658e-1, 8.761e-2, 1.027e-1, 6.487e-2,
                              1.042e-1, 1.045e-1, 8.898e-2, 9.722e-2,
                              1.003e-1, 1.608e-1, 1.591e-1, 1.580e-1],
    (100, 0.2, "frozen"):    [7.238e-2, 1.438e-1, 1.942e-1, 1.204e-1,
                              1.254e-1, 6.864e-2, 1.148e-1, 1.246e-1,
                              1.658e-1, 8.761e-2, 1.027e-1, 6.487e-2,
                              1.042e-1, 1.045e-1, 8.898e-2, 9.722e-2,
                              1.003e-1, 1.608e-1, 1.591e-1, 1.580e-1,
                              7.935e-2, 1.637e-1, 1.095e-1, 7.712e-2,
                              9.125e-2],
    (100, 1.0, "pchip"):     [4.716e-2, 2.413e-1, 3.569e-1, 4.272e-1,
                              3.079e-1, 2.679e-1, 4.222e-1, 4.352e-1,
                              1.806e-1],
}

# Reference: strict-h=0 solver stall trace (from smooth_strict/stall_trace.json)
def load_strict_stall_trace():
    p = (f"{ROOT}/projects/REZN/solved_fixed_points/dd_k3_overnight/"
         f"smooth_strict/stall_trace.json")
    j = json.load(open(p))
    return j.get("omega1.0_nq16", {}).get("F", [])


def fig1_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # Left: g100_t1.0
    ax = axes[0]
    strict_F = load_strict_stall_trace()
    ax.semilogy(range(len(strict_F)), strict_F, "k-", alpha=0.6, lw=0.8,
                  label="kernel-band strict (stall trace)")
    for mode, marker, color in [("recompute", "o", "tab:blue"),
                                  ("frozen", "s", "tab:green"),
                                  ("pchip", "^", "tab:red")]:
        key = (100, 1.0, mode)
        if key in HISTORY:
            F = HISTORY[key]
            ax.semilogy(range(len(F)), F, marker=marker, color=color,
                         label=f"zero-h ({mode})", ms=5, lw=1.0)
    ax.axhline(1e-10, color="gray", lw=0.5, ls=":", label="target 1e-10")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|F|_\infty$")
    ax.set_title(r"$\gamma=100,\;\tau=1.0$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(5e-3, 1)
    # Right: g100_t0.2
    ax = axes[1]
    for mode, marker, color in [("recompute", "o", "tab:blue"),
                                  ("frozen", "s", "tab:green")]:
        key = (100, 0.2, mode)
        if key in HISTORY:
            F = HISTORY[key]
            ax.semilogy(range(len(F)), F, marker=marker, color=color,
                         label=f"zero-h ({mode})", ms=5, lw=1.0)
    ax.axhline(1e-10, color="gray", lw=0.5, ls=":", label="target 1e-10")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|F|_\infty$")
    ax.set_title(r"$\gamma=100,\;\tau=0.2$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(5e-3, 1)
    fig.suptitle("Zero-h FP solver: |F| convergence vs the strict-h=0 "
                  "kernel-band stall", fontsize=10)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig1_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig2_mu_smoothing():
    """mu_smooth vs mu_raw for g=100, tau=1.0 at the WARM-START P_strict."""
    # Recompute the smoothing at the saved warm-start to verify exact match.
    sys.path.insert(0,
        "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
    from lin_cdf_strict import (build_mu_table_lin_strict, make_gl_for_u)
    from dd_k3_zero_h_solver import build_smoothed_mu_table
    from dd_k3_critpts import find_critical_points, critical_p_values
    warm_path = (f"{ROOT}/projects/REZN/solved_fixed_points/dd_k3_overnight/"
                 f"dd_k3_strict_fp_g100_t1.0000.npz")
    d_warm = np.load(warm_path)
    P0 = d_warm["P_strict"].astype(np.float64)
    u_grid = make_cdf_uniform_grid(11)
    p_grid = make_p_grid(121)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    mu_raw = build_mu_table_lin_strict(P0, u_grid, p_grid, gl_u, gl_du,
                                          1.0, 11, 16)
    cps = find_critical_points(P0, u_grid, tol=1e-8)
    p_cs = critical_p_values(cps)
    mu_smooth, infos = build_smoothed_mu_table(mu_raw, p_grid, p_cs,
                                                  deg_in=8)
    # use these arrays for the plots
    d = dict(mu_raw=mu_raw, mu_smooth=mu_smooth, p_cs=p_cs)
    p_grid = make_p_grid(121)
    u_grid = make_cdf_uniform_grid(11)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # (1) heatmap of |smooth - raw| (in-support only)
    fb_mask = np.abs(mu_raw - 0.5) < 1e-14
    err = np.where(fb_mask, np.nan, np.abs(mu_smooth - mu_raw))
    ax = axes[0, 0]
    im = ax.imshow(np.log10(err + 1e-20), aspect="auto", origin="lower",
                     extent=[u_grid[0], u_grid[-1], p_grid[0], p_grid[-1]],
                     cmap="viridis", vmin=-16, vmax=0)
    ax.set_xlabel("u_k"); ax.set_ylabel("p")
    ax.set_title("log10 |mu_smooth - mu_raw| in-support")
    plt.colorbar(im, ax=ax)
    # (2) histogram
    ax = axes[0, 1]
    e = err[~np.isnan(err)].ravel()
    if len(e) > 0:
        ax.hist(np.log10(e + 1e-20), bins=60, color="steelblue",
                  edgecolor="k")
    ax.set_xlabel("log10 |err|")
    ax.set_ylabel("# points")
    ax.set_title(f"in-support smoothing: max={np.nanmax(err):.2e}, "
                  f"median={np.nanmedian(err):.2e}")
    ax.grid(alpha=0.3)
    # (3) mu slice k=5
    k0 = 5
    ax = axes[1, 0]
    in_sup = ~fb_mask[:, k0]
    ax.plot(p_grid[in_sup], mu_raw[in_sup, k0], "ko", ms=4,
              label=f"raw (in-support)")
    ax.plot(p_grid, mu_smooth[:, k0], "r-", lw=1, alpha=0.7,
              label="smooth (Cheby)")
    # also plot critical points
    p_cs = d["p_cs"]
    for pc in p_cs:
        ax.axvline(pc, color="b", lw=0.4, alpha=0.3)
    ax.set_xlabel("p")
    ax.set_ylabel(f"mu(p, u_{k0}={u_grid[k0]:.3f})")
    ax.set_title(f"u_k={k0}: blue lines = p_c knots ({len(p_cs)} total)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (4) per-slice max err
    ax = axes[1, 1]
    err_per_k = np.nanmax(err, axis=0)
    ax.semilogy(u_grid, err_per_k, "o-", color="darkred")
    ax.set_xlabel("u_k")
    ax.set_ylabel("max |smooth - raw|")
    ax.set_title("Per-slice max smoothing error (in-support)")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle(r"$g{=}100,\tau{=}1.0$: zero-h pipeline smoothing accuracy",
                  fontsize=11)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig2_mu_smoothing.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def fig3_P_diff():
    """P_star vs warm-start P_0 difference, for g=100, tau=1.0 recompute."""
    npz_path = f"{OUT_DIR}/zero_h_fp_g100_t1.0000_recompute.npz"
    if not os.path.exists(npz_path):
        print(f"SKIP fig3: missing {npz_path}")
        return
    d = np.load(npz_path)
    P_star = d["P_star"]; P0 = d["P0"]
    diff = P_star - P0
    u_grid = make_cdf_uniform_grid(11)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # slices through u_1 = central index
    k_mid = 5
    for ax, sl, name in [(axes[0], diff[k_mid, :, :], r"diff slice $u_1{=}u_{5}$"),
                          (axes[1], P0[k_mid, :, :], r"$P_0$ (warm) slice"),
                          (axes[2], P_star[k_mid, :, :], r"$P_\star$ slice")]:
        im = ax.imshow(sl, origin="lower", cmap="coolwarm" if "diff" in name else "viridis",
                          extent=[u_grid[0], u_grid[-1], u_grid[0], u_grid[-1]])
        ax.set_xlabel("u_2"); ax.set_ylabel("u_3")
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"max |P_star - P_warm| = {np.max(np.abs(diff)):.3e}",
                  fontsize=10)
    fig.tight_layout()
    out = f"{FIG_DIR}/fig3_P_diff.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_convergence()
    fig2_mu_smoothing()
    fig3_P_diff()
