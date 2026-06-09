"""Figures and tables for the strict-h=0 deficit-ridge re-examination.

Produces:
  - figs/heatmap_strict.png  -- 2D deficit heatmap on (log gamma, tau), strict-h=0
  - figs/heatmap_R4.png      -- same, kernel-band R4 (same FPs as strict run)
  - figs/heatmap_compare.png -- side-by-side (shared colour scale)
  - figs/heatmap_ratio.png   -- ratio R4/strict (log scale)
  - figs/slice_g100.png      -- 1D slice of deficit vs tau at gamma=100
  - figs/slice_all.png       -- 1D slices for all gammas, faceted
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge"
FIGS = os.path.join(OUT_DIR, "figs")
R4_EXIST = "/tmp/dd_k3_sweep_fps"


def load_ridge_json():
    with open(os.path.join(OUT_DIR, "ridge.json")) as f:
        return json.load(f)


def grid_from_results(results, key):
    gammas = sorted({d["gamma"] for d in results.values()})
    taus   = sorted({d["tau"]   for d in results.values()})
    M = np.full((len(gammas), len(taus)), np.nan)
    for d in results.values():
        i = gammas.index(d["gamma"]); j = taus.index(d["tau"])
        M[i, j] = d.get(key, np.nan)
    return np.array(gammas), np.array(taus), M


def heatmap(ax, gammas, taus, M, title, vmin=None, vmax=None, log=True,
              cmap="viridis"):
    if log:
        # Avoid log(0): clip
        Mc = np.clip(M, 1e-12, None)
        im = ax.pcolormesh(taus, gammas, Mc, shading="nearest",
                              norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
    else:
        im = ax.pcolormesh(taus, gammas, M, shading="nearest",
                              vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\gamma$ (log)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def plot_heatmaps():
    results = load_ridge_json()
    gammas, taus, D_str = grid_from_results(results, "deficit_strict")
    _, _, D_R4 = grid_from_results(results, "deficit_R4")

    # Joint colour scale
    all_d = np.concatenate([D_str[~np.isnan(D_str)].ravel(),
                                D_R4[~np.isnan(D_R4)].ravel()])
    vmin = max(1e-6, float(all_d[all_d > 0].min()))
    vmax = float(all_d.max())

    # 1. Each separately
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    heatmap(ax, gammas, taus, D_str,
              r"Deficit (strict-$h{=}0$): $1 - R^2$ vs $T$-groups",
              vmin=vmin, vmax=vmax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "heatmap_strict.png"),
                                            dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    heatmap(ax, gammas, taus, D_R4,
              r"Deficit (kernel-band R4): $1 - R^2$ vs $T$-groups",
              vmin=vmin, vmax=vmax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "heatmap_R4.png"),
                                            dpi=140)
    plt.close(fig)

    # 2. Side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    heatmap(axes[0], gammas, taus, D_R4,
              r"R4 (biased) deficit", vmin=vmin, vmax=vmax)
    heatmap(axes[1], gammas, taus, D_str,
              r"Strict-$h{=}0$ (corrected) deficit",
              vmin=vmin, vmax=vmax)
    fig.suptitle(r"K=3 deficit on $(\gamma, \tau)$: same cells, two operators",
                    fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "heatmap_compare.png"),
                                            dpi=140)
    plt.close(fig)

    # 3. Ratio R4 / strict (collapse factor)
    ratio = D_R4 / np.maximum(D_str, 1e-12)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    Rc = np.clip(ratio, 1e-2, 1e4)
    im = ax.pcolormesh(taus, gammas, Rc, shading="nearest",
                            norm=LogNorm(vmin=1e-1, vmax=1e3),
                            cmap="RdBu_r")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\gamma$ (log)")
    ax.set_title(r"Collapse factor: $\mathrm{deficit}_{R4}\,/\,\mathrm{deficit}_{\mathrm{strict}}$")
    plt.colorbar(im, ax=ax, label="ratio (log)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "heatmap_ratio.png"),
                                            dpi=140)
    plt.close(fig)


def plot_slice_g100():
    results = load_ridge_json()
    cells = [(d["tau"], d["deficit_strict"], d["deficit_R4"])
                 for d in results.values() if abs(d["gamma"] - 100.0) < 1e-9]
    cells.sort()
    taus  = np.array([c[0] for c in cells])
    d_str = np.array([c[1] for c in cells])
    d_R4  = np.array([c[2] for c in cells])

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(taus, d_R4,  "o-", label=r"R4 (biased)", color="C3", lw=2)
    ax.plot(taus, d_str, "s-", label=r"strict-$h{=}0$", color="C0", lw=2)
    ax.set_xlabel(r"$\tau$ (signal precision)")
    ax.set_ylabel("deficit (one-to-one R² gap)")
    ax.set_yscale("log")
    ax.set_title(r"$\gamma = 100$: deficit vs $\tau$, R4 vs strict-$h{=}0$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "slice_g100.png"),
                                            dpi=140)
    plt.close(fig)


def plot_slice_all():
    results = load_ridge_json()
    gammas = sorted({d["gamma"] for d in results.values()})
    nrows = 2; ncols = (len(gammas)+1)//2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.6*nrows),
                                  sharex=True, sharey=True)
    axes = axes.ravel()
    for i, g in enumerate(gammas):
        cells = [(d["tau"], d["deficit_strict"], d["deficit_R4"])
                     for d in results.values() if abs(d["gamma"] - g) < 1e-9]
        cells.sort()
        taus  = np.array([c[0] for c in cells])
        d_str = np.array([c[1] for c in cells])
        d_R4  = np.array([c[2] for c in cells])
        ax = axes[i]
        ax.plot(taus, d_R4, "o-", color="C3", lw=1.6, label="R4")
        ax.plot(taus, d_str, "s-", color="C0", lw=1.6, label="strict")
        ax.set_yscale("log")
        ax.set_title(rf"$\gamma={g:g}$", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    for j in range(len(gammas), len(axes)): axes[j].axis("off")
    for ax in axes[-ncols:]: ax.set_xlabel(r"$\tau$")
    for k in range(0, len(axes), ncols): axes[k].set_ylabel("deficit")
    fig.suptitle("Deficit vs $\\tau$, faceted by $\\gamma$", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "slice_all.png"), dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("Plotting heatmaps...")
    plot_heatmaps()
    print("Plotting gamma=100 slice...")
    plot_slice_g100()
    print("Plotting faceted slices...")
    plot_slice_all()
    print(f"Figs -> {FIGS}")


if __name__ == "__main__":
    main()
