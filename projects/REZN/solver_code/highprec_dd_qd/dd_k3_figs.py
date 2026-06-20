"""Figures for the DD K=3 (gamma, tau) sweep."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_figs"
os.makedirs(OUT, exist_ok=True)
DATA = "/tmp/dd_k3_sweep.json"
FPS = "/tmp/dd_k3_sweep_fps"

results = json.load(open(DATA))
GAMMAS = sorted(set(v["gamma"] for v in results.values()))
TAUS   = sorted(set(v["tau"]   for v in results.values()))
nG, nT = len(GAMMAS), len(TAUS)
gi = {g: i for i, g in enumerate(GAMMAS)}
ti = {t: i for i, t in enumerate(TAUS)}

def grid(key, default=np.nan):
    M = np.full((nT, nG), default)
    for v in results.values():
        M[ti[v["tau"]], gi[v["gamma"]]] = v.get(key, default)
    return M

F = grid("F")
SLOPE = grid("slope")
DEFICIT = grid("deficit_oneToOne")
WALL = grid("t_warm") + grid("t_jac") + grid("t_nail")
WARM = grid("F_warm")

def heat(ax, M, title, cmap, fmt="{:.2f}", vmin=None, vmax=None):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:.3g}" for g in GAMMAS], rotation=45, fontsize=8)
    ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:.2f}" for t in TAUS], fontsize=8)
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(nT):
        for j in range(nG):
            if np.isfinite(M[i,j]):
                col = "white" if cmap in ("viridis", "magma", "cividis", "plasma") else "black"
                ax.text(j, i, fmt.format(M[i,j]), ha="center", va="center",
                        fontsize=5.5, color=col)

# 1. F floor (log10)
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, np.log10(np.where(F>0, F, np.nan)),
        r"$\log_{10}|F|_\infty$ at saved DD K=3 FP",
        "viridis", fmt="{:.1f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_floor.png", dpi=140); plt.close()

# 2. Slope alpha*
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, SLOPE, r"slope $\alpha^*$ (linear fit of logit($P$) vs $T$)",
        "magma", fmt="{:.2f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_slope.png", dpi=140); plt.close()

# 3. Deficit one-to-one
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, DEFICIT, r"one-to-one deficit $1-R^2_\mathrm{nonparam}$",
        "inferno", fmt="{:.3f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_deficit.png", dpi=140); plt.close()

# 4. Wall time per cell
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, WALL, "wall time per cell (s)", "plasma", fmt="{:.0f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_wall.png", dpi=140); plt.close()

# 5. Sample P contours (corners and centre)
g_pick = [GAMMAS[0], GAMMAS[nG//2], GAMMAS[-1]]
t_pick = [TAUS[0],   TAUS[nT//2],   TAUS[-1]]
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
for ri, t in enumerate(t_pick):
    for ci, g in enumerate(g_pick):
        ax = axs[2-ri, ci]
        key = f"g{g:.4g}_t{t:.4f}"
        try:
            d = np.load(f"{FPS}/{key}.npz")
            P = d["P"]
            # 2D projection: P(u_1, u_2, 0) - middle slice
            mid = P.shape[2]//2
            cs = ax.contourf(P[:,:,mid], levels=np.linspace(0,1,15), cmap="RdBu_r")
            ax.contour(P[:,:,mid], levels=[0.5], colors="k", linewidths=1.2)
            v = results.get(key, {})
            ax.set_title(f"$\\gamma$={g:.3g}, $\\tau$={t:.2f}\n"
                            f"F={v.get('F', np.nan):.1e}, slope={v.get('slope', np.nan):.2f}",
                          fontsize=9)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
        ax.set_xlabel("$u_1$ idx"); ax.set_ylabel("$u_2$ idx")
plt.suptitle(r"Sample $P(u_1, u_2, u_3{=}\mathrm{mid})$ contours at sweep corners and centre", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_contours.png", dpi=140); plt.close()

# 6. Main 4-panel
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
heat(axs[0,0], np.log10(np.where(F>0, F, np.nan)),
        r"$\log_{10}|F|_\infty$", "viridis", fmt="{:.1f}")
heat(axs[0,1], SLOPE, r"slope $\alpha^*$", "magma", fmt="{:.2f}")
heat(axs[1,0], DEFICIT, r"deficit $1-R^2$", "inferno", fmt="{:.3f}")
heat(axs[1,1], WALL,  "wall (s)", "plasma", fmt="{:.0f}")
plt.suptitle(r"DD K=3 ($\gamma, \tau$) sweep \-\- main 4-panel", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_main.png", dpi=140); plt.close()

# 7. Cross sections
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for j, g in enumerate(GAMMAS):
    axs[0].plot(TAUS, SLOPE[:, j], "o-", label=f"$\\gamma$={g:.3g}",
                 lw=0.8, ms=3, alpha=0.7)
    axs[1].plot(TAUS, DEFICIT[:, j], "o-", label=f"$\\gamma$={g:.3g}",
                 lw=0.8, ms=3, alpha=0.7)
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"slope $\alpha^*$")
axs[0].set_title("slope vs $\\tau$ at each $\\gamma$"); axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=7, loc="best", ncol=2)
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel(r"deficit")
axs[1].set_title("deficit vs $\\tau$ at each $\\gamma$"); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=7, loc="best", ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_cross.png", dpi=140); plt.close()

n_eps = int((F < 1e-25).sum())
n_med = int(((F < 1e-10) & (F >= 1e-25)).sum())
n_fail = int((F >= 1e-10).sum())
total = float(np.nansum(WALL))
print(json.dumps({
    "cells_total": int(F.size),
    "cells_DD": n_eps,
    "cells_OK": n_med,
    "cells_FAIL": n_fail,
    "total_wall_sec": total,
    "median_F": float(np.nanmedian(F)),
    "min_F": float(F[F > 0].min()),
    "max_F": float(F.max()),
}, indent=2))
print("Figures in", OUT)
