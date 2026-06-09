"""Figures for the Method C v2 (gamma, tau) grid."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_method_c_v2_grid_figs"; os.makedirs(OUT, exist_ok=True)
d = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_grid/grid.json"))
gammas = sorted(set(v["gamma"] for v in d.values()))
taus = sorted(set(v["tau"] for v in d.values()))
nG, nT = len(gammas), len(taus)
gi = {g: i for i, g in enumerate(gammas)}; ti = {t: i for i, t in enumerate(taus)}

def grid(key, fill=np.nan):
    M = np.full((nT, nG), fill)
    for v in d.values():
        M[ti[v["tau"]], gi[v["gamma"]]] = v.get(key, fill)
    return M

F = grid("F"); slope = grid("slope")
P_min = grid("P_min"); P_max = grid("P_max")

def heat(ax, M, title, cmap, fmt="{:.2f}"):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:g}" for g in gammas], rotation=45)
    ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:g}" for t in taus])
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(nT):
        for j in range(nG):
            if np.isfinite(M[i,j]):
                ax.text(j, i, fmt.format(M[i,j]), ha="center", va="center",
                        fontsize=6, color="white" if cmap=="viridis" else "black")

# Fig 1: F (structural non-additivity)
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, np.log10(np.where(F>0, F, np.nan)), r"$\log_{10}|F|_\infty$ (Method C v2 residual)",
        "viridis", fmt="{:.1f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_F.png", dpi=140); plt.close()

# Fig 2: slope
fig, ax = plt.subplots(figsize=(11, 7))
heat(ax, slope, r"slope $\alpha^*$", "magma", fmt="{:.2f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_slope.png", dpi=140); plt.close()

# Fig 3: P range (spread)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
heat(axs[0], P_min, r"$P_{\min}$", "RdBu_r", fmt="{:.2f}")
heat(axs[1], P_max, r"$P_{\max}$", "RdBu_r", fmt="{:.2f}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_Prange.png", dpi=140); plt.close()

# Fig 4: cross-sections
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for j, g in enumerate(gammas):
    axs[0].plot(taus, slope[:, j], "o-", lw=0.8, ms=3, alpha=0.7,
                   label=f"$\\gamma$={g:g}")
    axs[1].semilogy(taus, np.maximum(F[:, j], 1e-15), "o-", lw=0.8, ms=3, alpha=0.7)
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"$\alpha^*$")
axs[0].set_title("slope vs $\\tau$"); axs[0].grid(True, alpha=0.3); axs[0].legend(fontsize=7, ncol=2)
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel(r"$|F|_\infty$")
axs[1].set_title("residual (structural non-additivity) vs $\\tau$")
axs[1].grid(True, alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_cross.png", dpi=140); plt.close()

print(f"figs in {OUT}: {sorted(os.listdir(OUT))}")
print(f"F: median {np.nanmedian(F):.3e}, max {np.nanmax(F):.3e}")
