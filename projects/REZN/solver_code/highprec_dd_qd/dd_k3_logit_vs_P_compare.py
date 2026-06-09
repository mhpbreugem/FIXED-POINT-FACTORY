"""Compare Method C v2 with P-loss vs L-loss."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_logit_compare_figs"; os.makedirs(OUT, exist_ok=True)
d_P = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_grid/grid.json"))
d_L = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_logit/grid.json"))
gammas = sorted(set(v["gamma"] for v in d_P.values()))
taus = sorted(set(v["tau"] for v in d_P.values()))
nG, nT = len(gammas), len(taus); gi = {g: i for i, g in enumerate(gammas)}; ti = {t: i for i, t in enumerate(taus)}

def grid(d, key, fill=np.nan):
    M = np.full((nT, nG), fill)
    for v in d.values():
        if v["tau"] in ti and v["gamma"] in gi:
            M[ti[v["tau"]], gi[v["gamma"]]] = v.get(key, fill)
    return M

F_P_loss_P = grid(d_P, "F")
F_L_loss_P = grid(d_L, "F_P")    # P-space F from logit-loss optimizer
F_L_loss_L = grid(d_L, "F_L")    # L-space F from logit-loss optimizer

# Comparison plot: which loss gives smaller P-space F (the canonical residual)?
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
def heat(ax, M, title, vmin=None, vmax=None):
    im = ax.imshow(np.log10(np.where(M>0, M, np.nan)), origin="lower", aspect="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:g}" for g in gammas], rotation=45, fontsize=7)
    ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:g}" for t in taus], fontsize=7)
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(nT):
        for j in range(nG):
            if np.isfinite(M[i,j]):
                ax.text(j, i, f"{np.log10(M[i,j]):.1f}", ha="center", va="center",
                        fontsize=5.5, color="white")
heat(axs[0], F_P_loss_P, "$\\log_{10} F$ (P-loss optimizer)")
heat(axs[1], F_L_loss_P, "$\\log_{10} F_P$ (L-loss optimizer, measured in P)")
heat(axs[2], F_L_loss_L, "$\\log_{10} F_L$ (L-loss optimizer, measured in L)")
plt.suptitle("Method C v2 loss comparison: P-space vs L-space loss", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_loss_compare.png", dpi=140); plt.close()

# Ratio: how much does L-loss improve over P-loss in P-space?
ratio = F_P_loss_P / np.maximum(F_L_loss_P, 1e-30)
fig, ax = plt.subplots(figsize=(11, 7))
im = ax.imshow(np.log10(ratio), origin="lower", aspect="auto", cmap="RdBu_r",
                  vmin=-2, vmax=2)
ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:g}" for g in gammas], rotation=45)
ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:g}" for t in taus])
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$")
ax.set_title("$\\log_{10}(F_{P\\text{-loss}} / F_{L\\text{-loss}})$: positive = L-loss wins")
plt.colorbar(im, ax=ax, fraction=0.04)
for i in range(nT):
    for j in range(nG):
        if np.isfinite(ratio[i,j]):
            ax.text(j, i, f"{np.log10(ratio[i,j]):.1f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(np.log10(ratio[i,j]))>1 else "black")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_ratio.png", dpi=140); plt.close()
print(f"figs in {OUT}")
print(f"F_P (P-loss) median: {np.nanmedian(F_P_loss_P):.3e}")
print(f"F_P (L-loss) median: {np.nanmedian(F_L_loss_P):.3e}")
print(f"L-loss wins in P-space ratio: median = {np.nanmedian(ratio):.2f}")
