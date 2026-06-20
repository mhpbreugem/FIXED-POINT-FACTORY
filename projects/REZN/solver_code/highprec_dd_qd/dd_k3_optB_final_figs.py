"""Final overnight figures combining all Option B / Option B+ findings."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_optB_final_figs"
os.makedirs(OUT, exist_ok=True)


# --- Fig 1: Option B+ refinement convergence (with BC fix) vs strict-h=0 ---
# Recompute the data points from the run we did earlier
refinement_data = {
    "g=1000, tau=0.2": [(1, 0.312, 5.85e-3), (2, 0.306, 2.06e-3), (4, 0.304, 4.76e-4),
                             (8, 0.303, 1.89e-4), (16, 0.302, 5.16e-5)],
    "g=1000, tau=1.0": [(1, 0.542, 1.89e-1), (2, 0.529, 6.91e-2), (4, 0.537, 6.08e-2),
                             (8, 0.622, 5.87e-2), (16, 0.544, 5.75e-2)],
    "g=100, tau=0.2": [(1, 0.312, 5.86e-3), (2, 0.306, 2.07e-3), (4, 0.304, 4.95e-4),
                            (8, 0.303, 1.94e-4), (16, 0.302, 4.72e-5)],
    "g=100, tau=1.0": [(1, 0.541, 1.87e-1), (2, 0.528, 6.84e-2), (4, 0.537, 5.78e-2),
                            (8, 0.574, 5.55e-2), (16, 0.570, 5.86e-2)],
}
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for key, data in refinement_data.items():
    ns = [d[0] for d in data]
    maxs = [d[1] for d in data]; meds = [d[2] for d in data]
    axs[0].plot(ns, maxs, "o-", lw=1.5, ms=6, label=key)
    axs[1].plot(ns, meds, "o-", lw=1.5, ms=6, label=key)
for ax in axs:
    ax.set_xlabel("refinement factor $n_{\\rm sub}$"); ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
axs[0].set_ylabel(r"$\max|\mu_{B+} - \mu_{\rm strict}|$")
axs[0].set_title("Max error vs $n_{\\rm sub}$ (structural at tails)")
axs[1].set_ylabel(r"$\mathrm{median}|\mu_{B+} - \mu_{\rm strict}|$"); axs[1].set_yscale("log")
axs[1].set_title("Median error vs $n_{\\rm sub}$ (converges as $1/n_{\\rm sub}^p$)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_refinement.png", dpi=140); plt.close()


# --- Fig 2: Sweep results (max F, median F, slope, deficit) ---
try:
    sweep = json.load(open("/tmp/dd_k3_optB_sweep.json"))
    GAMMAS = sorted(set(v["gamma"] for v in sweep.values()))
    TAUS = sorted(set(v["tau"] for v in sweep.values()))
    nG = len(GAMMAS); nT = len(TAUS)
    gi = {g: i for i, g in enumerate(GAMMAS)}
    ti = {t: i for i, t in enumerate(TAUS)}
    def grid(key, default=np.nan):
        M = np.full((nT, nG), default)
        for v in sweep.values():
            M[ti[v["tau"]], gi[v["gamma"]]] = v.get(key, default)
        return M
    F_max = grid("F"); F_med = grid("F_med", default=np.nan)
    SLOPE = grid("slope"); DEFICIT = grid("deficit_oneToOne")

    def heat(ax, M, title, cmap, fmt="{:.2g}", vmin=None, vmax=None):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:.3g}" for g in GAMMAS], rotation=45, fontsize=8)
        ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:.2f}" for t in TAUS], fontsize=8)
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$"); ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for i in range(nT):
            for j in range(nG):
                if np.isfinite(M[i,j]):
                    col = "white" if cmap in ("viridis", "magma", "inferno", "cividis") else "black"
                    ax.text(j, i, fmt.format(M[i,j]), ha="center", va="center",
                            fontsize=6, color=col)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    heat(axs[0,0], np.log10(np.where(F_max>0, F_max, np.nan)),
            r"$\log_{10}|F|_{\rm max}$", "viridis", fmt="{:.1f}")
    heat(axs[0,1], np.log10(np.where(F_med>0, F_med, np.nan)),
            r"$\log_{10}|F|_{\rm med}$", "viridis", fmt="{:.1f}")
    heat(axs[1,0], SLOPE, r"slope $\alpha^*$", "magma", fmt="{:.2f}")
    heat(axs[1,1], DEFICIT, r"deficit", "inferno", fmt="{:.3f}")
    plt.suptitle("Option B+ (n_sub=8, BC-extrapolated) sweep result", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_sweep.png", dpi=140); plt.close()
except Exception as e:
    print("fig2 fail:", e)


# --- Fig 3: slope/deficit cross-sections ---
try:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for j, g in enumerate(GAMMAS):
        axs[0].plot(TAUS, SLOPE[:, j], "o-", lw=0.8, ms=3, alpha=0.7,
                       label=f"$\\gamma$={g:.3g}")
        axs[1].plot(TAUS, DEFICIT[:, j], "o-", lw=0.8, ms=3, alpha=0.7,
                       label=f"$\\gamma$={g:.3g}")
    axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"$\alpha^*$")
    axs[0].set_title("Slope vs $\\tau$ at each $\\gamma$ (Option B+)")
    axs[0].grid(True, alpha=0.3); axs[0].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel("deficit")
    axs[1].set_title("Deficit vs $\\tau$ at each $\\gamma$ (Option B+)")
    axs[1].grid(True, alpha=0.3); axs[1].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_cross.png", dpi=140); plt.close()
except Exception as e: print("fig3 fail:", e)


# --- Fig 4: Compare deficit Option B+ vs R4 baseline vs strict-h=0 ---
try:
    # Load R4 baseline from previous sweep
    r4 = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_sweep_2d/dd_k3_sweep.json"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for j, g in enumerate(GAMMAS):
        if g not in [v["gamma"] for v in r4.values()]: continue
        # R4 at this gamma
        r4_pts = sorted([(v["tau"], v["deficit_oneToOne"]) for v in r4.values()
                              if abs(v["gamma"]-g) < 0.01])
        if r4_pts:
            t_r, d_r = zip(*r4_pts)
            axs[0].plot(t_r, d_r, "s-", lw=1.2, ms=5, alpha=0.7, label=f"R4: $\\gamma$={g:.3g}")
        # Option B+ at this gamma
        b_pts = [(v["tau"], v["deficit_oneToOne"]) for v in sweep.values()
                    if abs(v["gamma"]-g) < 0.01]
        b_pts.sort()
        if b_pts:
            t_b, d_b = zip(*b_pts)
            axs[1].plot(t_b, d_b, "o-", lw=1.2, ms=5, alpha=0.7, label=f"B+: $\\gamma$={g:.3g}")
    axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel("deficit")
    axs[0].set_title(r"R4 baseline deficit (kernel-band artifact)")
    axs[0].grid(True, alpha=0.3); axs[0].legend(fontsize=7, ncol=2)
    axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel("deficit")
    axs[1].set_title(r"Option B+ deficit (closer to true)")
    axs[1].grid(True, alpha=0.3); axs[1].legend(fontsize=7, ncol=2)
    plt.suptitle("Deficit comparison: R4 baseline vs Option B+ refined lookup", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_deficit_compare.png", dpi=140); plt.close()
except Exception as e: print("fig4 fail:", e)


print("figures in", OUT)
print(sorted(os.listdir(OUT)))
