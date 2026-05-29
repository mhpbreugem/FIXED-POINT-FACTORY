"""Smoothness figure for the h-free co-area operator."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth"
d = np.load("/tmp/hfree_smooth_compare.npz")
res = json.load(open("/tmp/hfree_smooth_res.json"))
ps = d["ps"]; nodevals = d["nodevals"]
A1h = d["A1h"]; A1g = d["A1g"]; A1c = d["A1c"]
sc = res["slopejump_scaling"]
hl = np.array(sc["h_list"])

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), dpi=130)

# (a) A_v(p) curves with node-value markers
ax[0].plot(ps, A1h, "-", lw=1.6, label="h-free smooth (this work)")
ax[0].plot(ps, A1g / A1g.max() * A1h.max(), "-", lw=0.9, c="orange",
           alpha=0.8, label="grid marching-squares (scaled)")
for nv in nodevals:
    ax[0].axvline(nv, color="grey", ls=":", lw=0.5)
ax[0].set_xlabel("price p"); ax[0].set_ylabel("A_1(p)  (evidence under v=1)")
ax[0].set_title("(a) A_v(p): dotted = former grid-node prices\n"
                "h-free has no kink at nodes; grid does")
ax[0].legend(fontsize=8); ax[0].grid(ls=":")

# (b) zoom on dA/dp near node values
dp = ps[1] - ps[0]
d1h = np.gradient(A1h, dp)
d1g = np.gradient(A1g, dp)
ax[1].plot(ps, d1h, "-", lw=1.5, label="h-free dA/dp (smooth)")
ax[1].plot(ps, d1g, "-", lw=0.9, c="orange", alpha=0.8,
           label="grid dA/dp (sawtooth kinks)")
for nv in nodevals:
    ax[1].axvline(nv, color="grey", ls=":", lw=0.5)
ax[1].set_xlabel("price p"); ax[1].set_ylabel("dA_1/dp")
ax[1].set_title("(b) derivative: grid kinks AT node values,\n"
                "h-free is continuous")
ax[1].legend(fontsize=8); ax[1].grid(ls=":")

# (c) THE definitive scaling test
ax[2].loglog(hl, sc["hfree_median"], "o-", c="C0",
             label="h-free (this work)")
ax[2].loglog(hl, sc["grid_scan_median"], "s-", c="orange",
             label="grid marching-squares")
ax[2].loglog(hl, sc["grid_cdf_median"], "^-", c="red",
             label="grid CDF-derivative")
# reference slopes
ax[2].loglog(hl, sc["grid_scan_median"][0] * (hl[0] / hl), "k--", lw=0.8,
             alpha=0.6, label="~1/h (kink)")
ax[2].loglog(hl, sc["hfree_median"][-1] * (hl / hl[-1]), "k:", lw=0.8,
             alpha=0.6, label="~h (smooth)")
ax[2].invert_xaxis()
ax[2].set_xlabel("probe width h (decreasing ->)")
ax[2].set_ylabel("median slope-jump in A_v at node values")
ax[2].set_title("(c) SCALING TEST (definitive):\n"
                "h-free ->0 (SMOOTH); grid grows ~1/h (KINK)")
ax[2].legend(fontsize=7.5); ax[2].grid(ls=":", which="both")

plt.suptitle("K=3 CRRA REE: h-FREE co-area operator (NO kernel/bandwidth) is "
             "SMOOTH at grid-node prices; grid methods kink. tau=2 gamma=0.1",
             weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/smoothness.png", dpi=130, bbox_inches="tight")
print("wrote", f"{OUT}/smoothness.png")
