"""Figure for Option d adaptive p-grid results."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_optD_figs"
os.makedirs(OUT, exist_ok=True)
d = json.load(open("/tmp/dd_k3_optD_results.json"))
cells = sorted(d.keys())
monitors = ["uniform_logit", "alpha_a_v", "beta_dmu", "gamma_arc"]

maxs = np.array([[d[c][m]["max"] for m in monitors] for c in cells])
meds = np.array([[d[c][m]["median"] for m in monitors] for c in cells])
ncells, nm = maxs.shape

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(nm); width = 0.8 / ncells
for ci, c in enumerate(cells):
    axs[0].bar(x + (ci - ncells/2)*width, maxs[ci], width, label=c, alpha=0.7)
    axs[1].bar(x + (ci - ncells/2)*width, np.maximum(meds[ci], 1e-12), width, alpha=0.7)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(monitors, rotation=20)
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)
axs[0].set_ylabel(r"$\max|\mu_{\rm interp} - \mu_{\rm ref}|$")
axs[0].set_title("Max interpolation error vs reference (1001 pts)")
axs[1].set_ylabel(r"$\mathrm{median}|\mu_{\rm interp} - \mu_{\rm ref}|$")
axs[1].set_yscale("log"); axs[1].set_title("Median interpolation error")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_monitor_compare.png", dpi=140); plt.close()

avg_max = maxs.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(nm)
ax.bar(x, avg_max, color="C0", alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(monitors, rotation=15)
ax.set_ylabel("mean max error vs reference"); ax.grid(True, alpha=0.3)
for i, v in enumerate(avg_max):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_title(f"Average max error across 16 cells (G_p=121)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_avg.png", dpi=140); plt.close()

pdens = np.array([[d[c][m]["p_density_at_0p5"] for m in monitors] for c in cells])
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(np.arange(nm), pdens.mean(axis=0), color="C2", alpha=0.7)
ax.set_xticks(np.arange(nm)); ax.set_xticklabels(monitors, rotation=15)
ax.set_ylabel("avg # grid points in |p-0.5|<0.05")
ax.set_title("Where each monitor places its 121 grid points")
ax.grid(True, alpha=0.3)
for i, v in enumerate(pdens.mean(axis=0)):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_density.png", dpi=140); plt.close()

imp = (avg_max[0] - avg_max[1:]) / avg_max[0] * 100
print(f"\nImprovement vs uniform_logit baseline ({avg_max[0]:.3f}):")
for n, v, i in zip(monitors[1:], avg_max[1:], imp):
    print(f"  {n}: max|d| = {v:.3f}  improvement = {i:+.1f}%")
print(f"\nfigures in {OUT}")
