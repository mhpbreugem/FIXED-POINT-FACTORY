"""Figures for the alternative-lookup overnight report.

Inputs:
  /tmp/dd_k3_alts_results.json
  /tmp/dd_k3_alts_*.npz  (mu tables per cell)
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_alts_figs"
os.makedirs(OUT, exist_ok=True)

d = json.load(open("/tmp/dd_k3_alts_results.json"))
cells = sorted(d.keys())
ncells = len(cells)

# ---- fig 1: comparison of max|d| and median|d| across options ----
options = ["A_h=0.5", "A_h=0.3", "A_h=0.2", "A_h=0.1", "A_h=0.05", "B", "C", "D"]
def collect(field):
    rows = []
    for key in cells:
        v = d[key]
        row = []
        for o in options:
            if o.startswith("A_"):
                h = o.split("=")[1]
                k = f"h={h}"
                row.append(v["A"][k].get(field, np.nan) if k in v["A"] else np.nan)
            else:
                cell = v.get(o, {})
                row.append(cell.get(field, np.nan) if "error" not in cell else np.nan)
        rows.append(row)
    return np.array(rows)

maxD = collect("max"); medD = collect("med")

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
x = np.arange(len(options)); width = 0.8/ncells
for ci, key in enumerate(cells):
    axs[0].bar(x + (ci - ncells/2)*width, maxD[ci], width, label=key, alpha=0.7)
    axs[1].bar(x + (ci - ncells/2)*width, medD[ci], width, alpha=0.7)
axs[0].set_xticks(x); axs[0].set_xticklabels(options, rotation=30)
axs[0].set_yscale("log"); axs[0].set_ylabel(r"$\max|\mu_{\rm opt} - \mu_{\rm strict}|$")
axs[0].set_title(r"Maximum absolute discrepancy vs strict-$h{=}0$")
axs[0].grid(True, alpha=0.3, which="both")
axs[1].set_xticks(x); axs[1].set_xticklabels(options, rotation=30)
axs[1].set_yscale("log"); axs[1].set_ylabel(r"$\mathrm{median}|\mu_{\rm opt} - \mu_{\rm strict}|$")
axs[1].set_title(r"Median absolute discrepancy vs strict-$h{=}0$ (Option B median $\to 0$)")
axs[1].grid(True, alpha=0.3, which="both")
axs[0].legend(fontsize=6, ncol=2, loc="best")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_max_med.png", dpi=140); plt.close()


# ---- fig 2: per-option mu(p, u_k) overlay at a few cells ----
def load_cell(key):
    f = f"/tmp/dd_k3_alts_{key}.npz"
    if os.path.exists(f): return np.load(f)
    return None

picks = []
for key in cells:
    g = d[key]["gamma"]; t = d[key]["tau"]
    if abs(g - 100) < 0.01 and abs(t - 0.2) < 0.01: picks.append(key)
    if abs(g - 100) < 0.01 and abs(t - 1.0) < 0.01: picks.append(key)
    if abs(g - 10) < 0.01 and abs(t - 0.2) < 0.01: picks.append(key)
    if abs(g - 10) < 0.01 and abs(t - 1.0) < 0.01: picks.append(key)

if not picks: picks = cells[:4]

fig, axs = plt.subplots(2, len(picks), figsize=(5*len(picks), 8))
if len(picks) == 1: axs = axs[:, None]
for col, key in enumerate(picks):
    dd = load_cell(key)
    if dd is None: continue
    mu_s = dd["mu_strict"]; mu_A = dd["mu_A_h0p2"]; mu_B = dd["mu_B"]
    mu_C = dd["mu_C"]; mu_D = dd["mu_D"]
    G_p, G = mu_s.shape
    p_grid = np.linspace(0.0003, 0.9997, G_p)
    # Pick a central u_k
    kc = G // 2
    for ki, label in [(0, "u_k[0]"), (G//4, "u_k[G/4]"), (G//2, "u_k[G/2]")]:
        if col == 0:
            axs[0, col].plot(p_grid, mu_s[:, ki], "k-", lw=1.5, alpha=0.8, label=f"strict {label}")
        axs[0, col].plot(p_grid, mu_s[:, ki], "k-", lw=1.5, alpha=0.7)
        axs[0, col].plot(p_grid, mu_A[:, ki], "b--", lw=1.0, alpha=0.6, label=f"A (cube KB h=0.2)" if ki==G//2 and col==0 else None)
        axs[0, col].plot(p_grid, mu_B[:, ki], "g-.", lw=1.0, alpha=0.7, label=f"B (CDF density)" if ki==G//2 and col==0 else None)
    axs[0, col].set_title(f"{key}\nstrict-h=0 vs A,B")
    axs[0, col].set_xlabel("$p$"); axs[0, col].set_ylabel("$\\mu(p, u_k)$")
    axs[0, col].grid(True, alpha=0.3); axs[0, col].legend(fontsize=8)
    # Difference plot
    for ki, label in [(0, "u_k[0]"), (G//4, "u_k[G/4]"), (G//2, "u_k[G/2]")]:
        axs[1, col].plot(p_grid, mu_B[:, ki] - mu_s[:, ki], "g-", lw=1.0, alpha=0.7,
                            label=f"B - strict {label}" if col==0 else None)
        axs[1, col].plot(p_grid, mu_A[:, ki] - mu_s[:, ki], "b--", lw=0.8, alpha=0.5,
                            label=f"A - strict {label}" if col==0 else None)
    axs[1, col].axhline(0, color="k", lw=0.5)
    axs[1, col].set_xlabel("$p$"); axs[1, col].set_ylabel("difference vs strict")
    axs[1, col].grid(True, alpha=0.3); axs[1, col].legend(fontsize=7, loc="best")
plt.suptitle("Lookup-table options vs strict-$h{=}0$ ground truth", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_mu_curves.png", dpi=140); plt.close()


# ---- fig 3: CDF construction (Option B mechanism), demonstrated ----
key0 = picks[0] if picks else cells[0]
dd = load_cell(key0); v = d[key0]
P = dd["P"]; G_p = dd["mu_strict"].shape[0]; G = dd["mu_strict"].shape[1]
import importlib, sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
from lin_cdf_strict import make_cdf_uniform_grid
u_grid = make_cdf_uniform_grid(G)
du = np.diff(u_grid)
w_trap = np.empty(G); w_trap[0] = 0.5*du[0]; w_trap[-1] = 0.5*du[-1]
w_trap[1:-1] = 0.5*(du[:-1] + du[1:])
from cheby_numba import f_signal_jit
tau = v["tau"]
f0 = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
f1 = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
# Pick u_k = mid
k_node = G // 2
P_slice = P[k_node, :, :].ravel()
i_idx, j_idx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
i_idx = i_idx.ravel(); j_idx = j_idx.ravel()
w_cell = w_trap[i_idx] * w_trap[j_idx]
f0_cell = f0[i_idx] * f0[j_idx]
f1_cell = f1[i_idx] * f1[j_idx]
order = np.argsort(P_slice)
P_s = P_slice[order]; w_s = w_cell[order]
F0 = np.cumsum(w_s * f0_cell[order])
F1 = np.cumsum(w_s * f1_cell[order])
axs[0].step(P_s, F0, where="post", label=r"$F_0(p|u_k^{\rm mid})$ (v=0)")
axs[0].step(P_s, F1, where="post", label=r"$F_1(p|u_k^{\rm mid})$ (v=1)")
axs[0].set_xlabel(r"$p$"); axs[0].set_ylabel("cumulative weighted $f_v$")
axs[0].set_title(f"{key0}: empirical CDF on cube slice (G$^2$={G**2} cells)")
axs[0].grid(True, alpha=0.3); axs[0].legend()
# Density via PCHIP derivative
from scipy.interpolate import PchipInterpolator
Pu, inv_idx = np.unique(P_s, return_inverse=True)
F0u = np.zeros(len(Pu)); F1u = np.zeros(len(Pu))
for s in range(len(P_s)):
    F0u[inv_idx[s]] = F0[s]; F1u[inv_idx[s]] = F1[s]
p_eval = np.linspace(Pu[0]+1e-6, Pu[-1]-1e-6, 200)
if len(Pu) >= 4:
    p0 = PchipInterpolator(Pu, F0u, extrapolate=False).derivative()(p_eval)
    p1 = PchipInterpolator(Pu, F1u, extrapolate=False).derivative()(p_eval)
    axs[1].plot(p_eval, p0, label=r"$a_0 = \partial_p F_0$ (density)")
    axs[1].plot(p_eval, p1, label=r"$a_1 = \partial_p F_1$ (density)")
axs[1].set_xlabel(r"$p$"); axs[1].set_ylabel("density")
axs[1].set_title(r"Density via PCHIP derivative $\to$ feed Bayes ratio for $\mu$")
axs[1].grid(True, alpha=0.3); axs[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_cdf_construction.png", dpi=140); plt.close()


# ---- fig 4: Option E diagnostic -- distribution of mu across u_k ----
fig, axs = plt.subplots(1, len(picks[:4]), figsize=(5*len(picks[:4]), 4))
if len(picks) == 1: axs = [axs]
for ax, key in zip(axs, picks[:4]):
    v = d[key]["E"]
    p_grid = np.linspace(0.0003, 0.9997, len(v["mean"]))
    ax.plot(p_grid, v["mean"], "k-", lw=2, label="mean")
    ax.fill_between(p_grid, v["p25"], v["p75"], alpha=0.3, label="IQR")
    ax.fill_between(p_grid, np.array(v["mean"])-np.array(v["std"]),
                       np.array(v["mean"])+np.array(v["std"]), alpha=0.2, label=r"$\pm \sigma$")
    ax.set_xlabel("$p$"); ax.set_ylabel(r"$\mu(p, u_k)$ distribution over $u_k$")
    ax.set_title(f"{key}: spread of $\\mu$ across $u_k$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.suptitle("Option E: distribution of $\\mu$ across $u_k$ at each $p$ (strict-$h$=0)",
              fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_distribution.png", dpi=140); plt.close()


print(f"figures in {OUT}")
print(sorted(os.listdir(OUT)))
