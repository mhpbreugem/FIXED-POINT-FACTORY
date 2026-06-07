"""Figures for the overnight comparative report.

Inputs:
  /tmp/dd_k3_compare.json       -- variant comparator (8 FPs x 7 variants)
  /tmp/dd_k3_rconverge.json     -- Richardson-order convergence U-curve
  /tmp/dd_k3_small_h.json       -- small-h test (kernel-band convergence to strict)
  /tmp/dd_k3_strict_xv.json     -- strict-h=0 cross-validation (at R4 FPs)
  /tmp/dd_k3_strict_solve.json  -- strict-h=0 FP solve + bias quantification
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_overnight_figs"
os.makedirs(OUT, exist_ok=True)

# ---- Fig 1: Richardson U-curve (R2..R7 at FP, |F| at saved FP) ----
try:
    d = json.load(open("/tmp/dd_k3_rconverge.json"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, family in zip(axs, ["linear", "geom"]):
        for key, cell in d.items():
            ns, Fs, L1s = [], [], []
            for vname, v in cell.items():
                if v.get("family") != family: continue
                ns.append(v["n"]); Fs.append(v["F_inf"]); L1s.append(v["L1_norm"])
            order = np.argsort(ns)
            ns = np.array(ns)[order]; Fs = np.array(Fs)[order]; L1s = np.array(L1s)[order]
            ax.semilogy(ns, np.abs(Fs)+1e-30, "o-", lw=1.5, ms=6, label=key)
        ax.axhline(1e-15, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("Richardson order $R$"); ax.set_ylabel(r"$|F|_\infty$ at baseline FP")
        ax.set_title(f"Richardson U-curve ({family} h-spacing)")
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_richardson_ucurve.png", dpi=140); plt.close()
except Exception as e: print("fig1 fail:", e)

# ---- Fig 2: Variant comparator dmu_vs_truth ----
try:
    d = json.load(open("/tmp/dd_k3_compare.json"))
    variants = ["V1_base", "V2_pchip", "V3_neville", "V4_combined",
                  "V6_hi_NQK", "V7_hi_Gp"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for vi, v in enumerate(variants):
        ys = [cell[v]["dmu_vs_truth"] for cell in d.values() if v in cell]
        ax.plot([vi]*len(ys), ys, "o", alpha=0.6)
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, rotation=30)
    ax.set_ylabel(r"$\max|\mu_{\rm var} - \mu_{\rm truth}|$  (truth=V8 PCHIP+Neville+hi)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.set_title(
        "Variant vs truth-proxy lookup-table discrepancy")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_variant_compare.png", dpi=140); plt.close()
except Exception as e: print("fig2 fail:", e)

# ---- Fig 3: smaller h convergence to strict-h=0 ----
try:
    d = json.load(open("/tmp/dd_k3_small_h.json"))
    fig, ax = plt.subplots(figsize=(11, 6))
    scales = ["scale0_coarse", "scale1", "scale2", "scale3_fine"]
    x = range(len(scales))
    for key, cell in d.items():
        ys = [cell[s]["max_diff"] for s in scales]
        ax.plot(x, ys, "o-", lw=1.4, ms=6, label=key)
    ax.set_xticks(x); ax.set_xticklabels(["h~0.35\nNQK=16","h~0.18\nNQK=24",
                                                  "h~0.07\nNQK=32","h~0.035\nNQK=64"])
    ax.set_ylabel(r"$\max|\mu_{R4} - \mu_{\rm strict}|$ (at R4 FP)")
    ax.set_xlabel("kernel-band scale (h, NQK)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8); ax.set_title(
        "Does kernel-band R4 converge to strict-h=0 as h shrinks?\n"
        "(NO -- the integral isn't analytic in $h^2$ for piecewise-linear $P$)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_smallh.png", dpi=140); plt.close()
except Exception as e: print("fig3 fail:", e)

# ---- Fig 4: Bias |P_strict - P_R4| and deficit shift ----
try:
    d = json.load(open("/tmp/dd_k3_strict_solve.json"))
    rows = []
    for key, v in d.items():
        rows.append([v["gamma"], v["tau"], v["bias_P_inf"], v["mu_diff_R4_at_strict"],
                       v["deficit_R4"], v["deficit_strict"],
                       v["slope_R4"], v["slope_strict"]])
    rows = np.array(rows)
    rows = rows[np.lexsort((rows[:,1], rows[:,0]))]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    labels = [f"$\\gamma$={int(r[0]) if r[0].is_integer() else r[0]:g}, $\\tau$={r[1]:.1f}"
                 for r in rows]
    x = np.arange(len(rows))
    axs[0,0].bar(x, rows[:,2], color="C3")
    axs[0,0].set_ylabel(r"$|P_{\rm strict} - P_{R4}|_\infty$"); axs[0,0].set_title("FP bias")
    axs[0,0].set_xticks(x); axs[0,0].set_xticklabels(labels, rotation=45, fontsize=8)
    axs[0,0].grid(True, alpha=0.3)
    axs[0,1].bar(x, rows[:,3], color="C1")
    axs[0,1].set_ylabel(r"$|\mu_{R4}(P_{\rm strict}) - \mu_{\rm strict}(P_{\rm strict})|_\infty$")
    axs[0,1].set_title("Lookup table discrepancy at strict-h=0 FP")
    axs[0,1].set_xticks(x); axs[0,1].set_xticklabels(labels, rotation=45, fontsize=8)
    axs[0,1].grid(True, alpha=0.3)
    width = 0.4
    axs[1,0].bar(x - width/2, rows[:,4], width, color="C0", label="R4")
    axs[1,0].bar(x + width/2, rows[:,5], width, color="C2", label="strict-h=0")
    axs[1,0].set_ylabel("deficit $1-R^2_\\mathrm{nonparam}$"); axs[1,0].set_title(
        "Deficit at R4 FP vs strict-h=0 FP")
    axs[1,0].set_xticks(x); axs[1,0].set_xticklabels(labels, rotation=45, fontsize=8)
    axs[1,0].legend(); axs[1,0].grid(True, alpha=0.3)
    axs[1,1].bar(x - width/2, rows[:,6], width, color="C0", label="R4")
    axs[1,1].bar(x + width/2, rows[:,7], width, color="C2", label="strict-h=0")
    axs[1,1].set_ylabel("slope $\\alpha^*$"); axs[1,1].set_title(
        "Slope at R4 FP vs strict-h=0 FP")
    axs[1,1].set_xticks(x); axs[1,1].set_xticklabels(labels, rotation=45, fontsize=8)
    axs[1,1].legend(); axs[1,1].grid(True, alpha=0.3)
    plt.suptitle("R4 kernel-band vs strict-h=0: FP bias and observable shifts",
                  fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_strict_vs_R4.png", dpi=140); plt.close()
except Exception as e: print("fig4 fail:", e)

# ---- Fig 5: sample mu(p, u_k) curves at one cell, R4 vs strict ----
try:
    files = sorted(glob.glob("/tmp/dd_k3_strict_fp_*.npz"))
    if files:
        fig, axs = plt.subplots(2, 4, figsize=(16, 7))
        plot_keys = files[:4]      # first 4
        for col, f in enumerate(plot_keys):
            d = np.load(f)
            mu_s = d["mu_strict"]; mu_R = d["mu_R4_at_strict"]
            label = os.path.basename(f).replace("dd_k3_strict_fp_", "").replace(".npz", "")
            G_p, G = mu_s.shape
            # at u_k = central
            kc = G//2
            p_grid = np.linspace(0.005, 0.995, G_p)
            for ki in [0, G//4, G//2, 3*G//4, G-1]:
                axs[0, col].plot(p_grid, mu_s[:, ki], "-", lw=1.5,
                                     label=f"$u_k$#{ki}", alpha=0.7)
                axs[1, col].plot(p_grid, mu_R[:, ki] - mu_s[:, ki], "-", lw=1.5,
                                     alpha=0.7, label=f"$u_k$#{ki}")
            axs[0, col].set_title(label); axs[0, col].grid(True, alpha=0.3)
            axs[0, col].set_xlabel("$p$"); axs[0, col].set_ylabel("$\\mu_{\\rm strict}(p, u_k)$")
            axs[0, col].legend(fontsize=7)
            axs[1, col].set_xlabel("$p$"); axs[1, col].set_ylabel("$\\mu_{R4} - \\mu_{\\rm strict}$")
            axs[1, col].grid(True, alpha=0.3)
            axs[1, col].axhline(0, color="k", lw=0.5)
        plt.suptitle("Top: strict-h=0 lookup curves. Bottom: R4 bias vs strict.",
                      fontsize=12)
        plt.tight_layout(); plt.savefig(f"{OUT}/fig5_mu_curves.png", dpi=140); plt.close()
except Exception as e: print("fig5 fail:", e)

print("Figures in", OUT)
print(sorted(os.listdir(OUT)))
