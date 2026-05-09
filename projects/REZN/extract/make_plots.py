#!/usr/bin/env python3
"""Render Fig 7 (volume) and Fig 8 (V_info) — both as τ-on-x and γ-on-x.

Reads projects/REZN/figures/all_metrics.csv.
Writes PNGs and pgfplots .tex files to projects/REZN/figures/.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "projects/REZN/figures"

# Paper colour palette
GAMMA_COLORS = {0.1:(0.55,0.0,0.0), 0.25:(0.7,0.11,0.11), 0.35:(0.72,0.53,0.04),
                0.5:(0.0,0.0,0.0), 1.0:(0.11,0.35,0.02), 1.4:(0.0,0.20,0.42),
                2.0:(0.4,0.0,0.4), 4.0:(0.55,0.55,0.55)}
TAU_COLORS = {1.5:(0.55,0.55,0.55), 2.0:(0.7,0.11,0.11), 3.0:(0.72,0.53,0.04),
              5.0:(0.0,0.0,0.0), 10.0:(0.11,0.35,0.02), 15.0:(0.0,0.20,0.42),
              20.0:(0.4,0.0,0.4)}

def load():
    rows = []
    with open(FIG / "all_metrics.csv") as f:
        for row in csv.DictReader(f):
            rows.append((round(float(row["gamma"]),3), float(row["tau"]),
                         float(row["volume"]), float(row["V_info"])))
    return rows

def plot(by_key, x_label, y_label, title, fname, color_map, ymax=None, xlog=False, xlim=None):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=130)
    for k in sorted(by_key.keys()):
        pts = sorted(by_key[k])
        if len(pts) < 2: continue
        xs = np.array([x for x,_ in pts]); ys = np.array([y for _,y in pts])
        color = color_map.get(k, (0.5,0.5,0.5))
        label = f"τ = {k}" if "γ" in x_label else f"γ = {k}"
        ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=5, label=label)
    ax.set_xlabel(x_label, fontsize=11); ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    if xlog: ax.set_xscale("log")
    if xlim: ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0, top=ymax) if ymax else ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6, which="both")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / fname, dpi=140); plt.close()
    print(f"Wrote {fname}")

def emit_tex(by_key, fname, header):
    with open(FIG / fname, "w") as f:
        f.write(f"% {header}\n% Generated from projects/REZN/figures/all_metrics.csv\n\n")
        for k in sorted(by_key.keys()):
            pts = sorted(by_key[k])
            if len(pts) < 2: continue
            coords = " ".join(f"({x},{y:.6f})" for x, y in pts)
            prefix = "γ" if "vs_gamma" not in fname else "τ"
            f.write(f"% {prefix} curve: {k}\n")
            f.write(f"\\addplot coordinates {{{coords}}};\n")
            f.write(f"\\addlegendentry{{${prefix}={k}$}}\n\n")
    print(f"Wrote {fname}")

if __name__ == "__main__":
    rows = load()
    # τ on x: group by γ
    by_g_vol = defaultdict(list); by_g_vi = defaultdict(list)
    for g, t, vo, vi in rows:
        by_g_vol[g].append((t, vo)); by_g_vi[g].append((t, vi))
    # γ on x: group by τ
    by_t_vol = defaultdict(list); by_t_vi = defaultdict(list)
    for g, t, vo, vi in rows:
        by_t_vol[t].append((g, vo)); by_t_vi[t].append((g, vi))

    plot(by_g_vol, r"$\tau$", r"Aggregate trade volume $V$",
         "Trade volume in the REE", "fig7_volume_vs_tau.png", GAMMA_COLORS, ymax=5)
    plot(by_g_vi, r"$\tau$", r"Value of information $V$",
         "Per-agent value of one extra signal", "fig8_value_info_vs_tau.png", GAMMA_COLORS, ymax=0.32)
    plot(by_t_vol, r"$\gamma$", r"Aggregate trade volume $V$",
         "Trade volume vs CRRA risk aversion", "fig7_volume_vs_gamma.png",
         TAU_COLORS, ymax=5, xlog=True, xlim=(0.2, 5))
    plot(by_t_vi, r"$\gamma$", r"Value of information $V$",
         "Value of one extra signal vs CRRA risk aversion", "fig8_value_info_vs_gamma.png",
         TAU_COLORS, ymax=0.32, xlog=True, xlim=(0.2, 5))

    emit_tex(by_g_vol, "fig7_volume_vs_tau_pgfplots.tex", "Fig 7: trade volume vs τ at fixed γ")
    emit_tex(by_g_vi, "fig8_value_info_vs_tau_pgfplots.tex", "Fig 8: value of info vs τ at fixed γ")
    emit_tex(by_t_vol, "fig7_volume_vs_gamma_pgfplots.tex", "Fig 7B: trade volume vs γ at fixed τ")
    emit_tex(by_t_vi, "fig8_value_info_vs_gamma_pgfplots.tex", "Fig 8B: value of info vs γ at fixed τ")
