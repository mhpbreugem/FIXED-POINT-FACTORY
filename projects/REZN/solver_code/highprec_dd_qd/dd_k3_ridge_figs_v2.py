"""Regenerate ridge figures from the actually-completed cells."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

D = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge"
OUT = f"{D}/figs"
os.makedirs(OUT, exist_ok=True)

d = json.load(open(f"{D}/ridge.json"))
print(f"loaded {len(d)} cells")

# Collect points
rows = []
for k, v in d.items():
    rows.append((v['gamma'], v['tau'], v['deficit_R4'], v['deficit_strict'],
                 v['F_strict'], v.get('slope_R4', 0), v.get('slope_strict', 0)))
rows.sort()
gammas = sorted(set(r[0] for r in rows))
taus = sorted(set(r[1] for r in rows))
print(f"gammas: {gammas}")
print(f"taus: {taus}")

# Fig 1: tau-slice for each gamma we have
fig, axes = plt.subplots(1, len(gammas), figsize=(5*len(gammas), 4))
if len(gammas) == 1: axes = [axes]
for ax, g in zip(axes, gammas):
    cells = [(r[1], r[2], r[3]) for r in rows if r[0] == g]
    cells.sort()
    if cells:
        t = [c[0] for c in cells]
        dR4 = [c[1] for c in cells]
        dS = [c[2] for c in cells]
        ax.semilogy(t, dR4, 'r-o', label='R4 (biased)', ms=6)
        ax.semilogy(t, dS, 'b-s', label='strict-h=0', ms=6)
        ax.set_xlabel(r'$\tau$ (signal precision)')
        ax.set_ylabel('deficit')
        ax.set_title(f"$\\gamma = {g}$")
        ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{OUT}/slice_per_gamma.png", dpi=120)
plt.close()

# Fig 2: ratio R4/strict (collapse factor)
fig, ax = plt.subplots(figsize=(8, 5))
for g in gammas:
    cells = [(r[1], r[2], r[3]) for r in rows if r[0] == g]
    cells.sort()
    if cells:
        t = [c[0] for c in cells]
        ratio = [c[1]/c[2] if c[2] > 1e-30 else 0 for c in cells]
        ax.semilogy(t, ratio, '-o', label=f'$\\gamma = {g}$', ms=6)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'deficit$_{R4}$ / deficit$_{strict}$')
ax.set_title('Collapse factor (R4 over-states deficit)')
ax.axhline(1.0, color='k', ls=':', lw=0.5)
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{OUT}/ratio_per_gamma.png", dpi=120)
plt.close()

# Fig 3: side-by-side R4 vs strict per cell (bar-style)
fig, ax = plt.subplots(figsize=(12, 5))
labels = [f"g{int(r[0])}t{r[1]:.1f}" for r in rows]
x = np.arange(len(rows))
w = 0.35
ax.bar(x - w/2, [r[2] for r in rows], w, label='R4', color='red', alpha=0.7)
ax.bar(x + w/2, [r[3] for r in rows], w, label='strict-h=0', color='blue', alpha=0.7)
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('deficit')
ax.legend(); ax.grid(alpha=0.3, which='both', axis='y')
ax.set_title('Per-cell deficit: kernel-band R4 vs strict-h=0')
plt.tight_layout()
plt.savefig(f"{OUT}/per_cell_bars.png", dpi=120)
plt.close()

print(f"saved 3 figs to {OUT}/")
