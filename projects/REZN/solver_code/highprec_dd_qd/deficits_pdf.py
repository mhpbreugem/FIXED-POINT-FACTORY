"""Publishable deficits PDF.

Plots only the certified, trust-filtered deficit data:
  Page 1: title + headline
  Page 2: certified deficit vs gamma per tau (Jensen-wedge collapse), log-log
  Page 3: certified deficit map (tau x gamma heatmap)
  Page 4: deep G-ladder convergence per cell (nailed rungs only)
  Page 5: continuum extrapolations with honest +/- bars
"""
import os, json, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight'
OUT = ROOT
BUILD = '/tmp/deficits_build'
os.makedirs(BUILD, exist_ok=True)

# --- emin15 ACCEPT cells only ---
em = json.load(open(f"{ROOT}/emin15/emin15.json"))
acc = []
for v in em.values():
    if v.get('verdict') == 'ACCEPT':
        acc.append(dict(tau=float(v['tau']), gamma=float(v['gamma']),
                        deficit=float(v['deficit']), slope=float(v['slope']),
                        F_ld=float(v.get('F_ld')) if v.get('F_ld') else None,
                        rescued=bool(v.get('rescued', False))))
taus = sorted(set(r['tau'] for r in acc))
gammas_all = sorted(set(r['gamma'] for r in acc))
cmap = plt.get_cmap('viridis')

# --- deep ladder clean ---
dl_clean = json.load(open(f"{ROOT}/deep_ladder/extrapolation_clean.json"))
dl_full = json.load(open(f"{ROOT}/deep_ladder/results.json"))
UMAX = 4.0; C = 0.45
def h_of(G): return C*np.sqrt(2*UMAX/(G-1))

# ============ PAGE 1: title ============
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("Revelation deficit $1-R^2$ on the K=3 CRRA REE",
              fontsize=15, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = [
    "Definition: deficit = $1 - R^2$ of the linear regression of $\\mathrm{logit}\\,P$",
    "on $\\sum_k u_k$ over inner grid cells. Deficit 0 = price is a perfect",
    "sufficient statistic for $\\sum u$ (Jensen-wedge collapse to log-odds line);",
    "deficit 1 = price totally uninformative.",
    "",
    "Data source (trust filter):",
    f"  - 87 cells emin15-certified (extended-precision $\\|F\\|_\\infty \\leq 10^{{-15}}$)",
    f"    out of 91 attempted; 4 rejected (stalls, deficits quarantined)",
    f"  - 5 cells extended through deep G-ladders to $G\\!=\\!37$",
    f"    with continuum extrapolations",
    "",
    "Plots include only the certified cells; rejected cells are excluded.",
    "Continuum extrapolations use only NAILED rungs ($F\\!<\\!10^{-8}$);",
    "cells whose extrapolation overshoots to unphysical (negative)",
    "$d_\\infty$ are flagged unreliable and not used in synthesis claims.",
    "",
    "Headline numbers:",
    f"  - Lowest certified deficit: {min(r['deficit'] for r in acc):.2e}",
    f"    at $(\\tau,\\gamma)=({min((r for r in acc if r['deficit']==min(r2['deficit'] for r2 in acc)), key=lambda r:r['deficit'])['tau']}, "
    f"{min((r for r in acc if r['deficit']==min(r2['deficit'] for r2 in acc)), key=lambda r:r['deficit'])['gamma']})$",
    f"  - Highest certified deficit: {max(r['deficit'] for r in acc):.3f}",
    f"  - Immortal-anchor continuum: $d_\\infty = 0.268 \\pm 0.010$",
    f"    at $(\\tau,\\gamma)=(2.0, 0.098)$",
]
ax.text(0.05, 0.92, "\n".join(txt), fontsize=10.5, va='top', ha='left')
plt.savefig(f"{BUILD}/p1.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 2: deficit vs gamma, log-log per tau ============
fig, ax = plt.subplots(figsize=(8.5, 7))
for ti, tau in enumerate(taus):
    row = sorted([r for r in acc if r['tau'] == tau], key=lambda r: r['gamma'])
    if not row: continue
    g = [r['gamma'] for r in row]
    d = [max(r['deficit'], 1e-9) for r in row]
    color = cmap(ti/max(1, len(taus)-1))
    ax.loglog(g, d, 'o-', color=color, label=f'$\\tau={tau}$', markersize=6,
              linewidth=1.5)
    # mark rescued cells
    rescued = [r for r in row if r['rescued']]
    if rescued:
        ax.loglog([r['gamma'] for r in rescued],
                   [max(r['deficit'], 1e-9) for r in rescued],
                   'o', color=color, markerfacecolor='none',
                   markeredgecolor=color, markersize=11, markeredgewidth=1.5,
                   zorder=4)
# 1/gamma reference
gref = np.logspace(-1.5, 1.5, 50)
ax.loglog(gref, 0.05/gref, '--', color='gray', alpha=0.5)
ax.text(0.07, 0.4, r'reference: $1/\gamma$', color='gray', fontsize=10)
ax.set_xlabel(r'$\gamma$ (risk aversion)', fontsize=12)
ax.set_ylabel(r'$1 - R^2$  (revelation deficit)', fontsize=12)
ax.set_title('Page 2 -- Certified deficit vs $\\gamma$ per $\\tau$ '
              '(circles: $\\tau$-continuation rescues)', fontsize=12)
ax.legend(fontsize=10, loc='lower left'); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{BUILD}/p2.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 3: heatmap ============
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
A = np.full((len(taus), len(gammas_all)), np.nan)
for r in acc:
    i = taus.index(r['tau']); j = gammas_all.index(r['gamma'])
    A[i, j] = r['deficit']
g_log = np.log10(gammas_all)
imargs = dict(aspect='auto', origin='lower',
              extent=[g_log[0], g_log[-1], taus[0]-0.1, taus[-1]+0.1])
ax = axes[0]
im = ax.imshow(np.log10(np.where(A > 0, A, 1e-12)), cmap='viridis',
               vmin=-6, vmax=0, **imargs)
plt.colorbar(im, ax=ax, label=r'$\log_{10}$ deficit')
ax.set_xlabel(r'$\log_{10}\gamma$'); ax.set_ylabel(r'$\tau$')
ax.set_title('Certified deficit (log scale)')
# Slope map
ax = axes[1]
S = np.full((len(taus), len(gammas_all)), np.nan)
for r in acc:
    i = taus.index(r['tau']); j = gammas_all.index(r['gamma'])
    S[i, j] = r['slope']
im2 = ax.imshow(S, cmap='plasma', **imargs)
plt.colorbar(im2, ax=ax, label=r'slope on $\sum u$')
ax.set_xlabel(r'$\log_{10}\gamma$'); ax.set_ylabel(r'$\tau$')
ax.set_title('Log-odds slope (informativeness)')
plt.suptitle('Page 3 -- Maps over the $20\\times 5$ certified grid', fontsize=12)
plt.tight_layout()
plt.savefig(f"{BUILD}/p3.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 4: G-ladder convergence per cell ============
fig, axes = plt.subplots(1, 1, figsize=(8.5, 7))
ax = axes
for ci, (cell, info) in enumerate(dl_full.items()):
    rungs = info['ladder']
    Gs = np.array([r['G'] for r in rungs])
    F = np.array([r['F'] for r in rungs])
    d = np.array([r['deficit'] for r in rungs])
    nailed_mask = F < 1e-8
    color = plt.cm.tab10(ci)
    label = cell.replace('t', r'$\tau{=}').replace('_g', r'$,$\gamma{=}') + '$'
    ax.semilogy(Gs[nailed_mask], d[nailed_mask], 'o-', color=color, label=label,
                markersize=7, linewidth=1.5)
    if (~nailed_mask).any():
        ax.semilogy(Gs[~nailed_mask], d[~nailed_mask], 'x', color=color,
                    markersize=10, markeredgewidth=2)
ax.set_xlabel(r'grid size $G$', fontsize=12)
ax.set_ylabel(r'deficit $1 - R^2$', fontsize=12)
ax.set_title('Page 4 -- Deep-ladder convergence per cell '
              '(circles: nailed, $\\times$: stalled and quarantined)', fontsize=12)
ax.legend(fontsize=10, loc='best'); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{BUILD}/p4.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 5: continuum extrapolation ============
fig, ax = plt.subplots(figsize=(8.5, 7))
for ci, info in enumerate(dl_clean):
    cell = info['cell']
    if info['n_stalled'] > 0 and info.get('d_inf') == '--': continue
    label = cell.replace('t', r'$\tau{=}').replace('_g', r'$,$\gamma{=}') + '$'
    rungs = dl_full[cell]['ladder']
    nailed = [r for r in rungs if r['F'] < 1e-8]
    Gs = np.array([r['G'] for r in nailed])
    h = h_of(Gs)
    d = np.array([r['deficit'] for r in nailed])
    color = plt.cm.tab10(ci)
    is_trust = info['n_stalled'] == 0
    suffix = '' if is_trust else ' (unreliable, stalls)'
    ax.plot(h, d, 'o-' if is_trust else 'o--', color=color,
            label=label + suffix, markersize=7, linewidth=1.5)
    if isinstance(info['d_inf'], float) and is_trust:
        d_inf = info['d_inf']; err = info['err']
        # plot extrapolation point with error bar at h=0
        ax.errorbar([0.0], [d_inf], yerr=[err], fmt='s', color=color,
                    markersize=10, capsize=6, capthick=2, zorder=10)
        # connect last nailed to extrap
        ax.plot([0, h[-1]], [d_inf, d[-1]], ':', color=color, alpha=0.5)
ax.set_xlabel(r'kernel bandwidth $h = 0.45\,\sqrt{\Delta u}$', fontsize=12)
ax.set_ylabel(r'deficit $1 - R^2$', fontsize=12)
ax.set_title('Page 5 -- Continuum ($h{\\to}0$) deficit extrapolations '
              'with honest $\\pm$ bars', fontsize=12)
ax.legend(fontsize=10, loc='best'); ax.grid(alpha=0.3)
ax.set_xlim(left=-0.02)
plt.tight_layout()
plt.savefig(f"{BUILD}/p5.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- combine to PDF ----------
from PIL import Image
imgs = [Image.open(f"{BUILD}/p{p}.png").convert("RGB") for p in range(1, 6)]
dst = f"{OUT}/DEFICITS_publishable.pdf"
imgs[0].save(dst, save_all=True, append_images=imgs[1:])
print(f"saved {dst}")
