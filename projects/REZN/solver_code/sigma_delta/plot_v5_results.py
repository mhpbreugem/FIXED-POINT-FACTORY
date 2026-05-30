"""Plots for the V5 kernel co-area sigma-delta gamma-sweep + CARA test.

Loads:
  flint_v5_gamma_sweep.json     (G_inner=10 main sweep)
  flint_v5_gamma_low.json       (G_inner=10 low-gamma extension)
  flint_v5_G15_gamma_sweep.json (G_inner=15 sweep, if available)
  flint_v5_G15_cara.json        (G_inner=15 CARA test, if available)

Produces (saved in solved_fixed_points/plots/):
  v5_slope_vs_gamma.png    slope(gamma) -- CRRA -> CARA at high gamma
  v5_deficit_vs_gamma.png  1-R^2 and d_FR_w vs gamma (Jensen-gap law)
  v5_picard_traces.png     per-gamma iter histories at G=10 (and G=15 if available)
  v5_G_comparison.png      G=10 vs G=15 FP at each gamma
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points')
SD = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta')
OUT = ROOT / 'plots'
OUT.mkdir(exist_ok=True)

def safe_load(p):
    if not p.exists(): return None
    try: return json.load(open(p))
    except Exception: return None

main = safe_load(SD/'flint_v5_gamma_sweep.json')
low  = safe_load(SD/'flint_v5_gamma_low.json')
G15  = safe_load(SD/'flint_v5_G15_gamma_sweep.json')
G15_cara = safe_load(SD/'flint_v5_G15_cara.json')

def collect(d):
    if d is None: return [], [], [], []
    rs = d.get('results', [])
    g = [r['gamma'] for r in rs]
    s = [r['slope_final'] for r in rs]
    R = [r['one_minus_R2_final'] for r in rs]
    D = [r['d_FR_w_final'] for r in rs]
    return g, s, R, D

g_m, s_m, R_m, D_m = collect(main)
g_l, s_l, R_l, D_l = collect(low)
g15, s15, R15, D15 = collect(G15)

# Merge low + main for G=10
g_all_G10 = sorted(set(g_m + g_l))
s_all_G10 = []
R_all_G10 = []
D_all_G10 = []
for gv in g_all_G10:
    if gv in g_l:
        i = g_l.index(gv)
        s_all_G10.append(s_l[i]); R_all_G10.append(R_l[i]); D_all_G10.append(D_l[i])
    else:
        i = g_m.index(gv)
        s_all_G10.append(s_m[i]); R_all_G10.append(R_m[i]); D_all_G10.append(D_m[i])

# CARA reference at G=15 (from G15_cara if available, last iter)
cara_ref_G15 = None
if G15_cara:
    cara_ref_G15 = (G15_cara['slope'][-1], G15_cara['1mR2'][-1], G15_cara['d_FR_w'][-1])

# ============ 1. slope vs gamma ============
fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
if g_all_G10:
    ax.semilogx(g_all_G10, s_all_G10, 'o-', color='C0', lw=2, ms=10, label='σ-δ V5 kernel co-area, G=10')
if g15:
    ax.semilogx(g15, s15, 's-', color='C2', lw=2, ms=10, label='σ-δ V5 kernel co-area, G=15')
ax.axhline(1.0, color='C3', ls='--', lw=1.5, label='CARA / FR (slope = 1)')
if cara_ref_G15:
    ax.axhline(cara_ref_G15[0], color='C2', ls=':', alpha=0.7, label=f'V5 CARA G=15 ref (slope={cara_ref_G15[0]:.4f})')
ax.set_xlabel('risk aversion γ'); ax.set_ylabel('slope of logit(P) vs τΣu')
ax.set_title('σ-δ V5 kernel co-area: CRRA → CARA (slope → 1) as γ → ∞')
ax.legend(fontsize=10, loc='lower right'); ax.grid(ls=':', which='both', alpha=0.5)
plt.tight_layout(); plt.savefig(OUT/'v5_slope_vs_gamma.png', dpi=140, bbox_inches='tight'); plt.close()
print('wrote v5_slope_vs_gamma.png')

# ============ 2. deficit vs gamma (log-log) ============
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
if g_all_G10:
    ax[0].loglog(g_all_G10, R_all_G10, 'o-', color='C0', lw=2, ms=10, label='G=10')
if g15:
    ax[0].loglog(g15, R15, 's-', color='C2', lw=2, ms=10, label='G=15')
ax[0].set_xlabel('γ'); ax[0].set_ylabel('1−R² (Jensen-gap proxy)')
ax[0].set_title('Jensen gap 1−R² shrinks as γ→∞ (CRRA→CARA)')
ax[0].legend(fontsize=10); ax[0].grid(ls=':', which='both', alpha=0.5)
if g_all_G10:
    ax[1].loglog(g_all_G10, D_all_G10, 'o-', color='C0', lw=2, ms=10, label='G=10')
if g15:
    ax[1].loglog(g15, D15, 's-', color='C2', lw=2, ms=10, label='G=15')
ax[1].set_xlabel('γ'); ax[1].set_ylabel('d_FR_w (signal-weighted RMS dist to FR)')
ax[1].set_title('d_FR_w → 0 as γ → ∞')
ax[1].legend(fontsize=10); ax[1].grid(ls=':', which='both', alpha=0.5)
plt.suptitle('σ-δ V5 kernel co-area γ-sweep: Jensen gap closes ~1/γ to the CARA limit', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'v5_deficit_vs_gamma.png', dpi=140, bbox_inches='tight'); plt.close()
print('wrote v5_deficit_vs_gamma.png')

# ============ 3. Picard traces per gamma at G=10 ============
if main and main.get('results'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
    cmap = plt.cm.viridis
    gammas_sorted = sorted([(r['gamma'], r) for r in main['results']] +
                            [(r['gamma'], r) for r in (low['results'] if low else [])])
    n = len(gammas_sorted)
    for idx, (gv, r) in enumerate(gammas_sorted):
        c = cmap(idx / max(n-1, 1))
        slopes = r['iter_data']['slope']; deficits = r['iter_data']['1mR2']
        iters = list(range(1, len(slopes)+1))
        ax[0].plot(iters, slopes, '-', color=c, lw=1.5, label=f'γ={gv:g}')
        ax[1].plot(iters, deficits, '-', color=c, lw=1.5, label=f'γ={gv:g}')
    ax[0].axhline(1.0, color='C3', ls='--', alpha=0.6)
    ax[0].set_xlabel('Picard iter'); ax[0].set_ylabel('slope'); ax[0].set_title('slope vs iter')
    ax[0].legend(fontsize=8, loc='lower right', ncol=2); ax[0].grid(ls=':')
    ax[1].set_xlabel('Picard iter'); ax[1].set_ylabel('1−R²'); ax[1].set_title('1−R² vs iter')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=8, loc='upper right', ncol=2); ax[1].grid(ls=':', which='both', alpha=0.5)
    plt.suptitle('σ-δ V5 G=10 Picard traces per γ', weight='bold', fontsize=12)
    plt.tight_layout(); plt.savefig(OUT/'v5_picard_traces.png', dpi=140, bbox_inches='tight'); plt.close()
    print('wrote v5_picard_traces.png')

# ============ 4. G comparison ============
if g_all_G10 and g15:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    ax.semilogx(g_all_G10, [1-s for s in s_all_G10], 'o-', color='C0', lw=2, ms=10, label='G=10: 1−slope')
    ax.semilogx(g15, [1-s for s in s15], 's-', color='C2', lw=2, ms=10, label='G=15: 1−slope')
    ax.semilogx(g_all_G10, R_all_G10, 'o:', color='C0', alpha=0.7, label='G=10: 1−R²')
    ax.semilogx(g15, R15, 's:', color='C2', alpha=0.7, label='G=15: 1−R²')
    ax.set_xlabel('γ'); ax.set_ylabel('Jensen-gap metrics')
    ax.set_title('G=15 vs G=10: kernel co-area σ-δ FP at each γ')
    ax.legend(fontsize=10); ax.grid(ls=':', which='both', alpha=0.5)
    plt.tight_layout(); plt.savefig(OUT/'v5_G_comparison.png', dpi=140, bbox_inches='tight'); plt.close()
    print('wrote v5_G_comparison.png')

print(f'\nAll V5 plots in {OUT}')
