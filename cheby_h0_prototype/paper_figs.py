"""Paper-quality figures for 2D (tau, gamma) sweep."""
import json, os, sys
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib import cm
from lin_cdf_kern_tab import make_cdf_uniform_grid

FIGS = '/tmp/cheby_h0/figs/paper'
os.makedirs(FIGS, exist_ok=True)

d = json.load(open('/tmp/cheby_h0/paper_2d.json'))
items = sorted(d.items(), key=lambda x: (x[1]['tau'], x[1]['gamma']))

# Build 2D arrays
TAUS = sorted(set(v['tau'] for v in d.values()))
GAMMAS = sorted(set(v['gamma'] for v in d.values()))
nt = len(TAUS); ng = len(GAMMAS)
print(f'Grid: {nt} x {ng}')

# Map (tau, gamma) -> indices
tau_idx = {t: i for i, t in enumerate(TAUS)}
gam_idx = {g: i for i, g in enumerate(GAMMAS)}

# Build matrices
F_grid = np.full((nt, ng), np.nan)
slope_grid = np.full((nt, ng), np.nan)
def_grid = np.full((nt, ng), np.nan)
deflin_grid = np.full((nt, ng), np.nan)
rb_grid = np.full((nt, ng), np.nan)
for v in d.values():
    i, j = tau_idx[v['tau']], gam_idx[v['gamma']]
    F_grid[i, j] = v['F']
    slope_grid[i, j] = v['slope']
    def_grid[i, j] = v['deficit_oneToOne']
    deflin_grid[i, j] = v['deficit_lin']
    rb_grid[i, j] = v['romberg_err']

# Mask non-converged
mask_conv = F_grid < 1e-10

# === FIG 1: slope heatmap ===
fig, ax = plt.subplots(figsize=(10, 7))
slope_plot = np.where(mask_conv, slope_grid, np.nan)
im = ax.pcolormesh(GAMMAS, TAUS, slope_plot, cmap='viridis',
                      shading='auto', vmin=0.3, vmax=0.83)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'risk aversion $\gamma$', fontsize=13)
ax.set_ylabel(r'signal precision $\tau$', fontsize=13)
ax.set_title(r'Equilibrium slope $\alpha^*(\tau, \gamma)$' '\n'
              '(Lin-CDF R4 Richardson, 900-cell sweep at machine $\\varepsilon$)',
              fontsize=13)
cbar = plt.colorbar(im, ax=ax, label=r'$\alpha^*$')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_2d.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 01_slope_2d.png')

# === FIG 2: deficit heatmap ===
fig, ax = plt.subplots(figsize=(10, 7))
def_plot = np.where(mask_conv, def_grid, np.nan)
im = ax.pcolormesh(GAMMAS, TAUS, np.maximum(def_plot, 1e-5),
                      cmap='RdYlGn_r', shading='auto',
                      norm=LogNorm(vmin=1e-4, vmax=1e-1))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=13); ax.set_ylabel(r'$\tau$', fontsize=13)
ax.set_title(r'One-to-one deficit $1-R^2_\mathrm{nonparam}(\tau, \gamma)$' '\n'
              r'(real $P \leftrightarrow T$ breakdown)', fontsize=13)
cbar = plt.colorbar(im, ax=ax, label=r'$1-R^2_\mathrm{nonparam}$')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_deficit_2d.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 02_deficit_2d.png')

# === FIG 3: convergence floor heatmap ===
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.pcolormesh(GAMMAS, TAUS, np.log10(np.maximum(F_grid, 1e-18)),
                      cmap='RdYlGn_r', shading='auto', vmin=-16, vmax=-2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=13); ax.set_ylabel(r'$\tau$', fontsize=13)
ax.set_title(r'$\log_{10} \|F\|_\infty$ at converged FP', fontsize=13)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{FIGS}/03_floor_2d.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 03_floor_2d.png')

# === FIG 4: Romberg error bar ===
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.pcolormesh(GAMMAS, TAUS, np.log10(np.maximum(rb_grid, 1e-10)),
                      cmap='RdYlGn_r', shading='auto', vmin=-7, vmax=-1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=13); ax.set_ylabel(r'$\tau$', fontsize=13)
ax.set_title(r'$\log_{10} |T_5 - T_4|$  (Romberg error bound on R4)', fontsize=13)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_romberg_2d.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 04_romberg_2d.png')

# === FIG 5: cross-sections ===
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Slope vs gamma at fixed tau (multi-line)
ax = axes[0, 0]
sel_taus_idx = [0, len(TAUS)//4, len(TAUS)//2, 3*len(TAUS)//4, len(TAUS)-1]
for ti in sel_taus_idx:
    ok = mask_conv[ti, :]
    ax.semilogx(np.array(GAMMAS)[ok], slope_grid[ti, ok], 'o-',
                  markersize=4, label=rf'$\tau = {TAUS[ti]:.3g}$')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\alpha^*$')
ax.set_title(r'Slope vs $\gamma$ at selected $\tau$')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# Slope vs tau at fixed gamma
ax = axes[0, 1]
sel_gams_idx = [0, len(GAMMAS)//4, len(GAMMAS)//2, 3*len(GAMMAS)//4, len(GAMMAS)-1]
for gi in sel_gams_idx:
    ok = mask_conv[:, gi]
    ax.semilogx(np.array(TAUS)[ok], slope_grid[ok, gi], 's-',
                  markersize=4, label=rf'$\gamma = {GAMMAS[gi]:.3g}$')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$\alpha^*$')
ax.set_title(r'Slope vs $\tau$ at selected $\gamma$')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# Deficit vs gamma at fixed tau
ax = axes[1, 0]
for ti in sel_taus_idx:
    ok = mask_conv[ti, :]
    ax.loglog(np.array(GAMMAS)[ok], np.maximum(def_grid[ti, ok], 1e-6),
                'o-', markersize=4, label=rf'$\tau = {TAUS[ti]:.3g}$')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title(r'Deficit vs $\gamma$ at selected $\tau$')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# Deficit vs tau at fixed gamma
ax = axes[1, 1]
for gi in sel_gams_idx:
    ok = mask_conv[:, gi]
    ax.loglog(np.array(TAUS)[ok], np.maximum(def_grid[ok, gi], 1e-6),
                's-', markersize=4, label=rf'$\gamma = {GAMMAS[gi]:.3g}$')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title(r'Deficit vs $\tau$ at selected $\gamma$')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

plt.suptitle('2D phase diagram cross-sections: K=3 CRRA REE',
              fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_cross_sections.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 05_cross_sections.png')

# === FIG 6: combined 2-panel for paper main figure ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
im = ax.pcolormesh(GAMMAS, TAUS, slope_plot, cmap='viridis',
                      shading='auto')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=12); ax.set_ylabel(r'$\tau$', fontsize=12)
ax.set_title(r'(a) Slope $\alpha^*$', fontsize=13)
plt.colorbar(im, ax=ax)
ax = axes[1]
im = ax.pcolormesh(GAMMAS, TAUS, np.maximum(def_plot, 1e-5),
                      cmap='RdYlGn_r', shading='auto',
                      norm=LogNorm(vmin=1e-4, vmax=1e-1))
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$', fontsize=12); ax.set_ylabel(r'$\tau$', fontsize=12)
ax.set_title(r'(b) Deficit $1-R^2_\mathrm{nonparam}$', fontsize=13)
plt.colorbar(im, ax=ax)
plt.suptitle(r'2D phase diagram: K=3 CRRA strict-$h{=}0$ REE on Lin-CDF grid',
              fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_main_figure.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 06_main_figure.png')

# === FIG 7: sample contours from 6 corners/centers ===
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
G = 7
u_grid = make_cdf_uniform_grid(G)
mid = G // 2
# 9 sample points: (low/mid/high tau) x (low/mid/high gamma)
sample_tau_idx = [0, nt//2, nt-1]
sample_gam_idx = [0, ng//2, ng-1]
for i, ti in enumerate(sample_tau_idx):
    for j, gi in enumerate(sample_gam_idx):
        ax = axes[i, j]
        t = TAUS[ti]; g = GAMMAS[gi]
        path = f'/tmp/cheby_h0/fps_paper_2d/P_FP_t{t:.6e}_g{g:.6e}.npy'
        if not os.path.exists(path):
            ax.set_title(f't={t:.2g}, g={g:.2g}: no FP')
            continue
        P = np.load(path)
        slice2 = P[:, :, mid]
        cs = ax.contour(u_grid, u_grid, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                          cmap='RdBu_r', linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
        ax.set_title(rf'$\tau={t:.2g}$, $\gamma={g:.2g}$, '
                       rf'$\alpha^*={slope_grid[ti,gi]:.3f}$, '
                       rf'def={def_grid[ti,gi]:.3f}', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
plt.suptitle('FP contours $\\{P=p\\}$ at 9 (τ, γ) samples from the 2D sweep',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/07_contour_samples.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 07_contour_samples.png')

# Summary stats
print('\n=== Summary ===')
print(f'  Total cells: {nt*ng}={len(d)}')
print(f'  Machine eps: {int(np.sum(F_grid < 1e-12))}')
print(f'  Converged:   {int(np.sum(F_grid < 1e-10))}')
print(f'  Mean Romberg error: {np.nanmean(rb_grid):.2e}')
print(f'  Median Romberg error: {np.nanmedian(rb_grid):.2e}')
