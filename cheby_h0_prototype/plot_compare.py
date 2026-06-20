"""Comparison figures: Chebyshev POU vs Linear-CDF kernel-band."""
import sys, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

FIGS = '/tmp/cheby_h0/figs/compare'
os.makedirs(FIGS, exist_ok=True)

cheb = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
linc = json.load(open('/tmp/cheby_h0/lin_cdf_gamma.json'))
r2c = json.load(open('/tmp/cheby_h0/r2_correct.json'))

# Common gammas
gammas = sorted([d['gamma'] for d in linc.values()])

# ===== FIG 1: slopes side by side =====
fig, ax = plt.subplots(figsize=(11, 6))
slope_cheb = [cheb[f'gamma={g}']['slope'] for g in gammas]
slope_lin = [linc[f'gamma={g}']['slope'] for g in gammas]
F_cheb = [cheb[f'gamma={g}']['F_final'] for g in gammas]
F_lin = [linc[f'gamma={g}']['F'] for g in gammas]
ax.semilogx(gammas, slope_cheb, 'o-', markersize=10, color='tab:blue',
              label='Chebyshev (strict h=0, POU)')
ax.semilogx(gammas, slope_lin, 's-', markersize=10, color='tab:red',
              label='Linear-CDF (kernel band h=0.5)')
# mark machine-eps points
for g, sc, sl, fc, fl in zip(gammas, slope_cheb, slope_lin, F_cheb, F_lin):
    if fc < 1e-12:
        ax.plot(g, sc, '*', color='blue', markersize=20, markeredgecolor='black')
    if fl < 1e-12:
        ax.plot(g, sl, '*', color='red', markersize=18, markeredgecolor='black')
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title('Slope $\\alpha^*$ vs $\\gamma$: Chebyshev vs Linear-CDF\n'
              '$\\star$ = converged to machine $\\varepsilon$')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 01_slope_compare.png')

# ===== FIG 2: deficits =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
def_cheb_lin = [r2c[f'gamma={g}']['deficit_lin'] for g in gammas]
def_lin_lin = [linc[f'gamma={g}']['deficit_lin'] for g in gammas]
ax.semilogy(gammas, def_cheb_lin, 'o-', markersize=10, color='tab:blue',
              label='Chebyshev')
ax.semilogy(gammas, def_lin_lin, 's-', markersize=10, color='tab:red',
              label='Linear-CDF')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'1 - $R^2_\mathrm{linear}$')
ax.set_title('Linear deficit $1-R^2_\mathrm{lin}$')
ax.legend(); ax.grid(alpha=0.3, which='both')
ax = axes[1]
def_cheb_np = [r2c[f'gamma={g}']['deficit_oneToOne'] for g in gammas]
def_lin_np = [linc[f'gamma={g}']['deficit_oneToOne'] for g in gammas]
ax.semilogy(gammas, np.maximum(def_cheb_np, 1e-7), 'o-', markersize=10,
              color='tab:blue', label='Chebyshev (strict h=0)')
ax.semilogy(gammas, np.maximum(def_lin_np, 1e-7), 's-', markersize=10,
              color='tab:red', label='Linear-CDF (kernel h=0.5)')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'1 - $R^2_\mathrm{nonparam}$  (one-to-one breakdown)')
ax.set_title('Nonparametric deficit (one-to-one breakdown)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.suptitle('Deficit measures: Chebyshev strict h=0 vs Linear-CDF kernel band',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/02_deficit_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 02_deficit_compare.png')

# ===== FIG 3: convergence machine-eps comparison =====
fig, ax = plt.subplots(figsize=(11, 5))
xb = np.arange(len(gammas)) - 0.2
xc = np.arange(len(gammas)) + 0.2
ax.bar(xb, np.maximum(F_cheb, 1e-18), 0.4, color='tab:blue',
        label='Chebyshev (strict h=0)')
ax.bar(xc, np.maximum(F_lin, 1e-18), 0.4, color='tab:red',
        label='Linear-CDF (kernel band)')
ax.axhline(1e-15, color='black', linestyle=':', label='machine $\\varepsilon$')
ax.set_yscale('log')
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'{g}' for g in gammas])
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\|F\|_\infty$ at converged FP')
ax.set_title('Convergence floor per $\\gamma$: Chebyshev vs Linear-CDF')
ax.legend()
ax.grid(axis='y', which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/03_floors_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 03_floors_compare.png')

# ===== FIG 4: contour overlays at gamma=1 =====
# Compare P contours from Cheb FP vs Linear-CDF FP
from cheby_numba import U_NODES as ULOB
from lin_cdf_kern_tab import make_cdf_uniform_grid
ULIN = make_cdf_uniform_grid(7)

P_cheb_g1 = np.load('/tmp/cheby_h0/fps_gamma/P_FP_gamma1.0.npy')
P_lin_g1 = np.load('/tmp/cheby_h0/fps_lin_cdf/P_FP_lin_gamma1.0.npy')

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
# Cheb slice at u_3=0
G = 7; mid = G // 2
slice_cheb = P_cheb_g1[:, :, mid]
cs = ax.contour(ULOB, ULOB, slice_cheb.T, levels=np.arange(0.1, 1.0, 0.1),
                  cmap='RdBu_r', linewidths=1.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
ax.set_title(f'Chebyshev FP at $\\gamma=1$, $u_3=0$\n'
              f'slope $\\alpha^*={cheb["gamma=1.0"]["slope"]:.4f}$, '
              f'def$_{{1{{-}}1}}={r2c["gamma=1.0"]["deficit_oneToOne"]:.4f}$')
ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
ax.grid(alpha=0.3)

ax = axes[1]
slice_lin = P_lin_g1[:, :, mid]
cs = ax.contour(ULIN, ULIN, slice_lin.T, levels=np.arange(0.1, 1.0, 0.1),
                  cmap='RdBu_r', linewidths=1.5)
ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
ax.set_title(f'Linear-CDF FP at $\\gamma=1$, $u_3=0$\n'
              f'slope $\\alpha^*={linc["gamma=1.0"]["slope"]:.4f}$, '
              f'def$_{{1{{-}}1}}={linc["gamma=1.0"]["deficit_oneToOne"]:.4f}$')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
ax.grid(alpha=0.3)

plt.suptitle('Price contours $\\{P=p\\}$ at $u_3=0$, $\\gamma=1$\n'
              '(Chebyshev grid spans $|u|<10$; Linear-CDF grid spans $|u|<2.3$)',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_contours_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 04_contours_compare.png')

# ===== FIG 5: contours over all gammas, Linear-CDF =====
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
for ax, g in zip(axes.flat, gammas):
    P = np.load(f'/tmp/cheby_h0/fps_lin_cdf/P_FP_lin_gamma{g}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(ULIN, ULIN, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    r = linc[f'gamma={g}']
    ax.set_title(rf'$\gamma={g}$, $\alpha^*={r["slope"]:.3f}$, '
                   rf'def$_{{1{{-}}1}}={r["deficit_oneToOne"]:.4f}$',
                   fontsize=10)
    ax.grid(alpha=0.3)
plt.suptitle('Linear-CDF Cube FP: price contours $\\{P=p\\}$ at $u_3=0$ across $\\gamma$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_lin_cdf_contours_all.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 05_lin_cdf_contours_all.png')

# ===== FIG 6: Chebyshev contour gallery (already done in gamma_sweep) =====
# Just produce a cleaner 3x3 grid for the comparison

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
for ax, g in zip(axes.flat, gammas):
    P = np.load(f'/tmp/cheby_h0/fps_gamma/P_FP_gamma{g}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(ULOB, ULOB, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    r = cheb[f'gamma={g}']
    ax.set_title(rf'$\gamma={g}$, $\alpha^*={r["slope"]:.3f}$, '
                   rf'def$_{{1{{-}}1}}={r2c[f"gamma={g}"]["deficit_oneToOne"]:.4f}$',
                   fontsize=10)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.grid(alpha=0.3)
plt.suptitle('Chebyshev Cube FP: price contours $\\{P=p\\}$ at $u_3=0$ across $\\gamma$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_cheb_contours_all.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 06_cheb_contours_all.png')

# ===== FIG 7: logit(P) vs T at gamma=1 side by side =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
TAU = 1.0
U1c, U2c, U3c = np.meshgrid(ULOB, ULOB, ULOB, indexing='ij')
Tc = TAU*(U1c+U2c+U3c)
Pc_c = np.clip(P_cheb_g1, 1e-15, 1-1e-15)
Lc = np.log(Pc_c/(1-Pc_c)).ravel()
ax.scatter(Tc.ravel(), Lc, c=Pc_c.ravel(), cmap='RdBu_r', s=6, alpha=0.6)
slope = cheb['gamma=1.0']['slope']
ax.plot([Tc.min(), Tc.max()], [slope*Tc.min(), slope*Tc.max()], 'k--',
          label=f'lin fit: $\\alpha^*={slope:.4f}$')
ax.set_xlabel('T'); ax.set_ylabel('logit P')
ax.set_title(f'Chebyshev: def$_{{1{{-}}1}}={r2c["gamma=1.0"]["deficit_oneToOne"]:.4f}$')
ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
U1l, U2l, U3l = np.meshgrid(ULIN, ULIN, ULIN, indexing='ij')
Tl = TAU*(U1l+U2l+U3l)
Pc_l = np.clip(P_lin_g1, 1e-15, 1-1e-15)
Ll = np.log(Pc_l/(1-Pc_l)).ravel()
ax.scatter(Tl.ravel(), Ll, c=Pc_l.ravel(), cmap='RdBu_r', s=6, alpha=0.6)
slope = linc['gamma=1.0']['slope']
ax.plot([Tl.min(), Tl.max()], [slope*Tl.min(), slope*Tl.max()], 'k--',
          label=f'lin fit: $\\alpha^*={slope:.4f}$')
ax.set_xlabel('T'); ax.set_ylabel('logit P')
ax.set_title(f'Linear-CDF: def$_{{1{{-}}1}}={linc["gamma=1.0"]["deficit_oneToOne"]:.4f}$')
ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('logit$P$ vs $T$ at $\\gamma=1$: Chebyshev vs Linear-CDF',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/07_scatter_compare_g1.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 07_scatter_compare_g1.png')

print('\nAll comparison figures saved.')
