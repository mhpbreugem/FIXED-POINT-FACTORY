"""Three-way comparison: Cheb strict / Lin-CDF kernel / Lin-CDF strict."""
import sys, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

FIGS = '/tmp/cheby_h0/figs/three_way'
os.makedirs(FIGS, exist_ok=True)

cheb = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
linc_kern = json.load(open('/tmp/cheby_h0/lin_cdf_gamma.json'))
linc_strict = json.load(open('/tmp/cheby_h0/lin_strict_gamma.json'))
r2c = json.load(open('/tmp/cheby_h0/r2_correct.json'))

gammas = sorted([d['gamma'] for d in linc_strict.values()])

def get(d, g, k): return d[f'gamma={g}'][k]

# ===== Slope =====
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogx(gammas, [get(cheb, g, 'slope') for g in gammas], 'o-',
              markersize=10, color='tab:blue', label='Cheb strict $h{=}0$ POU')
ax.semilogx(gammas, [get(linc_kern, g, 'slope') for g in gammas], 's-',
              markersize=10, color='tab:red',
              label='Lin-CDF kernel band $h{=}0.5$')
ax.semilogx(gammas, [get(linc_strict, g, 'slope') for g in gammas], 'D-',
              markersize=10, color='tab:green',
              label='Lin-CDF strict $h{=}0$ POU (NEW)')
# Mark machine-eps points
for g in gammas:
    if get(cheb, g, 'F_final') < 1e-12:
        ax.plot(g, get(cheb, g, 'slope'), '*', color='blue',
                  markersize=18, markeredgecolor='black')
    if get(linc_kern, g, 'F') < 1e-12:
        ax.plot(g, get(linc_kern, g, 'slope'), '*', color='red',
                  markersize=18, markeredgecolor='black')
    if get(linc_strict, g, 'F') < 1e-12:
        ax.plot(g, get(linc_strict, g, 'slope'), '*', color='green',
                  markersize=18, markeredgecolor='black')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\gamma$: three operators (* = machine $\varepsilon$)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_three.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 01_slope_three.png')

# ===== Floor =====
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(gammas))
F_cheb = [get(cheb, g, 'F_final') for g in gammas]
F_kern = [get(linc_kern, g, 'F') for g in gammas]
F_strict = [get(linc_strict, g, 'F') for g in gammas]
ax.bar(x - 0.27, np.maximum(F_cheb, 1e-18), 0.27, color='tab:blue',
        label='Cheb strict $h{=}0$')
ax.bar(x + 0.0,  np.maximum(F_kern, 1e-18), 0.27, color='tab:red',
        label='Lin-CDF kernel $h{=}0.5$')
ax.bar(x + 0.27, np.maximum(F_strict, 1e-18), 0.27, color='tab:green',
        label='Lin-CDF strict $h{=}0$')
ax.axhline(1e-15, color='black', linestyle=':', label=r'machine $\varepsilon$')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels([f'{g}' for g in gammas])
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor per $\\gamma$: three operators')
ax.legend(loc='upper right'); ax.grid(axis='y', which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/02_floor_three.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 02_floor_three.png')

# ===== Deficits =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
ax.semilogy(gammas, [get(r2c, g, 'deficit_lin') for g in gammas], 'o-',
              color='tab:blue', markersize=10, label='Cheb strict')
ax.semilogy(gammas, [get(linc_kern, g, 'deficit_lin') for g in gammas], 's-',
              color='tab:red', markersize=10, label='Lin-CDF kernel')
ax.semilogy(gammas, [get(linc_strict, g, 'deficit_lin') for g in gammas], 'D-',
              color='tab:green', markersize=10, label='Lin-CDF strict')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$1-R^2_{linear}$')
ax.set_title('Linear deficit')
ax.legend(); ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.semilogy(gammas, np.maximum([get(r2c, g, 'deficit_oneToOne') for g in gammas], 1e-7),
              'o-', color='tab:blue', markersize=10, label='Cheb strict')
ax.semilogy(gammas, np.maximum([get(linc_kern, g, 'deficit_oneToOne') for g in gammas], 1e-7),
              's-', color='tab:red', markersize=10, label='Lin-CDF kernel')
ax.semilogy(gammas, np.maximum([get(linc_strict, g, 'deficit_oneToOne') for g in gammas], 1e-7),
              'D-', color='tab:green', markersize=10, label='Lin-CDF strict')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_{nonparam}$  (one-to-one breakdown)')
ax.set_title('Nonparametric deficit (one-to-one breakdown)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.suptitle('Deficits: three operators', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/03_deficit_three.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 03_deficit_three.png')

# ===== Contours: Lin-CDF strict =====
from lin_cdf_kern_tab import make_cdf_uniform_grid
ULIN = make_cdf_uniform_grid(7)
G = 7; mid = G//2

fig, axes = plt.subplots(3, 3, figsize=(15, 14))
for ax, g in zip(axes.flat, gammas):
    P = np.load(f'/tmp/cheby_h0/fps_lin_strict/P_FP_lin_strict_gamma{g}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(ULIN, ULIN, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    r = linc_strict[f'gamma={g}']
    ax.set_title(rf'$\gamma={g}$, $\alpha^*={r["slope"]:.3f}$, '
                   rf'$|F|={r["F"]:.1e}$, def$_{{1{{-}}1}}={r["deficit_oneToOne"]:.4f}$',
                   fontsize=10)
    ax.grid(alpha=0.3)
plt.suptitle('Lin-CDF strict $h{=}0$ POU: contours $\\{P=p\\}$ at $u_3=0$ across $\\gamma$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_lin_strict_contours.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 04_lin_strict_contours.png')

# ===== Side-by-side three contours at gamma=1 =====
from cheby_numba import U_NODES as ULOB
P_cheb_g1 = np.load('/tmp/cheby_h0/fps_gamma/P_FP_gamma1.0.npy')
P_kern_g1 = np.load('/tmp/cheby_h0/fps_lin_cdf/P_FP_lin_gamma1.0.npy')
P_strict_g1 = np.load('/tmp/cheby_h0/fps_lin_strict/P_FP_lin_strict_gamma1.0.npy')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, P, u_g, label, color in zip(axes,
                                          [P_cheb_g1, P_kern_g1, P_strict_g1],
                                          [ULOB, ULIN, ULIN],
                                          ['Cheb strict $h{=}0$',
                                           'Lin-CDF kernel $h{=}0.5$',
                                           'Lin-CDF strict $h{=}0$'],
                                          ['blue', 'red', 'green']):
    slice2 = P[:, :, mid]
    cs = ax.contour(u_g, u_g, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(label + f' at $\\gamma=1$', color=color)
    rng = u_g[-1]; ax.set_xlim(-rng, rng); ax.set_ylim(-rng, rng)
    ax.grid(alpha=0.3)
plt.suptitle('Price contours at $\\gamma=1$, $u_3=0$', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_contours_g1_three.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 05_contours_g1_three.png')

# ===== Scatter logit-T at gamma=1 three operators =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, P, u_g, label in zip(axes,
                                 [P_cheb_g1, P_kern_g1, P_strict_g1],
                                 [ULOB, ULIN, ULIN],
                                 ['Cheb strict', 'Lin-CDF kernel', 'Lin-CDF strict']):
    U1, U2, U3 = np.meshgrid(u_g, u_g, u_g, indexing='ij')
    T = U1+U2+U3
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    ax.scatter(T.ravel(), L, c=Pc.ravel(), cmap='RdBu_r', s=6, alpha=0.6)
    s = float(np.sum(L*T.ravel()) / np.sum(T.ravel()**2))
    ax.plot([T.min(), T.max()], [s*T.min(), s*T.max()], 'k--',
              label=f'lin fit $\\alpha^*={s:.4f}$')
    ax.set_xlabel('T'); ax.set_ylabel('logit P')
    ax.set_title(label)
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('logit-$P$ vs $T$ at $\\gamma=1$: three operators',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_scatter_three_g1.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 06_scatter_three_g1.png')

print('All comparison figures saved.')
