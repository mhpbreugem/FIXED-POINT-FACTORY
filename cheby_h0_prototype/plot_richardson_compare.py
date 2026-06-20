"""4-way operator comparison: Cheb-strict-POU / Lin-CDF-kernel /
Lin-CDF-strict-cspl / Cheb-Richardson."""
import sys, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cheby_numba import U_NODES as ULOB

FIGS = '/tmp/cheby_h0/figs/richardson_compare'
os.makedirs(FIGS, exist_ok=True)

cheb = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
r2c = json.load(open('/tmp/cheby_h0/r2_correct.json'))
linc_kern = json.load(open('/tmp/cheby_h0/lin_cdf_gamma.json'))
linc_strict = json.load(open('/tmp/cheby_h0/lin_strict_gamma.json'))
rich = json.load(open('/tmp/cheby_h0/richardson_gamma.json'))

gammas = sorted([d['gamma'] for d in rich.values()])

def g(d, key, field): return d[f'gamma={key}'][field]

# ===== FIG 1: slope =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogx(gammas, [g(cheb, k, 'slope') for k in gammas], 'o-',
              color='tab:blue', markersize=10, label='Cheb strict-h=0 POU')
ax.semilogx(gammas, [g(rich, k, 'slope') for k in gammas], 'D-',
              color='tab:purple', markersize=10,
              label='Cheb Richardson 3-pt (NEW, fast)')
ax.semilogx(gammas, [g(linc_kern, k, 'slope') for k in gammas], 's-',
              color='tab:red', markersize=10,
              label='Lin-CDF kernel h=0.5 (smoothed FP)')
ax.semilogx(gammas, [g(linc_strict, k, 'slope') for k in gammas], '^-',
              color='tab:green', markersize=10,
              label='Lin-CDF strict-h=0 (didnt converge)')
# Mark machine eps points
for k in gammas:
    if g(cheb, k, 'F_final') < 1e-12:
        ax.plot(k, g(cheb, k, 'slope'), '*', color='blue', markersize=20,
                  markeredgecolor='black')
    if g(rich, k, 'F') < 1e-12:
        ax.plot(k, g(rich, k, 'slope'), '*', color='purple', markersize=20,
                  markeredgecolor='black')
    if g(linc_kern, k, 'F') < 1e-12:
        ax.plot(k, g(linc_kern, k, 'slope'), '*', color='red', markersize=18,
                  markeredgecolor='black')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title('Slope $\\alpha^*$ vs $\\gamma$: 4 operators '
              '($\\star$ = machine $\\varepsilon$)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_4way.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 01_slope_4way.png')

# ===== FIG 2: deficits =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(gammas, np.maximum([g(r2c, k, 'deficit_oneToOne') for k in gammas], 1e-7),
              'o-', color='tab:blue', markersize=10, label='Cheb strict-h=0 POU')
ax.semilogy(gammas, np.maximum([g(rich, k, 'deficit_oneToOne') for k in gammas], 1e-7),
              'D-', color='tab:purple', markersize=10,
              label='Cheb Richardson 3-pt (NEW)')
ax.semilogy(gammas, np.maximum([g(linc_kern, k, 'deficit_oneToOne') for k in gammas], 1e-7),
              's-', color='tab:red', markersize=10,
              label='Lin-CDF kernel h=0.5 (SMOOTHED, fake $\\sim$0)')
ax.semilogy(gammas, np.maximum([g(linc_strict, k, 'deficit_oneToOne') for k in gammas], 1e-7),
              '^-', color='tab:green', markersize=10,
              label='Lin-CDF strict-h=0')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'1 - $R^2_\mathrm{nonparam}$ (real one-to-one breakdown)')
ax.set_title('Nonparametric deficit: Richardson recovers the REAL deficit\n'
              '(kernel smoothing alone gives fake $\\sim$0)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_deficit_4way.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 02_deficit_4way.png')

# ===== FIG 3: convergence floor + wall time =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
x = np.arange(len(gammas))
F_cheb = np.maximum([g(cheb, k, 'F_final') for k in gammas], 1e-18)
F_rich = np.maximum([g(rich, k, 'F') for k in gammas], 1e-18)
F_kern = np.maximum([g(linc_kern, k, 'F') for k in gammas], 1e-18)
F_strict = np.maximum([g(linc_strict, k, 'F') for k in gammas], 1e-18)
ax.bar(x - 0.30, F_cheb, 0.2, color='tab:blue', label='Cheb POU')
ax.bar(x - 0.10, F_rich, 0.2, color='tab:purple', label='Richardson (NEW)')
ax.bar(x + 0.10, F_kern, 0.2, color='tab:red', label='Lin-CDF kernel')
ax.bar(x + 0.30, F_strict, 0.2, color='tab:green', label='Lin-CDF strict')
ax.axhline(1e-15, color='black', linestyle=':', label='machine $\\varepsilon$')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([f'{k}' for k in gammas])
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor per $\\gamma$ (4 operators)')
ax.legend(fontsize=9); ax.grid(axis='y', which='both', alpha=0.3)

ax = axes[1]
t_cheb = [g(cheb, k, 't_solve') for k in gammas]
t_rich = [g(rich, k, 't_solve') for k in gammas]
t_kern = [g(linc_kern, k, 't_solve') for k in gammas]
ax.semilogy(gammas, t_cheb, 'o-', color='tab:blue', markersize=10,
              label='Cheb POU')
ax.semilogy(gammas, t_rich, 'D-', color='tab:purple', markersize=10,
              label='Richardson')
ax.semilogy(gammas, t_kern, 's-', color='tab:red', markersize=10,
              label='Lin-CDF kernel')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('solve time (s)')
ax.set_title('Wall time per $\\gamma$ solve')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/03_floor_time.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 03_floor_time.png')

# ===== FIG 4: contour plots Richardson all gammas =====
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
mid = 3
for ax, k in zip(axes.flat, gammas):
    P = np.load(f'/tmp/cheby_h0/fps_richardson/P_FP_rich_gamma{k}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(ULOB, ULOB, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    r = rich[f'gamma={k}']
    ax.set_title(rf'$\gamma={k}$, $\alpha^*={r["slope"]:.3f}$, '
                   rf'def$_{{1-1}}={r["deficit_oneToOne"]:.4f}$, '
                   rf'$|F|={r["F"]:.0e}$',
                   fontsize=10)
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.grid(alpha=0.3)
plt.suptitle('Richardson Cheb FP contours $\\{P=p\\}$ at $u_3=0$ across $\\gamma$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_rich_contours.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 04_rich_contours.png')

print('\nAll figures saved to', FIGS)
