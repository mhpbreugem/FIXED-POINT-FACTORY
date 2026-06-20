"""Build figures and PDF for Lin-CDF Richardson 100-γ report."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.insert(0, '/tmp/cheby_h0')
from lin_cdf_kern_tab import make_cdf_uniform_grid

FIGS = '/tmp/cheby_h0/figs/lin_r2_report'
os.makedirs(FIGS, exist_ok=True)

# Load data
dl = json.load(open('/tmp/cheby_h0/lin_r2_100.json'))
dc = json.load(open('/tmp/cheby_h0/r10_100gamma.json'))
il = sorted(dl.items(), key=lambda x: float(x[0]))
ic = sorted(dc.items(), key=lambda x: float(x[0]))
gl = np.array([v['gamma'] for k,v in il]); Fl = np.array([v['F'] for k,v in il])
sll = np.array([v['slope'] for k,v in il]); defl = np.array([v['deficit_oneToOne'] for k,v in il])
deflin = np.array([v['deficit_lin'] for k,v in il])
gc = np.array([v['gamma'] for k,v in ic]); Fc = np.array([v['F'] for k,v in ic])
slc = np.array([v['slope'] for k,v in ic]); defc = np.array([v['deficit_oneToOne'] for k,v in ic])

G = 7
u_grid = make_cdf_uniform_grid(G)

# ===== FIG 1: Headline slope + deficit =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.semilogx(gl, sll, 'o-', color='tab:red', markersize=5,
              label='Lin-CDF 2-pt Richardson (100/100 mach. eps)')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\gamma$')
ax.grid(alpha=0.3, which='both'); ax.legend()
ax = axes[1]
ax.loglog(gl, np.maximum(defl, 1e-7), 's-', color='tab:red', markersize=5)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title('One-to-one breakdown $1-R^2_\mathrm{nonparam}$ vs $\gamma$')
ax.grid(alpha=0.3, which='both')
plt.suptitle('Lin-CDF Richardson: equilibrium statistics, $\\tau=1$', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/01_headline.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 2: Convergence floor (all 100 at machine eps) =====
fig, ax = plt.subplots(figsize=(12, 5))
ax.loglog(gl, np.maximum(Fl, 1e-18), '.-', color='tab:red', markersize=6)
ax.axhline(1e-15, color='black', linestyle=':', label=r'machine $\varepsilon$')
ax.axhline(1e-12, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor across all 100 $\\gamma$ values (Lin-CDF 2-pt Richardson)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_floor.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 3: Linear vs nonparametric deficit =====
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogx(gl, deflin, 'o-', color='tab:blue', markersize=5,
              label=r'$1-R^2_\mathrm{lin}$ (vs $\alpha T$ linear fit)')
ax.semilogx(gl, defl, 's-', color='tab:red', markersize=5,
              label=r'$1-R^2_\mathrm{nonparam}$ (vs best $f(T)$, real breakdown)')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('deficit')
ax.set_title('Linear vs nonparametric deficit\n'
              '(linear deficit conflates nonlinearity in $f(T)$ with real breakdown)')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/03_deficits_both.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 4: Cheb vs Lin-CDF Richardson side by side =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ok_c = Fc < 1e-12; ok_l = Fl < 1e-12
ax = axes[0]
ax.semilogx(gc[ok_c], slc[ok_c], 'o-', color='tab:blue', markersize=5,
              label=f'Cheb 10-pt Richardson ({ok_c.sum()}/100 mach.eps)')
ax.semilogx(gl[ok_l], sll[ok_l], 's-', color='tab:red', markersize=5,
              label=f'Lin-CDF 2-pt Richardson (100/100)')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title('Slope: Cheb vs Lin-CDF Richardson FPs')
ax.legend(); ax.grid(alpha=0.3, which='both')
ax = axes[1]
ax.loglog(gc[ok_c], np.maximum(defc[ok_c], 1e-7), 'o-', color='tab:blue',
            markersize=5, label='Cheb')
ax.loglog(gl[ok_l], np.maximum(defl[ok_l], 1e-7), 's-', color='tab:red',
            markersize=5, label='Lin-CDF')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title('Deficit: Cheb persistent vs Lin-CDF decaying')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.suptitle('Cheb vs Lin-CDF Richardson on the same gamma sweep', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_cheb_vs_lin.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 5: Contour gallery at 9 selected gammas =====
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
sample_gammas = [0.01, 0.1, 0.32, 1.0, 3.2, 10, 32, 100, 1000]
mid = G // 2
for ax, gtarget in zip(axes.flat, sample_gammas):
    # Find closest gamma in our sweep
    idx = int(np.argmin(np.abs(gl - gtarget)))
    g_act = gl[idx]
    P = np.load(f'/tmp/cheby_h0/fps_lin_r2/P_FP_g{g_act:.6e}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(u_grid, u_grid, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(rf'$\gamma={g_act:.3g}$, $\alpha^*={sll[idx]:.3f}$, '
                   rf'def$_{{1-1}}={defl[idx]:.4f}$', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
plt.suptitle('Lin-CDF Richardson FP contours $\\{P=p\\}$ at $u_3=0$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_contours.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 6: solve time =====
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogx(gl, [v['t'] for k,v in il], 'o-', color='tab:red', markersize=4,
              label='Lin-CDF 2-pt Richardson')
ax.semilogx(gc, [v['t'] for k,v in ic], 's-', color='tab:blue', markersize=4,
              label='Cheb 10-pt Richardson')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('solve time (s)')
ax.set_title('Per-$\\gamma$ wall time (warm-start chain)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/06_time.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 7: logit-P vs T scatter for representative gammas =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
T = U1+U2+U3
for ax, gtarget in zip(axes.flat, [0.1, 0.3, 1.0, 3.0, 10, 100]):
    idx = int(np.argmin(np.abs(gl - gtarget)))
    g_act = gl[idx]
    P = np.load(f'/tmp/cheby_h0/fps_lin_r2/P_FP_g{g_act:.6e}.npy')
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = sll[idx]
    ax.scatter(T.ravel(), L, c=Pc.ravel(), cmap='RdBu_r', s=6, alpha=0.6)
    ax.plot([T.min(), T.max()], [s*T.min(), s*T.max()], 'k--',
              label=rf'$\alpha^*={s:.4f}$')
    ax.set_xlabel('T'); ax.set_ylabel('logit P')
    ax.set_title(rf'$\gamma={g_act:.3g}$, def$_{{1-1}}={defl[idx]:.4f}$')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.suptitle('logit-$P$ vs $T$ scatter (Lin-CDF Richardson FPs)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/07_logit_T.png', dpi=140, bbox_inches='tight')
plt.close()

print('All figures saved.')
