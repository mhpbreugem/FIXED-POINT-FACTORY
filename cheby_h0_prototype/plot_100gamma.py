"""Plot 100-gamma Richardson sweep."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

FIGS = '/tmp/cheby_h0/figs/r10_100gamma'
os.makedirs(FIGS, exist_ok=True)

d = json.load(open('/tmp/cheby_h0/r10_100gamma.json'))
items = sorted(d.items(), key=lambda x: float(x[0]))
g = np.array([v['gamma'] for k,v in items])
F = np.array([v['F'] for k,v in items])
slope = np.array([v['slope'] for k,v in items])
def1 = np.array([v['deficit_oneToOne'] for k,v in items])
deflin = np.array([v['deficit_lin'] for k,v in items])

# Mark convergence quality
ok_eps = F < 1e-13
ok_conv = (F >= 1e-13) & (F < 1e-10)
ok_loose = (F >= 1e-10) & (F < 1e-5)
fail = F >= 1e-5

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

ax = axes[0, 0]
ax.semilogx(g[ok_eps], slope[ok_eps], 'o', color='blue', markersize=6,
              label=f'machine eps ({ok_eps.sum()})')
ax.semilogx(g[ok_conv], slope[ok_conv], 's', color='green', markersize=5,
              label=f'<1e-10 ({ok_conv.sum()})')
ax.semilogx(g[ok_loose], slope[ok_loose], '^', color='orange', markersize=5,
              label=f'<1e-5 ({ok_loose.sum()})')
ax.semilogx(g[fail], slope[fail], 'x', color='red', markersize=8,
              label=f'failed ({fail.sum()})')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\gamma$ (100 points, 10-pt Richardson Cheb-tab)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=10)

ax = axes[0, 1]
ax.loglog(g[ok_eps], def1[ok_eps], 'o', color='blue', markersize=6)
ax.loglog(g[ok_conv], def1[ok_conv], 's', color='green', markersize=5)
ax.loglog(g[ok_loose], def1[ok_loose], '^', color='orange', markersize=5)
ax.loglog(g[fail], def1[fail], 'x', color='red', markersize=8)
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_{\rm nonparam}$ (one-to-one breakdown)')
ax.set_title('Deficit vs $\\gamma$ — persistent ~0.02-0.03 across 5 decades')
ax.grid(alpha=0.3, which='both')

ax = axes[1, 0]
ax.loglog(g, np.maximum(F, 1e-18), '.-', color='tab:purple', markersize=4)
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5, label='machine $\\varepsilon$')
ax.axhline(1e-10, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor per $\\gamma$')
ax.grid(alpha=0.3, which='both'); ax.legend()

ax = axes[1, 1]
ax.semilogx(g, deflin, 'o-', markersize=4, color='tab:red', label='linear deficit')
ax.semilogx(g, def1, 's-', markersize=4, color='tab:blue', label='nonparam deficit')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'deficit')
ax.set_title('Linear vs nonparametric deficit')
ax.grid(alpha=0.3, which='both'); ax.legend()

plt.suptitle('100-point gamma sweep, 10-pt Richardson Cheb-tab (G=7, $\\tau=1$)',
              fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGS}/sweep_100.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved sweep_100.png')

# Also: economics-style plot of slope and deficit
fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()
mask_good = F < 1e-8
ax1.semilogx(g[mask_good], slope[mask_good], 'o-', color='tab:blue',
                markersize=7, label=r'slope $\alpha^*$')
ax2.semilogx(g[mask_good], def1[mask_good], 's-', color='tab:red',
                markersize=7, label=r'1-$R^2_{\rm nonparam}$ (deficit)')
ax1.set_xlabel(r'risk aversion $\gamma$', fontsize=12)
ax1.set_ylabel(r'slope $\alpha^*$', color='tab:blue', fontsize=12)
ax2.set_ylabel(r'1-$R^2_{\rm nonparam}$ (one-to-one breakdown)',
                  color='tab:red', fontsize=12)
ax1.set_title('K=3 CRRA strict-$h{=}0$ REE: equilibrium statistics across $\\gamma$\n'
                '(100 gammas, 10-pt Richardson, $\\tau{=}1$, $G{=}7$)')
ax1.grid(alpha=0.3, which='both')
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIGS}/economics.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved economics.png')
