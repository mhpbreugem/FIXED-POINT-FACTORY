"""Plot Cheb vs Lin-CDF 100-gamma Richardson sweeps side by side."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS = '/tmp/cheby_h0/figs/both_100'
os.makedirs(FIGS, exist_ok=True)

dc = json.load(open('/tmp/cheby_h0/r10_100gamma.json'))      # Cheb 10-pt
dl = json.load(open('/tmp/cheby_h0/lin_r2_100.json'))        # Lin-CDF 2-pt

ic = sorted(dc.items(), key=lambda x: float(x[0]))
il = sorted(dl.items(), key=lambda x: float(x[0]))
gc = np.array([v['gamma'] for k,v in ic]); Fc = np.array([v['F'] for k,v in ic])
slc = np.array([v['slope'] for k,v in ic]); defc = np.array([v['deficit_oneToOne'] for k,v in ic])
gl = np.array([v['gamma'] for k,v in il]); Fl = np.array([v['F'] for k,v in il])
sll = np.array([v['slope'] for k,v in il]); defl = np.array([v['deficit_oneToOne'] for k,v in il])

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

ax = axes[0, 0]
ok_c = Fc < 1e-12; ok_l = Fl < 1e-12
ax.semilogx(gc[ok_c], slc[ok_c], 'o-', color='tab:blue', markersize=5,
              label=f'Cheb 10-pt Richardson ({ok_c.sum()}/100 at machine eps)')
ax.semilogx(gl[ok_l], sll[ok_l], 's-', color='tab:red', markersize=5,
              label=f'Lin-CDF 2-pt Richardson ({ok_l.sum()}/100 at machine eps)')
# show non-converged Cheb points faintly
not_c = ~ok_c
if not_c.any():
    ax.semilogx(gc[not_c], slc[not_c], 'x', color='gray', markersize=8,
                  label=f'Cheb not converged ({not_c.sum()})')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\gamma$')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

ax = axes[0, 1]
ax.loglog(gc[ok_c], np.maximum(defc[ok_c], 1e-7), 'o-', color='tab:blue',
            markersize=5, label='Cheb 10-pt Richardson')
ax.loglog(gl[ok_l], np.maximum(defl[ok_l], 1e-7), 's-', color='tab:red',
            markersize=5, label='Lin-CDF 2-pt Richardson')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_{\rm nonparam}$ (one-to-one breakdown)')
ax.set_title('Deficit vs $\\gamma$: Cheb persistent, Lin-CDF decays')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

ax = axes[1, 0]
ax.loglog(gc, np.maximum(Fc, 1e-18), '.-', color='tab:blue', markersize=4,
            label='Cheb')
ax.loglog(gl, np.maximum(Fl, 1e-18), '.-', color='tab:red', markersize=4,
            label='Lin-CDF')
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5, label='machine $\\varepsilon$')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\|F\|_\infty$ achieved')
ax.set_title('Convergence floor')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

ax = axes[1, 1]
ax.semilogx(gc, [v['t'] for k,v in ic], '.-', color='tab:blue', label='Cheb 10-pt')
ax.semilogx(gl, [v['t'] for k,v in il], '.-', color='tab:red', label='Lin-CDF 2-pt')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('solve time (s)')
ax.set_title('Wall time per $\\gamma$')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

plt.suptitle('100-point gamma sweep with Richardson extrapolation:\n'
              'Chebyshev 10-pt vs Linear-CDF 2-pt (G=7, $\\tau=1$)',
              fontsize=14)
plt.tight_layout()
plt.savefig(f'{FIGS}/cheb_vs_lin_100.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved cheb_vs_lin_100.png')
