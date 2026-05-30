"""Overlay CARA limit on the CRRA deficit-vs-gamma plot — smartly:
  - CRRA deficit from k3_cara test2 (gammas 1..128, tau=2) on log-log
  - CARA deficit (gamma->inf limit) as the horizontal asymptote (the quadrature floor)
  - power-law fit ~gamma^-0.94 from the data
Shows CRRA -> CARA as gamma -> inf: gap closes ~1/gamma toward the no-gap CARA benchmark."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
cara = json.load(open(os.path.join(os.path.dirname(HERE), 'k3_cara', 'report.json')))
t2 = cara['test2']; rows = t2['rows']; cara_def_t2 = t2['cara_deficit']; expo = t2['deficit_decay_exponent']; const = t2['deficit_decay_const']

# also pull CRRA at the consistent G=17 sweep for tau=2 (different op, slightly different deficit)
sweep = json.load(open(os.path.join(HERE, 'sweep2d.json')))['rows']
crra_g17 = sorted([(r['gamma'], r['deficit']) for r in sweep if r['tau'] == 2.0])

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)

# (a) main: CRRA deficit(gamma) at tau=2, with CARA floor + power-law fit
g_cara = np.array([r['gamma'] for r in rows]); d_cara = np.array([r['deficit'] for r in rows])
ax[0].loglog(g_cara, d_cara, 'o-', color='C0', lw=2, ms=7, label='CRRA  (k3_cara G=9, τ=2)')
g_sweep, d_sweep = zip(*crra_g17)
ax[0].loglog(g_sweep, d_sweep, 's--', color='C2', lw=1.5, ms=6, alpha=0.8, label='CRRA  (consistent sweep G=17, τ=2)')
gg = np.geomspace(0.5, 200, 100)
ax[0].loglog(gg, const * gg**expo, 'k:', lw=1.5, alpha=0.7, label=f'fit  ~γ^{expo:.2f}  (theory: γ^-1)')
ax[0].axhline(cara_def_t2, color='C3', ls='-', lw=2.0, label=f'CARA (γ→∞ limit) = {cara_def_t2:.4f}\n(no-Jensen-gap benchmark)')
ax[0].set_xlabel('risk aversion γ'); ax[0].set_ylabel('revelation deficit  1−R²')
ax[0].set_title('CRRA → CARA as γ→∞:  Jensen gap closes ~1/γ to the no-gap benchmark')
ax[0].legend(fontsize=9, loc='lower left'); ax[0].grid(ls=':', which='both', alpha=0.5)

# (b) linear scale to show CARA floor "smartly": deficit - cara_floor vs gamma (the residual gap above CARA)
gap = np.maximum(d_cara - cara_def_t2, 1e-9)
ax[1].loglog(g_cara, gap, 'D-', color='C0', lw=2, ms=7, label='CRRA deficit − CARA floor  (pure Jensen gap above CARA)')
ax[1].loglog(gg, (const * gg**expo - cara_def_t2).clip(min=1e-9), 'k:', lw=1.5, alpha=0.7, label='fit')
ax[1].set_xlabel('γ'); ax[1].set_ylabel('Jensen gap above CARA limit')
ax[1].set_title('The PURE CRRA Jensen contribution → 0 as γ→∞')
ax[1].legend(fontsize=9); ax[1].grid(ls=':', which='both', alpha=0.5)

plt.suptitle('CARA is the γ→∞ limit of CRRA — and it is the no-gap (FR) benchmark.  Partial revelation IS the CRRA wealth-curvature effect.', weight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'cara_limit_smart.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote cara_limit_smart.png')
