"""Compare Lin-CDF Richardson orders R2/R3/R4/R5 across 100 gammas."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS = '/tmp/cheby_h0/figs/orders_report'
os.makedirs(FIGS, exist_ok=True)

orders = {
    'R2': ('/tmp/cheby_h0/lin_r2_100.json', 'tab:blue', (0.5, 0.3)),
    'R3': ('/tmp/cheby_h0/lin_R3_100.json', 'tab:green', (0.5, 0.4, 0.3)),
    'R4': ('/tmp/cheby_h0/lin_R4_100.json', 'tab:purple', (0.5, 0.4, 0.3, 0.2)),
    'R5': ('/tmp/cheby_h0/lin_R5_100.json', 'tab:red', (0.6, 0.5, 0.4, 0.3, 0.2)),
}
data = {}
for k, (path, color, hs) in orders.items():
    d = json.load(open(path))
    items = sorted(d.items(), key=lambda x: float(x[0]))
    gs = np.array([v['gamma'] for k2,v in items])
    Fs = np.array([v['F'] for k2,v in items])
    sl = np.array([v['slope'] for k2,v in items])
    df = np.array([v['deficit_oneToOne'] for k2,v in items])
    dl = np.array([v['deficit_lin'] for k2,v in items])
    data[k] = dict(gammas=gs, F=Fs, slope=sl, def1to1=df, deflin=dl,
                     color=color, hs=hs,
                     eps=int(np.sum(Fs < 1e-12)),
                     conv=int(np.sum(Fs < 1e-10)))

# ===== FIG 1: slope comparison =====
fig, ax = plt.subplots(figsize=(12, 6))
for k, v in data.items():
    ok = v['F'] < 1e-10
    ax.semilogx(v['gammas'][ok], v['slope'][ok], 'o-', color=v['color'],
                  markersize=4,
                  label=f'{k} hs={v["hs"]} ({v["eps"]}/100 mach.eps)')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title('Slope $\\alpha^*$ vs $\\gamma$ across Richardson orders')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_orders.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 2: deficit comparison =====
fig, ax = plt.subplots(figsize=(12, 6))
for k, v in data.items():
    ok = v['F'] < 1e-10
    ax.loglog(v['gammas'][ok], np.maximum(v['def1to1'][ok], 1e-7),
                's-', color=v['color'], markersize=4,
                label=f'{k} hs={v["hs"]}')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title('One-to-one deficit vs $\\gamma$: Richardson order convergence')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_deficit_orders.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 3: convergence floor =====
fig, ax = plt.subplots(figsize=(12, 6))
for k, v in data.items():
    ax.loglog(v['gammas'], np.maximum(v['F'], 1e-18), '.-',
                color=v['color'], markersize=4,
                label=f'{k} ({v["eps"]}/100 mach.eps)')
ax.axhline(1e-15, color='black', linestyle=':', label=r'machine $\varepsilon$')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor per $\\gamma$ across Richardson orders')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/03_floor_orders.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 4: Richardson weights bar =====
fig, ax = plt.subplots(figsize=(12, 5))
xpos = 0
labels_pos = []
for k, v in data.items():
    from lin_cdf_richardson import richardson_weights
    w = richardson_weights(v['hs'])
    for i, (h, wi) in enumerate(zip(v['hs'], w)):
        ax.bar(xpos + i*0.5, wi, 0.4, color=v['color'])
    labels_pos.append((xpos + (len(w)-1)*0.25, f'{k}\nsum|w|={np.sum(np.abs(w)):.1f}'))
    xpos += len(w)*0.6 + 1.5
ax.set_xticks([p for p, _ in labels_pos])
ax.set_xticklabels([l for _, l in labels_pos])
ax.set_ylabel('Richardson weight')
ax.set_title('Richardson weights per order (larger |w| = numerical amplification)')
ax.axhline(0, color='black', lw=0.5)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_weights.png', dpi=140, bbox_inches='tight')
plt.close()

print('All figures saved.')
print()
print('Summary table:')
print(f'{"Order":>6} {"weights sum|w|":>15} {"machine eps":>13} {"slope max":>10} {"def_1to1 min":>14}')
import sys
sys.path.insert(0, '/tmp/cheby_h0')
from lin_cdf_richardson import richardson_weights
for k, v in data.items():
    w = richardson_weights(v['hs'])
    ok = v['F'] < 1e-10
    print(f'{k:>6} {np.sum(np.abs(w)):>15.2f} {v["eps"]:>10d}/100 '
          f'{v["slope"][ok].max():>10.4f} {v["def1to1"][ok].min():>14.4f}')
