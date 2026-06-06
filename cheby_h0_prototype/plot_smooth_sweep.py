"""Plot smoothed-operator h_bw sweep results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('/tmp/cheby_h0/smooth_sweep.json') as f:
    d = json.load(f)

hbw = sorted(d.keys(), key=float, reverse=True)
hbw_v = [float(h) for h in hbw]
and_floor = [d[h]['anderson_floor'] for h in hbw]
lm_floor = [d[h]['lm_floor'] for h in hbw]
lm_cost = [d[h]['lm_cost'] for h in hbw]

# Add reference: bisect floor (h_bw → 0 equivalent)
bi_floor = 2.703e-3   # from fp_analysis.json
cr_floor = 4.254e-3

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: floor vs h_bw
ax = axes[0]
ax.loglog(hbw_v, and_floor, 'o-', label='Anderson floor', color='tab:blue', markersize=8)
ax.loglog(hbw_v, lm_floor, 's-', label='Lev-Marq floor', color='tab:green', markersize=8)
ax.axhline(bi_floor, color='tab:red', linestyle='--', alpha=0.7,
            label=f'bisect (no smoothing) = {bi_floor:.2e}')
ax.axhline(cr_floor, color='tab:purple', linestyle=':', alpha=0.7,
            label=f'chebroots (no smoothing) = {cr_floor:.2e}')
ax.set_xlabel(r'smoothing bandwidth $h_{bw}$ (in $p$-units)')
ax.set_ylabel(r'min $\|F\|_\infty$ achieved')
ax.set_title('Option-3 boundary correction: floor vs bandwidth\n'
              '(floor essentially flat — boundary smoothing alone insufficient)')
ax.legend(fontsize=9, loc='best')
ax.grid(True, which='both', alpha=0.3)
ax.invert_xaxis()

# Right: LM cost (||F||_2^2 / 2)
ax = axes[1]
ax.semilogx(hbw_v, lm_cost, 'D-', color='tab:orange', markersize=8)
ax.set_xlabel(r'smoothing bandwidth $h_{bw}$')
ax.set_ylabel(r'LM cost $\frac{1}{2}\|F\|_2^2$')
ax.set_title('LM cost (genuine $\\|F\\|^2$ minimum)\n'
              'all bandwidths converge to similar minima ~5-10e-6')
ax.invert_xaxis()
ax.grid(True, alpha=0.3)
ax.axhline(min(lm_cost), color='gray', linestyle=':',
            label=f'best = {min(lm_cost):.2e} at $h_{{bw}}$={hbw_v[lm_cost.index(min(lm_cost))]:.0e}')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/fp_analysis_smooth_sweep.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved fp_analysis_smooth_sweep.png')

# Summary
print('\n=== Summary: Option-3 boundary corrections do NOT meaningfully drop the floor ===')
print(f'  best Anderson floor: {min(and_floor):.3e} at h_bw={hbw_v[and_floor.index(min(and_floor))]:.0e}')
print(f'  best LM floor:       {min(lm_floor):.3e} at h_bw={hbw_v[lm_floor.index(min(lm_floor))]:.0e}')
print(f'  reference (bisect, no smoothing): {bi_floor:.3e}')
print(f'  improvement factor: {bi_floor/min(lm_floor):.2f}x  --  marginal')
