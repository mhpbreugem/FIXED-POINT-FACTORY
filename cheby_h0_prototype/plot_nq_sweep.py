"""Plot NQ sweep and the sweet-spot behavior."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS = '/tmp/cheby_h0/figs'

# Original NQ sweep
nq_sweep = json.load(open('/tmp/cheby_h0/pou_nq_sweep.json'))
sweet = json.load(open('/tmp/cheby_h0/sweet_nq.json'))
warm = json.load(open('/tmp/cheby_h0/warm_nq_fast.json'))

nqs = sorted(int(k) for k in sweet.keys())
nks = [sweet[str(nq)]['nk'] for nq in nqs]
ands = [sweet[str(nq)]['anderson'] for nq in nqs]
slopes = [sweet[str(nq)]['slope'] for nq in nqs]

# Wider NQ range from initial sweep
nqs_wide = sorted(int(k) for k in nq_sweep.keys())
ands_wide = [nq_sweep[str(nq)]['anderson'] for nq in nqs_wide]
nks_wide = [nq_sweep[str(nq)]['nk'] for nq in nqs_wide]
merged_nq = sorted(set(nqs + nqs_wide))
merged_and = {}; merged_nk = {}
for nq in merged_nq:
    if str(nq) in sweet:
        merged_and[nq] = sweet[str(nq)]['anderson']
        merged_nk[nq] = sweet[str(nq)]['nk']
    else:
        merged_and[nq] = nq_sweep[str(nq)]['anderson']
        merged_nk[nq] = nq_sweep[str(nq)]['nk']

# Figure: NQ vs floor
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
nq_p = sorted(merged_and.keys())
ax.semilogy(nq_p, [max(merged_and[n], 1e-18) for n in nq_p], 'o-',
             color='tab:blue', markersize=8, label='Anderson floor')
ax.semilogy(nq_p, [max(merged_nk[n], 1e-18) for n in nq_p], 's-',
             color='tab:red', markersize=8, label='Newton-Krylov floor')
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5, label='machine $\\varepsilon$')
ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5,
            label='floor of generic-NQ POU')
ax.set_xlabel('NQ (GL quadrature order)')
ax.set_ylabel(r'min $\|F\|_\infty$ achieved')
ax.set_title('POU+chebroots at N=6 (G=7), $\\tau=\\gamma=1$:\n'
              'NQ=16 hits machine $\\varepsilon$ -- a sweet spot')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, which='both')

ax = axes[1]
nq_warm = sorted(int(k) for k in warm.keys())
ax.semilogy(nq_warm, [max(warm[str(n)]['warm'], 1e-18) for n in nq_warm], 'o-',
             color='tab:purple', markersize=8,
             label='|F| at NQ=16 FP, under each NQ')
ax.semilogy(nq_warm, [max(warm[str(n)]['anderson'], 1e-18) for n in nq_warm], 's-',
             color='tab:olive', markersize=8,
             label='After 30 Anderson from NQ=16 FP')
ax.set_xlabel('NQ (GL quadrature order)')
ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Warm-start from NQ=16 FP into other NQ operators:\n'
              'each NQ has its OWN FP (operator depends on NQ)')
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5)
# mark odd NQ
for nq in nq_warm:
    if nq % 2 == 1:
        ax.axvline(nq, color='red', alpha=0.1)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3, which='both')

plt.suptitle('NQ sweet spot for POU+chebroots Cheb-tab at G=7', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/pou_nq_sweet.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved pou_nq_sweet.png')

# Also slope at each NQ
fig, ax = plt.subplots(figsize=(10, 5))
nq_sl = sorted(int(k) for k in sweet.keys())
slopes = [sweet[str(n)]['slope'] for n in nq_sl]
ax.plot(nq_sl, slopes, 'o-', markersize=8)
ax.set_xlabel('NQ')
ax.set_ylabel(r'slope $\alpha^*$ at converged FP')
ax.set_title('FP slope $\\alpha^*$ varies with NQ\n'
              '(each NQ has slightly different operator FP)')
ax.grid(alpha=0.3)
ax.set_ylim(0.31, 0.37)
plt.tight_layout()
plt.savefig(f'{FIGS}/pou_nq_slope.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved pou_nq_slope.png')

# Visualize Lobatto-vs-GL coincidences
fig, ax = plt.subplots(figsize=(11, 5))
from cheby_numba import LOBATTO
NQs_to_show = [12, 13, 14, 15, 16, 17, 18, 19, 20]
for i, NQ in enumerate(NQs_to_show):
    gl, _ = np.polynomial.legendre.leggauss(NQ)
    y_offset = i * 0.5
    ax.scatter(gl, [y_offset]*len(gl), color='tab:blue', s=30, marker='o')
    # mark coincidences with Lobatto
    for lob in LOBATTO:
        if np.min(np.abs(gl - lob)) < 1e-12:
            idx = np.argmin(np.abs(gl - lob))
            ax.scatter(gl[idx], y_offset, color='red', s=100, marker='x', linewidth=2)
    ax.text(-1.1, y_offset, f'NQ={NQ}', ha='right', va='center', fontsize=10)
# overlay Lobatto
for lob in LOBATTO:
    ax.axvline(lob, color='gray', linestyle=':', alpha=0.5)
ax.set_xlim(-1.15, 1.05)
ax.set_xlabel(r'$\xi$ ∈ [-1, 1]')
ax.set_yticks([])
ax.set_title('GL nodes (blue) vs Lobatto nodes (vertical dashed lines)\n'
              'Red X = exact coincidence (occurs for odd NQ at G=7)')
plt.tight_layout()
plt.savefig(f'{FIGS}/pou_nq_nodes.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved pou_nq_nodes.png')
