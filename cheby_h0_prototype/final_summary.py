"""Produce the final summary figures for the auto-mode exploration."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS = '/tmp/cheby_h0/figs'

# Load all available results
files = ['pou_nq_sweep.json', 'sweet_nq.json', 'warm_nq_fast.json',
         'pou_N_NQ_v2.json', 'phase_NQ16.json', 'kern_tab_results.json',
         'pou_cr_jit_results.json', 'pou_dense_newton.json',
         'pou_gp_sweep.json', 'push_kern_h0.json']
data = {}
for f in files:
    try:
        data[f.replace('.json', '')] = json.load(open(f'/tmp/cheby_h0/{f}'))
    except: pass

# ===== Headline figure: methods comparison =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
methods = [
    ('Chebroots+FD-Newton\n(prior, fp_analysis)', 1.7e-3, 'tab:red'),
    ('POU scan-bisect+And', 5.7e-3, 'tab:orange'),
    ('POU scan-bisect+Newton', 2.7e-3, 'tab:orange'),
    ('POU chebroots+And', 4.4e-3, 'tab:olive'),
    ('POU chebroots+NK+FD-Newton', 2.1e-3, 'tab:olive'),
    ('Kernel-band h=0.3', 1.1e-16, 'tab:green'),
    ('POU chebroots NQ=16', 9.85e-16, 'tab:blue'),
]
labels = [m[0] for m in methods]
floors = [m[1] for m in methods]
colors = [m[2] for m in methods]
ax.barh(range(len(methods)), floors, color=colors, alpha=0.8)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(labels, fontsize=9)
ax.set_xscale('log')
ax.set_xlabel(r'min $\|F\|_\infty$ achieved')
ax.set_title('Residual floor across methods (N=6/G=7, $\\tau{=}\\gamma{=}1$)')
ax.axvline(1e-15, color='black', linestyle=':', label='machine $\\varepsilon$')
ax.grid(axis='x', which='both', alpha=0.3)
ax.legend(fontsize=10)
ax.invert_yaxis()

# NQ sweep
ax = axes[1]
if 'sweet_nq' in data:
    nqs = sorted(int(k) for k in data['sweet_nq'].keys())
    floors_nq = [data['sweet_nq'][str(n)]['nk'] for n in nqs]
    slopes = [data['sweet_nq'][str(n)]['slope'] for n in nqs]
    ax2 = ax.twinx()
    line1 = ax.semilogy(nqs, [max(f, 1e-18) for f in floors_nq], 'o-',
                          color='tab:blue', markersize=10, label='NK floor')
    line2 = ax2.plot(nqs, slopes, 's-', color='tab:red', markersize=10,
                       label='slope $\\alpha^*$')
    ax.set_xlabel('NQ (GL quadrature order in POU)')
    ax.set_ylabel(r'NK floor $\|F\|_\infty$', color='tab:blue')
    ax2.set_ylabel(r'slope $\alpha^*$', color='tab:red')
    ax.set_title('NQ sweet spot at G=7, $\\tau{=}\\gamma{=}1$\n'
                  '(POU+chebroots, NK from Anderson best)')
    ax.axhline(1e-15, color='black', linestyle=':')
    ax.grid(alpha=0.3, which='both')
    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGS}/final_summary.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved final_summary.png')

# ===== Strict-h=0 FP slice visualization =====
P_FP = np.load('/tmp/cheby_h0/P_FP_pou_nq16_n6.npy')
G = P_FP.shape[0]

import sys
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_numba import U_NODES, TAU

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, mid, title in zip(axes,
                              [0, G//2, G-1],
                              [f'$u_3 = u_{{\\rm min}} = {U_NODES[0]:+.1f}$',
                               f'$u_3 = 0$',
                               f'$u_3 = u_{{\\rm max}} = {U_NODES[-1]:+.1f}$']):
    im = ax.pcolormesh(U_NODES, U_NODES, P_FP[:, :, mid],
                         vmin=0, vmax=1, cmap='RdBu_r', shading='auto')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.suptitle('Strict-$h{=}0$ K=3 CRRA REE: $P(u_1, u_2, u_3)$ slices\n'
              '(POU+chebroots, NQ=16, $\\tau{=}\\gamma{=}1$, $G{=}7$, '
              '$\\|F\\|_\\infty = 9.85\\!\\times\\!10^{-16}$)',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/strict_h0_FP_slices.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved strict_h0_FP_slices.png')

# ===== logit(P) vs T regression check (slope α*) =====
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
Pc = np.clip(P_FP, 1e-15, 1-1e-15)
L = np.log(Pc/(1-Pc)).ravel()
slope = float(np.sum(L*T.ravel()) / np.sum(T.ravel()**2))

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(T.ravel(), L, c=Pc.ravel(), cmap='RdBu_r', s=10, alpha=0.7)
ax.plot([T.min(), T.max()], [slope*T.min(), slope*T.max()],
         'k--', lw=2, label=f'fit: $\\alpha^* = {slope:.4f}$')
ax.set_xlabel(r'$T = \tau(u_1 + u_2 + u_3)$')
ax.set_ylabel(r'$\mathrm{logit}\,P$')
ax.set_title(r'logit$P$ vs $T$ at the strict-$h{=}0$ REE FP'
              '\n(near-linear: a partially-revealing equilibrium)')
ax.grid(alpha=0.3); ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/strict_h0_logit_T.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved strict_h0_logit_T.png')
print(f'\nFP characterization:')
print(f'  slope α* = {slope:.6f}')
print(f'  deficit (1-R²) = {1 - np.corrcoef(L, T.ravel())[0,1]**2:.4f}')
