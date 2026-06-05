"""Build figures from the overnight rank-1 sweep + bisection benchmarks."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

data = np.load('/tmp/cheby_h0/rank1_phase.npz')
tau_grid = data['tau_grid']
gamma_grid = data['gamma_grid']
alpha = data['alpha']
deficit = data['deficit']
d_FR = data['d_FR']

with open('/tmp/cheby_h0/rank1_overnight_results.json') as f:
    results = json.load(f)

# === Figure 1: Phase diagram (3 panels) ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Alpha heatmap
ax = axes[0]
im = ax.pcolormesh(gamma_grid, tau_grid, alpha,
                     shading='auto', cmap='RdBu_r', vmin=0.4, vmax=1.05)
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$ (CRRA risk aversion)')
ax.set_ylabel(r'$\tau$ (signal precision)')
ax.set_title(r'Rank-1 FP slope $\alpha^*(\tau, \gamma)$' + '\n' + r'$P = \sigma(\alpha^* T)$, $G=15$')
cb = plt.colorbar(im, ax=ax)
cb.set_label(r'$\alpha^*$')

# Deficit heatmap (1 - R^2 of regression)
ax = axes[1]
im = ax.pcolormesh(gamma_grid, tau_grid, np.log10(deficit + 1e-10),
                     shading='auto', cmap='viridis')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\tau$')
ax.set_title('Rank-1 deficit $\\log_{10}(1 - R^2)$' + '\n' + 'how poorly the operator output fits rank-1 form')
plt.colorbar(im, ax=ax)

# d_FR heatmap
ax = axes[2]
im = ax.pcolormesh(gamma_grid, tau_grid, d_FR,
                     shading='auto', cmap='magma')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\tau$')
ax.set_title(r'$d_{FR}$ (RMS distance from FR $\sigma(T)$)' + '\n' + 'small ⇒ near full revelation')
plt.colorbar(im, ax=ax)

plt.suptitle(r'Rank-1 ($h \equiv 0$) phase diagram', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/rank1_phase.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved /tmp/cheby_h0/figs/rank1_phase.png')

# === Figure 2: Grid invariance ===
fig, ax = plt.subplots(figsize=(11, 6))
B = results['B']
for key, rows in B.items():
    Gs = [r['G'] for r in rows]
    alphas = [r['alpha'] for r in rows]
    ax.plot(Gs, alphas, 'o-', label=key.replace('_', ', '))
ax.set_xlabel('G (Lobatto grid size)')
ax.set_ylabel(r'$\alpha^*$')
ax.set_title('Rank-1 $\\alpha^*$ vs grid resolution G\n(should be invariant at large G)')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/rank1_grid_invariance.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved /tmp/cheby_h0/figs/rank1_grid_invariance.png')

# === Figure 3: Bisection speedup ===
N_list = [6, 8]
chebroots_t = [0.163, 0.610]
bisection_t = [0.0154, 0.0370]
speedup = [c/b for c, b in zip(chebroots_t, bisection_t)]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(N_list))
ax.bar(x - 0.2, chebroots_t, width=0.4, color='tab:red',
        label='chebroots (current operator)')
ax.bar(x + 0.2, bisection_t, width=0.4, color='tab:green',
        label='bisection on polynomial')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels([f'N={N}' for N in N_list])
ax.set_ylabel('per-Φ wall time (s)')
ax.set_title(f'Bisection vs chebroots in actual operator\n'
              f'speedup: {speedup[0]:.1f}× at N=6, {speedup[1]:.1f}× at N=8')
ax.legend()
ax.grid(alpha=0.3, axis='y', which='both')
for i, (cr, bi, sp) in enumerate(zip(chebroots_t, bisection_t, speedup)):
    ax.text(i - 0.2, cr*1.1, f'{cr:.3f}s', ha='center', fontsize=9)
    ax.text(i + 0.2, bi*1.1, f'{bi:.4f}s', ha='center', fontsize=9)
    ax.text(i, max(cr, bi) * 1.3, f'{sp:.1f}×', ha='center', fontsize=12,
              fontweight='bold', color='tab:blue')
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/bisection_speedup.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved /tmp/cheby_h0/figs/bisection_speedup.png')

print('\n=== Summary ===')
print(f'Rank-1 alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]')
print(f'Rank-1 deficit range: [{deficit.min():.3e}, {deficit.max():.3e}]')
print(f'Mean alpha (geo): {np.exp(np.log(alpha).mean()):.4f}')
print(f'Cells with alpha > 0.99 (near FR): {int((alpha > 0.99).sum())}/{alpha.size}')
print(f'Cells with alpha < 0.7 (significant PR): {int((alpha < 0.7).sum())}/{alpha.size}')
