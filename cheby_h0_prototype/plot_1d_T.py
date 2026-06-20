"""Build figures from the 1D-in-T overnight."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

data = np.load('/tmp/cheby_h0/onedT_phase.npz')
tau_grid = data['tau_grid']
gamma_grid = data['gamma_grid']
m_list = [1, 3, 5, 7]
alpha = {m: data[f'alpha_m{m}'] for m in m_list}
deficit = {m: data[f'deficit_m{m}'] for m in m_list}
with open('/tmp/cheby_h0/onedT_results.json') as f:
    results = json.load(f)

# ===== Figure 1: deficit heatmaps for each m =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, m in enumerate(m_list):
    ax = axes[idx // 2, idx % 2]
    log_def = np.log10(deficit[m] + 1e-12)
    im = ax.pcolormesh(gamma_grid, tau_grid, log_def,
                         shading='auto', cmap='viridis',
                         vmin=-7, vmax=0)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\tau$')
    ax.set_title(f'$\\log_{{10}}$(deficit), m={m} '
                  f'({"rank-1" if m==1 else f"1D-in-T degree {m}"})')
    plt.colorbar(im, ax=ax)
plt.suptitle(r'Deficit reduction as ansatz order $m$ grows', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/onedT_deficit_panels.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved onedT_deficit_panels.png')

# ===== Figure 2: deficit reduction ratio (m=7 / m=1) — where rank-1 was wrong =====
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ratio = deficit[7] / np.maximum(deficit[1], 1e-12)
ax = axes[0]
im = ax.pcolormesh(gamma_grid, tau_grid, np.log10(ratio + 1e-10),
                     shading='auto', cmap='RdBu_r', vmin=-4, vmax=0)
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\tau$')
ax.set_title(r'$\log_{10}(\mathrm{def}_{m=7}/\mathrm{def}_{m=1})$' + '\n' + r'how much $m=7$ improves over rank-1')
plt.colorbar(im, ax=ax)

# Slope at m=7 (the "true" 1D-in-T slope)
ax = axes[1]
im = ax.pcolormesh(gamma_grid, tau_grid, alpha[7],
                     shading='auto', cmap='RdBu_r', vmin=0.3, vmax=1.05)
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\tau$')
ax.set_title(r'$\alpha^*$ at $m=7$ (richer 1D-in-T slope)' + '\n' + 'compare with rank-1 slope from earlier')
plt.colorbar(im, ax=ax)

# Difference: m=7 alpha vs rank-1 alpha
ax = axes[2]
im = ax.pcolormesh(gamma_grid, tau_grid, alpha[7] - alpha[1],
                     shading='auto', cmap='coolwarm', vmin=-0.5, vmax=0.5)
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'$\tau$')
ax.set_title(r'$\alpha^*_{m=7} - \alpha^*_{m=1}$' + '\n' + 'where the slope estimate shifts')
plt.colorbar(im, ax=ax)

plt.suptitle('1D-in-T vs rank-1: deficit and slope comparisons', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/onedT_vs_rank1.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved onedT_vs_rank1.png')

# ===== Figure 3: P̃(T) shapes at anchor points =====
anchors = results['anchors']
n_anchors = len(anchors)
ncols = 3; nrows = (n_anchors + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
m_colors = {1: 'tab:red', 3: 'tab:blue', 5: 'tab:green', 7: 'tab:purple', 10: 'black'}
for idx, anchor in enumerate(anchors):
    ax = axes[idx // ncols, idx % ncols]
    tau = anchor['tau']; gamma = anchor['gamma']
    for m, shape in anchor['shapes'].items():
        m_int = int(m)
        T_vals = np.array(shape['T_vals'])
        P_t = np.array(shape['P_tilde'])
        ax.plot(T_vals, P_t, color=m_colors.get(m_int, 'gray'), lw=1.5,
                 label=f'm={m_int}, def={shape["deficit"]:.2e}')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('T')
    ax.set_ylabel(r'$\tilde P(T)$')
    ax.set_title(f'$\\tau$={tau}, $\\gamma$={gamma}', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
plt.suptitle(r'$\tilde P(T)$ shapes across $(\tau, \gamma)$ — how rank-1 ($m=1$, red) compares to richer ansatze', fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/onedT_shapes.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved onedT_shapes.png')

# Summary table
print('\n=== Summary ===')
for m in m_list:
    d = deficit[m]
    print(f'  m={m}:  alpha range [{alpha[m].min():.3f}, {alpha[m].max():.3f}], '
          f'deficit range [{d.min():.2e}, {d.max():.2e}]')
print(f'\nMean deficit reduction (m=7/m=1): {np.exp(np.log(deficit[7]/np.maximum(deficit[1],1e-12)).mean()):.2e}')
print(f'Max deficit reduction (m=7/m=1): {(deficit[7]/np.maximum(deficit[1],1e-12)).min():.2e}')
