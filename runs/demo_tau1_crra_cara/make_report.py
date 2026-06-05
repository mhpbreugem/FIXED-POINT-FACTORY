"""Generate plots + LaTeX report PDF for the CRRA + CARA Newton run."""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = '/tmp/mizn_run'
os.makedirs(f'{HERE}/figs', exist_ok=True)

data = json.load(open(f'{HERE}/summary.json'))
P_crra = np.load(f'{HERE}/P_crra.npy')
P_cara = np.load(f'{HERE}/P_cara.npy')

TAU = data['config']['TAU']
G = data['config']['G_inner']
UMAX = data['config']['U_max']
H_KERNEL = data['config']['h_kernel']
ui = np.linspace(-UMAX, UMAX, G)

# ===== Fig 1: convergence trajectory =====
fig, ax = plt.subplots(figsize=(9, 5.5))
for key, color, marker in [('crra', 'tab:blue', 'o'), ('cara', 'tab:red', 's')]:
    h = data[key]['history']
    label = data[key]['label']
    ax.semilogy(h['iter'], h['F_norm'], marker=marker, lw=2, markersize=12, color=color, label=label)
ax.axhline(1e-9, color='black', linestyle=':', alpha=0.5, label='tolerance 1e-9')
ax.set_xlabel('outer Newton iteration', fontsize=12)
ax.set_ylabel(r'$\|F\|_\infty$', fontsize=12)
ax.set_title('Pure (dense) Newton convergence at G=9, τ=1\nBoth cases: quadratic convergence in 2 iterations from Picard warm-start')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{HERE}/figs/01_convergence.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 2: P slices side by side =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# CRRA at u1=0
ax = axes[0]
i = G // 2
slc = P_crra[i, :, :]
im = ax.imshow(slc.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                 extent=[ui[0], ui[-1], ui[0], ui[-1]], aspect='equal')
U2g, U3g = np.meshgrid(ui, ui, indexing='ij')
ax.contour(U2g, U3g, slc, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.5)
ax.set_title(f'CRRA γ=1\nP(u₂, u₃) at u₁={ui[i]:.2f}', fontsize=11)
ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
plt.colorbar(im, ax=ax, shrink=0.8)

# CARA at u1=0
ax = axes[1]
slc = P_cara[i, :, :]
im = ax.imshow(slc.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                 extent=[ui[0], ui[-1], ui[0], ui[-1]], aspect='equal')
ax.contour(U2g, U3g, slc, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.5)
ax.set_title(f'CARA γ=100\nP(u₂, u₃) at u₁={ui[i]:.2f}', fontsize=11)
ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
plt.colorbar(im, ax=ax, shrink=0.8)

# Difference (CARA - CRRA)
ax = axes[2]
diff = P_cara[i, :, :] - P_crra[i, :, :]
im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r',
                 vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                 extent=[ui[0], ui[-1], ui[0], ui[-1]], aspect='equal')
ax.contour(U2g, U3g, diff, levels=[-0.05, 0, 0.05], colors='black', linewidths=0.8)
ax.set_title(f'Difference: CARA - CRRA at u₁={ui[i]:.2f}\nmax|Δ|={np.abs(diff).max():.4f}', fontsize=11)
ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(f'{HERE}/figs/02_slices.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 3: logit(P) vs T regression =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU * (U1 + U2 + U3)
for ax, P, label, color in [(axes[0], P_crra, 'CRRA γ=1', 'tab:blue'),
                               (axes[1], P_cara, 'CARA γ=100', 'tab:red')]:
    Pc = np.clip(P, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    Trange = np.linspace(T.min(), T.max(), 100)
    pred = a[0]*Trange + a[1]
    ax.scatter(T.ravel(), y, s=8, c=color, alpha=0.3, label='cell values')
    ax.plot(Trange, pred, 'k-', lw=2, label=f'fit: slope={a[0]:.4f}')
    ax.plot(Trange, Trange, 'k--', lw=1, alpha=0.5, label='FR slope=1')
    ax.set_xlabel('T = τ·Σu'); ax.set_ylabel('logit(P)')
    ax.set_title(f'{label}: logit(P) vs T regression\n(slope=1 ⟹ FR; slope<1 ⟹ PR)')
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{HERE}/figs/03_logit_regression.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 4: per-iter metrics =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, key in [(axes[0], 'slope_T'), (axes[1], 'deficit'), (axes[2], 'd_FR')]:
    for cs, color, marker in [('crra','tab:blue','o'),('cara','tab:red','s')]:
        h = data[cs]['history']; label = data[cs]['label']
        vals = [data[cs]['metrics_per_iter'][i][key] for i in range(len(h['iter']))]
        ax.plot(h['iter'], vals, marker=marker, lw=2, markersize=10, color=color, label=label)
    ax.set_xlabel('Newton iteration'); ax.set_ylabel(key)
    ax.set_title(f'{key} per iteration')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{HERE}/figs/04_metrics.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 5: slope vs gamma (this point only) compared to prior γ-sweep =====
# Just CRRA point and CARA point on top of the prior session's sweep
fig, ax = plt.subplots(figsize=(10, 5.5))
# Prior sweep at tau=2 (this is from session)
g_prior = [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.225, 0.259]
slope_prior = [0.3599, 0.3601, 0.3603, 0.3611, 0.3626, 0.3641, 0.3668, 0.3708, 0.3728]
ax.semilogx(g_prior, slope_prior, 'go-', lw=1.5, alpha=0.6, label='Prior γ-sweep (τ=2, G=9 hfree strict h=0)')
# This run
ax.semilogx([1.0], [data['crra']['final_metrics']['slope_T']], 'b*', markersize=20, label=f'This run: CRRA γ=1, τ=1 (kernel h≈{H_KERNEL:.3f})')
ax.semilogx([100.0], [data['cara']['final_metrics']['slope_T']], 'r*', markersize=20, label=f'This run: CARA γ=100, τ=1')
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='FR (slope=1)')
ax.set_xlabel('γ'); ax.set_ylabel('slope_T')
ax.set_title('This MIZN run vs prior session reference: slope vs γ\n(different τ and operator class — qualitative comparison only)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
ax.set_ylim(0.2, 1.05)
plt.tight_layout()
plt.savefig(f'{HERE}/figs/05_comparison.png', dpi=140, bbox_inches='tight')
plt.close()

print('All 5 figures generated.')
