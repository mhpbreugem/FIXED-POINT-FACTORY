"""Non-uniform ξ grid: regular spacing 0.1 from -0.9..0.9, then dense 0.01
spacing near edges (0.95, 0.96, ..., 0.99, 1.0 and mirror).
Zero-order ('uniform') extrapolation at the δ̂ edges.
First 5 Picard iters from no-learn IC.
"""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior, crra_clear_sym)

# Build the non-uniform ξ grid: dense edges + regular interior
xi_neg_dense = np.array([-1.0, -0.99, -0.98, -0.97, -0.96, -0.95])
xi_regular = np.arange(-0.9, 0.91, 0.1)      # -0.9, -0.8, ..., 0.9
xi_pos_dense = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
xi_full = np.concatenate([xi_neg_dense, xi_regular, xi_pos_dense])
G_FULL = len(xi_full)
INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = G_FULL - 2

TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
N_SNAPSHOTS = 6

print(f'Grid G_FULL={G_FULL}, G_INNER={G_INNER}')
print(f'ξ_full ({G_FULL} pts): {xi_full}')
print(f'Δξ neighbour gaps: min={np.min(np.diff(xi_full)):.3f}, max={np.max(np.diff(xi_full)):.3f}')

xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
xi_inner = xi_full[INNER_LO:INNER_HI]
u_phys = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
S_full = U1m + U2m + U3m

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

P_FR_in = sigmoid(TAU * S_full)
mu1_NL = sigmoid(TAU * U1m); mu2_NL = sigmoid(TAU * U2m); mu3_NL = sigmoid(TAU * U3m)
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_sym(mu1_NL[i,j,k], mu2_NL[i,j,k], mu3_NL[i,j,k], GAMMA, W)


def set_boundary_uniform(P):
    """Zero-order (=uniform) extrap at δ̂ edges; FR at u_1, Σ̂ edges."""
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]       # zero-order δ̂
    P[:, :, G-1] = P[:, :, G-2]   # zero-order δ̂
    return np.clip(P, 1e-30, 1 - 1e-30)


P = np.zeros((G_FULL,)*3)
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = set_boundary_uniform(P)

print('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT compile + 1 step: {time.time()-t0:.1f}s')

snapshots = [P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()]
ferr_trace = [0.0]
for it in range(1, N_SNAPSHOTS):
    t0 = time.time()
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_uniform(P_new)
    ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P = P_new
    snapshots.append(P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy())
    ferr_trace.append(ferr)
    print(f'  iter {it}: ferr={ferr:.3e}  ({time.time()-t0:.1f}s)')

# Plot
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
mid = G_INNER // 2

# Use pcolormesh with the actual ξ-grid (non-uniform) so cell sizes reflect spacing
fig, axes = plt.subplots(2, N_SNAPSHOTS, figsize=(2.8*N_SNAPSHOTS, 6.5), dpi=140)
for col, P_in in enumerate(snapshots):
    sl = P_in[mid, :, :]
    sl_FR = P_FR_in[mid, :, :]
    ax = axes[0, col]
    pm = ax.pcolormesh(xi_inner, xi_inner, sl.T, cmap='RdBu_r', vmin=0, vmax=1, shading='nearest')
    ax.axvline(-1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axvline(+1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(-1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(+1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    # mark dense edge transitions
    for x_mark in [-0.95, -0.9, 0.9, 0.95]:
        ax.axvline(x_mark, color='lightgrey', lw=0.4, alpha=0.6)
        ax.axhline(x_mark, color='lightgrey', lw=0.4, alpha=0.6)
    if col == 0: ax.set_ylabel(r'$\hat\delta$')
    title = 'iter 0 (no-learn)' if col == 0 else f'iter {col}\nferr={ferr_trace[col]:.2e}'
    ax.set_title(title, fontsize=10)
    if col == N_SNAPSHOTS - 1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P')
    ax = axes[1, col]
    dev = sl - sl_FR
    vmax = max(abs(dev).max(), 1e-10)
    pm = ax.pcolormesh(xi_inner, xi_inner, dev.T, cmap='PiYG', vmin=-vmax, vmax=vmax, shading='nearest')
    for x_mark in [-0.95, -0.9, 0.9, 0.95]:
        ax.axvline(x_mark, color='lightgrey', lw=0.4, alpha=0.6)
        ax.axhline(x_mark, color='lightgrey', lw=0.4, alpha=0.6)
    ax.axvline(-1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axvline(+1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(-1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(+1.0, color='black', lw=0.6, ls='--', alpha=0.6)
    ax.set_xlabel(r'$\hat\Sigma$')
    if col == 0: ax.set_ylabel(r'$\hat\delta$')
    ax.set_title(f'max|dev|={vmax:.2e}', fontsize=10)
    if col == N_SNAPSHOTS - 1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P − P^FR')

plt.suptitle(rf'Non-uniform ξ grid (regular 0.1 + dense 0.01 at edges, G={G_FULL}), zero-order (uniform) δ̂ extrap.'
              + f'\nFirst {N_SNAPSHOTS-1} Picard iters from no-learn IC, slice $\\hat u_1=0$. Light grey: dense-region boundaries.',
              fontsize=10.5, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/dense_edge_first5_nolearn.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/dense_edge_first5_nolearn.png')

# Plot the grid layout
fig, ax = plt.subplots(figsize=(10, 3), dpi=140)
for x in xi_full:
    color = 'red' if abs(x) > 0.94 else 'C0'
    ax.scatter(x, 0, s=80, c=color, edgecolors='black', lw=0.4)
ax.axvline(-1.0, color='black', lw=1.5)
ax.axvline(+1.0, color='black', lw=1.5)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.4, 0.4); ax.set_yticks([])
ax.set_xlabel('ξ')
ax.set_title(f'Non-uniform ξ grid: {G_FULL} pts ({len(xi_neg_dense)+len(xi_pos_dense)} dense red + {len(xi_regular)} regular blue)')
plt.tight_layout()
plt.savefig(f'{FIG}/dense_edge_grid_layout.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/dense_edge_grid_layout.png')

# ferr trajectory
fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
ax.semilogy(range(N_SNAPSHOTS), [max(f, 1e-30) for f in ferr_trace], 'o-', lw=2, ms=8, label='dense-edge G=31')
# overlay the uniform-step G=21 result for reference (ferrs from earlier run)
uniform_ferr = [0.0, 3.322e-01, 1.860e-01, 1.292e-01, 9.216e-02, 6.099e-02]
ax.semilogy(range(N_SNAPSHOTS), [max(f, 1e-30) for f in uniform_ferr], 's--', lw=2, ms=8, alpha=0.7, label='uniform G=21')
ax.set_xlabel('Picard iteration')
ax.set_ylabel('ferr')
ax.set_title('First 5 iters: dense-edge G=31 vs uniform G=21 (both zero-order δ̂)')
ax.grid(True, ls=':', alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/dense_edge_ferr_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/dense_edge_ferr_compare.png')
print('done')
