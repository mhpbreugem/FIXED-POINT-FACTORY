"""Uniform G=31 (ξ in linspace(-1, 1, 31)), zero-order δ̂ extrap.
First 5 Picard iters from no-learn IC. Compare with dense-edge G=31."""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior, crra_clear_sym)

G_FULL = 31
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
N_SNAPSHOTS = 6

xi_full = np.linspace(-1.0, 1.0, G_FULL)
xi_inner = xi_full[INNER_LO:INNER_HI]
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
print(f'Uniform G={G_FULL}, Δξ = {xi_full[1]-xi_full[0]:.4f}')

u_phys = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
S_full = U1m + U2m + U3m
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sigmoid(TAU * S_full)
mu1 = sigmoid(TAU * U1m); mu2 = sigmoid(TAU * U2m); mu3 = sigmoid(TAU * U3m)
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_sym(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA, W)


def set_boundary_uniform(P):
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return np.clip(P, 1e-30, 1 - 1e-30)


P = np.zeros((G_FULL,)*3)
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = set_boundary_uniform(P)

print('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT: {time.time()-t0:.1f}s')

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

FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
mid = G_INNER // 2

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
    if col == 0: ax.set_ylabel('δ̂')
    title = 'iter 0 (no-learn)' if col == 0 else f'iter {col}\nferr={ferr_trace[col]:.2e}'
    ax.set_title(title, fontsize=10)
    if col == N_SNAPSHOTS - 1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P')
    ax = axes[1, col]
    dev = sl - sl_FR
    vmax = max(abs(dev).max(), 1e-10)
    pm = ax.pcolormesh(xi_inner, xi_inner, dev.T, cmap='PiYG', vmin=-vmax, vmax=vmax, shading='nearest')
    ax.set_xlabel('Σ̂')
    if col == 0: ax.set_ylabel('δ̂')
    ax.set_title(f'max|dev|={vmax:.2e}', fontsize=10)
    if col == N_SNAPSHOTS - 1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P − P^FR')

plt.suptitle(f'Uniform G={G_FULL} (Δξ=0.067), zero-order δ̂ extrap. First 5 Picard iters from no-learn IC.',
              fontsize=11, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/uniform_g31_first5.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/uniform_g31_first5.png')

# Three-way ferr comparison
fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
# This run
ax.semilogy(range(N_SNAPSHOTS), [max(f, 1e-30) for f in ferr_trace], 'o-', lw=2, ms=8, label='uniform G=31 (Δξ=0.067)')
# Earlier results
uniform_g21 = [0.0, 3.322e-01, 1.860e-01, 1.292e-01, 9.216e-02, 6.099e-02]
dense_g31 = [0.0, 3.331e-01, 2.419e-01, 1.373e-01, 1.161e-01, 8.972e-02]
ax.semilogy(range(N_SNAPSHOTS), [max(f, 1e-30) for f in uniform_g21], 's--', lw=2, ms=8, alpha=0.7,
            label='uniform G=21 (Δξ=0.1)')
ax.semilogy(range(N_SNAPSHOTS), [max(f, 1e-30) for f in dense_g31], '^--', lw=2, ms=8, alpha=0.7,
            label='dense-edge G=31 (Δξ=0.1 bulk + 0.01 edges)')
ax.set_xlabel('Picard iteration')
ax.set_ylabel('ferr')
ax.set_title('First 5 iters: 3 grid types, zero-order δ̂')
ax.grid(True, ls=':', alpha=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIG}/grid_comparison_ferr.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/grid_comparison_ferr.png')

print(f'\nUniform G=31 ferr trajectory: {[f"{f:.3e}" for f in ferr_trace]}')
print(f'Uniform G=21 ferr trajectory: {[f"{f:.3e}" for f in uniform_g21]}')
print(f'Dense  G=31 ferr trajectory: {[f"{f:.3e}" for f in dense_g31]}')
print('done')
