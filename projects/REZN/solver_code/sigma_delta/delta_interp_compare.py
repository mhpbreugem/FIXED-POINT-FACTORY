"""G=21 with different δ̂ edge interpolation methods. Compare first 5 iters."""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior, crra_clear_sym)

G_FULL = 21
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
N_SNAPSHOTS = 6  # iter 0 + 1..5

xi_full = np.linspace(-1.0, 1.0, G_FULL)
xi_inner = xi_full[INNER_LO:INNER_HI]
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

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


def set_boundary_method(P, method):
    """Apply BCs:
       - u_1 = ±∞ (i=0, G-1): P=0/1 (FR limit)
       - Σ̂ = ±∞ (j=0, G-1): P=0/1 (FR limit)
       - δ̂ = ±∞ (k=0, G-1): per method
    """
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    if method == 'zero':
        P[:, :, 0] = P[:, :, 1]
        P[:, :, G-1] = P[:, :, G-2]
    elif method == 'linear':
        P[:, :, 0] = 2*P[:, :, 1] - P[:, :, 2]
        P[:, :, G-1] = 2*P[:, :, G-2] - P[:, :, G-3]
    elif method == 'quadratic':
        # 3-point Lagrange extrapolation at offset h with grid points 1,2,3 (from boundary)
        # P[0] = 3 P[1] - 3 P[2] + P[3]
        P[:, :, 0] = 3*P[:, :, 1] - 3*P[:, :, 2] + P[:, :, 3]
        P[:, :, G-1] = 3*P[:, :, G-2] - 3*P[:, :, G-3] + P[:, :, G-4]
    elif method == 'cubic':
        # 4-point Lagrange extrapolation
        # P[0] = 4 P[1] - 6 P[2] + 4 P[3] - P[4]
        P[:, :, 0] = 4*P[:, :, 1] - 6*P[:, :, 2] + 4*P[:, :, 3] - P[:, :, 4]
        P[:, :, G-1] = 4*P[:, :, G-2] - 6*P[:, :, G-3] + 4*P[:, :, G-4] - P[:, :, G-5]
    else:
        raise ValueError(method)
    P = np.clip(P, 1e-30, 1 - 1e-30)
    return P


methods = ['zero', 'linear', 'quadratic', 'cubic']
results = {}

# JIT warmup
P_warm = np.zeros((G_FULL,)*3)
P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_warm = set_boundary_method(P_warm, 'zero')
print('JIT warmup...')
_ = phi_sigmadelta(P_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print('done')

for method in methods:
    print(f'\n=== {method} ===')
    P = np.zeros((G_FULL,)*3)
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
    P = set_boundary_method(P, method)
    snapshots = [P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()]
    ferr_trace = [0.0]
    for it in range(1, N_SNAPSHOTS):
        P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                                INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_new = set_boundary_method(P_new, method)
        ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P = P_new
        snapshots.append(P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy())
        ferr_trace.append(ferr)
        print(f'  iter {it}: ferr={ferr:.3e}')
    results[method] = {'snapshots': snapshots, 'ferr': ferr_trace}

# ============ PLOTS ============
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
mid = G_INNER // 2
extent = [xi_inner[0], xi_inner[-1], xi_inner[0], xi_inner[-1]]

# Plot 1: rows = methods, cols = iters
fig, axes = plt.subplots(len(methods), N_SNAPSHOTS, figsize=(2.8*N_SNAPSHOTS, 2.8*len(methods)), dpi=140)
for ri, method in enumerate(methods):
    for ci, P_in in enumerate(results[method]['snapshots']):
        ax = axes[ri, ci]
        sl = P_in[mid, :, :]
        im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                       extent=extent, aspect='auto')
        if ci == 0:
            ax.set_ylabel(f'{method}\nδ̂')
        if ri == len(methods)-1:
            ax.set_xlabel('Σ̂')
        if ri == 0:
            ax.set_title(f'iter {ci}' + (f' (IC)' if ci==0 else f' ferr={results[method]["ferr"][ci]:.2e}'),
                          fontsize=10)
        else:
            ax.set_title(f'ferr={results[method]["ferr"][ci]:.2e}' if ci > 0 else 'IC',
                          fontsize=9)
        if ci == N_SNAPSHOTS-1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P')
plt.suptitle(rf'G=21 no-learn IC, first {N_SNAPSHOTS-1} iters, slice $\hat u_1=0$. δ̂-edge interpolation comparison.',
              fontsize=12, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/delta_interp_compare_P.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/delta_interp_compare_P.png')

# Plot 2: deviation from FR
fig, axes = plt.subplots(len(methods), N_SNAPSHOTS, figsize=(2.8*N_SNAPSHOTS, 2.8*len(methods)), dpi=140)
for ri, method in enumerate(methods):
    for ci, P_in in enumerate(results[method]['snapshots']):
        ax = axes[ri, ci]
        sl_FR = P_FR_in[mid, :, :]
        dev = P_in[mid, :, :] - sl_FR
        vmax = max(abs(dev).max(), 1e-10)
        im = ax.imshow(dev.T, origin='lower', cmap='PiYG', vmin=-vmax, vmax=vmax,
                       extent=extent, aspect='auto')
        if ci == 0: ax.set_ylabel(f'{method}\nδ̂')
        if ri == len(methods)-1: ax.set_xlabel('Σ̂')
        ax.set_title(f'iter {ci}\nmax|dev|={vmax:.2e}', fontsize=9)
        if ci == N_SNAPSHOTS-1:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P − P^FR')
plt.suptitle(rf'G=21 deviation from FR, no-learn IC, first {N_SNAPSHOTS-1} iters, slice $\hat u_1=0$',
              fontsize=12, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/delta_interp_compare_dev.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/delta_interp_compare_dev.png')

# Plot 3: ferr trajectories
fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
markers = {'zero': 'o', 'linear': 's', 'quadratic': '^', 'cubic': 'D'}
for method in methods:
    ax.semilogy(range(N_SNAPSHOTS),
                [max(f, 1e-30) for f in results[method]['ferr']],
                f'{markers[method]}-', lw=2, ms=8, label=method)
ax.set_xlabel('Picard iteration')
ax.set_ylabel('ferr')
ax.set_title('First 5 iters: ferr trajectories for δ̂-edge interpolation methods (G=21)')
ax.grid(True, ls=':', alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/delta_interp_compare_ferr.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/delta_interp_compare_ferr.png')

print('\n=== SUMMARY ===')
for method in methods:
    print(f'  {method:10s}: ferrs = {[f"{f:.2e}" for f in results[method]["ferr"]]}')
