"""Remake P-slice figure with smooth Chebyshev evaluation on fine grid.
Compares: raw 7x7 Lobatto pixels (left) vs smooth 100x100 Chebyshev eval (right)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval, chebval2d

HERE = '/tmp/cheby_h0'
results = json.load(open(f'{HERE}/results_sym2.json'))
P_final = np.load(f'{HERE}/P_final_sym2.npy')
cfg = results['config']
N = cfg['N']; G = N + 1
TAU = cfg['tau']; GAMMA = cfg['gamma']; C = cfg['c']
LOBATTO = np.array(cfg['lobatto'])
U_NODES = np.array(cfg['u_nodes'])

# Build 3D Chebyshev coefficients (vals → coeffs along each axis)
P_coeffs = P_final.copy()
for ax_idx in range(3):
    coeffs_new = np.empty_like(P_coeffs)
    for i in range(G):
        for j in range(G):
            if ax_idx == 0: idx = (slice(None), i, j)
            elif ax_idx == 1: idx = (i, slice(None), j)
            else: idx = (i, j, slice(None))
            coeffs_new[idx] = chebfit(LOBATTO, P_coeffs[idx], N)
    P_coeffs = coeffs_new

# Fine grid in ξ-space (and corresponding u-space via atanh inverse)
N_FINE = 100
xi_fine = np.linspace(-0.99, 0.99, N_FINE)  # avoid exact ±1 (atanh blowup)
u_fine = C * np.arctanh(xi_fine)

def evaluate_3d(xi1, xi2, xi3):
    """Evaluate the 3D Chebyshev expansion at (xi1, xi2, xi3)."""
    # Reduce along axis 0 first
    # coefficients are P_coeffs[i,j,k] with T_i(xi1)·T_j(xi2)·T_k(xi3)
    # Convert each (j,k) slice's coefficients into a function of xi1, evaluate
    T1 = np.array([chebval(xi1, np.eye(1, G, n).ravel()) for n in range(G)])
    T2 = np.array([chebval(xi2, np.eye(1, G, n).ravel()) for n in range(G)])
    T3 = np.array([chebval(xi3, np.eye(1, G, n).ravel()) for n in range(G)])
    return float(np.einsum('ijk,i,j,k->', P_coeffs, T1, T2, T3))

# For a 2D slice at fixed xi1 = ξ_target
def smooth_slice(xi1_val):
    """Evaluate the 3D Chebyshev at fixed ξ_1 = xi1_val, on (xi2, xi3) fine grid."""
    T1 = np.array([chebval(xi1_val, np.eye(1, G, n).ravel()) for n in range(G)])
    # 2D coeffs after reducing axis 0
    coeffs_2d = np.einsum('ijk,i->jk', P_coeffs, T1)
    # Evaluate on fine grid
    # chebval2d wants a flat array but we want a meshgrid
    # Simpler: pre-compute T_n(xi_fine) for each fine point, then einsum
    T_fine_2 = np.zeros((G, N_FINE))
    T_fine_3 = np.zeros((G, N_FINE))
    for n in range(G):
        basis = np.zeros(G); basis[n] = 1
        T_fine_2[n] = chebval(xi_fine, basis)
        T_fine_3[n] = chebval(xi_fine, basis)
    return np.einsum('jk,jm,kn->mn', coeffs_2d, T_fine_2, T_fine_3)

# Three slices: u_1 ≈ -1.1, 0, +1.1 (corresponds to ξ_1 ≈ -0.5, 0, +0.5 at c=2)
slice_xi1_vals = [-0.5, 0.0, 0.5]
slice_u1_vals = [C * np.arctanh(x) for x in slice_xi1_vals]

# Build 2 rows: top = raw imshow (7x7), bottom = smooth Chebyshev (100x100)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
U2g, U3g = np.meshgrid(U_NODES, U_NODES, indexing='ij')
U2f, U3f = np.meshgrid(u_fine, u_fine, indexing='ij')

for col, (xi1_val, u1_val) in enumerate(zip(slice_xi1_vals, slice_u1_vals)):
    # TOP: raw at nearest Lobatto node
    i_nearest = int(np.argmin(np.abs(LOBATTO - xi1_val)))
    raw_slc = P_final[i_nearest, :, :]
    ax = axes[0, col]
    im = ax.imshow(raw_slc.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                     extent=[U_NODES[0], U_NODES[-1], U_NODES[0], U_NODES[-1]], aspect='auto')
    ax.contour(U2g, U3g, raw_slc, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.5)
    ax.set_title(f'RAW: P at Lobatto node u₁={U_NODES[i_nearest]:+.2f}\n(7×7 pixels)', fontsize=11)
    ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # BOTTOM: smooth Chebyshev evaluation at exact ξ_1=xi1_val (mapped to u_1=u1_val)
    smooth = smooth_slice(xi1_val)
    ax = axes[1, col]
    # Clip for display (raw Chebyshev can overshoot slightly outside [0,1])
    smooth_disp = np.clip(smooth, 0, 1)
    im = ax.imshow(smooth_disp.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                     extent=[u_fine[0], u_fine[-1], u_fine[0], u_fine[-1]], aspect='auto')
    ax.contour(U2f, U3f, smooth, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.5)
    overshoot_max = float(max(smooth.max() - 1, 0, -smooth.min(), 0))
    ax.set_title(f'SMOOTH Chebyshev: u₁={u1_val:+.2f}\n(100×100 polynomial eval; overshoot {overshoot_max:.3f})', fontsize=11)
    ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle(f'Chebyshev h=0 FP slices: raw Lobatto pixels (top) vs smooth Chebyshev polynomial (bottom)\n'
              f'N={N}, τ={TAU}, γ={GAMMA}, c={C}', fontsize=12, y=1.00)
plt.tight_layout()
out_path = f'{HERE}/figs/03_slices_smooth.png'
plt.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {out_path}')
