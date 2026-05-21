#!/usr/bin/env python3
"""
Build Fig 3B contour lines from G=18 mp300 converged posterior.

Uses the posterior μ*(u, p) to solve market clearing at each (u₂, u₃)
on a fine grid, then extracts contour lines via matplotlib.
"""

import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import expit as Lam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load converged posterior
with open('results/full_ree/posterior_v3_G18_mp300_notrim.json') as f:
    data = json.load(f)

G = data['G']
tau = data['tau']
gamma = data['gamma']
u_grid = np.array([float(x) for x in data['u_grid']])
p_grids = [[float(x) for x in row] for row in data['p_grid']]
mu_vals = [[float(x) for x in row] for row in data['mu_strings']]

print(f"G={G}, tau={tau}, gamma={gamma}")
print(f"F_max = {data['F_max'][:20]}...e{data['F_max'].split('e')[-1] if 'e' in data['F_max'] else ''}")
print(f"u_grid: [{u_grid[0]:.2f}, ..., {u_grid[-1]:.2f}]")
print()

def logit(p):
    return np.log(p / (1 - p))

def crra_demand(mu, p):
    if mu < 1e-15 or mu > 1-1e-15 or p < 1e-15 or p > 1-1e-15:
        return 0.0
    z = logit(mu) - logit(p)
    R = np.exp(z / gamma)
    return (R - 1) / ((1 - p) + R * p)

# Build interpolator for mu*(u, p)
# Since p_grid varies per u, we need per-u interpolators
mu_interps = []
for i in range(G):
    p_arr = np.array(p_grids[i])
    mu_arr = np.array(mu_vals[i])
    # Ensure monotonicity
    mu_interps.append((p_arr, mu_arr))

def interp_mu(u_val, p_val):
    """Interpolate mu*(u, p) using bilinear on the adaptive grid."""
    # Find bracketing u indices
    if u_val <= u_grid[0]:
        i_lo, i_hi = 0, 1
    elif u_val >= u_grid[-1]:
        i_lo, i_hi = G-2, G-1
    else:
        i_lo = np.searchsorted(u_grid, u_val) - 1
        i_hi = i_lo + 1
    
    # Interpolate in p at each u
    def mu_at_u_idx(idx):
        p_arr, mu_arr = mu_interps[idx]
        if p_val <= p_arr[0]:
            return mu_arr[0]
        if p_val >= p_arr[-1]:
            return mu_arr[-1]
        j = np.searchsorted(p_arr, p_val) - 1
        j = max(0, min(j, len(p_arr)-2))
        frac = (p_val - p_arr[j]) / (p_arr[j+1] - p_arr[j] + 1e-30)
        return mu_arr[j] + frac * (mu_arr[j+1] - mu_arr[j])
    
    mu_lo = mu_at_u_idx(i_lo)
    mu_hi = mu_at_u_idx(i_hi)
    
    # Interpolate in u
    frac_u = (u_val - u_grid[i_lo]) / (u_grid[i_hi] - u_grid[i_lo] + 1e-30)
    frac_u = max(0, min(1, frac_u))
    return mu_lo + frac_u * (mu_hi - mu_lo)

# Fix u₁ = closest grid point to 1.0
u1_idx = np.argmin(np.abs(u_grid - 1.0))
u1 = u_grid[u1_idx]
print(f"u₁ = {u1:.4f} (grid index {u1_idx})")

# Build price surface on fine grid
N_fine = 200
u2_fine = np.linspace(-3.5, 3.5, N_fine)
u3_fine = np.linspace(-3.5, 3.5, N_fine)
P_surface = np.full((N_fine, N_fine), np.nan)

print(f"Computing {N_fine}x{N_fine} price surface...")
n_done = 0
for i2 in range(N_fine):
    for i3 in range(N_fine):
        u2, u3 = u2_fine[i2], u3_fine[i3]
        
        def excess(p):
            mu1 = interp_mu(u1, p)
            mu2 = interp_mu(u2, p)
            mu3 = interp_mu(u3, p)
            return crra_demand(mu1, p) + crra_demand(mu2, p) + crra_demand(mu3, p)
        
        try:
            p = brentq(excess, 0.001, 0.999, xtol=1e-12)
            P_surface[i2, i3] = p
        except:
            pass
    
    n_done += 1
    if n_done % 20 == 0:
        print(f"  {n_done}/{N_fine} rows done")

print("Done.")

# Smooth the price surface to remove interpolation artifacts
from scipy.ndimage import gaussian_filter
P_smooth = P_surface.copy()
# Fill NaN with nearest-neighbor before smoothing
from scipy.interpolate import NearestNDInterpolator
valid = ~np.isnan(P_smooth)
if not np.all(valid):
    coords = np.array(np.where(valid)).T
    values = P_smooth[valid]
    interp_nn = NearestNDInterpolator(coords, values)
    invalid = np.where(~valid)
    P_smooth[invalid] = interp_nn(np.array(invalid).T)

# Light Gaussian smoothing (sigma=2 pixels = 2/200 * 7 = 0.07 in u-space)
P_smooth = gaussian_filter(P_smooth, sigma=2.0)

# Extract contours
levels = [0.2, 0.3, 0.5, 0.7, 0.8]

fig, ax = plt.subplots(figsize=(6,6))
CS = ax.contour(u2_fine, u3_fine, P_smooth.T, levels=levels)

# Save pgfplots coordinates
with open('figures/fig3B_G18_pgfplots.tex', 'w') as f:
    f.write(f"% Fig 3B contours from G={G} mp300 (F_max ~ 1e-131)\n")
    f.write(f"% u1={u1:.4f}, tau={tau}, gamma={gamma}\n\n")
    
    for idx, level in enumerate(levels):
        segs = CS.allsegs[idx]
        if not segs:
            continue
        # Take the longest segment
        longest = max(segs, key=lambda s: len(s))
        verts = np.array(longest)
        
        # Clip to plot region
        mask = (verts[:,0] >= -3.5) & (verts[:,0] <= 3.5) & \
               (verts[:,1] >= -3.5) & (verts[:,1] <= 3.5)
        verts = verts[mask]
        
        # Thin to ~40 points via arc-length resampling
        if len(verts) > 5:
            ds = np.sqrt(np.sum(np.diff(verts, axis=0)**2, axis=1))
            s = np.concatenate([[0], np.cumsum(ds)])
            s_target = np.linspace(0, s[-1], min(40, len(verts)))
            v_thin = np.column_stack([
                np.interp(s_target, s, verts[:,0]),
                np.interp(s_target, s, verts[:,1])
            ])
        else:
            v_thin = verts
        
        coords = "".join([f"({x:.3f},{y:.3f})" for x, y in v_thin])
        f.write(f"% p={level}\n")
        f.write(f"\\addplot coordinates {{{coords}}};\n\n")
        
        print(f"p={level}: {len(verts)} → {len(v_thin)} points")

plt.close()

# Also save a preview
fig, ax = plt.subplots(figsize=(6,6))
colors = ['#004C6D', '#00628A', '#B31C1C', '#D94040', '#E87070']
for level, c in zip(levels, colors):
    CS = ax.contour(u2_fine, u3_fine, P_smooth.T, levels=[level], colors=[c], linewidths=2)
ax.set_xlabel('$u_2$')
ax.set_ylabel('$u_3$')
ax.set_title(f'CRRA contours (G={G}, mp300, $\\gamma$={gamma})')
ax.set_aspect('equal')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
fig.tight_layout()
fig.savefig('figures/fig3B_G18_preview.png', dpi=150)
fig.savefig('figures/fig3B_G18_preview.pdf')
plt.close()

print("\nSaved: figures/fig3B_G18_pgfplots.tex")
print("Saved: figures/fig3B_G18_preview.png")
