"""Automatic Richardson order selection via Romberg convergence diagnostic.

For each (p, u_k) table entry, compute mu at orders R1, R2, ..., R_N.
The error of T_n is approximately |T_n - T_{n-1}|. Stop when this drops
below tolerance, OR when |T_n - T_{n-1}| > |T_{n-1} - T_{n-2}| (noise
starts growing -> numerical instability).

Output: per-(p, u_k) "converged order" map + global recommendation.
"""
import sys, time, os, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lin_cdf_richardson import phi_lin_richardson, richardson_weights
from lin_cdf_kern_tab import (make_cdf_uniform_grid, make_p_grid,
                                  build_mu_table_lin_kern, make_gl_for_u)
from cheby_numba import C_STRETCH

# Setup
G = 7
u_grid = make_cdf_uniform_grid(G)
TAU = 1.0; GAMMA = 1.0
p_grid = make_p_grid(121)
gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)

# Use the converged FP from earlier R4 sweep
P_fp = np.load('/tmp/cheby_h0/fps_lin_R4/P_FP_g1.047616e+00.npy')
print(f'Using R4 converged FP at gamma=1; P range [{P_fp.min():.4f}, {P_fp.max():.4f}]')

# Compute mu_h tables at many h values (single evaluation per h)
HS = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])
mu_per_h = []
for h in HS:
    mu_h = build_mu_table_lin_kern(P_fp, u_grid, p_grid, gl_u, gl_du,
                                       TAU, G, 16, float(h))
    mu_per_h.append(mu_h)
mu_per_h = np.array(mu_per_h)  # (n_h, G_p, G)

# Build Richardson tables at each (p, u_k) and at orders 1..n_h
# T[n] = Richardson estimate using first n+1 h values
n_h = len(HS)
T_per_order = np.empty((n_h, 121, G))
for n in range(1, n_h + 1):
    hs_subset = tuple(HS[:n])
    w = richardson_weights(hs_subset)
    # Apply weights to mu_per_h[:n]
    for ip in range(121):
        for k in range(G):
            T_per_order[n-1, ip, k] = float(np.dot(w, mu_per_h[:n, ip, k]))

# For each (p, u_k), find the order at which |T_n - T_{n-1}| stops decreasing
# i.e. find argmin over n of |T_n - T_{n-1}|
errors = np.abs(np.diff(T_per_order, axis=0))  # (n_h-1, G_p, G)
best_order = np.argmin(errors, axis=0) + 2  # +2 since errors[0] = R2 vs R1
# Convergence achievable: minimum error reached
best_error = errors.min(axis=0)

print(f'\nPer-(p, u_k) optimal Richardson order:')
print(f'  Most common order: R{int(np.median(best_order))}')
print(f'  Order distribution:')
for o in range(2, n_h+1):
    print(f'    R{o}: {int(np.sum(best_order == o))} of {121*G}')
print(f'\n  Average minimum |T_n - T_{{n-1}}| (across table): '
      f'{best_error.mean():.2e}')
print(f'  Max minimum error: {best_error.max():.2e}')

# What error if we just use R4 globally?
err_at_R4 = errors[2]   # |T_4 - T_3|
err_at_R5 = errors[3]   # |T_5 - T_4|
err_at_R6 = errors[4]   # |T_6 - T_5|
print(f'\n  Global order error estimates:')
print(f'    R3 vs R2 mean |delta|: {errors[0].mean():.2e}')
print(f'    R4 vs R3 mean |delta|: {errors[1].mean():.2e}')
print(f'    R5 vs R4 mean |delta|: {errors[2].mean():.2e}')
print(f'    R6 vs R5 mean |delta|: {errors[3].mean():.2e}')
print(f'    R7 vs R6 mean |delta|: {errors[4].mean():.2e}')
print(f'    R8 vs R7 mean |delta|: {errors[5].mean():.2e}')

# Plot: for each (p_idx, u_k) show convergence
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax = axes[0, 0]
# 2D heatmap of best order
im = ax.pcolormesh(np.arange(G), p_grid, best_order, shading='auto',
                      cmap='viridis', vmin=2, vmax=n_h)
ax.set_xlabel(r'$u_k$ index'); ax.set_ylabel(r'$p$')
ax.set_title('Optimal Richardson order per $(p, u_k)$ entry')
plt.colorbar(im, ax=ax, label='best order n')

ax = axes[0, 1]
# 2D heatmap of best error
im = ax.pcolormesh(np.arange(G), p_grid, np.log10(np.maximum(best_error, 1e-18)),
                      shading='auto', cmap='RdYlGn_r', vmin=-15, vmax=-3)
ax.set_xlabel(r'$u_k$ index'); ax.set_ylabel(r'$p$')
ax.set_title(r'$\log_{10} \min_n |T_n - T_{n-1}|$  (achievable accuracy)')
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
# Convergence curves for several sample (p, u_k) entries
samples = [(60, 3), (30, 1), (90, 5), (10, 6), (110, 0)]
for ip, k in samples:
    seq = T_per_order[:, ip, k]
    diffs = np.abs(np.diff(seq))
    ax.semilogy(range(2, n_h+1), diffs, 'o-',
                  label=f'(p={p_grid[ip]:.3f}, u_k_idx={k})')
ax.set_xlabel('Richardson order n')
ax.set_ylabel(r'$|T_n - T_{n-1}|$ (error estimate)')
ax.set_title('Convergence of Richardson series at sample $(p, u_k)$ entries')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

ax = axes[1, 1]
# Histogram of best orders
ax.hist(best_order.ravel(), bins=range(2, n_h+2),
          color='tab:blue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Best Richardson order')
ax.set_ylabel('Count of (p, u_k) entries')
ax.set_title('Distribution of optimal Richardson order across table')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Romberg convergence diagnostic for Lin-CDF Richardson at $\\gamma{=}1$',
              fontsize=13)
plt.tight_layout()
os.makedirs('/tmp/cheby_h0/figs/romberg', exist_ok=True)
plt.savefig('/tmp/cheby_h0/figs/romberg/diagnostic.png', dpi=140, bbox_inches='tight')
plt.close()
print('\nsaved diagnostic.png')

# Save summary
np.save('/tmp/cheby_h0/romberg_best_order.npy', best_order)
np.save('/tmp/cheby_h0/romberg_best_error.npy', best_error)
print('saved romberg_*.npy')
