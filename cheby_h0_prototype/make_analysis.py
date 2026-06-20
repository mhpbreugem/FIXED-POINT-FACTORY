"""Generate analysis figures + LaTeX PDF report for the Chebyshev h=0 run."""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit

HERE = '/tmp/cheby_h0'
FIGS = f'{HERE}/figs'
os.makedirs(FIGS, exist_ok=True)

results = json.load(open(f'{HERE}/results_sym2.json'))
P_final = np.load(f'{HERE}/P_final_sym2.npy')
P_IC = np.load(f'{HERE}/P_IC.npy') if os.path.exists(f'{HERE}/P_IC.npy') else None
P_after_one_phi = np.load(f'{HERE}/P_after_one_phi.npy') if os.path.exists(f'{HERE}/P_after_one_phi.npy') else None

cfg = results['config']
N = cfg['N']; G = N + 1
TAU = cfg['tau']; GAMMA = cfg['gamma']; C = cfg['c']
LOBATTO = np.array(cfg['lobatto'])
U_NODES = np.array(cfg['u_nodes'])

# ===== Fig 1: convergence trajectory =====
fig, ax = plt.subplots(figsize=(10, 5.5))
picard_ferrs = results['picard']['ferrs']
newton_ferrs = results['newton']['ferrs']
total_iters = list(range(len(picard_ferrs))) + list(range(len(picard_ferrs)-1, len(picard_ferrs)-1+len(newton_ferrs)))
all_ferrs = picard_ferrs + newton_ferrs
ax.semilogy(range(1, len(picard_ferrs)+1), picard_ferrs, 'o-', color='tab:blue', lw=2, markersize=10, label='Damped Picard (ω=0.5)')
nx = list(range(len(picard_ferrs), len(picard_ferrs)+len(newton_ferrs)))
ax.semilogy(nx, newton_ferrs, 's-', color='tab:red', lw=2, markersize=10, label='Pure dense Newton')
ax.axvline(len(picard_ferrs), color='gray', linestyle=':', alpha=0.5)
ax.text(len(picard_ferrs)+0.1, 1e-4, 'Newton starts', fontsize=10, alpha=0.6)
ax.axhline(1e-9, color='black', linestyle=':', alpha=0.5, label='target tol 1e-9')
ax.set_xlabel('iteration'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title(f'Chebyshev h=0 convergence at N={N} (G={G}, {G**3} cells)\nτ={TAU}, γ={GAMMA}, atanh c={C}')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/01_convergence.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 2: metrics evolution =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
all_metrics = results['picard']['metrics'] + results['newton']['metrics']
keys = ['slope_T', 'deficit', 'd_FR']
for ax, k in zip(axes, keys):
    p_vals = [m[k] for m in results['picard']['metrics']]
    n_vals = [m[k] for m in results['newton']['metrics']]
    ax.plot(range(1, len(p_vals)+1), p_vals, 'o-', color='tab:blue', label='Picard')
    nx = list(range(len(p_vals), len(p_vals)+len(n_vals)))
    ax.plot(nx, n_vals, 's-', color='tab:red', label='Newton')
    ax.set_xlabel('iteration'); ax.set_ylabel(k)
    ax.set_title(f'{k} per iteration')
    ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS}/02_metrics.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 3: final P slices =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
i_center = G // 2
for col, i_slice in enumerate([1, i_center, G-2]):
    ax = axes[col]
    slc = P_final[i_slice, :, :]
    im = ax.imshow(slc.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                     extent=[U_NODES[0], U_NODES[-1], U_NODES[0], U_NODES[-1]], aspect='auto')
    U2g, U3g = np.meshgrid(U_NODES, U_NODES, indexing='ij')
    try:
        ax.contour(U2g, U3g, slc, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.5)
    except Exception: pass
    ax.set_title(f'P(u₂, u₃) at u₁={U_NODES[i_slice]:+.2f}', fontsize=11)
    ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle(f'Chebyshev h=0 FP slices (N={N}, τ={TAU}, γ={GAMMA})')
plt.tight_layout()
plt.savefig(f'{FIGS}/03_slices.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 4: spectral coefficient decay =====
# Compute Chebyshev coefficients via chebfit per axis
P_coeffs = P_final.copy()
for ax_idx in range(3):
    coeffs_new = np.empty_like(P_coeffs)
    for i in range(G):
        for j in range(G):
            slicer = [slice(None)]*3
            if ax_idx == 0: slicer[1] = i; slicer[2] = j; idx = (slice(None), i, j)
            elif ax_idx == 1: slicer[0] = i; slicer[2] = j; idx = (i, slice(None), j)
            else: slicer[0] = i; slicer[1] = j; idx = (i, j, slice(None))
            try:
                c = chebfit(LOBATTO, P_coeffs[idx], N)
            except Exception:
                c = P_coeffs[idx]
            coeffs_new[idx] = c
    P_coeffs = coeffs_new

fig, ax = plt.subplots(figsize=(10, 5.5))
mag = np.abs(P_coeffs)
mag_by_order = []
order_max = N
for total_order in range(0, 3*N+1):
    mags_here = []
    for i in range(G):
        for j in range(G):
            for k in range(G):
                if i+j+k == total_order:
                    mags_here.append(mag[i,j,k])
    if mags_here:
        mag_by_order.append((total_order, max(mags_here)))
orders = [m[0] for m in mag_by_order]
maxmags = [m[1] for m in mag_by_order]
ax.semilogy(orders, maxmags, 'o-', lw=2, markersize=10, color='tab:blue', label=f'max |a_{{ijk}}| at total order n=i+j+k')
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5, label='machine precision')
# Reference exponential decay line
if len(maxmags) >= 3 and maxmags[2] > 1e-10:
    rate_est = np.log(maxmags[0]/maxmags[2])/2 if maxmags[2] > 0 else 1
    if rate_est > 0:
        ref = [maxmags[0]*np.exp(-rate_est*o) for o in orders]
        ax.semilogy(orders, ref, 'k--', alpha=0.4, label=f'reference: exp(-{rate_est:.2f}·n)')
ax.set_xlabel('total Chebyshev order n = i+j+k')
ax.set_ylabel('|a_{ijk}| (max over indices with same total order)')
ax.set_title('Spectral coefficient decay (diagnostic of smoothness)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/04_spectral_decay.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 5: logit(P) vs T regression =====
fig, ax = plt.subplots(figsize=(10, 5.5))
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU * (U1 + U2 + U3)
Pc = np.clip(P_final, 1e-12, 1-1e-12)
y = np.log(Pc/(1-Pc)).ravel()
a = np.polyfit(T.ravel(), y, 1)
Trange = np.linspace(T.min(), T.max(), 100)
pred = a[0]*Trange + a[1]
ax.scatter(T.ravel(), y, s=20, c='tab:blue', alpha=0.5, label=f'{G**3} cube cells')
ax.plot(Trange, pred, 'k-', lw=2, label=f'fit: slope={a[0]:.4f}')
ax.plot(Trange, Trange, 'k--', lw=1, alpha=0.5, label='FR slope=1')
ax.set_xlabel(r'$T = \tau \sum u_k$'); ax.set_ylabel('logit(P)')
ax.set_title('Logit(P) vs T regression — slope < 1 ⟹ PR')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/05_logit_regression.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig 6: comparison to prior session =====
fig, ax = plt.subplots(figsize=(10, 5.5))
# Prior γ-sweep deep PR branch (τ=2, hfree strict h=0)
g_prior = [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.225, 0.259]
slope_prior = [0.3599, 0.3601, 0.3603, 0.3611, 0.3626, 0.3641, 0.3668, 0.3708, 0.3728]
ax.semilogx(g_prior, slope_prior, 'go-', lw=1.5, alpha=0.6, label='Prior: hfree strict h=0, τ=2, G=9 (PR branch)')
# This run
this_slope = results['final_metrics']['slope_T']
ax.semilogx([GAMMA], [this_slope], 'r*', markersize=22, label=f'THIS run: Chebyshev h=0, τ={TAU}, N={N} → slope={this_slope:.4f}')
# Demo kernel run for context
ax.semilogx([1.0], [0.2680], 'b*', markersize=18, alpha=0.6, label='Earlier demo: kernel co-area τ=1, γ=1 → slope=0.268')
ax.semilogx([100.0], [0.3594], 'm*', markersize=18, alpha=0.6, label='Earlier demo: kernel CARA τ=1, γ=100 → slope=0.359')
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='FR (slope=1)')
ax.set_xlabel('γ'); ax.set_ylabel('slope_T')
ax.set_title('Chebyshev h=0 result in context')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/06_comparison.png', dpi=140, bbox_inches='tight')
plt.close()

print('All 6 analysis figures generated.')
