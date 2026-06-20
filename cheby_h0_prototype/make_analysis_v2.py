"""Generate analysis figures + LaTeX PDF for the upgraded Chebyshev run.

This is the v2 analysis (after numba JIT + sigmoid lift + smooth plotting).
Compares unlifted (raw S₃×Z₂ Newton) vs lifted (σ(αT+h) + S₃×Z₂ Newton) at τ=1, γ=1, N=6.
"""
import os, json, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

HERE = '/tmp/cheby_h0'
FIGS = f'{HERE}/figs'
os.makedirs(FIGS, exist_ok=True)

# Load results
res_unlifted = json.load(open(f'{HERE}/results_sym2.json'))
res_lifted = json.load(open(f'{HERE}/results_lifted_numba.json'))
P_unlifted = np.load(f'{HERE}/P_final_sym2.npy')
P_lifted = np.load(f'{HERE}/P_final_lifted_numba.npy')

cfg = res_unlifted['config']
N = cfg['N']; G = N + 1
TAU = cfg['tau']; GAMMA = cfg['gamma']; C = cfg['c']
LOBATTO = np.array(cfg['lobatto'])
U_NODES = np.array(cfg['u_nodes'])


# ===== Fig A: convergence comparison =====
fig, ax = plt.subplots(figsize=(11, 6))
# Unlifted
u_p = res_unlifted['picard']['ferrs']; u_n = res_unlifted['newton']['ferrs']
n_p = len(u_p); n_n = len(u_n)
ax.semilogy(range(1, n_p+1), u_p, 'o--', color='tab:blue', alpha=0.6, markersize=8,
              label=f'Unlifted Picard (n={n_p})')
ax.semilogy(range(n_p, n_p+n_n), u_n, 's--', color='tab:red', alpha=0.6, markersize=8,
              label=f'Unlifted Newton (n={n_n})')
# Lifted
l_p = res_lifted['picard']['ferrs']; l_n = res_lifted['newton']['ferrs']
n_lp = len(l_p); n_ln = len(l_n)
ax.semilogy(range(1, n_lp+1), l_p, 'o-', color='tab:blue', markersize=10,
              label=f'Lifted Picard (n={n_lp})')
ax.semilogy(range(n_lp, n_lp+n_ln), l_n, 's-', color='tab:red', markersize=10,
              label=f'Lifted Newton (n={n_ln})')
ax.axhline(1e-9, color='black', linestyle=':', alpha=0.5, label='target tol 1e-9')
ax.set_xlabel('iteration'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title(f'Convergence: unlifted (P-cells, 40 DOF) vs lifted (α,h, 41 DOF)\n'
              f'τ={TAU}, γ={GAMMA}, N={N}, numba JIT (~10× speedup)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/A_convergence_v2.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig B: metrics comparison =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
all_u = res_unlifted['picard']['metrics'] + res_unlifted['newton']['metrics']
all_l = res_lifted['picard']['metrics'] + res_lifted['newton']['metrics']
keys = ['slope_T', 'deficit', 'd_FR']
titles = ['slope (logit P vs T)', 'deficit (1 − R²)', 'd_FR (RMS dist from FR)']
for ax, k, ttl in zip(axes, keys, titles):
    u_p = [m[k] for m in res_unlifted['picard']['metrics']]
    u_n = [m[k] for m in res_unlifted['newton']['metrics']]
    l_p = [m[k] for m in res_lifted['picard']['metrics']]
    l_n = [m[k] for m in res_lifted['newton']['metrics']]
    ax.plot(range(1, len(u_p)+1), u_p, 'o--', color='tab:blue', alpha=0.5)
    ax.plot(range(len(u_p), len(u_p)+len(u_n)), u_n, 's--', color='tab:red', alpha=0.5, label='unlifted')
    ax.plot(range(1, len(l_p)+1), l_p, 'o-', color='tab:blue')
    ax.plot(range(len(l_p), len(l_p)+len(l_n)), l_n, 's-', color='tab:red', label='lifted')
    ax.set_xlabel('iteration'); ax.set_ylabel(k)
    ax.set_title(ttl)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.suptitle(f'Convergence of FP metrics: unlifted vs lifted (τ={TAU}, γ={GAMMA}, N={N})')
plt.tight_layout()
plt.savefig(f'{FIGS}/B_metrics_v2.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig C: smooth Chebyshev P-slices for the lifted solution =====
P_lifted_coeffs = P_lifted.copy()
for ax_idx in range(3):
    coeffs_new = np.empty_like(P_lifted_coeffs)
    for i in range(G):
        for j in range(G):
            if ax_idx == 0: idx = (slice(None), i, j)
            elif ax_idx == 1: idx = (i, slice(None), j)
            else: idx = (i, j, slice(None))
            coeffs_new[idx] = chebfit(LOBATTO, P_lifted_coeffs[idx], N)
    P_lifted_coeffs = coeffs_new

N_FINE = 100
xi_fine = np.linspace(-0.99, 0.99, N_FINE)
u_fine = C * np.arctanh(xi_fine)

T_fine_2 = np.zeros((G, N_FINE)); T_fine_3 = np.zeros((G, N_FINE))
for n in range(G):
    basis = np.zeros(G); basis[n] = 1
    T_fine_2[n] = chebval(xi_fine, basis); T_fine_3[n] = chebval(xi_fine, basis)

def smooth_slice(coeffs, xi1_val):
    T1 = np.array([chebval(xi1_val, np.eye(1, G, n).ravel()) for n in range(G)])
    coeffs_2d = np.einsum('ijk,i->jk', coeffs, T1)
    return np.einsum('jk,jm,kn->mn', coeffs_2d, T_fine_2, T_fine_3)

slice_xi1_vals = [-0.5, 0.0, 0.5]
slice_u1_vals = [C*np.arctanh(x) for x in slice_xi1_vals]
U2f, U3f = np.meshgrid(u_fine, u_fine, indexing='ij')

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for col, (xi1_val, u1_val) in enumerate(zip(slice_xi1_vals, slice_u1_vals)):
    ax = axes[col]
    smooth = smooth_slice(P_lifted_coeffs, xi1_val)
    smooth_disp = np.clip(smooth, 0, 1)
    im = ax.imshow(smooth_disp.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                     extent=[u_fine[0], u_fine[-1], u_fine[0], u_fine[-1]], aspect='auto')
    ax.contour(U2f, U3f, smooth, levels=[0.25, 0.5, 0.75], colors='yellow', linewidths=1.8)
    overshoot = float(max(smooth.max() - 1, 0, -smooth.min(), 0))
    ax.set_title(f'u₁={u1_val:+.2f} (ξ₁={xi1_val:+.2f})\nsmooth Chebyshev 100×100  overshoot {overshoot:.3f}', fontsize=10)
    ax.set_xlabel('u₂'); ax.set_ylabel('u₃')
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.suptitle(f'Lifted P-slices (smooth Chebyshev) — τ={TAU}, γ={GAMMA}, N={N}', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIGS}/C_slices_lifted_smooth.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig D: spectral decay comparison =====
def coeffs_3d(P_vals):
    out = P_vals.copy()
    for ax_idx in range(3):
        new = np.empty_like(out)
        for i in range(G):
            for j in range(G):
                if ax_idx == 0: idx = (slice(None), i, j)
                elif ax_idx == 1: idx = (i, slice(None), j)
                else: idx = (i, j, slice(None))
                new[idx] = chebfit(LOBATTO, out[idx], N)
        out = new
    return out

P_u_c = coeffs_3d(P_unlifted)
P_l_c = coeffs_3d(P_lifted)

def decay(C3):
    mag = np.abs(C3)
    orders = []; maxmags = []
    for tot in range(3*N+1):
        m = []
        for i in range(G):
            for j in range(G):
                for k in range(G):
                    if i+j+k == tot: m.append(mag[i,j,k])
        if m:
            orders.append(tot); maxmags.append(max(m))
    return orders, maxmags

o_u, m_u = decay(P_u_c)
o_l, m_l = decay(P_l_c)

fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(o_u, m_u, 'o-', color='tab:blue', lw=2, markersize=10, label='Unlifted (P cells)')
ax.semilogy(o_l, m_l, 's-', color='tab:red', lw=2, markersize=10, label='Lifted (σ(αT+h))')
ax.axhline(1e-15, color='black', linestyle=':', alpha=0.5, label='machine ε')
ax.set_xlabel('total Chebyshev order n = i+j+k')
ax.set_ylabel('max |a_{ijk}|')
ax.set_title('Spectral coefficient decay (smaller = better resolved)')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/D_spectral_decay_v2.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig E: per-call timing (numba speedup) =====
# Hardcoded from numba_test.log measurements
labels = ['Pure Python\nchebroots+\nchebval/numpy', 'Numba JIT\n(inlined)']
times = [3.1, 0.27]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, times, color=['tab:blue', 'tab:red'], width=0.5)
for b, t in zip(bars, times):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02, f'{t:.2f}s', ha='center', fontsize=11)
ax.set_ylabel('per-Φ call (s)')
ax.set_title(f'Numba speedup per Φ evaluation (N={N}, G³={G**3} cells)\n{times[0]/times[1]:.1f}× faster ⇒ Newton iter dropped from ~130s to ~13s')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{FIGS}/E_numba_speedup.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== Fig F: comparison with prior session =====
fig, ax = plt.subplots(figsize=(11, 6))
g_prior = [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.225, 0.259]
slope_prior = [0.3599, 0.3601, 0.3603, 0.3611, 0.3626, 0.3641, 0.3668, 0.3708, 0.3728]
ax.semilogx(g_prior, slope_prior, 'go-', lw=1.5, alpha=0.5, label='Prior: hfree strict h=0 spline, τ=2 (PR branch)')
# Unlifted Chebyshev
ax.semilogx([GAMMA], [res_unlifted['final_metrics']['slope_T']], 'b*', markersize=22,
              label=f'UNLIFTED Chebyshev: τ={TAU}, γ={GAMMA}, N={N} → slope={res_unlifted["final_metrics"]["slope_T"]:.4f}')
# Lifted Chebyshev
ax.semilogx([GAMMA], [res_lifted['final_metrics']['slope_T']], 'r*', markersize=22,
              label=f'LIFTED Chebyshev: τ={TAU}, γ={GAMMA}, N={N} → slope={res_lifted["final_metrics"]["slope_T"]:.4f}')
ax.semilogx([1.0], [0.2680], 'b*', markersize=14, alpha=0.4, label='Earlier kernel demo τ=1,γ=1: 0.268')
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='FR (slope=1)')
ax.set_xlabel('γ'); ax.set_ylabel('slope$_T$')
ax.set_title('Chebyshev h=0 results vs prior PR branch')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/F_comparison_v2.png', dpi=140, bbox_inches='tight')
plt.close()

print('Generated v2 analysis figures:')
print('  A_convergence_v2.png   — Picard+Newton residual norm, unlifted vs lifted')
print('  B_metrics_v2.png       — slope/deficit/d_FR evolution')
print('  C_slices_lifted_smooth.png — smooth Chebyshev evaluation of P at τ=1,γ=1')
print('  D_spectral_decay_v2.png— coeff decay, unlifted vs lifted')
print('  E_numba_speedup.png    — per-Φ timing')
print('  F_comparison_v2.png    — slope vs γ comparison with prior session')
