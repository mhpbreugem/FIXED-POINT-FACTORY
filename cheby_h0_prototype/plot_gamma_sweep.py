"""Build all figures for the 20-page gamma-sweep PDF."""
import sys, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

from cheby_numba import U_NODES, TAU, LOBATTO, C_STRETCH, V_INV, N_GRID
from cheby_pou_cr_jit import make_p_grid
from cheby_numba_kern_tab_N import make_grid_N

FIGS = '/tmp/cheby_h0/figs/gamma_sweep'
os.makedirs(FIGS, exist_ok=True)
FPS_DIR = '/tmp/cheby_h0/fps_gamma'

data = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
gammas = sorted([d['gamma'] for d in data.values()])
G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
G_p = 121
p_grid = make_p_grid(G_p)

# Color map for gammas
norm = plt.Normalize(vmin=min(gammas), vmax=max(gammas))
cmap = cm.viridis
def gcolor(g): return cmap(norm(g))

# ===== FIG 1: Slope and deficit vs gamma =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
slopes = [data[f'gamma={g}']['slope'] for g in gammas]
deficits = [data[f'gamma={g}']['deficit'] for g in gammas]
ax = axes[0]
ax.semilogx(gammas, slopes, 'o-', markersize=10, color='tab:blue')
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel(r'slope $\alpha^*$ (logit-$P$ vs $T$)')
ax.set_title(r'Revealed information slope $\alpha^*$ vs $\gamma$')
ax.grid(alpha=0.3, which='both')
ax = axes[1]
ax.semilogx(gammas, deficits, 's-', markersize=10, color='tab:red')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'deficit $1-R^2$')
ax.set_title(r'Information deficit $1-R^2$ vs $\gamma$')
ax.grid(alpha=0.3, which='both')
plt.suptitle(r'K=3 CRRA REE: $(\alpha^*, 1{-}R^2)$ vs $\gamma$ at $\tau=1$, $G=7$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/01_slope_deficit.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 01_slope_deficit.png')

# ===== FIG 2: Convergence trajectories =====
fig, ax = plt.subplots(figsize=(11, 6))
for g in gammas:
    Fs = data[f'gamma={g}']['anderson_history']
    ax.semilogy(range(1, len(Fs)+1), np.maximum(Fs, 1e-18), 'o-',
                  markersize=4, color=gcolor(g), label=f'$\\gamma={g}$')
ax.axhline(1e-15, color='black', linestyle=':', label='machine $\\varepsilon$')
ax.set_xlabel('Anderson iteration')
ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Anderson convergence at each $\\gamma$ (POU+chebroots, NQ auto-selected)')
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_convergence.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 02_convergence.png')

# ===== FIG 3: Final residuals per gamma =====
fig, ax = plt.subplots(figsize=(10, 5))
fs = [data[f'gamma={g}']['F_final'] for g in gammas]
nqs = [data[f'gamma={g}']['NQ'] for g in gammas]
bars = ax.bar(range(len(gammas)),
                np.maximum(fs, 1e-18),
                color=[gcolor(g) for g in gammas])
for i, (b, f, nq) in enumerate(zip(bars, fs, nqs)):
    ax.text(i, max(f, 1e-18)*1.5, f'NQ={nq}\nF={f:.1e}',
             ha='center', fontsize=8)
ax.set_yscale('log')
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'{g}' for g in gammas])
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\|F\|_\infty$ at best NQ')
ax.set_title('Convergence floor per $\\gamma$ (auto-selected NQ)')
ax.axhline(1e-15, color='black', linestyle=':')
ax.grid(axis='y', which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/03_floors_per_gamma.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 03_floors_per_gamma.png')

# ===== FIGS 4-12: contour plots at fixed u_3, for each gamma =====
# Use ~9 gammas, one per page; each shows {P=p} for several p values
def contour_at_u3(P, u3_value, p_levels):
    """Compute contour lines {P(u1,u2,u3_value) = p} for various p.
    Returns dict {p: list of (u1, u2) arrays}."""
    from cheby_numba import _T_basis, chebval_jit
    G = P.shape[0]
    # Need slice2d(u1, u2) at u_3 = u3_value
    coeffs = np.zeros((G, G, G))
    # Convert P → coeffs via V_INV (along all 3 axes)
    from cheby_numba import vals_to_coeffs_3d_jit
    coeffs = vals_to_coeffs_3d_jit(P, V_INV)
    # 1D Cheb evaluation along u_3 at u3_value
    xi3 = float(np.tanh(u3_value / C_STRETCH))
    Tk = np.empty(G); _T_basis(xi3, G, Tk)
    slice2 = np.einsum('ijk,k->ij', coeffs, Tk)  # in Cheb basis along axes 0,1
    # Now slice2 is Cheb 2D in (xi_1, xi_2). Evaluate on a fine grid.
    NF = 121
    xi_f = np.linspace(-0.9999, 0.9999, NF)
    u_f = C_STRETCH * np.arctanh(xi_f)
    # 2D evaluation
    Ti_f = np.empty((NF, G)); Tj_f = np.empty((NF, G))
    for q in range(NF):
        _T_basis(xi_f[q], G, Ti_f[q])
        _T_basis(xi_f[q], G, Tj_f[q])
    # slice2_fine[i, j] = sum_{m,n} slice2[m, n] T_m(xi_f[i]) T_n(xi_f[j])
    slice2_fine = Ti_f @ slice2 @ Tj_f.T
    return u_f, slice2_fine


for g in gammas:
    P = np.load(f'{FPS_DIR}/P_FP_gamma{g}.npy')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, u3v, title in zip(axes,
                                 [U_NODES[0]*0.9, 0.0, U_NODES[-1]*0.9],
                                 [f'$u_3 \\approx -8.9$ (low signal 3)',
                                  f'$u_3 = 0$ (neutral)',
                                  f'$u_3 \\approx +8.9$ (high signal 3)']):
        u_f, slice2 = contour_at_u3(P, u3v, None)
        cs = ax.contour(u_f, u_f, slice2.T, levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                          cmap='RdBu_r', linewidths=1.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    plt.suptitle(f'Price contours $\\{{P(u_1, u_2, u_3) = p\\}}$ at $\\gamma={g}$, $\\tau=1$\n'
                  f'slope $\\alpha^*={data[f"gamma={g}"]["slope"]:.4f}$, '
                  f'deficit$={data[f"gamma={g}"]["deficit"]:.4f}$',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGS}/04_contours_gamma{g}.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved 04_contours_gamma{g}.png')

# ===== FIGS for mu(p, u_k) heatmaps =====
for g in gammas:
    mu = np.load(f'{FPS_DIR}/mu_table_gamma{g}.npy')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    im = ax.pcolormesh(np.arange(G), p_grid, mu, cmap='RdBu_r',
                          vmin=0, vmax=1, shading='auto')
    ax.set_xlabel(r'$u_k$ index')
    ax.set_ylabel(r'price $p$')
    ax.set_title(f'$\\mu(p, u_k)$ table at $\\gamma={g}$')
    plt.colorbar(im, ax=ax)
    ax = axes[1]
    colors2 = cm.coolwarm(np.linspace(0, 1, G))
    for j in range(G):
        ax.plot(p_grid, mu[:, j], color=colors2[j],
                  label=f'$u_k$={U_NODES[j]:+.1f}', lw=1.5)
    ax.plot([0,1],[0,1], 'k--', alpha=0.3, label='$\\mu=p$')
    ax.set_xlabel(r'$p$'); ax.set_ylabel(r'$\mu(p, u_k)$')
    ax.set_title(r'$\mu$ vs $p$ slices')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.suptitle(f'Posterior table at $\\gamma={g}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGS}/05_mu_gamma{g}.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved 05_mu_gamma{g}.png')

# ===== logit-T regression scatter at each gamma =====
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, g in zip(axes.flat, gammas):
    P = np.load(f'{FPS_DIR}/P_FP_gamma{g}.npy')
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = data[f'gamma={g}']['slope']
    ax.scatter(T.ravel(), L, c=Pc.ravel(), cmap='RdBu_r', s=4, alpha=0.6)
    ax.plot([T.min(), T.max()], [s*T.min(), s*T.max()], 'k--', lw=1.5,
              label=f'$\\alpha^*={s:.4f}$')
    ax.set_xlabel(r'$T$', fontsize=9); ax.set_ylabel(r'logit$P$', fontsize=9)
    ax.set_title(f'$\\gamma={g}$, $1{{-}}R^2={data[f"gamma={g}"]["deficit"]:.3f}$',
                   fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('logit$P$ vs $T$ scatter at each $\\gamma$\n'
              '(slope $\\alpha^*$ and information deficit $1{-}R^2$)',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_logit_scatter.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved 06_logit_scatter.png')

# ===== 3D slice visualizations =====
for g in gammas:
    P = np.load(f'{FPS_DIR}/P_FP_gamma{g}.npy')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mid, title in zip(axes,
                                  [0, G//2, G-1],
                                  [f'$u_3 = {U_NODES[0]:+.1f}$',
                                   f'$u_3 = 0$',
                                   f'$u_3 = {U_NODES[-1]:+.1f}$']):
        im = ax.pcolormesh(U_NODES, U_NODES, P[:, :, mid],
                              vmin=0, vmax=1, cmap='RdBu_r', shading='auto')
        ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.suptitle(f'$P(u_1, u_2, u_3)$ slices at $\\gamma={g}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGS}/07_slices_gamma{g}.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'saved 07_slices_gamma{g}.png')

print('\nAll figures saved.')
