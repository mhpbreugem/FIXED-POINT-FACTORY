"""Diagnose whether the strict-h=0 Cheb FP is 'curly' between nodes."""
import sys, os, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cheby_numba import C_STRETCH, _T_basis, vals_to_coeffs_3d_jit
from cheby_numba_kern_tab_N import make_grid_N

def eval_cheb_3d_at(coeffs, xi1, xi2, xi3, n_grid):
    G = n_grid
    T1 = np.empty(G); T2 = np.empty(G); T3 = np.empty(G)
    _T_basis(xi1, G, T1); _T_basis(xi2, G, T2); _T_basis(xi3, G, T3)
    s = 0.0
    for i in range(G):
        for j in range(G):
            for k in range(G):
                s += coeffs[i,j,k] * T1[i]*T2[j]*T3[k]
    return s

FIGS = '/tmp/cheby_h0/figs/curliness'
os.makedirs(FIGS, exist_ok=True)

# Plot Cheb FP at gamma=1 for various N, on a FINE u-grid (much denser than nodes)
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
gamma = 1.0
for col, N in enumerate([6, 8, 10, 12]):
    G, lobatto, u_nodes, V_inv = make_grid_N(N)
    # Load FP (use the strict h=0 FP if available)
    fp_paths = [
        f'/tmp/cheby_h0/fps_high_order/P_FP_N{N}_gamma{gamma}.npy',
        f'/tmp/cheby_h0/fps_gamma/P_FP_gamma{gamma}.npy',   # only N=6
    ]
    P = None
    for path in fp_paths:
        if os.path.exists(path):
            P_try = np.load(path)
            if P_try.shape == (G, G, G):
                P = P_try
                break
    if P is None:
        print(f'No FP at N={N}, gamma={gamma}')
        continue
    coeffs = vals_to_coeffs_3d_jit(P, V_inv)
    # Fine xi grid for evaluation
    xi_fine = np.linspace(-0.9999, 0.9999, 401)
    u_fine = C_STRETCH * np.arctanh(xi_fine)
    # 1D slice at u_2 = u_3 = 0
    xi0_idx = G // 2
    P_fine_1d = np.array([eval_cheb_3d_at(coeffs, xi, 0.0, 0.0, G) for xi in xi_fine])
    P_at_nodes_1d = np.array([P[i, G//2, G//2] for i in range(G)])

    ax = axes[0, col]
    ax.plot(u_fine, P_fine_1d, '-', color='tab:blue', lw=1.5,
              label='Cheb poly (fine eval)')
    ax.plot(u_nodes, P_at_nodes_1d, 'o', color='red', markersize=8,
              label=f'Lobatto nodes (N={N})')
    ax.set_xlim(-10, 10)
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$P(u_1, 0, 0)$')
    ax.set_title(f'N={N}: 1D slice at $u_2=u_3=0$')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2D slice at u_3=0, fine
    NF = 81
    xi_f2 = np.linspace(-0.9999, 0.9999, NF)
    u_f2 = C_STRETCH * np.arctanh(xi_f2)
    P_2d = np.empty((NF, NF))
    for i, x1 in enumerate(xi_f2):
        for j, x2 in enumerate(xi_f2):
            P_2d[i, j] = eval_cheb_3d_at(coeffs, x1, x2, 0.0, G)
    ax = axes[1, col]
    cs = ax.contour(u_f2, u_f2, P_2d.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')
    # Mark the Lobatto nodes
    for un in u_nodes:
        ax.axvline(un, color='gray', alpha=0.2, lw=0.5)
        ax.axhline(un, color='gray', alpha=0.2, lw=0.5)
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(f'N={N}: 2D contours at $u_3=0$')
    ax.grid(alpha=0.3)

plt.suptitle('Cheb FP "curliness" check at $\\gamma=1$: fine-grid evaluation '
              'vs Lobatto-node values\n'
              '(red dots = node values; blue = Cheb poly between nodes; '
              'gray = node positions)',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/cheb_curliness.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved cheb_curliness.png')

# Also: max |P_fine - linear-between-nodes|, as a "curliness" metric
print('\n=== Curliness metric ===')
print('  max |P_cheb(u) - linear-interp(u, node values)| over [-8, 8]')
for N in [6, 8, 10, 12]:
    G, lobatto, u_nodes, V_inv = make_grid_N(N)
    fp_paths = [
        f'/tmp/cheby_h0/fps_high_order/P_FP_N{N}_gamma{gamma}.npy',
        f'/tmp/cheby_h0/fps_gamma/P_FP_gamma{gamma}.npy',
    ]
    P = None
    for path in fp_paths:
        if os.path.exists(path):
            P_try = np.load(path)
            if P_try.shape == (G, G, G):
                P = P_try; break
    if P is None: continue
    coeffs = vals_to_coeffs_3d_jit(P, V_inv)
    # Compare Cheb to linear interp along u_1 axis at center
    u_test = np.linspace(-8, 8, 401)
    xi_test = np.tanh(u_test / C_STRETCH)
    P_cheb = np.array([eval_cheb_3d_at(coeffs, x, 0.0, 0.0, G) for x in xi_test])
    P_at_nodes = np.array([P[i, G//2, G//2] for i in range(G)])
    P_lin = np.interp(u_test, u_nodes, P_at_nodes)
    max_diff = float(np.max(np.abs(P_cheb - P_lin)))
    rms_diff = float(np.sqrt(np.mean((P_cheb - P_lin)**2)))
    # Range
    P_range = float(P_cheb.max() - P_cheb.min())
    print(f'  N={N:>3}: max|cheb-linear|={max_diff:.3e}, '
          f'rms={rms_diff:.3e}, range_of_P={P_range:.3f}')
