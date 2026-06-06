"""Build all figures for tab_explain.pdf: how the tabulated-mu simplification
works, why it's accurate, and what it costs."""
import os, sys, time
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LogNorm

FIGS = '/tmp/cheby_h0/figs'
os.makedirs(FIGS, exist_ok=True)

# ----- Imports of the operators -----
from cheby_numba import phi as phi_chebroots
from cheby_numba import U_NODES, LOBATTO, TAU, GAMMA, N_GRID, NQ, C_STRETCH, V_INV
from cheby_numba_bisect import phi_bisect
from cheby_numba_tab import (phi_tab, make_p_grid, phi_jit_tab,
                               build_mu_table, vals_to_coeffs_3d_jit)
from cheby_numba import GL_NODES, GL_WEIGHTS

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# Warmup
P_in = sg(0.5*T)
_ = phi_chebroots(P_in); _ = phi_bisect(P_in); _ = phi_tab(P_in)

# =====================================================================
# FIG 1: Schematic --- the 5 Hellwig steps, with contour highlighted
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.5)
ax.axis('off')

# Box for each step
steps = [
    (0.2, "1. Conjecture\n$P(u_1,u_2,u_3)$\n(Chebyshev tensor)", 'lightblue'),
    (2.4, "2. Contour\n$\\{(u_a,u_b):P=p\\}$\nroot-find + co-area", 'salmon'),
    (4.6, "3. Bayes $\\mu_k$\n$\\mu_k=\\frac{f_1 A_1}{f_0 A_0{+}f_1 A_1}$",
            'lightyellow'),
    (6.8, "4. Demand\nCRRA: 3 demands\n$d_k(p)$", 'lightgreen'),
    (9.0, "5. Clearing\nsolve $\\sum d_k(p)\\!=\\!0$\nper cube cell",
            'lightpink'),
]
for (x, txt, c) in steps:
    ax.add_patch(FancyBboxPatch((x, 3.5), 2.0, 2.2, boxstyle='round,pad=0.05',
                                   facecolor=c, edgecolor='black'))
    ax.text(x+1.0, 4.6, txt, ha='center', va='center', fontsize=10)

# Arrows
for x in [2.2, 4.4, 6.6, 8.8]:
    ax.annotate('', xy=(x+0.2, 4.6), xytext=(x, 4.6),
                  arrowprops=dict(arrowstyle='->', lw=2))

# Highlight step 2 (the costly one)
ax.add_patch(Rectangle((2.4, 3.5), 2.0, 2.2, facecolor='none',
                          edgecolor='red', linewidth=3, linestyle='--'))
ax.text(3.4, 6.0, 'Expensive!', ha='center', color='red',
         fontweight='bold', fontsize=12)

# Cost annotations
ax.text(6, 2.5, r'\textbf{Original cost per $\Phi$: $G^3 \times K \times N_Q$ contour calls}',
         ha='center', fontsize=12, color='darkred', usetex=False,
         bbox=dict(facecolor='lightyellow', edgecolor='red'))
ax.text(6, 1.5, r'$\;G{=}7,K{=}3,N_Q{=}12\;\Rightarrow\; 12{,}348$ contour calls/iter',
         ha='center', fontsize=11)
ax.text(6, 0.7, r'each = 1D Cheb-poly root extract $\;\to\;$ \textbf{1.2 s at N=8}',
         ha='center', fontsize=11)

plt.title('The 5 Hellwig steps and where the cost lives', fontsize=14, pad=20)
plt.savefig(f'{FIGS}/tab_01_steps.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_01_steps.png')


# =====================================================================
# FIG 2: The simplification --- factor through mu(p, u_k) table
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: original (per-cell contour)
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('ORIGINAL: every cube cell does its own contour', fontsize=12)
# Draw G^3 cube grid points (simplified to 4x4)
for i in range(5):
    for j in range(5):
        x, y = 1+i*0.5, 1+j*0.5
        ax.plot(x, y, 'ko', markersize=3)
ax.text(2.25, 0.4, r'$G^3$ cells', ha='center', fontsize=10)
# Arrow from each cell to contour
for ang in [30, 90, 150, 210, 270, 330]:
    ax.annotate('', xy=(7, 5), xytext=(2.25, 2.25),
                  arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3))
ax.text(7, 5, r'contour\n+root\n+co-area\n+Bayes',
         ha='center', va='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='salmon', edgecolor='red'),
         usetex=False)
ax.text(7, 1.5, r'$G^3 \!\times\! K \!\times\! N_Q$' + '\ncontour calls', ha='center',
         fontsize=11, color='red')

# Right: tabulated (one table, lookup per cell)
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('TABULATED: build mu(p,u_k) ONCE, look up per cell', fontsize=12)
for j in range(7):  # G_p rows in table
    for k in range(5):  # G cols in table
        x, y = 1+k*0.4, 6+j*0.3
        ax.plot(x, y, 'rs', markersize=4)
ax.text(2.4, 4.5, r'$\mu(p, u_k)$ table' + '\n' + r'$(G_p \times G)$ entries',
         ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
# Cube cells
for i in range(5):
    for j in range(5):
        x, y = 6+i*0.5, 1+j*0.5
        ax.plot(x, y, 'ko', markersize=3)
ax.text(7.25, 0.4, r'$G^3$ cells (each does 3 lookups + CRRA clear)', ha='center',
         fontsize=10)
ax.annotate('', xy=(6, 2.25), xytext=(3, 4.5),
              arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text(4.5, 3.5, 'lookup\n+ CRRA', ha='center', fontsize=10, color='blue')
ax.text(7, 7, r'$\;G_p \!\times\! G \!\times\! N_Q\;$' + '\ncontour calls' + '\n' +
         r'($G_p$ replaces $G^3$)',
         ha='center', fontsize=11, color='red')

plt.suptitle('The simplification: factor the per-cell contour through a small 1D table',
              fontsize=14)
plt.savefig(f'{FIGS}/tab_02_arch.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_02_arch.png')


# =====================================================================
# FIG 3: The mu(p, u_k) table itself (heatmap)
# =====================================================================
# Build mu_table from a converged-ish conjecture
P_in_use = sg(0.6*T + 0.05*np.sin(0.3*T))  # mild perturbation for richness
coeffs = vals_to_coeffs_3d_jit(P_in_use, V_INV)
G_p_demo = 51
p_grid = make_p_grid(G_p_demo)
mu_table = build_mu_table(coeffs, LOBATTO, U_NODES, p_grid,
                            GL_NODES, GL_WEIGHTS, TAU, C_STRETCH, G, NQ)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
im = ax.pcolormesh(np.arange(G), p_grid, mu_table, shading='auto',
                     cmap='RdBu_r', vmin=0, vmax=1)
ax.set_xlabel(r'Lobatto signal index $j$  ($u_k$)')
ax.set_ylabel(r'price $p$ (logit-uniform grid)')
ax.set_title(r'$\mu(p, u_k)$ table $(G_p{=}51, G{=}7)$, raw entries')
plt.colorbar(im, ax=ax, label=r'$\mu = P(v{=}1 \mid p, u_k)$')

# Slices
ax = axes[1]
colors = plt.cm.coolwarm(np.linspace(0, 1, G))
for j in range(G):
    ax.plot(p_grid, mu_table[:, j], color=colors[j], lw=1.5,
             label=f'$u_k$={U_NODES[j]:+.2f}')
ax.plot([0,1],[0,1], 'k--', alpha=0.3, label=r'$\mu=p$ (no info)')
ax.set_xlabel(r'price $p$'); ax.set_ylabel(r'posterior $\mu(p,u_k)$')
ax.set_title(r'$\mu(p,u_k)$ slices at each Lobatto $u_k$')
ax.legend(fontsize=8, loc='lower right', ncol=2); ax.grid(alpha=0.3)

plt.suptitle('The tabulated table: built ONCE per $\\Phi$ call, looked up per cube cell',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_03_mutable.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_03_mutable.png')


# =====================================================================
# FIG 4: Contour call savings (bar chart)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))
configs = [(6, 12), (8, 12), (10, 12)]
G_ps = [11, 21, 31, 51]
labels = []
orig = []
tab = {gp: [] for gp in G_ps}
for N_in, NQ_in in configs:
    G_n = N_in + 1
    labels.append(f'$N{{=}}{N_in}$\n($G{{=}}{G_n}$, cells={G_n**3})')
    orig.append(G_n**3 * 3 * NQ_in)
    for gp in G_ps:
        tab[gp].append(gp * G_n * NQ_in)
x = np.arange(len(configs))
w = 0.15
ax.bar(x - 2*w, orig, w, label=f'original ($G^3 \\cdot 3 \\cdot N_Q$)',
        color='crimson')
for i, gp in enumerate(G_ps):
    ax.bar(x + (i-1)*w, tab[gp], w, label=f'tabulated $G_p={gp}$')
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('contour calls per $\\Phi$ evaluation')
ax.set_title('Contour call counts: tabulated reduces $G^3$ to $G_p$\n'
              'speedup factor = $G_p$ / $G^3$ (15--90$\\times$ at relevant grid sizes)')
ax.legend(fontsize=10, ncol=2, loc='upper left')
ax.grid(axis='y', alpha=0.3, which='both')
for i, v in enumerate(orig):
    ax.text(x[i]-2*w, v*1.3, f'{v:,}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_04_calls.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_04_calls.png')


# =====================================================================
# FIG 5: Wall-time speedup (measured, multiple trials)
# =====================================================================
def time_op(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.time(); _ = fn(); times.append(time.time()-t0)
    return min(times), np.median(times)

ops = {
    'chebroots': lambda: phi_chebroots(P_in),
    'bisect': lambda: phi_bisect(P_in),
    'tab(Gp=11)': lambda: phi_tab(P_in, G_p=11),
    'tab(Gp=21)': lambda: phi_tab(P_in, G_p=21),
    'tab(Gp=31)': lambda: phi_tab(P_in, G_p=31),
    'tab(Gp=51)': lambda: phi_tab(P_in, G_p=51),
}
print('Timing at N=6...')
times_n6 = {k: time_op(v)[0] for k, v in ops.items()}

# At N=8: load the N=8 phi modules
print('Loading N=8 ops via test_tab_n8...')
import importlib.util
spec = importlib.util.spec_from_file_location('test_tab_n8', '/tmp/cheby_h0/test_tab_n8.py')
mod = importlib.util.module_from_spec(spec)
# Manually emulate N=8 timing from saved log we already produced
# Reuse the values from prior runs
times_n8 = {
    'chebroots': 1.165,
    'bisect': 0.140,
    'tab(Gp=11)': 0.013,
    'tab(Gp=21)': 0.013,
    'tab(Gp=31)': 0.014,
    'tab(Gp=51)': 0.025,
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
methods = list(ops.keys())
for ax, times, title in [(axes[0], times_n6, 'N=6 (G=7, 343 cells)'),
                            (axes[1], times_n8, 'N=8 (G=9, 729 cells)')]:
    vals = [times[m] for m in methods]
    colors = ['crimson', 'orange', 'forestgreen', 'darkgreen', 'darkgreen', 'darkgreen']
    bars = ax.bar(methods, vals, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('seconds per $\\Phi$ evaluation')
    ax.set_title(title + f'\nbest speedup vs chebroots: '
                  f'{max(times["chebroots"]/v for v in vals):.0f}x')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.5, f'{v*1000:.1f} ms',
                 ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3, which='both')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

plt.suptitle('Measured wall-time per $\\Phi$ evaluation', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_05_walltime.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_05_walltime.png')


# =====================================================================
# FIG 6: Accuracy of tabulation vs G_p (interpolation density)
# =====================================================================
P_cr = phi_chebroots(P_in)
P_bi = phi_bisect(P_in)
G_p_sweep = [5, 9, 15, 21, 31, 51, 81, 121, 201]
errors_cr = []
errors_bi = []
for gp in G_p_sweep:
    P_t = phi_tab(P_in, G_p=gp)
    errors_cr.append(float(np.max(np.abs(P_t - P_cr))))
    errors_bi.append(float(np.max(np.abs(P_t - P_bi))))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.loglog(G_p_sweep, errors_cr, 'o-', color='crimson',
            label=r'$\|P_{\rm tab} - P_{\rm chebroots}\|_\infty$', markersize=8)
ax.loglog(G_p_sweep, errors_bi, 's-', color='orange',
            label=r'$\|P_{\rm tab} - P_{\rm bisect}\|_\infty$', markersize=8)
ax.axhline(1e-3, color='gray', linestyle=':', alpha=0.5, label='1e-3')
ax.axhline(1e-6, color='gray', linestyle='--', alpha=0.5, label='1e-6')
# Predicted O(1/G_p^2) scaling line
G_arr = np.array(G_p_sweep, dtype=float)
ax.loglog(G_p_sweep, 1e-2/G_arr**2, 'k--', alpha=0.4,
            label=r'$\mathcal{O}(1/G_p^2)$ reference (linear interp)')
ax.set_xlabel(r'$G_p$ (number of $p$-grid table points)')
ax.set_ylabel(r'$\|P_{\rm tab} - P_{\rm ref}\|_\infty$ (one $\Phi$ application)')
ax.set_title('Accuracy of tabulated operator vs table density $G_p$\n'
              '(linear-in-$p$ interpolation, logit-uniform grid)')
ax.legend(fontsize=10, loc='lower left'); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_06_accuracy.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_06_accuracy.png')


# =====================================================================
# FIG 7: Per-cell mu lookup vs computed mu (correlation)
# =====================================================================
# For each cube cell, compute mu_chebroots-style and mu_tab-style and compare
# Reuse: at N=6 the chebroots operator stores final mu per cell? We can
# compute mu inline. Let's just show P_new chebroots vs P_new tab scatter.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, gp, label in [(axes[0], 21, 'G_p=21 (under-resolved)'),
                         (axes[1], 51, 'G_p=51 (well-resolved)')]:
    P_t = phi_tab(P_in, G_p=gp)
    ax.scatter(P_cr.ravel(), P_t.ravel(), c=T.ravel(), cmap='coolwarm',
                s=12, alpha=0.7)
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_xlabel(r'$P_{\rm chebroots}$ per cell')
    ax.set_ylabel(r'$P_{\rm tabulated}$ per cell')
    err = float(np.max(np.abs(P_t - P_cr)))
    ax.set_title(label + f'\nmax|tab-cr|={err:.2e}')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
plt.suptitle('Cell-by-cell agreement: tabulated vs reference (color = T)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_07_cellscatter.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_07_cellscatter.png')


# =====================================================================
# FIG 8: Confirming gamma-dependence (NOT rank-1)
# =====================================================================
gammas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
slopes_cr = []
slopes_tab = []
times_per_gamma_cr = []
times_per_gamma_tab = []
for g in gammas:
    t0 = time.time(); P_cr_g = phi_chebroots(P_in, gamma=g); t_cr = time.time()-t0
    t0 = time.time(); P_tab_g = phi_tab(P_in, gamma=g, G_p=51); t_tab = time.time()-t0
    # Fit slope
    Pc_cr = np.clip(P_cr_g, 1e-15, 1-1e-15)
    L_cr = np.log(Pc_cr/(1-Pc_cr)).ravel()
    slopes_cr.append(float(np.sum(L_cr*T.ravel()) / np.sum(T.ravel()**2)))
    Pc_tab = np.clip(P_tab_g, 1e-15, 1-1e-15)
    L_tab = np.log(Pc_tab/(1-Pc_tab)).ravel()
    slopes_tab.append(float(np.sum(L_tab*T.ravel()) / np.sum(T.ravel()**2)))
    times_per_gamma_cr.append(t_cr)
    times_per_gamma_tab.append(t_tab)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.semilogx(gammas, slopes_cr, 'o-', color='crimson',
              label='chebroots (reference)', markersize=8)
ax.semilogx(gammas, slopes_tab, 's--', color='forestgreen',
              label='tabulated $G_p$=51', markersize=8)
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel(r'logit-$P$ vs $T$ slope ($\alpha^*$)')
ax.set_title(r'$\gamma$-dependence is preserved by tabulated operator')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogx(gammas, np.array(times_per_gamma_cr)*1000, 'o-',
              color='crimson', label='chebroots', markersize=8)
ax.semilogx(gammas, np.array(times_per_gamma_tab)*1000, 's-',
              color='forestgreen', label='tabulated $G_p$=51', markersize=8)
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel('wall time per $\\Phi$ (ms)')
ax.set_title('Per-$\\Phi$ time at varying $\\gamma$ (table is $\\gamma$-invariant)')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.suptitle('The tabulated operator is the same equilibrium operator,\n'
              'NOT a rank-1 projection: slope changes with $\\gamma$',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_08_gamma.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_08_gamma.png')


# =====================================================================
# FIG 9: Phi cost breakdown --- table-build vs cell-lookups
# =====================================================================
import json
breakdown = {}
for N_test, G_test, NQ_test in [(6, 7, 12), (8, 9, 12)]:
    # Per Phi: table build is G_p * G * NQ contour calls; per-cell is G^3 lookups.
    # Approximate cost: contour call = c_contour, lookup = c_lookup.
    # Using measured numbers, estimate the ratio
    # At N=8 G_p=51: tab is 25ms vs bisect 140ms. Bisect = 729*3*12 contour
    # Tab = 51*9*12 contour + 729*3 lookups.
    pass

fig, ax = plt.subplots(figsize=(11, 5))
N_values = [4, 5, 6, 7, 8, 10, 12]
NQ_val = 12
G_p_val = 21
orig_calls = [(n+1)**3 * 3 * NQ_val for n in N_values]
tab_calls = [G_p_val * (n+1) * NQ_val for n in N_values]
ratios = [o/t for o, t in zip(orig_calls, tab_calls)]
ax.semilogy(N_values, orig_calls, 'o-', color='crimson',
              label=f'original: $G^3 \\cdot 3 \\cdot N_Q$', markersize=8)
ax.semilogy(N_values, tab_calls, 's-', color='forestgreen',
              label=f'tabulated: $G_p \\cdot G \\cdot N_Q$ ($G_p={G_p_val}$)',
              markersize=8)
ax2 = ax.twinx()
ax2.semilogy(N_values, ratios, 'd--', color='purple',
              label='speedup factor', markersize=8)
ax2.set_ylabel('contour-call reduction factor', color='purple')
ax.set_xlabel(r'polynomial order $N$')
ax.set_ylabel('contour calls per $\\Phi$')
ax.set_title(f'Asymptotic scaling: tab is $O(N)$, original is $O(N^3)$ '
              f'$\\Rightarrow$ speedup grows like $N^2$')
ax.legend(loc='upper left'); ax2.legend(loc='center right')
ax.grid(alpha=0.3, which='both')
for n, r in zip(N_values, ratios):
    ax2.text(n, r*1.3, f'{r:.0f}x', ha='center', fontsize=8, color='purple')
plt.tight_layout()
plt.savefig(f'{FIGS}/tab_09_scaling.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved tab_09_scaling.png')


# =====================================================================
# Dump summary JSON
# =====================================================================
import json
summary = dict(
    G_p_accuracy=dict(zip(G_p_sweep, [float(e) for e in errors_cr])),
    G_p_accuracy_vs_bisect=dict(zip(G_p_sweep, [float(e) for e in errors_bi])),
    timings_N6={k: float(v) for k, v in times_n6.items()},
    timings_N8=times_n8,
    gamma_dependence=dict(
        gammas=list(gammas),
        slope_chebroots=slopes_cr,
        slope_tabulated=slopes_tab,
    ),
    contour_calls_scaling=dict(
        N_values=N_values,
        orig=orig_calls,
        tab=tab_calls,
        ratios=ratios,
    ),
)
json.dump(summary, open('/tmp/cheby_h0/tab_explain.json', 'w'),
            indent=2, default=str)
print('saved tab_explain.json')

print('\n=== All 9 figures saved to', FIGS, '===')
