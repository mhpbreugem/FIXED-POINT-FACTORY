"""Build figures for the 3-way comparison:
  A. Cheb tab            (Lobatto nodes + atanh GL + Cheb interp)
  B. Linear-on-Lobatto   (Lobatto nodes + atanh GL + linear interp)  -- isolates interp
  C. Linear uniform-u    (uniform-u nodes + uniform-u GL + linear interp) -- pure CDF-grid
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cheby_numba import (TAU, GAMMA, NQ, U_NODES as U_LOB, N_GRID as G_LOB)
from cheby_numba_tab import phi_tab
from cheby_linear_op import phi_linear, make_uniform_u_grid
from cheby_linterp_on_lob import phi_linterp_lob

def sg(x): return 1/(1+np.exp(-x))

FIGS = '/tmp/cheby_h0/figs'

# Common warmup
U1L, U2L, U3L = np.meshgrid(U_LOB, U_LOB, U_LOB, indexing='ij')
T_LOB = TAU*(U1L+U2L+U3L)
P0_LOB = sg(0.5*T_LOB)
_ = phi_tab(P0_LOB); _ = phi_linterp_lob(P0_LOB)
G_LIN_DEFAULT = 21
U_MAX_DEFAULT = 6.0
u_lin = make_uniform_u_grid(G_LIN_DEFAULT, U_MAX_DEFAULT)
U1U, U2U, U3U = np.meshgrid(u_lin, u_lin, u_lin, indexing='ij')
T_UNI = TAU*(U1U+U2U+U3U)
P0_UNI = sg(0.5*T_UNI)
_ = phi_linear(P0_UNI, u_lin, G_p=51)

# ========== Anderson ==========
def anderson(op, x0, n_iter=80, m=8):
    x = x0.copy().reshape(-1)
    Xh, Gh = [], []; Fs = []
    sh = x0.shape
    for it in range(n_iter):
        gx = op(x.reshape(sh)).reshape(-1)
        F = gx - x; Fs.append(float(np.max(np.abs(F))))
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > m: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    return Fs

# ========== Run all three ==========
print('Running 3 operators...')
F_A = anderson(lambda P: phi_tab(P, G_p=51), P0_LOB)
print(f'  A. Cheb tab:           floor {min(F_A):.3e}')
F_B = anderson(lambda P: phi_linterp_lob(P, G_p=51), P0_LOB)
print(f'  B. Linear-on-Lobatto:  floor {min(F_B):.3e}')
F_C = anderson(lambda P: phi_linear(P, u_lin, G_p=51), P0_UNI)
print(f'  C. Linear uniform-u:   floor {min(F_C):.3e}')

# Wall times (median of 5)
print('\nWall times (median of 5):')
def med_time(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.time(); fn(); ts.append(time.time()-t0)
    return float(np.median(ts))
t_A = med_time(lambda: phi_tab(P0_LOB, G_p=51))
t_B = med_time(lambda: phi_linterp_lob(P0_LOB, G_p=51))
t_C = med_time(lambda: phi_linear(P0_UNI, u_lin, G_p=51))
print(f'  A. Cheb tab:           {t_A*1000:.1f} ms')
print(f'  B. Linear-on-Lobatto:  {t_B*1000:.1f} ms')
print(f'  C. Linear uniform-u:   {t_C*1000:.1f} ms')

# gamma dependence
gammas = [0.3, 0.5, 1.0, 2.0, 5.0]
slope_data = {'A': [], 'B': [], 'C': []}
def slope(P_out, T):
    Lc = np.log(np.clip(P_out, 1e-15, 1-1e-15)/(1-np.clip(P_out, 1e-15, 1-1e-15))).ravel()
    return float(np.sum(Lc*T.ravel())/np.sum(T.ravel()**2))
print('\ngamma sweep:')
for g in gammas:
    P_A = P0_LOB.copy();
    for _ in range(80): P_A = phi_tab(P_A, gamma=g, G_p=51)
    P_B = P0_LOB.copy();
    for _ in range(80): P_B = phi_linterp_lob(P_B, gamma=g, G_p=51)
    P_C = P0_UNI.copy();
    for _ in range(80): P_C = phi_linear(P_C, u_lin, gamma=g, G_p=51)
    slope_data['A'].append(slope(P_A, T_LOB))
    slope_data['B'].append(slope(P_B, T_LOB))
    slope_data['C'].append(slope(P_C, T_UNI))
    print(f'  gamma={g}: A={slope_data["A"][-1]:.4f}, B={slope_data["B"][-1]:.4f}, '
          f'C={slope_data["C"][-1]:.4f}')

# Accuracy at single Phi (A vs B vs C)
print('\nSingle-Phi: ||C(B,C) - C(A)||_inf (at Lobatto nodes):')
P_A1 = phi_tab(P0_LOB, G_p=51)
P_B1 = phi_linterp_lob(P0_LOB, G_p=51)
diff_BA = float(np.max(np.abs(P_B1 - P_A1)))
# For C, resample uniform-u output to Lobatto via trilinear
from scipy.interpolate import RegularGridInterpolator
P_C1 = phi_linear(P0_UNI, u_lin, G_p=51)
interp = RegularGridInterpolator((u_lin,u_lin,u_lin), P_C1, method='linear',
                                   bounds_error=False, fill_value=0.5)
U1Lc = np.clip(U1L, u_lin[0], u_lin[-1])
U2Lc = np.clip(U2L, u_lin[0], u_lin[-1])
U3Lc = np.clip(U3L, u_lin[0], u_lin[-1])
P_C_at_lob = interp(np.stack([U1Lc.ravel(), U2Lc.ravel(), U3Lc.ravel()], axis=1)).reshape(G_LOB,G_LOB,G_LOB)
diff_CA = float(np.max(np.abs(P_C_at_lob - P_A1)))
print(f'  B (linterp Lob) vs A (Cheb): {diff_BA:.3e}')
print(f'  C (linear uniform) vs A:     {diff_CA:.3e}')

# Sweep G_lin (for variant C only) to show non-convergence
print('\nVariant C floor as G_lin grows:')
G_lin_sweep = [9, 13, 17, 21, 25, 33]
C_floors = []
C_times = []
for g_lin in G_lin_sweep:
    u_l = make_uniform_u_grid(g_lin, U_MAX_DEFAULT)
    U1u, U2u, U3u = np.meshgrid(u_l, u_l, u_l, indexing='ij')
    P0u = sg(0.5*TAU*(U1u+U2u+U3u))
    _ = phi_linear(P0u, u_l, G_p=51)  # warmup
    F = anderson(lambda P: phi_linear(P, u_l, G_p=51), P0u, n_iter=40)
    t = med_time(lambda: phi_linear(P0u, u_l, G_p=51), n=3)
    C_floors.append(float(min(F)))
    C_times.append(t*1000)
    print(f'  G_lin={g_lin:>3}: floor={min(F):.3e}, time={t*1000:.1f}ms')

# Save data
data = dict(
    floor_A=float(min(F_A)),
    floor_B=float(min(F_B)),
    floor_C=float(min(F_C)),
    floor_C_sweep=dict(G=G_lin_sweep, floors=C_floors, times=C_times),
    time_A=t_A*1000, time_B=t_B*1000, time_C=t_C*1000,
    F_A=F_A, F_B=F_B, F_C=F_C,
    gammas=gammas, slope_data=slope_data,
    diff_BA=diff_BA, diff_CA=diff_CA,
)
json.dump(data, open('/tmp/cheby_h0/three_ops.json', 'w'),
            indent=2, default=str)

# ===================================================================
# FIGURE 1: Floor under iteration (the headline)
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(range(1, len(F_A)+1), F_A, 'o-', color='tab:red',
              label=f'A. Cheb tab (Lob+atanh quad, Cheb interp)\nfloor={min(F_A):.2e}',
              markersize=4)
ax.semilogy(range(1, len(F_B)+1), F_B, 's-', color='tab:orange',
              label=f'B. Lin interp on Lob (SAME quad, linear interp)\nfloor={min(F_B):.2e}',
              markersize=4)
ax.semilogy(range(1, len(F_C)+1), F_C, '^-', color='tab:blue',
              label=f'C. Linear uniform-u (uniform-u quad+interp)\nfloor={min(F_C):.2e}',
              markersize=4)
ax.set_xlabel('Anderson iteration')
ax.set_ylabel(r'$\|F\|_\infty = \|\Phi(P)-P\|_\infty$')
ax.set_title('Convergence floor: Chebyshev vs Linear interpolation\n'
              '(B isolates linear-interp effect; C also drops the atanh quadrature)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_01_floor.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_01_floor.png')


# ===================================================================
# FIGURE 2: Wall time comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ops = ['A. Cheb tab\n(Cheb interp)',
       'B. Lin on Lobatto\n(same nodes, linear)',
       'C. Linear uniform-u\n(uniform grid + linear)']
times = [t_A*1000, t_B*1000, t_C*1000]
floors = [min(F_A), min(F_B), min(F_C)]
colors = ['tab:red', 'tab:orange', 'tab:blue']
bars = ax.bar(ops, times, color=colors, alpha=0.8)
ax.set_ylabel('wall time per $\\Phi$ (ms)')
ax.set_title('Per-$\\Phi$ wall time at $G{=}7$, $G_p{=}51$ ($G_{\\rm lin}{=}21$ for variant C)')
for b, t, f in zip(bars, times, floors):
    ax.text(b.get_x()+b.get_width()/2, t+1, f'{t:.1f} ms\n(floor {f:.1e})',
             ha='center', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_02_walltime.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_02_walltime.png')


# ===================================================================
# FIGURE 3: gamma dependence
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(gammas, slope_data['A'], 'o-', color='tab:red', markersize=8,
              label='A. Cheb tab')
ax.semilogx(gammas, slope_data['B'], 's-', color='tab:orange', markersize=8,
              label='B. Linear on Lobatto')
ax.semilogx(gammas, slope_data['C'], '^-', color='tab:blue', markersize=8,
              label='C. Linear uniform-u')
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel(r'$\alpha^*$ = logit-$P$ vs $T$ slope')
ax.set_title(r'$\gamma$-dependence at the converged FP')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_03_gamma.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_03_gamma.png')


# ===================================================================
# FIGURE 4: Variant C floor vs G_lin (non-convergence proof)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.semilogy(G_lin_sweep, C_floors, 'o-', color='tab:blue', markersize=8)
ax.axhline(min(F_A), color='tab:red', linestyle='--',
            label=f'Cheb tab floor = {min(F_A):.2e}')
ax.axhline(min(F_B), color='tab:orange', linestyle='--',
            label=f'Linear-on-Lob floor = {min(F_B):.2e}')
ax.set_xlabel(r'linear grid size $G_{\rm lin}$')
ax.set_ylabel(r'min $\|F\|_\infty$ over 40 Anderson iters')
ax.set_title('Variant C floor does NOT decrease with $G_{\\rm lin}$\n'
              '(uniform-u quadrature is the wrong measure for $\\int f_v$)')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.loglog(G_lin_sweep, C_times, 'o-', color='tab:blue', markersize=8,
            label='Linear (C) wall time')
ax.axhline(t_A*1000, color='tab:red', linestyle='--',
            label=f'Cheb tab time = {t_A*1000:.1f} ms')
# G^3 reference for scaling
Gs = np.array(G_lin_sweep, dtype=float)
ax.loglog(G_lin_sweep, 0.04*Gs**3, 'k--', alpha=0.4,
            label=r'$\mathcal{O}(G^3)$')
ax.set_xlabel(r'$G_{\rm lin}$')
ax.set_ylabel('wall time (ms) per $\\Phi$')
ax.set_title('Variant C wall time scales like $G^3$ from per-cube CRRA')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_04_lin_sweep.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_04_lin_sweep.png')


# ===================================================================
# FIGURE 5: Slice of the operator state (A vs B vs C)
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# Show u_2 = 0 slice of P after one Phi
# Take middle Lobatto/uniform node
mid_lob = G_LOB // 2
mid_lin = G_LIN_DEFAULT // 2
sliceA = P_A1[:, mid_lob, :]
sliceB = P_B1[:, mid_lob, :]
sliceC = P_C1[:, mid_lin, :]
for ax, sli, title, xax in zip(axes, [sliceA, sliceB, sliceC],
                                  ['A. Cheb tab', 'B. Linear on Lobatto',
                                   'C. Linear uniform-u'],
                                  [U_LOB, U_LOB, u_lin]):
    im = ax.pcolormesh(xax, xax, sli, vmin=0, vmax=1, cmap='RdBu_r',
                          shading='auto')
    ax.set_xlabel('$u_3$'); ax.set_ylabel('$u_1$')
    ax.set_title(f'{title}\nslice at middle $u_2$ after one $\\Phi$')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_05_slices.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_05_slices.png')


# ===================================================================
# FIGURE 6: Summary bar chart of floors and times together
# ===================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = ops
floors_v = floors
ax.bar(np.arange(3) - 0.2, floors_v, 0.4, label='residual floor',
        color='tab:red', alpha=0.7)
ax2 = ax.twinx()
ax2.bar(np.arange(3) + 0.2, times, 0.4, label='wall time (ms)',
         color='tab:green', alpha=0.7)
ax.set_yscale('log')
ax.set_xticks(range(3)); ax.set_xticklabels(labels, rotation=0, fontsize=9)
ax.set_ylabel(r'min $\|F\|_\infty$', color='tab:red')
ax2.set_ylabel('wall time (ms)', color='tab:green')
ax.set_title('Summary: Chebyshev wins on floor, linear-on-Lobatto wins on time')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/three_ops_06_summary.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved three_ops_06_summary.png')

print('\nsaved three_ops.json')
print('=== All figures saved ===')
