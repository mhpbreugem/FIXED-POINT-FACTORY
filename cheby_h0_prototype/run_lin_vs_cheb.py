"""Head-to-head: Chebyshev-tab vs Linear-grid-tab operators.

Compares same architecture (tabulated mu(p, u_k)) with two different
P representations: tensor-Chebyshev on Lobatto-stretched nodes vs
piecewise-linear on uniform-u nodes.

Measures:
  1. Per-Phi accuracy vs grid size
  2. Per-Phi wall time
  3. Floor under Anderson-accelerated iteration
  4. Sensitivity to gamma
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cheby_numba import (TAU, GAMMA, C_STRETCH, NQ, GL_NODES, GL_WEIGHTS,
                            U_NODES as U_NODES_LOB, N_GRID as G_CHEB)
from cheby_numba_tab import phi_tab
from cheby_linear_op import phi_linear, make_uniform_u_grid

def sigmoid(x): return 1/(1+np.exp(-x))
def P_analytic(u1, u2, u3, alpha=0.5):
    """The reference conjecture: sigmoid(alpha * T) sampled exactly."""
    return sigmoid(alpha * TAU * (u1 + u2 + u3))

# ========== Experiment 1: per-Phi accuracy vs grid size ==========
print('=== Experiment 1: per-Phi output, Chebyshev vs Linear ===\n')

# Reference: Chebyshev at N=6 (G=7) with G_p=51 — this is what we already
# know is accurate to ~5e-3 vs full chebroots.
U1c, U2c, U3c = np.meshgrid(U_NODES_LOB, U_NODES_LOB, U_NODES_LOB, indexing='ij')
P_in_cheb = P_analytic(U1c, U2c, U3c)
# warmup
_ = phi_tab(P_in_cheb, G_p=51)
print(f'Reference: Cheb tab at G={G_CHEB}, G_p=51')
t_cheb = []
for _ in range(5):
    t0 = time.time(); P_out_cheb = phi_tab(P_in_cheb, G_p=51); t_cheb.append(time.time()-t0)
t_cheb_med = float(np.median(t_cheb))
print(f'  wall time: {t_cheb_med*1000:.1f} ms')
print(f'  ||P_out - P_in||_inf = {float(np.max(np.abs(P_out_cheb - P_in_cheb))):.3e}\n')

# Linear at various grid sizes
G_lins = [7, 9, 13, 17, 21, 25, 33, 49]
U_max = 6.0  # MUST exceed Lobatto-stretched range at N=6 (~5.3) to avoid extrap
lin_results = []
print(f'Linear-grid op (uniform-in-u, U_max={U_max}, G_p=51):')
print(f'  {"G_lin":>6} {"wall(ms)":>10} {"||P-P_ref||_inf @ Lob nodes":>30}')
for G_lin in G_lins:
    u_lin = make_uniform_u_grid(G_lin, U_max)
    U1l, U2l, U3l = np.meshgrid(u_lin, u_lin, u_lin, indexing='ij')
    P_in_lin = P_analytic(U1l, U2l, U3l)
    _ = phi_linear(P_in_lin, u_lin, G_p=51)  # warmup
    ts = []
    for _ in range(5):
        t0 = time.time(); P_out_lin = phi_linear(P_in_lin, u_lin, G_p=51); ts.append(time.time()-t0)
    t_lin = float(np.median(ts))
    # Resample Linear output to Lobatto cube grid for comparison
    # Use trilinear interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((u_lin, u_lin, u_lin), P_out_lin,
                                       method='linear', bounds_error=False,
                                       fill_value=0.5)  # safe fallback
    # Clip Lobatto points to inside the linear grid
    U1c_clip = np.clip(U1c, u_lin[0], u_lin[-1])
    U2c_clip = np.clip(U2c, u_lin[0], u_lin[-1])
    U3c_clip = np.clip(U3c, u_lin[0], u_lin[-1])
    pts = np.stack([U1c_clip.ravel(), U2c_clip.ravel(), U3c_clip.ravel()], axis=1)
    P_lin_at_lob = interp(pts).reshape(G_CHEB, G_CHEB, G_CHEB)
    err = float(np.max(np.abs(P_lin_at_lob - P_out_cheb)))
    print(f'  {G_lin:>6} {t_lin*1000:>10.1f} {err:>30.3e}')
    lin_results.append(dict(G_lin=G_lin, time_ms=t_lin*1000, err_vs_cheb=err))

# ========== Experiment 2: Floor under iteration ==========
print('\n=== Experiment 2: residual floor under Anderson iteration ===\n')

def anderson(op_func, x0_shape, x0_init_func, n_iter=60, m_hist=8):
    """Generic Anderson on full-cube unknown (no symmetric reduction)."""
    x = x0_init_func().reshape(-1)
    Xh, Gh = [], []
    Ferrs = []
    for it in range(n_iter):
        Px = op_func(x.reshape(x0_shape))
        gx = Px.reshape(-1)
        F = gx - x
        Ferrs.append(float(np.max(np.abs(F))))
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > m_hist:
            Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1:
            x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1])
                                    for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    return Ferrs

print('Cheb tab (G=7, G_p=51):')
F_cheb = anderson(lambda P: phi_tab(P, G_p=51), (G_CHEB, G_CHEB, G_CHEB),
                    lambda: P_analytic(U1c, U2c, U3c))
print(f'  min ||F||_inf over 60 iters = {min(F_cheb):.3e}')

print('Linear tab (various G_lin):')
floor_lin = {}
for G_lin in [7, 13, 21, 33]:
    u_lin = make_uniform_u_grid(G_lin, U_max)
    U1l, U2l, U3l = np.meshgrid(u_lin, u_lin, u_lin, indexing='ij')
    F_lin = anderson(lambda P: phi_linear(P, u_lin, G_p=51),
                      (G_lin, G_lin, G_lin),
                      lambda: P_analytic(U1l, U2l, U3l))
    print(f'  G_lin={G_lin:>3}: min ||F||_inf = {min(F_lin):.3e}')
    floor_lin[G_lin] = dict(min_F=float(min(F_lin)), Ferrs=F_lin)

# ========== Experiment 3: gamma-sensitivity ==========
print('\n=== Experiment 3: gamma-dependence ===\n')
gammas = [0.3, 0.5, 1.0, 2.0, 5.0]
print(f'{"gamma":>8} {"Cheb slope":>15} {"Lin slope (G=21)":>20} {"diff":>12}')
gamma_results = []
u_lin21 = make_uniform_u_grid(21, U_max)
U1l21, U2l21, U3l21 = np.meshgrid(u_lin21, u_lin21, u_lin21, indexing='ij')
T_cheb = TAU*(U1c+U2c+U3c)
T_lin21 = TAU*(U1l21+U2l21+U3l21)
for g in gammas:
    P_out_cheb_g = phi_tab(P_in_cheb, gamma=g, G_p=51)
    P_in_lin21 = P_analytic(U1l21, U2l21, U3l21)
    P_out_lin_g = phi_linear(P_in_lin21, u_lin21, gamma=g, G_p=51)
    Lc = np.log(np.clip(P_out_cheb_g, 1e-15, 1-1e-15) /
                  (1-np.clip(P_out_cheb_g, 1e-15, 1-1e-15))).ravel()
    sc = float(np.sum(Lc*T_cheb.ravel()) / np.sum(T_cheb.ravel()**2))
    Ll = np.log(np.clip(P_out_lin_g, 1e-15, 1-1e-15) /
                  (1-np.clip(P_out_lin_g, 1e-15, 1-1e-15))).ravel()
    sl = float(np.sum(Ll*T_lin21.ravel()) / np.sum(T_lin21.ravel()**2))
    print(f'{g:>8.2f} {sc:>15.4f} {sl:>20.4f} {abs(sc-sl):>12.2e}')
    gamma_results.append(dict(gamma=g, cheb_slope=sc, lin_slope=sl,
                                diff=abs(sc-sl)))

# ========== Save raw data ==========
data = dict(
    cheb_ref=dict(G=G_CHEB, time_ms=t_cheb_med*1000,
                  F_iter=F_cheb,
                  min_F=float(min(F_cheb))),
    lin_results=lin_results,
    lin_floor=floor_lin,
    gamma=gamma_results,
)
json.dump(data, open('/tmp/cheby_h0/lin_vs_cheb.json', 'w'),
            indent=2, default=str)
print('\nsaved lin_vs_cheb.json')

# ========== Plots ==========
import matplotlib.pyplot as plt
FIGS = '/tmp/cheby_h0/figs'

# Plot 1: accuracy and timing vs G_lin
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
G_list = [r['G_lin'] for r in lin_results]
errs = [r['err_vs_cheb'] for r in lin_results]
times = [r['time_ms'] for r in lin_results]
ax.loglog(G_list, errs, 'o-', color='tab:blue', markersize=8,
            label=r'$\|P_{\rm lin} - P_{\rm cheb}\|_\infty$ at Lobatto nodes')
ax.axhline(min(F_cheb), color='tab:red', linestyle='--', alpha=0.7,
            label=f'Cheb iteration floor = {min(F_cheb):.2e}')
G_arr = np.array(G_list, dtype=float)
ax.loglog(G_list, 0.5/G_arr**2, 'k--', alpha=0.4,
            label=r'$O(G^{-2})$ reference')
ax.set_xlabel(r'linear grid size $G_{\rm lin}$')
ax.set_ylabel(r'$\|P_{\rm lin}-P_{\rm cheb}\|_\infty$ at $u$-Lobatto nodes')
ax.set_title('Per-$\\Phi$ accuracy: linear-grid op converges to Cheb-op as $G_{\\rm lin}$ grows')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.semilogy(G_list, times, 'o-', color='tab:green', markersize=8,
              label='Linear-grid wall time')
ax.axhline(t_cheb_med*1000, color='tab:red', linestyle='--',
            label=f'Cheb tab time = {t_cheb_med*1000:.1f} ms')
ax.set_xlabel(r'$G_{\rm lin}$')
ax.set_ylabel('wall time (ms) per $\\Phi$')
ax.set_title('Per-$\\Phi$ wall time')
ax.legend(fontsize=10); ax.grid(True, which='both', alpha=0.3)
plt.suptitle('Head-to-head: linear-grid vs Chebyshev (same $\\mu$-table architecture)',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/lin_vs_cheb_01_acc_time.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved lin_vs_cheb_01_acc_time.png')

# Plot 2: Floor under iteration for various G_lin
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.semilogy(range(1, len(F_cheb)+1), F_cheb, 'o-', color='tab:red',
              label=f'Cheb tab G={G_CHEB} (floor={min(F_cheb):.2e})',
              markersize=4)
for G_lin in [7, 13, 21, 33]:
    F = floor_lin[G_lin]['Ferrs']
    ax.semilogy(range(1, len(F)+1), F, 'o-', markersize=4,
                  label=f'Linear G={G_lin} (floor={min(F):.2e})')
ax.set_xlabel('Anderson iteration')
ax.set_ylabel(r'$\|F\|_\infty$ = $\|\Phi(P)-P\|_\infty$ per iter')
ax.set_title(r'Convergence floor: Chebyshev vs Linear (both at $G_p=51$ tab)')
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/lin_vs_cheb_02_floor.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved lin_vs_cheb_02_floor.png')

# Plot 3: gamma-dependence
fig, ax = plt.subplots(figsize=(10, 5))
gs = [r['gamma'] for r in gamma_results]
sc = [r['cheb_slope'] for r in gamma_results]
sl = [r['lin_slope'] for r in gamma_results]
ax.semilogx(gs, sc, 'o-', color='tab:red', markersize=8, label='Cheb tab G=7')
ax.semilogx(gs, sl, 's--', color='tab:blue', markersize=8,
              label='Linear G=21')
ax.set_xlabel(r'risk aversion $\gamma$')
ax.set_ylabel(r'$\alpha^*$ = logit-$P$ vs $T$ slope')
ax.set_title(r'$\gamma$-dependence preserved by linear-grid operator')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/lin_vs_cheb_03_gamma.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved lin_vs_cheb_03_gamma.png')

# Plot 4: combined summary
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(['Cheb tab G=7\n(40 sym DOF, 343 cells)'], [min(F_cheb)],
         color='tab:red', alpha=0.8, label='Cheb floor')
for G_lin in [7, 13, 21, 33]:
    ax.barh([f'Lin G={G_lin}\n({G_lin**3} cells)'],
             [floor_lin[G_lin]['min_F']],
             color='tab:blue', alpha=0.6)
ax.set_xscale('log')
ax.set_xlabel(r'min $\|F\|_\infty$ over 60 Anderson iters')
ax.set_title('Final convergence floor: linear vs Chebyshev (same mu-table)')
ax.grid(axis='x', which='both', alpha=0.3)
ax.axvline(1e-13, color='black', linestyle=':', alpha=0.5, label='target')
plt.tight_layout()
plt.savefig(f'{FIGS}/lin_vs_cheb_04_summary.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved lin_vs_cheb_04_summary.png')

print('\n=== All experiments complete ===')
