"""Build tau-sweep figures and report."""
import json, os, sys
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lin_cdf_kern_tab import make_cdf_uniform_grid

FIGS = '/tmp/cheby_h0/figs/tau_report'
os.makedirs(FIGS, exist_ok=True)

d = json.load(open('/tmp/cheby_h0/lin_r2_100tau.json'))
items = sorted(d.items(), key=lambda x: float(x[0]))
tau = np.array([v['tau'] for k,v in items])
F = np.array([v['F'] for k,v in items])
slope = np.array([v['slope'] for k,v in items])
def1 = np.array([v['deficit_oneToOne'] for k,v in items])
deflin = np.array([v['deficit_lin'] for k,v in items])

# Mark "valid" range where clipping is mild (logit cap = ln((1-1e-15)/1e-15) = 34.5)
# Estimate: clipping bites when true slope*max_T > 34.5
# At gamma=1, true_slope_approx = some function of tau
# Heuristic: clipping severe when tau > 3 (when slope drops)
G = 7
u_grid = make_cdf_uniform_grid(G)
T_max = 3 * 2.33

# ===== FIG 1: headline =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
ax.semilogx(tau, slope, 'o-', color='tab:red', markersize=5)
ax.axvspan(5, 1000, alpha=0.15, color='gray',
              label=r'$\tau \gtrsim 5$: clipping artifact')
ax.set_xlabel(r'$\tau$ (signal precision)')
ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\tau$ at $\gamma=1$')
ax.legend(); ax.grid(alpha=0.3, which='both')
ax = axes[1]
ax.loglog(tau, np.maximum(def1, 1e-7), 's-', color='tab:red', markersize=5)
ax.axvspan(5, 1000, alpha=0.15, color='gray',
              label=r'$\tau \gtrsim 5$: clipping artifact')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title('One-to-one breakdown vs $\\tau$')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.suptitle('Lin-CDF Richardson 2-pt, 100-$\\tau$ sweep at $\\gamma=1$',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/01_headline.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 2: convergence floor =====
fig, ax = plt.subplots(figsize=(11, 5))
ax.loglog(tau, np.maximum(F, 1e-18), '.-', color='tab:red', markersize=5)
ax.axhline(1e-15, color='black', linestyle=':', label=r'machine $\varepsilon$')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Convergence floor (99/100 reach machine $\\varepsilon$)')
ax.legend(); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/02_floor.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 3: zoomed slope and deficit, valid tau range =====
mask = tau < 5
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.semilogx(tau[mask], slope[mask], 'o-', color='tab:red', markersize=6)
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'slope $\alpha^*$')
ax.set_title(r'Slope $\alpha^*$ vs $\tau$, valid range $\tau<5$')
ax.grid(alpha=0.3, which='both')
ax = axes[1]
ax.semilogx(tau[mask], def1[mask], 's-', color='tab:red', markersize=6)
ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$1-R^2_\mathrm{nonparam}$')
ax.set_title('One-to-one breakdown, valid range $\\tau<5$')
ax.grid(alpha=0.3, which='both')
plt.suptitle(r'Lin-CDF Richardson, $\tau$ sweep restricted to clipping-free range',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/03_valid_range.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 4: contour gallery =====
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
sample_taus = [0.01, 0.1, 0.32, 1.0, 1.8, 2.5, 3.5, 5.0, 8.0]
mid = G // 2
for ax, ttarget in zip(axes.flat, sample_taus):
    idx = int(np.argmin(np.abs(tau - ttarget)))
    t_act = tau[idx]
    P = np.load(f'/tmp/cheby_h0/fps_lin_r2_tau/P_FP_tau{t_act:.6e}.npy')
    slice2 = P[:, :, mid]
    cs = ax.contour(u_grid, u_grid, slice2.T, levels=np.arange(0.1, 1.0, 0.1),
                      cmap='RdBu_r', linewidths=1.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(rf'$\tau={t_act:.3g}$, $\alpha^*={slope[idx]:.3f}$, '
                   rf'def$_{{1-1}}={def1[idx]:.4f}$', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
plt.suptitle('Lin-CDF Richardson FP contours vs $\\tau$ ($\\gamma=1$)',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/04_contours.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 5: linear vs nonparam deficit =====
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogx(tau, deflin, 'o-', color='tab:blue', markersize=5,
              label=r'$1-R^2_\mathrm{lin}$')
ax.semilogx(tau, def1, 's-', color='tab:red', markersize=5,
              label=r'$1-R^2_\mathrm{nonparam}$ (real breakdown)')
ax.axvspan(5, 1000, alpha=0.15, color='gray', label='clipping artifact')
ax.set_xlabel(r'$\tau$'); ax.set_ylabel('deficit')
ax.set_title('Linear vs nonparametric deficit across $\\tau$')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{FIGS}/05_deficits.png', dpi=140, bbox_inches='tight')
plt.close()

# ===== FIG 6: logit-T scatter at selected taus =====
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
for ax, ttarget in zip(axes.flat, [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]):
    idx = int(np.argmin(np.abs(tau - ttarget)))
    t_act = tau[idx]
    P = np.load(f'/tmp/cheby_h0/fps_lin_r2_tau/P_FP_tau{t_act:.6e}.npy')
    T = t_act*(U1+U2+U3)
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = slope[idx]
    ax.scatter(T.ravel(), L, c=Pc.ravel(), cmap='RdBu_r', s=6, alpha=0.6)
    ax.plot([T.min(), T.max()], [s*T.min(), s*T.max()], 'k--',
              label=rf'$\alpha^*={s:.4f}$')
    ax.set_xlabel('T'); ax.set_ylabel('logit P')
    ax.set_title(rf'$\tau={t_act:.3g}$, def$_{{1-1}}={def1[idx]:.4f}$')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.suptitle('logit-$P$ vs $T$ scatter ($\\tau$ sweep at $\\gamma=1$)',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/06_logitT.png', dpi=140, bbox_inches='tight')
plt.close()

print('All tau figures saved.')
