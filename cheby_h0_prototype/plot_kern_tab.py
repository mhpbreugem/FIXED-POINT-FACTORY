"""Plot: convergence to machine eps with kernel-band Cheb-tab.
Also compare FP shape (slope, deficit) against chebroots/bisect floor results.
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov

from cheby_numba_kern_tab import phi_kern_tab
from cheby_numba import phi as phi_chebroots, U_NODES, TAU, GAMMA, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))
_ = phi_kern_tab(sg(0.5*T), kernel_h=0.3)

REPO = '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype'
FIGS = '/tmp/cheby_h0/figs'

# ===== Track ||F|| per iter from cold and from warm =====
def F_sym(x, h, G_p):
    return contract(phi_kern_tab(expand(x), kernel_h=h, G_p=G_p)) - x

def anderson_track(F_func, x0, n_iter=60, m=10):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < 1e-15: break
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

x_cold = contract(sg(0.5*T))
x_warm = contract(np.load(f'{REPO}/P_final_sym2.npy'))
print('Cold-vs-warm convergence at various h:')
curves_cold = {}
curves_warm = {}
for h in [0.5, 0.3, 0.2, 0.15]:
    F_c = anderson_track(lambda x, h=h: F_sym(x, h, 121), x_cold, n_iter=60)
    F_w = anderson_track(lambda x, h=h: F_sym(x, h, 121), x_warm, n_iter=60)
    curves_cold[h] = F_c; curves_warm[h] = F_w
    print(f'  h={h}: cold floor {min(F_c):.3e}, warm floor {min(F_w):.3e}')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for h, F in curves_cold.items():
    ax.semilogy(range(1, len(F)+1), np.maximum(F, 1e-18), '-o',
                  markersize=4, label=f'h={h}')
ax.axhline(1.1e-16, color='black', linestyle=':', alpha=0.5, label='machine eps')
ax.set_xlabel('Anderson iteration')
ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Cold start ($P_0=\\sigma(0.5T)$)')
ax.legend(); ax.grid(alpha=0.3, which='both')
ax.set_ylim(1e-18, 2)

ax = axes[1]
for h, F in curves_warm.items():
    ax.semilogy(range(1, len(F)+1), np.maximum(F, 1e-18), '-o',
                  markersize=4, label=f'h={h}')
ax.axhline(1.1e-16, color='black', linestyle=':', alpha=0.5, label='machine eps')
ax.set_xlabel('Anderson iteration')
ax.set_ylabel(r'$\|F\|_\infty$')
ax.set_title('Warm start (P_final_sym2.npy)')
ax.legend(); ax.grid(alpha=0.3, which='both')
ax.set_ylim(1e-18, 2)
plt.suptitle('Kernel-band Cheb-tab: reaches machine $\\varepsilon$ from both starts\n'
              '(h=0.15 too narrow at G=7 -- band approaches the discontinuous h=0 limit)',
              fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIGS}/kern_tab_01_convergence.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved kern_tab_01_convergence.png')

# ===== FP characterization: slope and deficit across h =====
print('\nFP shape across h (G_p=121):')
print(f'{"h":>6} {"slope":>10} {"deficit":>12} {"P range":>30}')
hist = {}
for h in [0.5, 0.4, 0.3, 0.25, 0.20]:
    F_w = anderson_track(lambda x: F_sym(x, h, 121), x_warm, n_iter=80)
    # Get x at end
    x = x_warm.copy()
    Xh, Gh = [], []
    for it in range(80):
        F = F_sym(x, h, 121); gx = F + x
        if float(np.max(np.abs(F))) < 1e-15: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
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
    P = expand(x)
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    slope = float(np.sum(L*T.ravel()) / np.sum(T.ravel()**2))
    # Deficit = 1 - R^2 of L vs T
    pred = slope * T.ravel()
    intercept = np.mean(L - pred)
    pred += intercept
    res = np.sum((L - pred)**2)
    tot = np.sum((L - L.mean())**2)
    deficit = float(res / max(tot, 1e-30))
    hist[h] = dict(slope=slope, deficit=deficit, F_final=float(min(F_w)))
    print(f'{h:>6.2f} {slope:>10.4f} {deficit:>12.4e}  [{P.min():.4f}, {P.max():.4f}]')

# Plot slope and deficit vs h
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
hs = sorted(hist.keys(), reverse=True)
slopes = [hist[h]['slope'] for h in hs]
defs = [hist[h]['deficit'] for h in hs]
ax = axes[0]
ax.plot(hs, slopes, 'o-', markersize=8)
ax.set_xlabel(r'kernel bandwidth $h$ (band width in $p$-units)')
ax.set_ylabel(r'$\alpha^*$ (logit-$P$ vs $T$ slope)')
ax.set_title(r'FP slope vs $h$ (kernel-band Cheb-tab, $G_p=121$)')
ax.grid(alpha=0.3)
ax.invert_xaxis()  # show smaller h on the right (→ true h=0 limit)
ax = axes[1]
ax.semilogy(hs, defs, 's-', markersize=8, color='tab:red')
ax.set_xlabel(r'kernel bandwidth $h$')
ax.set_ylabel(r'information deficit $1-R^2$')
ax.set_title('FP deficit vs $h$')
ax.grid(alpha=0.3, which='both')
ax.invert_xaxis()
plt.suptitle('Kernel-band Cheb-tab: FP shape converges as $h \\to 0$\n'
              '(h>0 is a regularization; the joint h→0 limit recovers the true h=0 REE)',
              fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGS}/kern_tab_02_FPshape.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved kern_tab_02_FPshape.png')

# Save data
data = dict(
    cold_curves={f'h={h}': F for h, F in curves_cold.items()},
    warm_curves={f'h={h}': F for h, F in curves_warm.items()},
    FP_vs_h={f'h={h}': v for h, v in hist.items()},
)
json.dump(data, open('/tmp/cheby_h0/kern_tab_full.json', 'w'),
            indent=2, default=str)
print('saved kern_tab_full.json')
