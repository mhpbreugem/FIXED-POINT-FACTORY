"""(Σ̂, δ̂) cube at G=41 with STRICT h=0 contour Φ, no kernel smoothing.

The phi_sigmadelta operator (dd_phi_sigma_delta.py) already implements
strict linear-interp level-set crossings — no smoothing kernel anywhere.
h=0 is the model parameter: NO noise traders, NO supply shocks.

FR-ansatz IC, 20 Picard iters.
"""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior)

G_FULL = 41
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
MAX_ITER = 20

xi_full = np.linspace(-1.0, 1.0, G_FULL)
xi_inner = xi_full[INNER_LO:INNER_HI]
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

u_phys = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
S_full = U1m + U2m + U3m
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sigmoid(TAU * S_full)
Tstar = TAU * S_full
f_v_at = lambda U, v: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(U-v)**2)
F0_full = f_v_at(U1m,-0.5)*f_v_at(U2m,-0.5)*f_v_at(U3m,-0.5)
F1_full = f_v_at(U1m,+0.5)*f_v_at(U2m,+0.5)*f_v_at(U3m,+0.5)
Wd_inner = 0.5*F0_full + 0.5*F1_full; Wd_inner /= max(Wd_inner.sum(), 1e-30)


def set_boundary_hardwired(P):
    """h=0 hardwired boundary: FR=0/1 at u_1/Σ̂ infinities, zero-order δ̂."""
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return np.clip(P, 1e-30, 1 - 1e-30)


def weighted_R2_T(P_in):
    eps = 1e-30
    Pc = np.clip(P_in, eps, 1-eps); lp = np.log(Pc/(1-Pc))
    fl_t = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd_inner.flatten()
    slope, intercept = np.polyfit(fl_t, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_t + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan'))


P = np.zeros((G_FULL,)*3)
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P = set_boundary_hardwired(P)

LOG = '/tmp/sigmadelta_g41_strict.log'
open(LOG, 'w').close()
def lg(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

lg(f'STRICT-CONTOUR (Σ̂, δ̂) cube, h=0 HARDWIRED, G={G_FULL}, γ={GAMMA}, FR IC, MAX={MAX_ITER}')
lg('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
lg(f'  JIT: {time.time()-t0:.1f}s')

snapshots = [P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()]
ferr_t = [0.0]
omR2_t = [weighted_R2_T(P_FR_in)]
d_FR_t = [0.0]

t_iter = time.time()
for it in range(1, MAX_ITER+1):
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_hardwired(P_new)
    ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P = P_new
    P_in = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    snapshots.append(P_in.copy()); ferr_t.append(ferr); omR2_t.append(omR2); d_FR_t.append(d_FR)
    lg(f'  iter {it:2d}: ferr={ferr:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}')

lg(f'Total {(time.time()-t_iter)/60:.1f} min')

# ============ PLOT ============
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
mid = G_INNER // 2

NCOLS = 7   # 21 panels → 3 rows of 7 = 21
NROWS = int(np.ceil((MAX_ITER+1) / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(2.5*NCOLS, 2.5*NROWS), dpi=130)
axes = axes.ravel()
for it_idx in range(MAX_ITER+1):
    ax = axes[it_idx]
    sl = snapshots[it_idx][mid, :, :]
    im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                   extent=[xi_inner[0], xi_inner[-1], xi_inner[0], xi_inner[-1]],
                   aspect='auto')
    title = 'iter 0\n(FR IC)' if it_idx == 0 else f'iter {it_idx}\nferr={ferr_t[it_idx]:.2e}'
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
for k in range(MAX_ITER+1, len(axes)):
    axes[k].axis('off')
plt.suptitle(f'STRICT (Σ̂, δ̂), h=0 HARDWIRED, G={G_FULL}, γ={GAMMA}, FR IC, slice $\\hat u_1=0$.  '
              f'All {MAX_ITER+1} P snapshots',
              fontsize=11, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/sigmadelta_g41_strict_all_P.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/sigmadelta_g41_strict_all_P.png')

# Deviation slices
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(2.5*NCOLS, 2.5*NROWS), dpi=130)
axes = axes.ravel()
for it_idx in range(MAX_ITER+1):
    ax = axes[it_idx]
    dev = snapshots[it_idx][mid, :, :] - P_FR_in[mid, :, :]
    vmax = max(abs(dev).max(), 1e-200)
    im = ax.imshow(dev.T, origin='lower', cmap='PiYG', vmin=-vmax, vmax=vmax,
                   extent=[xi_inner[0], xi_inner[-1], xi_inner[0], xi_inner[-1]],
                   aspect='auto')
    title = f'iter {it_idx}\nmax|dev|={vmax:.1e}'
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
for k in range(MAX_ITER+1, len(axes)):
    axes[k].axis('off')
plt.suptitle(f'Deviation P − P^FR, same setup, all snapshots',
              fontsize=11, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/sigmadelta_g41_strict_all_dev.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/sigmadelta_g41_strict_all_dev.png')

# Trajectory plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
iters = np.arange(MAX_ITER+1)
axes[0].semilogy(iters, np.maximum(ferr_t, 1e-200), 'o-', lw=2, ms=7)
axes[0].set_xlabel('iter'); axes[0].set_ylabel('ferr'); axes[0].set_title('ferr')
axes[0].grid(True, ls=':', alpha=0.5)
axes[1].semilogy(iters, np.maximum(omR2_t, 1e-200), 's-', lw=2, ms=7, color='C1')
axes[1].set_xlabel('iter'); axes[1].set_ylabel('1−R²(T*)'); axes[1].set_title('Jensen wedge')
axes[1].grid(True, ls=':', alpha=0.5)
axes[2].semilogy(iters, np.maximum(d_FR_t, 1e-200), '^-', lw=2, ms=7, color='C2')
axes[2].set_xlabel('iter'); axes[2].set_ylabel('d_FR'); axes[2].set_title('distance to FR')
axes[2].grid(True, ls=':', alpha=0.5)
plt.suptitle(f'(Σ̂, δ̂) cube, G={G_FULL}, γ={GAMMA}, FR IC, STRICT h=0.  Trajectories.',
              fontsize=11.5, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/sigmadelta_g41_strict_trajectory.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/sigmadelta_g41_strict_trajectory.png')

import json
np.save('/tmp/sigmadelta_g41_strict_snaps.npy', np.array(snapshots))
with open('/tmp/sigmadelta_g41_strict.json', 'w') as f:
    json.dump({'ferr': ferr_t, 'omR2': omR2_t, 'd_FR': d_FR_t,
                'G_FULL': G_FULL, 'GAMMA': GAMMA, 'MAX_ITER': MAX_ITER,
                'mode': 'strict h=0 (Σ̂, δ̂)'}, f, indent=2)
print('saved snapshots + json')
