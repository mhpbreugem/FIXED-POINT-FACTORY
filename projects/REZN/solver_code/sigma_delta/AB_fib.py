"""A+B from no-learning IC, 100 iter max, plots at Fibonacci iters."""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior, crra_clear_sym)

G_FULL = 31
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
MAX_ITER = 100

FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
snapshot_iters = [0] + FIB + [100]

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

mu1 = sigmoid(TAU * U1m); mu2 = sigmoid(TAU * U2m); mu3 = sigmoid(TAU * U3m)
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_sym(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA, W)


def set_boundary_tail(P):
    G = P.shape[0]; P = P.copy(); eps = 1e-10
    def le(p1, p2):
        p1c = np.clip(p1, eps, 1-eps); p2c = np.clip(p2, eps, 1-eps)
        l1 = np.log(p1c/(1-p1c)); l2 = np.log(p2c/(1-p2c))
        y = np.clip(2*l1 - l2, -50, 50)
        return 1.0/(1.0 + np.exp(-y))
    P[0,:,:] = le(P[1,:,:], P[2,:,:]); P[G-1,:,:] = le(P[G-2,:,:], P[G-3,:,:])
    P[:,0,:] = le(P[:,1,:], P[:,2,:]); P[:,G-1,:] = le(P[:,G-2,:], P[:,G-3,:])
    P[:,:,0] = P[:,:,1]; P[:,:,G-1] = P[:,:,G-2]
    return np.clip(P, 1e-30, 1-1e-30)

def symmetrize(P):
    return 0.25 * (P + P[:,:,::-1] + (1.0 - P[::-1,::-1,:]) + (1.0 - P[::-1,::-1,::-1]))

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
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = set_boundary_tail(P); P = symmetrize(P); P = set_boundary_tail(P)

print('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT: {time.time()-t0:.1f}s')

snapshots = {0: P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()}
ferr_t = [0.0]; omR2_t = [weighted_R2_T(P_NL_in)]
d_FR_t = [float(np.sqrt(np.sum((P_NL_in - P_FR_in)**2 * Wd_inner)))]

for it in range(1, MAX_ITER+1):
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_tail(P_new); P_new = symmetrize(P_new); P_new = set_boundary_tail(P_new)
    ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P = P_new
    P_in = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    ferr_t.append(ferr); omR2_t.append(omR2); d_FR_t.append(d_FR)
    if it in snapshot_iters:
        snapshots[it] = P_in.copy()
    if it in snapshot_iters or it % 20 == 0 or it == 1:
        print(f'iter {it:3d}: ferr={ferr:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}')

FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
mid = G_INNER // 2
extent = [xi_inner[0], xi_inner[-1], xi_inner[0], xi_inner[-1]]
fib_iters_to_plot = [0] + FIB + [100]
ncols = len(fib_iters_to_plot)
fig, axes = plt.subplots(2, ncols, figsize=(2.4*ncols, 5.5), dpi=140)
for ci, it in enumerate(fib_iters_to_plot):
    P_in = snapshots[it]
    sl = P_in[mid, :, :]
    sl_FR = P_FR_in[mid, :, :]
    ax = axes[0, ci]
    pm = ax.pcolormesh(xi_inner, xi_inner, sl.T, cmap='RdBu_r', vmin=0, vmax=1, shading='nearest')
    if ci == 0: ax.set_ylabel('δ̂')
    if it == 0:
        ax.set_title('iter 0\n(no-learn IC)', fontsize=9)
    else:
        ax.set_title(f'iter {it}\nferr={ferr_t[it]:.2e}', fontsize=9)
    if ci == ncols-1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P')
    ax = axes[1, ci]
    dev = sl - sl_FR
    vmax = max(abs(dev).max(), 1e-10)
    pm = ax.pcolormesh(xi_inner, xi_inner, dev.T, cmap='PiYG', vmin=-vmax, vmax=vmax, shading='nearest')
    ax.set_xlabel('Σ̂')
    if ci == 0: ax.set_ylabel('δ̂')
    ax.set_title(f'max|dev|={vmax:.1e}', fontsize=9)
    if ci == ncols-1:
        plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label='P − P^FR')

plt.suptitle(f'A+B, no-learn IC, G={G_FULL}, γ={GAMMA}, MAX_ITER={MAX_ITER}. Snapshots at Fibonacci iters + 100.',
              fontsize=11.5, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/AB_no_learn_fib.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/AB_no_learn_fib.png')

# Full trajectory plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
iters = list(range(MAX_ITER+1))
metrics = [('ferr', ferr_t, 'ferr'),
           ('omR2', omR2_t, '1 − R²(T*)'),
           ('d_FR', d_FR_t, 'd_FR')]
for ax, (key, ys, title) in zip(axes, metrics):
    ax.semilogy(iters, [max(y, 1e-30) for y in ys], '-', lw=1.5)
    for f in FIB + [100]:
        ax.axvline(f, color='red', alpha=0.25, lw=0.8)
    ax.set_xlabel('Picard iter'); ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.5)
plt.suptitle('A+B, no-learn IC, 100 iters — red lines: Fibonacci snapshots',
              fontsize=12, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/AB_no_learn_100iter_trajectory.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/AB_no_learn_100iter_trajectory.png')

print('\n--- Values at Fibonacci iters ---')
for it in [0] + FIB + [100]:
    print(f'  iter {it:3d}: ferr={ferr_t[it]:.3e}  1-R²={omR2_t[it]:.3e}  d_FR={d_FR_t[it]:.3e}')
