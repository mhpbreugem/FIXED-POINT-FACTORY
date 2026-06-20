"""Uniform G=31, FR IC, first 20 Picard iters with 1-R²(T*) trajectory."""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior)

G_FULL = 31
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
N_ITERS = 20

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


def set_boundary_uniform(P):
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
    Pc = np.clip(P_in, eps, 1-eps)
    lp = np.log(Pc/(1-Pc))
    fl_t = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd_inner.flatten()
    slope, intercept = np.polyfit(fl_t, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_t + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan'))


P = np.zeros((G_FULL,)*3)
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P = set_boundary_uniform(P)

print('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT: {time.time()-t0:.1f}s')

ferr_trace = [0.0]
omR2_trace = [weighted_R2_T(P_FR_in)]  # for FR IC, 1-R² is basically zero (linear logit P in T*)
d_FR_trace = [0.0]
print(f'  iter 0: 1-R²={omR2_trace[0]:.3e}')

for it in range(1, N_ITERS+1):
    t0 = time.time()
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_uniform(P_new)
    ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P = P_new
    P_in = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    ferr_trace.append(ferr)
    omR2_trace.append(omR2)
    d_FR_trace.append(d_FR)
    print(f'  iter {it:2d}: ferr={ferr:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}  ({time.time()-t0:.1f}s)')

FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'

# 3-panel plot: ferr, 1-R², d_FR
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
iters = list(range(N_ITERS+1))

ax = axes[0]
ax.semilogy(iters, [max(f, 1e-30) for f in ferr_trace], 'o-', lw=2, ms=7, color='C0')
ax.set_xlabel('Picard iteration')
ax.set_ylabel('ferr')
ax.set_title('Φ-residual ferr')
ax.grid(True, ls=':', alpha=0.5)

ax = axes[1]
ax.semilogy(iters, [max(r, 1e-30) for r in omR2_trace], 's-', lw=2, ms=7, color='C1')
ax.set_xlabel('Picard iteration')
ax.set_ylabel('1 − R²(T*)')
ax.set_title('Non-linear-in-T* content (Jensen wedge)')
ax.grid(True, ls=':', alpha=0.5)

ax = axes[2]
ax.semilogy(iters, [max(d, 1e-30) for d in d_FR_trace], '^-', lw=2, ms=7, color='C2')
ax.set_xlabel('Picard iteration')
ax.set_ylabel('d_FR')
ax.set_title('weighted L² distance to FR ansatz')
ax.grid(True, ls=':', alpha=0.5)

plt.suptitle(f'Uniform G={G_FULL}, FR-ansatz IC, first {N_ITERS} Picard iters.  γ={GAMMA}, τ={TAU}, K=3 symm.',
              fontsize=12, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/uniform_g31_FR_20iters.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/uniform_g31_FR_20iters.png')

# Also a numerical table
print('\n=== Trajectory table ===')
print(f'{"iter":>4} {"ferr":>12} {"1-R²(T*)":>12} {"d_FR":>12}')
for i in range(N_ITERS+1):
    print(f'{i:>4} {ferr_trace[i]:>12.3e} {omR2_trace[i]:>12.3e} {d_FR_trace[i]:>12.3e}')
print('done')
