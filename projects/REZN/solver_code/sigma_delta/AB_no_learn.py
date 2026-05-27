"""Fix A+B (tail BC + symmetry projection) with NO-LEARNING IC, 20 iters."""
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

# No-learn IC
mu1 = sigmoid(TAU * U1m); mu2 = sigmoid(TAU * U2m); mu3 = sigmoid(TAU * U3m)
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_sym(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA, W)


# Fix A: tail-aware boundary (logit-linear extrap at u_1 and Σ̂ boundaries)
def set_boundary_tail(P):
    G = P.shape[0]
    P = P.copy()
    eps = 1e-10
    def logit_extrap(p1, p2):
        p1c = np.clip(p1, eps, 1-eps); p2c = np.clip(p2, eps, 1-eps)
        l1 = np.log(p1c/(1-p1c)); l2 = np.log(p2c/(1-p2c))
        y = np.clip(2*l1 - l2, -50, 50)
        return 1.0/(1.0 + np.exp(-y))
    P[0, :, :] = logit_extrap(P[1, :, :], P[2, :, :])
    P[G-1, :, :] = logit_extrap(P[G-2, :, :], P[G-3, :, :])
    P[:, 0, :] = logit_extrap(P[:, 1, :], P[:, 2, :])
    P[:, G-1, :] = logit_extrap(P[:, G-2, :], P[:, G-3, :])
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return np.clip(P, 1e-30, 1 - 1e-30)


# Fix B: symmetry projection
def symmetrize(P):
    Pa = P
    Pb = P[:, :, ::-1]
    Pc = 1.0 - P[::-1, ::-1, :]
    Pd = 1.0 - P[::-1, ::-1, ::-1]
    return 0.25 * (Pa + Pb + Pc + Pd)


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

def measure_asymmetry(P):
    return np.max(np.abs(P - symmetrize(P)))

# Initialize with NO-LEARNING IC
P = np.zeros((G_FULL,)*3)
P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = set_boundary_tail(P)
P = symmetrize(P)
P = set_boundary_tail(P)

print('JIT warmup...')
t0 = time.time()
_ = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT: {time.time()-t0:.1f}s')

ferr_t = [0.0]; omR2_t = [weighted_R2_T(P_NL_in)]
d_FR_t = [float(np.sqrt(np.sum((P_NL_in - P_FR_in)**2 * Wd_inner)))]
asym_t = [measure_asymmetry(P)]

print(f'\n--- Fix A+B, no-learn IC, G={G_FULL}, γ={GAMMA} ---')
print(f'iter 0: 1-R²={omR2_t[0]:.3e}  d_FR={d_FR_t[0]:.4e}  asym={asym_t[0]:.3e}')

for it in range(1, N_ITERS+1):
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_tail(P_new)
    P_new = symmetrize(P_new)
    P_new = set_boundary_tail(P_new)
    ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P = P_new
    P_in = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    asym = measure_asymmetry(P)
    ferr_t.append(ferr); omR2_t.append(omR2); d_FR_t.append(d_FR); asym_t.append(asym)
    print(f'iter {it:2d}: ferr={ferr:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}  asym={asym:.3e}')

# Plot
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
iters = list(range(N_ITERS+1))
metrics = [
    ('ferr', 0, 0, ferr_t, 'ferr = ‖Φ(P) − P‖_∞'),
    ('omR2', 0, 1, omR2_t, '1 − R²(T*)'),
    ('d_FR', 1, 0, d_FR_t, 'd_FR'),
    ('asym', 1, 1, asym_t, 'symmetry violation'),
]
for key, ri, ci, ys, title in metrics:
    ax = axes[ri, ci]
    ax.semilogy(iters, [max(y, 1e-30) for y in ys], 'o-', lw=2, ms=7, color='C0')
    ax.set_xlabel('Picard iteration')
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.5)
plt.suptitle(f'Fix A+B (tail BC + sym projection), NO-LEARN IC, G={G_FULL}, γ={GAMMA}, K=3 sym, 20 iters',
              fontsize=12, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/AB_no_learn_20iters.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'\nwrote {FIG}/AB_no_learn_20iters.png')

# Compare to baseline + FR IC trajectory (from previous run)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
# previous baseline FR-IC
fr_base = [0.0, 9.086e-02, 3.585e-02, 2.501e-02, 1.862e-02, 1.441e-02, 1.406e-02,
            1.633e-02, 1.615e-02, 1.560e-02, 1.342e-02, 1.283e-02, 1.171e-02,
            1.081e-02, 1.102e-02, 9.769e-03, 9.130e-03, 8.964e-03, 1.544e-02,
            2.958e-02, 4.003e-02]
ax = axes[0]
ax.semilogy(iters, [max(f, 1e-30) for f in ferr_t], 'o-', lw=2, ms=7, label='+A+B, no-learn IC')
ax.semilogy(iters, [max(f, 1e-30) for f in fr_base], 's--', lw=2, ms=7, alpha=0.6,
            label='baseline, FR IC (ref)')
ax.set_xlabel('Picard iter'); ax.set_ylabel('ferr'); ax.set_title('ferr')
ax.grid(True, ls=':', alpha=0.5); ax.legend(fontsize=9)

ax = axes[1]
ax.semilogy(iters, [max(r, 1e-30) for r in omR2_t], 'o-', lw=2, ms=7, label='+A+B, no-learn')
ax.set_xlabel('Picard iter'); ax.set_ylabel('1 − R²(T*)')
ax.set_title('Jensen wedge content')
ax.grid(True, ls=':', alpha=0.5); ax.legend(fontsize=9)

ax = axes[2]
ax.semilogy(iters, [max(d, 1e-30) for d in d_FR_t], 'o-', lw=2, ms=7, label='+A+B, no-learn')
ax.set_xlabel('Picard iter'); ax.set_ylabel('d_FR')
ax.set_title('distance to FR')
ax.grid(True, ls=':', alpha=0.5); ax.legend(fontsize=9)

plt.suptitle('A+B from no-learn IC vs baseline from FR IC (reference)',
              fontsize=12, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/AB_no_learn_vs_FR.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/AB_no_learn_vs_FR.png')
print('done')
