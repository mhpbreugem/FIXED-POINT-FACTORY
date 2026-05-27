"""Edge-accuracy + symmetry fixes for the (Σ̂, δ̂) contour-Φ.

Implements:
  Fix A (tail-aware boundary): replace the hard P=0/1 at the boundary
    nodes with a logit-linear extrapolation from the two nearest interior
    cells. The smooth FR limit Λ(τS)→0,1 at S→±∞ is preserved as an
    asymptote but the local logit P gradient is continuous across the
    boundary, eliminating the discrete "cliff" the contour scan sees
    when level sets approach ±∞.

  Fix B (symmetry projection): after each Φ step, project P onto its
    symmetric subspace via averaging over the two model symmetries:
      δ-sym:        P(i,j,k) = P(i,j,G-1-k)
      v↔1-v sym:    P(i,j,k) = 1 - P(G-1-i,G-1-j,k)
    The averaged P is bit-exactly symmetric, killing any numerical drift.

  Fix C (folded storage / symmetric scan): see phi_sigma_delta_folded.py.
    Stores only 1/4 of the cube; contour scan reads via reflection so
    crossings at +δ̂ and -δ̂ are exact mirrors by construction.

Run a 20-iter G=31 FR-IC comparison: baseline / +A / +B / +A+B / folded.
"""
import os, sys, time, math, json
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior)


def sigmoid(x): return 1.0/(1.0+np.exp(-x))


# ===== Fix A: tail-aware boundary =====
def set_boundary_tail(P):
    """Logit-linear extrapolation at boundary cells (Fix A).

    For each boundary cell:
      logit_P[boundary] = 2*logit_P[nearest interior] - logit_P[2nd-nearest]
    Convert back to P. Naturally tends to 0/1 because logit is unbounded
    while P is in (0,1); the FR limit at ±∞ is preserved asymptotically.
    """
    G = P.shape[0]
    P = P.copy()
    eps = 1e-10
    def logit_extrap(p1, p2):
        """y = 2*log(p1/(1-p1)) - log(p2/(1-p2)); return sigmoid(y)."""
        p1c = np.clip(p1, eps, 1-eps); p2c = np.clip(p2, eps, 1-eps)
        l1 = np.log(p1c/(1-p1c)); l2 = np.log(p2c/(1-p2c))
        y = 2*l1 - l2
        # clip to avoid overflow when sigmoiding
        y = np.clip(y, -50, 50)
        return 1.0/(1.0 + np.exp(-y))
    # u_1 = ±∞
    P[0, :, :] = logit_extrap(P[1, :, :], P[2, :, :])
    P[G-1, :, :] = logit_extrap(P[G-2, :, :], P[G-3, :, :])
    # Σ̂ = ±∞
    P[:, 0, :] = logit_extrap(P[:, 1, :], P[:, 2, :])
    P[:, G-1, :] = logit_extrap(P[:, G-2, :], P[:, G-3, :])
    # δ̂ = ±∞ (zero-order: keep as is)
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return np.clip(P, 1e-30, 1 - 1e-30)


def set_boundary_baseline(P):
    """Baseline (hard FR=0/1 at ±∞, zero-order δ̂)."""
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return np.clip(P, 1e-30, 1 - 1e-30)


# ===== Fix B: symmetry projection =====
def symmetrize(P):
    """Project P onto the (δ-sym + v↔1-v) symmetric subspace.

    P_proj = (1/4) [ P + P_δflip + (1 - P_full_flip) + (1 - P_δflip_full_flip) ]
    where full_flip means reflect axes 0,1 (u_1, Σ).
    """
    Pa = P
    Pb = P[:, :, ::-1]                  # δ flip (axis 2)
    Pc = 1.0 - P[::-1, ::-1, :]         # u_1 + Σ flip, value flip
    Pd = 1.0 - P[::-1, ::-1, ::-1]      # all three flipped (combine sym A·B)
    return 0.25 * (Pa + Pb + Pc + Pd)


# ===== run battery: baseline / +A / +B / +AB =====
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
P_FR_in = sigmoid(TAU * S_full)
Tstar = TAU * S_full
f_v_at = lambda U, v: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(U-v)**2)
F0_full = f_v_at(U1m,-0.5)*f_v_at(U2m,-0.5)*f_v_at(U3m,-0.5)
F1_full = f_v_at(U1m,+0.5)*f_v_at(U2m,+0.5)*f_v_at(U3m,+0.5)
Wd_inner = 0.5*F0_full + 0.5*F1_full; Wd_inner /= max(Wd_inner.sum(), 1e-30)


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
    """Measure symmetry violation: ||P − project(P)||_∞."""
    return np.max(np.abs(P - symmetrize(P)))


print('JIT warmup...')
P_warm = np.zeros((G_FULL,)*3)
P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_warm = set_boundary_baseline(P_warm)
t0 = time.time()
_ = phi_sigmadelta(P_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
print(f'  JIT: {time.time()-t0:.1f}s')


def run_variant(variant_label, use_tail_bc, use_sym):
    """Run N_ITERS Picard with given fix settings; return trajectories."""
    P = np.zeros((G_FULL,)*3)
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
    bc = set_boundary_tail if use_tail_bc else set_boundary_baseline
    P = bc(P)
    if use_sym:
        P = symmetrize(P)
    ferr_t = [0.0]; omR2_t = [weighted_R2_T(P_FR_in)]
    d_FR_t = [0.0]; asym_t = [measure_asymmetry(P)]
    for it in range(1, N_ITERS+1):
        P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                                INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_new = bc(P_new)
        if use_sym:
            P_new = symmetrize(P_new)
            P_new = bc(P_new)   # re-apply BC after sym (idempotent for sym BCs)
        ferr = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P = P_new
        P_in = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
        omR2 = weighted_R2_T(P_in)
        asym = measure_asymmetry(P)
        ferr_t.append(ferr); omR2_t.append(omR2); d_FR_t.append(d_FR); asym_t.append(asym)
    return ferr_t, omR2_t, d_FR_t, asym_t


variants = {
    'baseline': dict(use_tail_bc=False, use_sym=False),
    '+A (tail BC)': dict(use_tail_bc=True, use_sym=False),
    '+B (sym proj)': dict(use_tail_bc=False, use_sym=True),
    '+A+B': dict(use_tail_bc=True, use_sym=True),
}
results = {}
for label, kw in variants.items():
    print(f'\n=== {label} ===')
    ferr_t, omR2_t, d_FR_t, asym_t = run_variant(label, **kw)
    results[label] = {'ferr': ferr_t, 'omR2': omR2_t, 'd_FR': d_FR_t, 'asym': asym_t}
    for it in [1, 5, 10, 15, 20]:
        print(f'  iter {it:2d}: ferr={ferr_t[it]:.3e}  1-R²={omR2_t[it]:.3e}  '
              f'd_FR={d_FR_t[it]:.4e}  asym={asym_t[it]:.3e}')


# ============ PLOT ============
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
iters = list(range(N_ITERS+1))
metrics = [
    ('ferr', 0, 0, 'log'),
    ('omR2', 0, 1, 'log'),
    ('d_FR', 1, 0, 'log'),
    ('asym', 1, 1, 'log'),
]
titles = {
    'ferr': 'ferr = ‖Φ(P) − P‖_∞',
    'omR2': '1 − R²(T*) — Jensen wedge content',
    'd_FR': 'd_FR — distance to FR ansatz',
    'asym': 'symmetry violation = ‖P − project(P)‖_∞',
}
markers = {'baseline': 'o-', '+A (tail BC)': 's--', '+B (sym proj)': '^:', '+A+B': 'D-.'}
for key, ri, ci, scale in metrics:
    ax = axes[ri, ci]
    for label in variants:
        ys = results[label][key]
        ys_plot = [max(y, 1e-30) for y in ys]
        ax.plot(iters, ys_plot, markers[label], lw=2, ms=6, label=label)
    if scale == 'log': ax.set_yscale('log')
    ax.set_xlabel('Picard iteration')
    ax.set_title(titles[key])
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=10)
plt.suptitle(f'Edge-accuracy fixes: G=31 uniform, FR-ansatz IC, 20 Picard iters at γ={GAMMA}',
              fontsize=12, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/edge_fixes_compare.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'\nwrote {FIG}/edge_fixes_compare.png')


# Save summary numerics
summary_lines = ['Variant | iter 1 ferr | iter 20 ferr | iter 20 1-R² | iter 20 d_FR | iter 20 asym']
summary_lines.append('-' * 100)
for label in variants:
    r = results[label]
    summary_lines.append(f'{label:18s} | {r["ferr"][1]:.3e} | {r["ferr"][20]:.3e} | '
                          f'{r["omR2"][20]:.3e} | {r["d_FR"][20]:.3e} | {r["asym"][20]:.3e}')
summary_text = '\n'.join(summary_lines)
with open('/tmp/edge_fixes_summary.txt', 'w') as f:
    f.write(summary_text)
print('\n' + summary_text)
print('done')
