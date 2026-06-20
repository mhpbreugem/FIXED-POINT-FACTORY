"""Inspect the NK plateau: compute the Jacobian of Phi at the current near-FP
iterate via forward-difference, find its singular values, and check
||I - J|| condition. Tests:

A) Jacobian spectrum at best Anderson iterate.
B) Compare Phi computed at float64 vs flint arb (50-digit) on the SAME P_in,
   to confirm arithmetic precision is not the source of residual.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

from cdf_cube_numba import (phi_cdf_numba, build_zeta_grid, gauss_legendre,
                              crra_clear_nb, metrics, TAU, GAMMA, TAB_Z, TAB_U,
                              TAB_DUDZ)

G = 13; NQ = 64
zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
print(f'spectral diagnostic: G={G}, NQ={NQ}, N_unk={G**3}', flush=True)

# Load best Anderson iterate at γ=0.1 (use the one from G=13 NQ=64 sweep)
P_path = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/cdf_cube_NB_G13_NQ64_g0.1.npy'
P_fp = np.load(P_path)
print(f'loaded {P_path}, P range [{P_fp.min():.4f}, {P_fp.max():.4f}]', flush=True)
m = metrics(P_fp, u_arr)
print(f'this P: deficit={m["deficit"]:.4f} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}')

def Phi(P): return phi_cdf_numba(P, zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                                    TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)

# JIT warmup
print('JIT warmup...', flush=True); t=time.time()
_ = Phi(P_fp.copy())
print(f'  done {time.time()-t:.1f}s', flush=True)

PhP = Phi(P_fp.copy())
F = PhP - P_fp
ferr = float(np.max(np.abs(F)))
rms = float(np.sqrt(np.mean(F**2)))
print(f'\nPhi(P) - P: max={ferr:.3e}, RMS={rms:.3e}')
# Identify which cells have biggest residual
idx_sorted = np.argsort(-np.abs(F).ravel())[:10]
print(f'  Top-10 residual cells:')
for k, idx in enumerate(idx_sorted):
    i, j, l = np.unravel_index(idx, F.shape)
    print(f'    ({i:2d},{j:2d},{l:2d}) u=({u_arr[i]:+.3f},{u_arr[j]:+.3f},{u_arr[l]:+.3f}) '
          f'P={P_fp[i,j,l]:.4f} Phi(P)={PhP[i,j,l]:.4f} F={F[i,j,l]:+.3e}')

# === A) Sparse Jacobian estimate: sample 30 random columns ===
print(f'\n=== Jacobian spectrum (forward-diff, 30 random columns) ===', flush=True)
N = G**3
rng = np.random.default_rng(42)
sample_cols = rng.choice(N, size=30, replace=False)
eps_fd = 1e-6
J_cols = np.empty((N, len(sample_cols)))
for c, idx in enumerate(sample_cols):
    P_pert = P_fp.copy()
    i, j, l = np.unravel_index(idx, P_fp.shape)
    P_pert[i, j, l] += eps_fd
    PhP_pert = Phi(P_pert)
    J_cols[:, c] = (PhP_pert - PhP).ravel() / eps_fd
# Approximate sigma_min / sigma_max from sampled columns
U, S, Vt = np.linalg.svd(J_cols, full_matrices=False)
print(f'  sampled-J singular values: max={S.max():.4f}, min={S.min():.4f}, cond={S.max()/max(S.min(),1e-30):.2e}')
# Approximate I - J: identity rows where sampled cols
IJ_cols = np.eye(N)[:, sample_cols] - J_cols
U2, S2, V2 = np.linalg.svd(IJ_cols, full_matrices=False)
print(f'  sampled-(I-J) singular values: max={S2.max():.4f}, min={S2.min():.4f}, cond={S2.max()/max(S2.min(),1e-30):.2e}')
if S2.min() < 1e-3:
    print(f'  >>> SINGULAR DIRECTION FOUND: I-J has σ_min={S2.min():.3e} -- NK ill-conditioned <<<')
elif S2.min() < 0.1:
    print(f'  >>> NEAR-SINGULAR: I-J σ_min={S2.min():.3e} -- NK slow <<<')
else:
    print(f'  I-J σ_min={S2.min():.3f} is healthy; NK should converge')

# === B) Flint arb sanity at one cell ===
print(f'\n=== B) flint arb 50-digit Phi comparison at worst cell ===', flush=True)
import flint
flint.ctx.prec = 165   # ~50 decimal digits
print(f'  flint arb prec: {flint.ctx.prec} bits (~{int(flint.ctx.prec*0.301)} digits)')

# Simplest check: f_signal at u_worst
def f_signal_f64(u, vm, tau): return math.sqrt(tau/(2*math.pi))*math.exp(-0.5*tau*(u-vm)**2)
def f_signal_arb(u, vm, tau):
    pi = flint.arb.pi()
    return (tau/(2*pi)).sqrt() * (-(tau*(u-vm)**2)/2).exp()
i, j, l = np.unravel_index(idx_sorted[0], F.shape)
u1, u2, u3 = u_arr[i], u_arr[j], u_arr[l]
print(f'  worst cell: ({i},{j},{l}) u=({u1:+.3f},{u2:+.3f},{u3:+.3f})')
for v_mean in (-0.5, 0.5):
    f_f64 = f_signal_f64(u1, v_mean, 2.0)
    f_arb = f_signal_arb(flint.arb(str(u1)), flint.arb(str(v_mean)), flint.arb('2.0'))
    print(f'    f(u_1={u1:+.3f}, v_mean={v_mean}): float64={f_f64:.12e}, arb={f_arb}')
    diff = float(f_arb) - f_f64
    print(f'      diff = {diff:+.3e}  (expected ~1e-16)')

# Same for crra_clear
print('\n  crra_clear at sample (mu0, mu1, mu2)=(0.6, 0.7, 0.5):')
p_f64 = crra_clear_nb(0.6, 0.7, 0.5, 0.1, 120)
print(f'    float64 = {p_f64:.16e}')
# arb version
def crra_clear_arb(m0, m1, m2, gamma, steps=200):
    eps = flint.arb(10)**-50
    a = eps; b = flint.arb(1) - eps
    me = [m if m > eps else eps for m in (m0, m1, m2)]
    me = [m if m < (flint.arb(1)-eps) else (flint.arb(1)-eps) for m in me]
    lm = [(m/(flint.arb(1)-m)).log() for m in me]
    for _ in range(steps):
        m = (a+b)/2
        lp = (m/(flint.arb(1)-m)).log()
        e = flint.arb(0)
        for lmk in lm:
            arg = (lmk - lp)/gamma
            if arg > flint.arb(700):
                e += flint.arb(1)/m
            else:
                R = arg.exp()
                e += (R - flint.arb(1))/((flint.arb(1)-m) + R*m)
        # arb comparison
        if e.mid() > 0: a = m
        else: b = m
    return (a+b)/2
p_arb = crra_clear_arb(flint.arb('0.6'), flint.arb('0.7'), flint.arb('0.5'), flint.arb('0.1'))
print(f'    arb     = {p_arb}')
diff = float(p_arb) - p_f64
print(f'    diff    = {diff:+.3e}  (if ~1e-16, arithmetic is fine)')

# Final verdict
print(f'\n=== VERDICT ===')
if S2.min() < 0.1:
    print(f'  NK plateau is due to (I - dΦ) σ_min = {S2.min():.3e} ≪ 1 — Jacobian near-singular')
    print(f'  High-precision arithmetic CANNOT fix this; need a different method')
    print(f'  (regularized Newton, trust-region, or different operator).')
else:
    print(f'  Jacobian healthy (σ_min = {S2.min():.3f}); NK should converge.')
    print(f'  Plateau may be from operator non-smoothness; flint arb test result above.')

json.dump({'P_path':P_path, 'ferr':ferr, 'rms':rms,
            'sigma_J_min':float(S.min()), 'sigma_J_max':float(S.max()),
            'sigma_IJ_min':float(S2.min()), 'sigma_IJ_max':float(S2.max()),
            'IJ_cond':float(S2.max()/max(S2.min(),1e-30))},
          open(os.path.join(HERE,'spectral_diag.json'),'w'), indent=2, default=str)
print('saved')
