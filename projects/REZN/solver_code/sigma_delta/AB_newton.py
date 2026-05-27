"""Newton-Krylov solver on (u_1, Σ̂, δ̂) cube with A+B fixes.

F(P) = Φ(P) − P;  J = ∂F/∂P = ∂Φ/∂P − I.

Newton step:  J · δ = −F,  P ← P + δ.

J is too large to form. Use scipy.sparse.linalg.lgmres with matvec via
finite-difference Jacobian-vector products:

    J · v ≈ [F(P + ε·v) − F(P)] / ε

Each matvec costs one extra Φ call.  GMRES typically needs ~50-200
matvecs per Newton step; near a fixed point Newton has quadratic
convergence (~5 steps to machine eps).

Warm-start with a few Picard iters to land near a fixed point, then
switch to Newton.
"""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, lgmres
from dd_phi_sigma_delta import (phi_sigmadelta, finf_interior, crra_clear_sym)

G_FULL = 31
G_INNER = G_FULL - 2
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0

# Number of Picard warmup iters and Newton steps
N_PICARD_WARMUP = 5
N_NEWTON = 4
LGMRES_MAXITER = 6   # cap to ensure each Newton step is bounded
LGMRES_INNER_M = 15

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


def Phi_full(P_full):
    """One Φ step with Fix A (tail BC) + Fix B (sym projection)."""
    P_new = phi_sigmadelta(P_full, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_new = set_boundary_tail(P_new)
    P_new = symmetrize(P_new)
    P_new = set_boundary_tail(P_new)
    return P_new


def F_inner(P_full):
    """F = Φ(P) − P on inner block only (boundaries are not unknowns)."""
    P_new = Phi_full(P_full)
    P_in_old = P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    P_in_new = P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    return (P_in_new - P_in_old).ravel()


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


# IC: FR ansatz
P_full = np.zeros((G_FULL,)*3)
P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_full = set_boundary_tail(P_full)
P_full = symmetrize(P_full)
P_full = set_boundary_tail(P_full)

print('JIT warmup...')
t0 = time.time()
_ = Phi_full(P_full)
print(f'  JIT: {time.time()-t0:.1f}s')


# ===== Picard warmup =====
print(f'\n--- Picard warmup ({N_PICARD_WARMUP} iters) ---')
ferr_history = [0.0]
omR2_history = [weighted_R2_T(P_FR_in)]
d_FR_history = [0.0]
for it in range(1, N_PICARD_WARMUP+1):
    P_new = Phi_full(P_full)
    ferr = finf_interior(P_new, P_full, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_full = P_new
    P_in = P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    ferr_history.append(ferr); omR2_history.append(omR2); d_FR_history.append(d_FR)
    print(f'  Picard {it:2d}: ferr={ferr:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}')


# ===== Newton-Krylov =====
print(f'\n--- Newton-Krylov ({N_NEWTON} steps) ---')
n_inner = G_INNER ** 3
EPS_FD = 1e-6  # finite-difference step for Jv

def make_matvec_op(P_full_at_step, F0_vec):
    """Build LinearOperator for J·v at the current P_full."""
    def matvec(v):
        # FD: J·v ≈ (F(P + ε v) − F(P)) / ε   where F is computed on inner block
        v_inner = v.reshape((G_INNER,)*3)
        P_pert = P_full_at_step.copy()
        P_pert[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] += EPS_FD * v_inner
        # Re-apply boundary (so the perturbation respects BCs)
        P_pert = set_boundary_tail(P_pert)
        F_pert = F_inner(P_pert)
        return (F_pert - F0_vec) / EPS_FD
    return LinearOperator((n_inner, n_inner), matvec=matvec, dtype=np.float64)


for nstep in range(1, N_NEWTON+1):
    t0 = time.time()
    F0_vec = F_inner(P_full)
    F_norm = np.max(np.abs(F0_vec))
    print(f'  Newton {nstep}: ||F||_inf = {F_norm:.3e}')
    if F_norm < 1e-12:
        print('  CONVERGED'); break
    J_op = make_matvec_op(P_full.copy(), F0_vec)
    # Solve J·δ = −F
    delta, info = lgmres(J_op, -F0_vec, atol=max(F_norm * 0.05, 1e-9),
                          maxiter=LGMRES_MAXITER, inner_m=LGMRES_INNER_M)
    print(f'    LGMRES info={info}, ||δ||_inf={np.max(np.abs(delta)):.3e}')
    # Apply update
    delta_3D = delta.reshape((G_INNER,)*3)
    P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] += delta_3D
    # Clip + reapply BC + symmetrize
    P_full = np.clip(P_full, 1e-12, 1-1e-12)
    P_full = set_boundary_tail(P_full)
    P_full = symmetrize(P_full)
    P_full = set_boundary_tail(P_full)
    # Diagnostics
    P_in = P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_FR = float(np.sqrt(np.sum((P_in - P_FR_in)**2 * Wd_inner)))
    omR2 = weighted_R2_T(P_in)
    # Re-evaluate F at the new P for next iteration's records
    F_new = F_inner(P_full)
    F_new_norm = np.max(np.abs(F_new))
    ferr_history.append(F_new_norm); omR2_history.append(omR2); d_FR_history.append(d_FR)
    print(f'    after update: ||F||_inf={F_new_norm:.3e}  1-R²={omR2:.3e}  d_FR={d_FR:.4e}  ({time.time()-t0:.1f}s)')


# ============ PLOTS ============
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
nsteps_total = len(ferr_history) - 1
iters = list(range(nsteps_total + 1))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
labels = ['Picard' if i <= N_PICARD_WARMUP else 'Newton'
           for i in range(len(ferr_history))]

ax = axes[0]
ax.semilogy(iters, [max(f, 1e-30) for f in ferr_history], 'o-', lw=2, ms=7)
ax.axvline(N_PICARD_WARMUP + 0.5, color='red', ls='--', alpha=0.5, label='Newton starts')
ax.set_xlabel('step'); ax.set_ylabel('||F||_∞'); ax.set_title('||F = Φ(P)−P||_∞')
ax.grid(True, ls=':', alpha=0.5); ax.legend()

ax = axes[1]
ax.semilogy(iters, [max(r, 1e-30) for r in omR2_history], 's-', lw=2, ms=7, color='C1')
ax.axvline(N_PICARD_WARMUP + 0.5, color='red', ls='--', alpha=0.5)
ax.set_xlabel('step'); ax.set_ylabel('1−R²(T*)'); ax.set_title('Jensen wedge')
ax.grid(True, ls=':', alpha=0.5)

ax = axes[2]
ax.semilogy(iters, [max(d, 1e-30) for d in d_FR_history], '^-', lw=2, ms=7, color='C2')
ax.axvline(N_PICARD_WARMUP + 0.5, color='red', ls='--', alpha=0.5)
ax.set_xlabel('step'); ax.set_ylabel('d_FR'); ax.set_title('distance to FR')
ax.grid(True, ls=':', alpha=0.5)

plt.suptitle(f'Newton-Krylov on (u_1, Σ̂, δ̂), G={G_FULL}, γ={GAMMA}.  '
              f'{N_PICARD_WARMUP} Picard warmup → {N_NEWTON} Newton steps',
              fontsize=11.5, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/AB_newton_trajectory.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'\nwrote {FIG}/AB_newton_trajectory.png')
