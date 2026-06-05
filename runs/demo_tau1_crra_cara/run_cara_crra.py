"""MIZN demo run: dense ('pure') Newton on the K=3 kernel co-area operator.

Setup:
  - τ = 1 (all agents, low signal precision)
  - CRRA case: γ = 1 (all agents)
  - CARA case: γ = 100 (all agents, effectively CARA limit)
  - G_inner = 9 (low resolution)
  - kernel h = 0.45 * sqrt(du)  (the standard rule from k3_coarea_sweep)

Operator: phi_K3_halo_smooth from contour_K3_halo.py (kernel co-area, h > 0).
This is the SAME operator that nails γ=0.1, τ=2 to ||F|| < 1e-8 in the prior
session. With h>0 it is smooth and well-conditioned; dense Newton converges
quadratically from a Picard-preconditioned warm start.

Solver: dense Newton with forward-difference Jacobian, Armijo line search.
'pure' = explicit J, direct LU solve (NOT Newton-Krylov, NOT Anderson).

Produces:
  - Convergence trajectories (||F||_∞ vs iter) for both cases
  - Final FP slices P(u_2, u_3) at u_1 = 0
  - Comparison summary (slope, deficit, d_FR for each case)
"""
import os, sys, time, json
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
from code.contour_K3_halo import phi_K3_halo_smooth, init_no_learning_K3

TAU = 1.0
UMAX = 4.0
G_INNER = 9; PAD = 2; G_FULL = G_INNER + 2*PAD
DU = 2*UMAX / (G_INNER - 1)
H_KERNEL = 0.45 * np.sqrt(DU)
u_full = np.array([-UMAX + (q - PAD)*DU for q in range(G_FULL)])
LO, HI = PAD, PAD + G_INNER
slc = (slice(LO, HI),)*3

print(f'MIZN demo: K=3 kernel co-area + pure (dense) Newton')
print(f'  τ = {TAU}, G_inner = {G_INNER}, du = {DU:.4f}, h_kernel = {H_KERNEL:.4f}')

W = np.full(3, 1.0)

def F_residual(x, gamma_val):
    """Residual F(P_inner) = phi(P) - P on inner block."""
    tau_vec = np.full(3, TAU); gam_vec = np.full(3, gamma_val)
    halo = init_no_learning_K3(u_full, tau_vec, gam_vec, W)  # halo = no-learning ansatz
    P_full = halo.copy()
    P_full[slc] = x.reshape((G_INNER,)*3)
    return (phi_K3_halo_smooth(P_full, u_full, LO, HI, tau_vec, gam_vec, W, H_KERNEL) - P_full)[slc].ravel()

def metrics(P_inner):
    ui = u_full[LO:HI]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P_inner, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0]*T.ravel() + a[1]
    defi = float(np.sum((y-pr)**2) / max(np.sum((y-y.mean())**2), 1e-30))
    P_FR = 1.0 / (1.0 + np.exp(-T))
    d_FR = float(np.sqrt(np.mean((P_inner - P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(gamma_val, label, max_iter=12, tol=1e-10, eps_fd=1e-6, verbose=True):
    """Pure (dense) Newton with forward-difference Jacobian + Armijo line search."""
    tau_vec = np.full(3, TAU); gam_vec = np.full(3, gamma_val)
    # IC: no-learning ansatz, then 5 Picard iters to get into basin
    P_full = init_no_learning_K3(u_full, tau_vec, gam_vec, W)
    for _ in range(5):
        P_full = 0.7*P_full + 0.3*phi_K3_halo_smooth(P_full, u_full, LO, HI, tau_vec, gam_vec, W, H_KERNEL)
    x = P_full[slc].ravel().copy()
    N = x.size
    F = F_residual(x, gamma_val)
    F_norm = float(np.max(np.abs(F)))
    if verbose:
        print(f'\n=== Pure Newton: {label} (γ={gamma_val}) ===')
        print(f'  unknowns: {N}, initial ||F||_∞ = {F_norm:.3e}')

    history = {'iter': [0], 'F_norm': [F_norm], 'wall_time': [0.0]}
    metrics_history = [metrics(x.reshape((G_INNER,)*3))]
    t_start = time.time()

    for it in range(1, max_iter+1):
        # Build Jacobian via forward difference
        t_build = time.time()
        J = np.empty((N, N))
        for j in range(N):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (F_residual(xp, gamma_val) - F) / eps_fd
        t_build = time.time() - t_build

        # Solve linear system
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)

        # Armijo line search
        alpha = 1.0
        best = (None, 1e100)
        for _ in range(20):
            xn = np.clip(x + alpha*dx, 1e-12, 1-1e-12)
            Fn = F_residual(xn, gamma_val); norm_n = float(np.max(np.abs(Fn)))
            if norm_n < best[1]: best = (xn, norm_n, Fn, alpha)
            if norm_n < (1 - 0.5*alpha) * F_norm:
                break
            alpha *= 0.5
            if alpha < 1e-12: break

        xn, F_norm_new, F_new, alpha = best
        wall = time.time() - t_start
        if verbose:
            print(f'  it {it:2d}  ||F||_∞ = {F_norm_new:.3e}  α = {alpha:.3g}  '
                  f'(J build {t_build:.1f}s, total {wall:.1f}s)')

        x = xn; F = F_new; F_norm = F_norm_new
        history['iter'].append(it)
        history['F_norm'].append(F_norm)
        history['wall_time'].append(wall)
        metrics_history.append(metrics(x.reshape((G_INNER,)*3)))

        if F_norm < tol:
            if verbose: print(f'  CONVERGED to tol {tol:.0e}')
            break

    final_metrics = metrics(x.reshape((G_INNER,)*3))
    return dict(
        x_final=x, F_norm_final=F_norm,
        iters=history['iter'][-1],
        history=history, metrics_history=metrics_history,
        final_metrics=final_metrics, gamma=gamma_val, label=label
    )

# ===== Run both =====
results = {}
print('\n' + '='*60)
print('PHASE 1: CRRA at γ=1')
print('='*60)
results['crra'] = dense_newton(gamma_val=1.0, label='CRRA γ=1', max_iter=8, tol=1e-9)

print('\n' + '='*60)
print('PHASE 2: CARA (γ→∞, using γ=100)')
print('='*60)
results['cara'] = dense_newton(gamma_val=100.0, label='CARA γ=100', max_iter=8, tol=1e-9)

# ===== Save =====
np.save(os.path.join(HERE, 'P_crra.npy'), results['crra']['x_final'].reshape((G_INNER,)*3))
np.save(os.path.join(HERE, 'P_cara.npy'), results['cara']['x_final'].reshape((G_INNER,)*3))

# JSON summary (no numpy arrays)
def to_json(r):
    return dict(
        label=r['label'], gamma=r['gamma'],
        F_norm_final=r['F_norm_final'], iters=r['iters'],
        final_metrics=r['final_metrics'],
        history={'iter': r['history']['iter'], 'F_norm': r['history']['F_norm'], 'wall_time': r['history']['wall_time']},
        metrics_per_iter=r['metrics_history']
    )
summary = {
    'config': dict(TAU=TAU, G_inner=G_INNER, U_max=UMAX, h_kernel=H_KERNEL, solver='dense Newton'),
    'crra': to_json(results['crra']),
    'cara': to_json(results['cara']),
}
json.dump(summary, open(os.path.join(HERE, 'summary.json'), 'w'), indent=2, default=str)
print('\nResults saved.')
print(f"\nCRRA γ=1: slope_T={results['crra']['final_metrics']['slope_T']:.4f}, "
      f"deficit={results['crra']['final_metrics']['deficit']:.4f}, "
      f"d_FR={results['crra']['final_metrics']['d_FR']:.4f}, "
      f"||F||={results['crra']['F_norm_final']:.2e}")
print(f"CARA γ=100: slope_T={results['cara']['final_metrics']['slope_T']:.4f}, "
      f"deficit={results['cara']['final_metrics']['deficit']:.4f}, "
      f"d_FR={results['cara']['final_metrics']['d_FR']:.4f}, "
      f"||F||={results['cara']['F_norm_final']:.2e}")
