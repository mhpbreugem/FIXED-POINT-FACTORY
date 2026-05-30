"""Two deliverables in one run (consistent kernel co-area operator, K=3, tau=2, gamma=0.1, G=17):

  (A) CONTOUR PLOT of price-surface slices over the learning dynamics:
      iter 0 (no-learning IC) -> iter 1 = Phi(NL) -> iter 2 = Phi(Phi(NL)) -> nailed fixed point.
      Visualizes how the price function evolves under one Picard step at a time, and contrasts
      with the equilibrium reached by Newton.

  (B) PICARD STABILITY of the fixed point: compute the spectral radius of Phi'(P*) by
      Arnoldi (matrix-free FD JVP) on the symmetric subspace. If rho > 1 -> P* is a SADDLE
      of the Picard map, NOT reachable by undamped Picard (only by damped Picard or Newton).
      Reports rho, the largest eigenvalues, and the verdict.

Operator + solver: reuses the consistent kernel co-area op (reznsrc/contour_K3_halo.smooth)
and the SymReducer3+newton_krylov pattern from k3_coarea_2dsweep / k3_hfree_fast."""
import json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(os.path.dirname(HERE), 'k3_coarea_2dsweep')
sys.path.insert(0, SWEEP)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from scipy.sparse.linalg import LinearOperator, eigs

UMAX = 4.0; PAD = 2; G = 17; TAU = 2.0; GAMMA = 0.1; C_H = 0.45
du = 2*UMAX/(G-1); h = C_H*np.sqrt(du)
uf = np.array([-UMAX + (q-PAD)*du for q in range(G + 2*PAD)])
lo, hi = PAD, PAD + G; slc = (slice(lo, hi),)*3
tv = np.full(3, TAU); gv = np.full(3, GAMMA); W = np.full(3, 1.0)

# halo = no-learning IC over full padded grid
halo = init_no_learning_K3(uf, tv, gv, W)
P_NL = halo.copy()  # full grid (halo + inner = NL everywhere)

def apply_phi(P_full):
    return phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, W, h)

# (A) Learning iterations
print("Applying Phi twice to NL...", flush=True)
P_iter1 = apply_phi(P_NL); P_iter2 = apply_phi(P_iter1)

# Nail the fixed point via NK from NL inner
print("Nailing the fixed point with Newton-Krylov...", flush=True)
def resid(x):
    P = halo.copy(); P[slc] = x.reshape((G,)*3); return (apply_phi(P) - P)[slc].ravel()
x0 = halo[slc].ravel().copy()
try:
    sol = newton_krylov(resid, x0, f_tol=1e-9, maxiter=150, method='lgmres')
    conv = True
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); conv = False
Finf = float(np.max(np.abs(resid(sol))))
P_star_full = halo.copy(); P_star_full[slc] = sol.reshape((G,)*3)
print(f"  nailed: ||F||={Finf:.2e} conv={conv}", flush=True)

# CONTOUR PLOT — slice at u1 = middle interior
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mid = lo + G // 2
ui = uf[lo:hi]
def slice_inner(Pf):
    return Pf[mid, lo:hi, lo:hi]  # (u2, u3)
slices = [(slice_inner(P_NL), 'iter 0  (no-learning IC)'),
          (slice_inner(P_iter1), 'iter 1 = Φ(NL)'),
          (slice_inner(P_iter2), 'iter 2 = Φ²(NL)'),
          (slice_inner(P_star_full), 'fixed point  (Newton)')]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.0), dpi=140)
levels = np.linspace(0.05, 0.95, 19)
for ax, (S, title) in zip(axes, slices):
    cs = ax.contourf(ui, ui, S.T, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(ui, ui, S.T, levels=[0.5], colors='k', linewidths=1.5)
    ax.set_xlabel('u₂'); ax.set_ylabel('u₃' if ax is axes[0] else '')
    ax.set_title(title); ax.set_aspect('equal')
plt.colorbar(cs, ax=axes, shrink=0.8, label='price P')
plt.suptitle(f'K=3 learning dynamics: P(u₂,u₃) at u₁=0 slice  (τ={TAU}, γ={GAMMA}, consistent operator, G={G})\n'
             f'one Picard step at a time vs the Newton-nailed fixed point (‖F‖={Finf:.1e})', weight='bold', fontsize=11, y=1.02)
plt.savefig(os.path.join(HERE, 'picard_iterations_contour.png'), dpi=140, bbox_inches='tight')
plt.close()
print('wrote picard_iterations_contour.png', flush=True)

# Also save per-iteration residuals (Φ−P)
res_data = {'tau': TAU, 'gamma': GAMMA, 'G': G, 'kernel_h': float(h), 'nailed_Finf': Finf,
            'iter0_to_iter1_change_inf': float(np.max(np.abs((P_iter1 - P_NL)[slc]))),
            'iter1_to_iter2_change_inf': float(np.max(np.abs((P_iter2 - P_iter1)[slc]))),
            'iter2_to_FP_dist_inf': float(np.max(np.abs((P_star_full - P_iter2)[slc])))}

# (B) Picard stability — Jacobian of Phi at the fixed point, largest eigenvalues
# Use the FULL inner block (G^3 = 4913) but matrix-free; eigs computes top-k by Arnoldi.
print("Building Jacobian-vector product and computing spectral radius (Arnoldi)...", flush=True)
N = G**3
P_star_inner = P_star_full[slc].copy().ravel()
eps_fd = 1e-6

def Jv(v):
    """matvec: J_Phi @ v  by central FD on Phi."""
    Pp = halo.copy(); Pp[slc] = (P_star_inner + eps_fd * v).reshape((G,)*3)
    Pm = halo.copy(); Pm[slc] = (P_star_inner - eps_fd * v).reshape((G,)*3)
    return ((apply_phi(Pp) - apply_phi(Pm))[slc].ravel()) / (2*eps_fd)

t = time.time()
L = LinearOperator((N, N), matvec=Jv, dtype=np.float64)
try:
    vals, _ = eigs(L, k=8, which='LM', tol=1e-5, maxiter=300)
except Exception as e:
    print("eigs failed:", e); vals = np.array([np.nan])
rho = float(np.max(np.abs(vals)))
print(f"  spectral radius rho(Phi') = {rho:.4f}  ({time.time()-t:.0f}s)", flush=True)
print("  top eigenvalues |λ|:", [f"{abs(v):.4f}" for v in sorted(vals, key=lambda z: -abs(z))], flush=True)

res_data.update({'spectral_radius': rho,
                 'top_eigenvalues_abs': sorted([float(abs(v)) for v in vals], reverse=True),
                 'top_eigenvalues_complex': [str(v) for v in sorted(vals, key=lambda z: -abs(z))],
                 'picard_stable_if_rho_less_1': bool(rho < 1.0),
                 'verdict_picard_reachability': (
                     f'rho={rho:.3f} > 1 -> the fixed point is a SADDLE of the Picard map: '
                     'undamped Picard CANNOT reach it (will oscillate/diverge). Damped Picard with '
                     f'omega < {2/(1+rho):.3f} (so the damped Jacobian has spectral radius <1) CAN '
                     'converge to it. Newton finds it directly (it is a root, regardless of dynamics).'
                 ) if rho >= 1 else (
                     f'rho={rho:.3f} < 1 -> the fixed point is a Picard ATTRACTOR: '
                     'undamped Picard will converge to it from a neighborhood.')})
json.dump(res_data, open(os.path.join(HERE, 'picard_dynamics.json'), 'w'), indent=2)
print('wrote picard_dynamics.json', flush=True)
print('VERDICT:', res_data['verdict_picard_reachability'])
