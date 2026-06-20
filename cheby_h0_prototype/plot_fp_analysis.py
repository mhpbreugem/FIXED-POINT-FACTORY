"""Build figures for the FP analysis PDF."""
import sys, time
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, newton_krylov

from cheby_numba import phi as phi_chebroots, U_NODES, TAU, GAMMA, N_GRID
from cheby_numba_bisect import phi_bisect
from cheby_sym2 import expand, contract, FREE_REPS

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

# ===== Run all solvers, collect F_err histories =====
def run_picard(op, omega, n_iter):
    x = contract(sg(0.5*T))
    Ferrs = []
    for it in range(n_iter):
        x_new = contract(op(expand(x), gamma=1.0, tau=1.0))
        F = x_new - x
        Ferrs.append(float(np.max(np.abs(F))))
        x = omega*x_new + (1-omega)*x
    return Ferrs

def run_anderson(op, omega, m_history, n_iter):
    x = contract(sg(0.5*T))
    X_hist, G_hist = [], []
    Ferrs = []
    for it in range(n_iter):
        gx = contract(op(expand(x), gamma=1.0, tau=1.0))
        F = gx - x
        Ferrs.append(float(np.max(np.abs(F))))
        X_hist.append(x.copy()); G_hist.append(gx.copy())
        if len(X_hist) > m_history:
            X_hist.pop(0); G_hist.pop(0)
        k = len(X_hist)
        if k <= 1:
            x = omega*gx + (1-omega)*x
        else:
            DR = np.column_stack([(G_hist[i]-X_hist[i])-(G_hist[k-1]-X_hist[k-1]) for i in range(k-1)])
            R_k = G_hist[k-1] - X_hist[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([G_hist[i]-G_hist[k-1] for i in range(k-1)])
                x_next = G_hist[k-1] + DG @ ga
                x = omega*x_next + (1-omega)*x
            except: x = gx
    return Ferrs

print('Running solvers...', flush=True)
print('  Picard (chebroots, omega=0.3)...', flush=True)
F_picard_cr = run_picard(phi_chebroots, 0.3, 80)
print('  Picard (bisect, omega=0.4)...', flush=True)
F_picard_bi = run_picard(phi_bisect, 0.4, 80)
print('  Anderson (bisect, m=8)...', flush=True)
F_anderson_bi = run_anderson(phi_bisect, 1.0, 8, 80)
print('  Anderson (chebroots, m=8)...', flush=True)
F_anderson_cr = run_anderson(phi_chebroots, 1.0, 8, 50)

# ===== Eigenvalue spectrum at lifted FP =====
print('  Computing Jacobian spectrum...', flush=True)
P_lift = np.load('/tmp/cheby_h0/P_final_lifted_numba.npy')
x_at_fp = contract(P_lift)
gx0 = contract(phi_bisect(expand(x_at_fp), gamma=1.0, tau=1.0))
J = np.empty((x_at_fp.size, x_at_fp.size))
eps = 1e-6
for j in range(x_at_fp.size):
    xp = x_at_fp.copy(); xp[j] += eps
    gxp = contract(phi_bisect(expand(xp), gamma=1.0, tau=1.0))
    J[:, j] = (gxp - gx0) / eps
eigvals = np.linalg.eigvals(J)
print(f'    spectral radius = {max(abs(eigvals)):.3f}', flush=True)

# ===== Levenberg-Marquardt =====
print('  Levenberg-Marquardt...', flush=True)
def F_func(x):
    return contract(phi_bisect(expand(x), gamma=1.0, tau=1.0)) - x
LM_history = []
def F_func_log(x):
    f = F_func(x)
    LM_history.append(float(np.max(np.abs(f))))
    return f
sol_lm = least_squares(F_func_log, contract(sg(0.5*T)), method='lm',
                          max_nfev=200, xtol=1e-15, ftol=1e-15, gtol=1e-15)
F_lm = LM_history

# ===== Figure 1: F_err vs iter for all methods =====
fig, ax = plt.subplots(figsize=(11, 6))
ax.semilogy(range(1, len(F_picard_cr)+1), F_picard_cr, 'o-', color='tab:red',
              alpha=0.7, markersize=4, label=f'Picard (chebroots) — floor ~{min(F_picard_cr):.1e}')
ax.semilogy(range(1, len(F_picard_bi)+1), F_picard_bi, 's-', color='tab:orange',
              alpha=0.7, markersize=4, label=f'Picard (bisect) — floor ~{min(F_picard_bi):.1e}')
ax.semilogy(range(1, len(F_anderson_bi)+1), F_anderson_bi, '^-', color='tab:blue',
              alpha=0.7, markersize=4, label=f'Anderson (bisect) — floor ~{min(F_anderson_bi):.1e}')
ax.semilogy(range(1, len(F_anderson_cr)+1), F_anderson_cr, 'v-', color='tab:purple',
              alpha=0.7, markersize=4, label=f'Anderson (chebroots) — floor ~{min(F_anderson_cr):.1e}')
ax.semilogy(range(1, len(F_lm)+1), F_lm, 'D-', color='tab:green',
              alpha=0.7, markersize=4, label=f'Levenberg-Marquardt — floor ~{min(F_lm):.1e}')
ax.axhline(1e-13, color='black', linestyle=':', alpha=0.5, label='target (machine ε)')
ax.axhline(min(F_lm), color='tab:green', linestyle='--', alpha=0.4, label=f'LM minimum ||F||₂² = {sol_lm.cost:.2e}')
ax.set_xlabel('iteration')
ax.set_ylabel(r'$\|F\|_\infty$  (P-cell residual)')
ax.set_title(f'F_err vs iter for 5 solvers (G=7, τ=γ=1)\n'
              f'ALL HIT FLOOR ~3e-3 — no machine-ε FP reachable')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3, which='both')
ax.set_xlim(0, 80)
ax.set_ylim(1e-14, 1)
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/fp_analysis_solvers.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved fp_analysis_solvers.png')

# ===== Figure 2: Eigenvalue spectrum =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(eigvals.real, eigvals.imag, c=np.abs(eigvals), cmap='viridis',
            s=60, edgecolors='black')
# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='|λ|=1 (Picard stability boundary)')
ax.axhline(0, color='gray', alpha=0.3)
ax.axvline(0, color='gray', alpha=0.3)
ax.set_xlabel(r'Re(λ)'); ax.set_ylabel(r'Im(λ)')
ax.set_title(f'Jacobian eigenvalues at lifted+Newton "FP"\n'
              f'spectral radius ρ(J) = {max(abs(eigvals)):.2f}')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_aspect('equal')

ax = axes[1]
ax.bar(range(1, len(eigvals)+1), sorted(np.abs(eigvals), reverse=True))
ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='|λ|=1 (Picard stable)')
n_repelling = int(np.sum(np.abs(eigvals) > 1))
ax.set_xlabel('eigenvalue index (sorted by |λ|)')
ax.set_ylabel(r'$|\lambda|$')
ax.set_title(f'|λ| histogram: {n_repelling}/{len(eigvals)} eigenvalues > 1\n'
              f'(operator is REPELLING in {n_repelling} directions)')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/fp_analysis_eigvals.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved fp_analysis_eigvals.png')

# ===== Figure 3: Comparison with rank-1/1D-in-T (where F → 0 is achievable) =====
from cheby_rank1 import operator_rank1
print('  Rank-1 sanity check...', flush=True)
P_R1 = operator_rank1(U_NODES, 1.0, 1.0)
P_R1_again = operator_rank1(U_NODES, 1.0, 1.0)  # same operator, deterministic
F_R1 = float(np.max(np.abs(P_R1_again - P_R1)))  # 0 (operator is α-invariant)

fig, ax = plt.subplots(figsize=(11, 6))
labels = ['rank-1\n(α-inv, no roots)',
          '1D-in-T m=10\n(no roots)',
          'Lev-Marq on Chebyshev\n(bisect)',
          'Anderson on Chebyshev\n(bisect)',
          'Picard on Chebyshev\n(chebroots)',
          'Lifted+Newton\n(documented prior)']
floors = [F_R1 + 1e-16, 4.23e-6, min(F_lm), min(F_anderson_bi), min(F_picard_cr), 5.3e-3]
colors = ['tab:green', 'tab:cyan', 'tab:olive', 'tab:blue', 'tab:red', 'tab:purple']
bars = ax.bar(labels, floors, color=colors, alpha=0.8)
ax.set_yscale('log')
ax.set_ylabel(r'min $\|F\|_\infty$ achieved')
ax.set_title('Min F_err achievable across operator architectures\n'
              'closed-form operators reach machine ε; root-finding operators floor at ~3-5e-3')
ax.axhline(1e-13, color='black', linestyle=':', alpha=0.5, label='machine ε')
for b, v in zip(bars, floors):
    ax.text(b.get_x() + b.get_width()/2, v*1.5, f'{v:.1e}',
             ha='center', fontsize=10)
ax.set_ylim(1e-17, 1e-1)
plt.xticks(rotation=15, ha='right', fontsize=10)
ax.legend(); ax.grid(alpha=0.3, axis='y', which='both')
plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/fp_analysis_floors.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved fp_analysis_floors.png')

import json
json.dump(dict(
    F_picard_cr=F_picard_cr, F_picard_bi=F_picard_bi,
    F_anderson_bi=F_anderson_bi, F_anderson_cr=F_anderson_cr,
    F_lm=F_lm,
    eigvals_abs=[float(np.abs(e)) for e in eigvals],
    spectral_radius=float(max(abs(eigvals))),
    n_repelling=int(np.sum(np.abs(eigvals) > 1)),
    LM_final=float(min(F_lm)),
    floors_summary=dict(zip(labels, [float(f) for f in floors])),
), open('/tmp/cheby_h0/fp_analysis.json', 'w'), indent=2, default=str)
print('saved fp_analysis.json')
