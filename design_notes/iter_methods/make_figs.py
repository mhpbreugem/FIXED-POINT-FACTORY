"""Figures for the 'does NK go away?' PDF."""
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = '/tmp/iter_pdf/figs'

# ===== Fig 1: Why NK existed in the first place =====
# Side-by-side: dense Newton (forms full J) vs Newton-Krylov (matrix-free)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.set_title('Dense Newton\n(form J explicitly, solve J·Δ = -F)', fontsize=12)
# Show a big NxN matrix block
N = 729
ax.add_patch(mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='tab:blue', alpha=0.3, edgecolor='black'))
ax.text(0.5, 0.5, f'J: {N}×{N}\n= {N*N:,} entries\n\n{N} Φ evaluations to build\nthen LU solve O({N}³)',
        ha='center', va='center', fontsize=11)
ax.text(0.5, 0.02, 'For G=9 grid (prior session): N=729 → ~531k Jacobian entries,\n~3 GB memory (dense), Jacobian build cost prohibitive.',
        ha='center', va='top', fontsize=9, style='italic')
ax.set_xlim(0, 1); ax.set_ylim(-0.15, 1); ax.axis('off')

ax = axes[1]
ax.set_title('Newton-Krylov (LGMRES)\n(matrix-free: only need J·v for vectors v)', fontsize=12)
# Show a sparse-ish small subspace
ax.add_patch(mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='lightgray', alpha=0.5, edgecolor='black'))
for k in range(8):
    y = 0.85 - k*0.09
    ax.plot([0.15, 0.85], [y, y], 'b-', lw=2, alpha=0.7)
ax.text(0.5, 0.02, 'Build Krylov subspace iteratively: J·v_1, J·v_2, …\nEach J·v costs ONE Φ-evaluation (finite-difference).\nNo full J stored.',
        ha='center', va='top', fontsize=9, style='italic')
ax.text(0.5, 0.5, 'Each row: one Krylov vector\n(typically 10-50)\n\nInner LGMRES tol → "inexact Newton"',
        ha='center', va='center', fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(-0.15, 1); ax.axis('off')

plt.suptitle('NK was invented BECAUSE forming J was too expensive for big problems.\n'
              'Question: now that our problem is small (symmetric Chebyshev), is NK still needed?',
              fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/01_why_NK.png', dpi=130, bbox_inches='tight')
plt.close()
print('01_why_NK.png')

# ===== Fig 2: Problem size comparison =====
fig, ax = plt.subplots(figsize=(11, 6))
labels = ['Spline\nG=9\n(prior session)',
           'Spline\nG=13',
           'Spline\nG=17',
           'Chebyshev\nN=8\nno sym',
           'Chebyshev\nN=12\nno sym',
           'Cheby +\nS₃ N=12',
           'Cheby +\nS₃×Z₂\nN=12',
           'Cheby +\nS₃×Z₂\nN=24']
n_unknowns = [729, 2197, 4913, 729, 2197, 455, 228, 1462]
colors = ['tab:red']*3 + ['tab:orange']*2 + ['tab:olive', 'tab:green', 'tab:cyan']
bars = ax.bar(labels, n_unknowns, color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, n_unknowns):
    ax.text(bar.get_x()+bar.get_width()/2, val+100, f'{val:,}', ha='center', fontsize=10)
ax.axhline(500, color='black', linestyle='--', alpha=0.5)
ax.text(7.5, 520, 'Dense Newton "easy" threshold (~500)', fontsize=10, ha='right', style='italic')
ax.set_ylabel('Number of unknowns (= Jacobian side length)')
ax.set_title('Problem size: spline approaches vs Chebyshev + symmetry\n'
              'Below ~500: dense Newton is cheap. Above: NK was the only option.')
ax.set_ylim(0, 5500)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/02_problem_size.png', dpi=130, bbox_inches='tight')
plt.close()
print('02_problem_size.png')

# ===== Fig 3: Per-iteration cost breakdown =====
fig, ax = plt.subplots(figsize=(11, 6))
methods = ['Picard\n(no Jacobian)',
            'Anderson(m=8)\n(approx Jacobian)',
            'Newton-Krylov\n(matrix-free)',
            'Dense Newton\n(form J explicitly)']
# Per outer iter: Phi-eval count for each method at N=228 (Cheby+sym)
n_unknowns = 228
phi_per_iter = [1, 1, 25, n_unknowns]  # Picard 1, Anderson 1, NK ~25 inner (rough avg), Dense N+1
linsolve_per_iter_ops = [0, n_unknowns*8, 25*n_unknowns, n_unknowns**3/3]  # rough

# Total cost (Phi-evals dominate since each is ~ms-seconds)
costs = phi_per_iter
bars = ax.bar(methods, costs, color=['tab:blue','tab:cyan','tab:orange','tab:red'], alpha=0.7, edgecolor='black')
for bar, val, m in zip(bars, costs, methods):
    label = f'{val} Φ-eval' + ('s' if val > 1 else '')
    ax.text(bar.get_x()+bar.get_width()/2, val+5, label, ha='center', fontsize=11)
ax.set_ylabel('Φ-evaluations per outer iteration\n(dominant cost; each Φ is expensive)')
ax.set_title('Per-iteration cost at problem size N=228 (Cheby + S₃×Z₂)\n'
              'Dense Newton does N+1 evaluations to build J, then SOLVE is free in float64')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{OUT}/03_per_iter_cost.png', dpi=130, bbox_inches='tight')
plt.close()
print('03_per_iter_cost.png')

# ===== Fig 4: Convergence rates compared =====
fig, ax = plt.subplots(figsize=(11, 6))
# Hypothetical convergence trajectories
iters = np.arange(0, 25)
# Picard: linear, contraction rate 0.7
picard = 1.0 * 0.7**iters
# Anderson: superlinear ~rate 1.2
anderson = 1.0 * np.array([0.7**min(i, 3) for i in iters]) * np.exp(-0.1*iters**1.5)
# NK: superlinear, ~1.7
nk = np.array([1.0 if i==0 else 1.0 * (1e-2)**(min(i,1.7)*0.3) for i in iters])
# Dense Newton: quadratic
def quad_traj(n=20):
    x = 1.0
    out = [x]
    for _ in range(n):
        x = x**2
        out.append(x)
    return np.array(out[:n+1])
dense_newton = quad_traj(len(iters)-1)

ax.semilogy(iters, picard, 'o-', label='Picard (linear, ~0.7×/iter)', color='tab:blue')
ax.semilogy(iters, anderson, 's-', label='Anderson (superlinear)', color='tab:cyan')
ax.semilogy(iters, nk, '^-', label='Newton-Krylov (superlinear, inexact)', color='tab:orange')
ax.semilogy(iters, dense_newton, 'D-', label='Dense Newton (QUADRATIC)', color='tab:red', lw=2)
ax.axhline(1e-15, color='black', linestyle=':', label='machine precision')
ax.set_xlabel('outer iteration')
ax.set_ylabel('‖F‖∞ (log)')
ax.set_title('Convergence rates: schematic\n'
              'Dense Newton converges quadratically — doubles correct digits each step.')
ax.set_ylim(1e-17, 2)
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/04_convergence_rates.png', dpi=130, bbox_inches='tight')
plt.close()
print('04_convergence_rates.png')

# ===== Fig 5: How dense Newton works (one-iter walkthrough) =====
fig, ax = plt.subplots(figsize=(11, 6))
# Build a "diagram" of dense Newton step
ax.text(0.5, 0.95, 'Dense Newton: one outer iteration', ha='center', fontsize=14, weight='bold', transform=ax.transAxes)

steps = [
    ('1. Current iterate', 'P (= α, b_{ijk}) — small array of 228 numbers'),
    ('2. Compute residual', 'F(P) = Φ(P) − P     ← 1 Φ evaluation'),
    ('3. Build Jacobian J_{ij} = ∂F_i/∂P_j',
     'Forward-difference: J[:,j] = (F(P + ε·eⱼ) − F(P))/ε for j=1..228\n  ⟹ 228 Φ evaluations'),
    ('4. Solve J·Δ = −F', 'np.linalg.solve(J, -F)   ← O(228³) ≈ 12M flops ≈ instant'),
    ('5. Line search', 'Try P + α·Δ for α ∈ {1, ½, ¼, …}, keep best'),
    ('6. Update', 'P ← P + α_best·Δ\n‖F‖∞ → typically (‖F_prev‖∞)² near the FP — QUADRATIC convergence')
]
for i, (label, body) in enumerate(steps):
    y = 0.83 - i*0.13
    ax.text(0.05, y, label, fontsize=12, weight='bold', transform=ax.transAxes, color='tab:blue')
    ax.text(0.05, y-0.04, body, fontsize=10, transform=ax.transAxes, family='monospace')

ax.text(0.5, 0.02, 'Total cost per outer iter: 229 Φ-evals + tiny linsolve.\n'
                     'For Cheby + sym at N=12 (228 unknowns): completely feasible in float64.',
        ha='center', va='bottom', fontsize=11, style='italic', transform=ax.transAxes,
        bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round'))
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/05_dense_newton_workflow.png', dpi=130, bbox_inches='tight')
plt.close()
print('05_dense_newton_workflow.png')

# ===== Fig 6: When to use each method (decision flowchart-style) =====
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_title('Decision: which iteration method?', fontsize=14, weight='bold')

# Three regions in a grid
sizes = ['<300', '300-3000', '>3000']
for i, size in enumerate(sizes):
    x = 0.1 + i*0.30
    ax.add_patch(mpatches.Rectangle((x, 0.55), 0.25, 0.4, facecolor=['lightgreen','khaki','lightcoral'][i], alpha=0.5, edgecolor='black'))
    ax.text(x+0.125, 0.92, f'N_unk: {size}', ha='center', fontsize=11, weight='bold')
    method = ['DENSE NEWTON ★\n(quadratic, simple)',
               'Hybrid: Anderson → DENSE NEWTON\n(Anderson to get close, dense\nNewton for the last 4-5 digits)',
               'Newton-Krylov OR Anderson\n(dense J too costly)'][i]
    ax.text(x+0.125, 0.72, method, ha='center', fontsize=10)
    note = ['e.g. Cheby + S₃×Z₂\nat N≤14',
             'e.g. Cheby + S₃×Z₂ at N=16-20\nor non-symmetric Cheby at N=12',
             'e.g. unrestricted spline G≥13'][i]
    ax.text(x+0.125, 0.59, note, ha='center', fontsize=9, style='italic')

# Bottom: secondary considerations
ax.text(0.5, 0.42, 'Secondary considerations:', ha='center', fontsize=12, weight='bold')
considerations = [
    '• If Jacobian is well-conditioned (verified σ_min not tiny) → Newton works; if ill-conditioned → Levenberg-Marquardt or trust-region',
    '• If function has discrete jumps (spline contour topology) → NK plateaus; switch to Cheby first',
    '• Anderson is always cheap to add on top of Picard as a free warm-starter',
    '• For γ-continuation, ONE good Newton step from previous-γ FP is usually enough (warm-start so good that quadratic kicks in)',
]
for i, c in enumerate(considerations):
    ax.text(0.05, 0.35-i*0.06, c, fontsize=10)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/06_decision_tree.png', dpi=130, bbox_inches='tight')
plt.close()
print('06_decision_tree.png')

# ===== Fig 7: NK vs Dense Newton — total time to converge =====
fig, ax = plt.subplots(figsize=(11, 6))
N_problem = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
# Assume Phi-eval cost ~1ms for now (rough)
phi_cost_ms = 1.0
# Dense Newton: (N+1) Phi-evals to build J + tiny linsolve
# Per outer iter: cost = (N+1) * phi_cost + N^3/3 / 1e9 ms (assuming 1 GFLOP/s)
# Total outer iters to reach 1e-12 from good warm start: ~5 (quadratic)
dense_total_ms = [(N+1)*phi_cost_ms*5 + (N**3/3/1e9*1000)*5 for N in N_problem]
# NK: ~25 inner Krylov per outer, ~10 outer iters (superlinear inexact)
nk_total_ms = [25*phi_cost_ms*10 + N*1e-6*1000*10 for N in N_problem]  # ignore inner linsolve

ax.loglog(N_problem, dense_total_ms, 'o-', lw=2, markersize=10, label='Dense Newton (5 outer iters, quadratic)', color='tab:red')
ax.loglog(N_problem, nk_total_ms, '^-', lw=2, markersize=10, label='Newton-Krylov (~10 outer × 25 inner)', color='tab:orange')
# Crossover
crossover = None
for i, N in enumerate(N_problem):
    if dense_total_ms[i] > nk_total_ms[i] and crossover is None:
        crossover = N
ax.axvline(crossover or 5000, color='black', linestyle='--', alpha=0.5)
ax.text((crossover or 5000)*1.1, 1e3, f'Dense Newton wins\nfor N ≲ {crossover or 5000}', fontsize=11)
ax.set_xlabel('N (problem size = number of unknowns)')
ax.set_ylabel('Total time to converge (ms, schematic)')
ax.set_title('Dense Newton vs NK: total time to converge to 1e-12 from a warm start\n'
              '(assuming Φ-eval cost = 1 ms; the crossover scales with this cost)')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/07_total_time_compare.png', dpi=130, bbox_inches='tight')
plt.close()
print('07_total_time_compare.png')

# ===== Fig 8: Memory comparison =====
fig, ax = plt.subplots(figsize=(11, 5))
N_vals = [100, 228, 500, 1000, 2000, 5000, 10000]
# Dense Newton: needs N² × 8 bytes (float64)
dense_mem = [N**2 * 8 / 1e6 for N in N_vals]  # MB
# NK: needs ~25 vectors of size N
nk_mem = [25 * N * 8 / 1e6 for N in N_vals]
ax.loglog(N_vals, dense_mem, 'o-', lw=2, markersize=10, label='Dense Newton (full Jacobian N²)', color='tab:red')
ax.loglog(N_vals, nk_mem, '^-', lw=2, markersize=10, label='Newton-Krylov (Krylov subspace 25N)', color='tab:orange')
ax.axhline(1000, color='gray', linestyle=':', label='1 GB')
ax.axhline(8000, color='black', linestyle=':', label='8 GB (laptop RAM cap)')
ax.axvline(228, color='green', linestyle='--', alpha=0.5)
ax.text(240, 5, 'N=228 (Cheby+sym, N=12)', fontsize=10)
ax.set_xlabel('N (problem size)')
ax.set_ylabel('Memory (MB)')
ax.set_title('Memory footprint: Dense Newton vs NK\n'
              'At N=228 (our recommended Chebyshev+sym), Dense J fits in 0.4 MB — trivial.')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/08_memory_compare.png', dpi=130, bbox_inches='tight')
plt.close()
print('08_memory_compare.png')

print('\nALL FIGURES GENERATED')
