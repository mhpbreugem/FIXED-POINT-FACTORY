"""Overnight: rank-1 (h=0) solver — parameter sweep + comparisons.

1. Rank-1 (tau, gamma) phase diagram at high resolution.
2. Grid-invariance verification at sample points.
3. Comparison with chebroots-operator slope (where it converges).
4. Save NPZ + JSON + figures.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_rank1 import operator_rank1, fit_alpha, deficit_R2

C_STRETCH = 2.0

def solve(tau, gamma, G=15):
    LOBATTO = -np.cos(np.pi * np.arange(G) / (G-1))
    U = C_STRETCH * np.arctanh(np.clip(LOBATTO, -0.9999, 0.9999))
    U1, U2, U3 = np.meshgrid(U, U, U, indexing='ij')
    T = tau * (U1 + U2 + U3)
    P = operator_rank1(U, tau, gamma)
    alpha = fit_alpha(P, T)
    deficit, _ = deficit_R2(P, T)
    # Distance from FR (alpha=1)
    P_FR = 1/(1+np.exp(-T))
    d_FR = float(np.sqrt(np.mean((P - P_FR)**2)))
    # Distance from NL (no learning: P depends only on weighted average of own signals)
    # Actually NL would use only own signal; for our setup baseline is P=σ(τ u_k_only)... skip
    return alpha, deficit, d_FR

# JIT warmup
print('JIT warmup...', flush=True)
t0 = time.time()
_ = solve(1.0, 1.0, G=7)
print(f'  warmup {time.time()-t0:.1f}s\n', flush=True)

# ===== Part A: (tau, gamma) phase diagram =====
print('=== A: (tau, gamma) phase diagram, G=15, 60x60 grid ===', flush=True)
tau_grid = np.linspace(0.3, 4.0, 60)
gamma_grid = np.logspace(np.log10(0.05), np.log10(10), 60)
A_alpha = np.empty((len(tau_grid), len(gamma_grid)))
A_deficit = np.empty((len(tau_grid), len(gamma_grid)))
A_dFR = np.empty((len(tau_grid), len(gamma_grid)))
t0 = time.time()
for i, tau in enumerate(tau_grid):
    for j, gamma in enumerate(gamma_grid):
        alpha, d, dFR = solve(tau, gamma, G=15)
        A_alpha[i, j] = alpha
        A_deficit[i, j] = d
        A_dFR[i, j] = dFR
    if (i+1) % 5 == 0 or i == 0:
        print(f'  row {i+1}/{len(tau_grid)} tau={tau:.3f}  '
              f'(elapsed {time.time()-t0:.1f}s, '
              f'remaining ~{(time.time()-t0)*(len(tau_grid)-i-1)/(i+1):.0f}s)',
              flush=True)
print(f'  Phase diagram complete: {time.time()-t0:.1f}s\n', flush=True)
np.savez('/tmp/cheby_h0/rank1_phase.npz',
          tau_grid=tau_grid, gamma_grid=gamma_grid,
          alpha=A_alpha, deficit=A_deficit, d_FR=A_dFR)

# ===== Part B: grid-invariance verification =====
print('=== B: Grid-invariance at 6 sample points ===', flush=True)
sample_pts = [(1.0, 1.0), (1.0, 0.1), (1.0, 10.0),
                (2.0, 1.0), (0.5, 1.0), (3.0, 0.2)]
B_results = {}
for (tau, gamma) in sample_pts:
    rows = []
    for G in [3, 5, 7, 9, 11, 15, 21, 31, 41]:
        t = time.time()
        alpha, d, dFR = solve(tau, gamma, G=G)
        rows.append(dict(G=G, alpha=alpha, deficit=d, d_FR=dFR, time=time.time()-t))
    B_results[f'tau{tau}_gamma{gamma}'] = rows
    print(f'  (tau={tau}, gamma={gamma}):')
    for r in rows:
        print(f'    G={r["G"]:>3}  alpha*={r["alpha"]:.10f}  deficit={r["deficit"]:.3e}'
              f'  d_FR={r["d_FR"]:.3e}  t={r["time"]:.3f}s', flush=True)
print()

# ===== Part C: chebroots-operator slope where it converges =====
print('=== C: Chebroots operator output slope (G=7, single application to sigma(0.5*T)) ===', flush=True)
try:
    from cheby_numba import phi as phi_chebroots, U_NODES as U_NODES_CR
    G_CR = 7
    U1_CR, U2_CR, U3_CR = np.meshgrid(U_NODES_CR, U_NODES_CR, U_NODES_CR, indexing='ij')
    sample_pts_C = [(1.0, 1.0), (1.0, 0.1), (1.0, 10.0), (2.0, 1.0), (0.5, 1.0), (3.0, 0.2)]
    C_results = []
    def sg(x): return 1/(1+np.exp(-x))
    for (tau, gamma) in sample_pts_C:
        T = tau*(U1_CR+U2_CR+U3_CR)
        # Rank-1 alpha (best basis)
        rank1_alpha, rank1_def, _ = solve(tau, gamma, G=15)
        # Chebroots: apply to rank-1 conjecture sigma(rank1_alpha * T)
        P_in = sg(rank1_alpha * T)
        t0 = time.time()
        P_out = phi_chebroots(P_in, gamma=gamma, tau=tau)
        t_cr = time.time() - t0
        L = np.log(np.clip(P_out, 1e-15, 1-1e-15)/(1-np.clip(P_out, 1e-15, 1-1e-15))).ravel()
        Tf = T.ravel()
        ap_cr = float(np.sum(L*Tf)/np.sum(Tf**2))
        # Quick R^2 deficit
        pred = ap_cr*Tf
        defi = float(np.sum((L-pred)**2)/max(np.sum((L-L.mean())**2),1e-30))
        C_results.append(dict(tau=tau, gamma=gamma,
                                rank1_alpha=rank1_alpha, rank1_deficit=rank1_def,
                                chebroots_alpha_from_rank1IC=ap_cr,
                                chebroots_deficit=defi, t_chebroots=t_cr))
        print(f'  tau={tau} gamma={gamma}:  rank-1 alpha*={rank1_alpha:.6f}, '
              f'chebroots-from-rank1IC slope={ap_cr:.6f}, chebroots time={t_cr:.3f}s',
              flush=True)
except Exception as e:
    print(f'  Skipped (error: {e})')
    C_results = []
print()

# ===== Save everything =====
out = dict(
    A=dict(tau_grid=tau_grid.tolist(), gamma_grid=gamma_grid.tolist(),
            alpha_min=float(A_alpha.min()), alpha_max=float(A_alpha.max()),
            deficit_min=float(A_deficit.min()), deficit_max=float(A_deficit.max())),
    B=B_results,
    C=C_results,
)
json.dump(out, open('/tmp/cheby_h0/rank1_overnight_results.json','w'), indent=2, default=str)
print('=== DONE. Saved rank1_phase.npz, rank1_overnight_results.json ===')
print(f'Total time: {time.time()-t0:.0f}s')
