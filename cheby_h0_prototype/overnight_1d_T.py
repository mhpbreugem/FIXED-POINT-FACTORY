"""Overnight: 1D-in-T phase diagram at multiple m values.

Builds a deficit-reduction map: at each (tau, gamma), shows how much deficit
remains as we increase the 1D ansatz order m.

Total work: 1 operator eval per (tau, gamma) point + cheap fits for each m.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_rank1 import operator_rank1
from cheby_1d_T import fit_1d_in_T, evaluate_P_tilde, design_matrix

C_STRETCH_U = 2.0

def solve_multi_m(tau, gamma, G, m_list):
    """One operator eval, multiple fits."""
    LOBATTO = -np.cos(np.pi * np.arange(G) / (G-1))
    U_NODES = C_STRETCH_U * np.arctanh(np.clip(LOBATTO, -0.9999, 0.9999))
    u_max = U_NODES.max()
    T_max = 3 * tau * u_max
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T_grid = tau * (U1 + U2 + U3)
    P_op = operator_rank1(U_NODES, tau, gamma)
    results = {}
    for m in m_list:
        coeffs, deficit, _, _ = fit_1d_in_T(P_op, T_grid, m, T_max, weighted=True,
                                              edge_trim=0.95)
        results[m] = dict(alpha=float(coeffs[0]),
                           h_coeffs=[float(c) for c in coeffs[1:]],
                           deficit=float(deficit), T_max=float(T_max))
    return results


# JIT warmup
print('JIT warmup...', flush=True)
t0 = time.time()
_ = solve_multi_m(1.0, 1.0, G=7, m_list=[1, 3])
print(f'  warmup {time.time()-t0:.1f}s\n', flush=True)

# ===== Part A: (tau, gamma) phase diagram at m = 1, 3, 5, 7 =====
print('=== A: (tau, gamma) phase diagram, G=21, 80x80 grid, m=[1,3,5,7] ===', flush=True)
tau_grid = np.linspace(0.3, 4.0, 80)
gamma_grid = np.logspace(np.log10(0.05), np.log10(10), 80)
m_list = [1, 3, 5, 7]

A_alpha = {m: np.empty((len(tau_grid), len(gamma_grid))) for m in m_list}
A_deficit = {m: np.empty((len(tau_grid), len(gamma_grid))) for m in m_list}

t0 = time.time()
for i, tau in enumerate(tau_grid):
    for j, gamma in enumerate(gamma_grid):
        r = solve_multi_m(tau, gamma, G=21, m_list=m_list)
        for m in m_list:
            A_alpha[m][i, j] = r[m]['alpha']
            A_deficit[m][i, j] = r[m]['deficit']
    if (i+1) % 8 == 0 or i == 0:
        elapsed = time.time() - t0
        remaining = elapsed * (len(tau_grid) - i - 1) / (i + 1)
        print(f'  row {i+1}/{len(tau_grid)} tau={tau:.3f}  '
              f'(elapsed {elapsed:.1f}s, remaining ~{remaining:.0f}s)', flush=True)
total = time.time() - t0
print(f'\n  Phase diagram complete: {total:.1f}s\n', flush=True)

np.savez('/tmp/cheby_h0/onedT_phase.npz',
          tau_grid=tau_grid, gamma_grid=gamma_grid,
          **{f'alpha_m{m}': A_alpha[m] for m in m_list},
          **{f'deficit_m{m}': A_deficit[m] for m in m_list})

# ===== Part B: P̃(T) shape at representative points =====
print('=== B: P̃(T) shapes at 9 anchor points ===', flush=True)
anchors = [
    (0.5, 0.1), (0.5, 1.0), (0.5, 10.0),
    (1.0, 0.1), (1.0, 1.0), (1.0, 10.0),
    (3.0, 0.1), (3.0, 1.0), (3.0, 10.0),
]
B_results = []
for tau, gamma in anchors:
    r_full = solve_multi_m(tau, gamma, G=31, m_list=[1, 3, 5, 7, 10])
    T_max = r_full[5]['T_max']
    T_vals = np.linspace(-T_max*0.99, T_max*0.99, 200)
    shapes = {}
    for m in [1, 3, 5, 7, 10]:
        coeffs_full = np.array([r_full[m]['alpha']] + r_full[m]['h_coeffs'])
        P_tilde_vals = evaluate_P_tilde(coeffs_full, T_vals, T_max)
        shapes[m] = dict(alpha=r_full[m]['alpha'],
                          deficit=r_full[m]['deficit'],
                          T_vals=T_vals.tolist(), P_tilde=P_tilde_vals.tolist())
    B_results.append(dict(tau=tau, gamma=gamma, T_max=T_max, shapes=shapes))
    print(f'  (tau={tau}, gamma={gamma}): '
          f'rank1 def={r_full[1]["deficit"]:.2e}, '
          f'm=5 def={r_full[5]["deficit"]:.2e}, '
          f'm=10 def={r_full[10]["deficit"]:.2e}',
          flush=True)

# ===== Save everything =====
json.dump(dict(
    tau_grid=tau_grid.tolist(), gamma_grid=gamma_grid.tolist(),
    m_list=m_list, anchors=B_results,
), open('/tmp/cheby_h0/onedT_results.json', 'w'), indent=2, default=str)
print('\n=== DONE. Saved onedT_phase.npz, onedT_results.json ===')
print(f'Total time: {time.time()-t0:.0f}s')
