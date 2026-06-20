"""Scan α'(α) vs α and find genuine FPs (where α'(α) = α)."""
import numpy as np
import sys, time
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_rank1 import operator_rank1, fit_alpha, deficit_R2

C_STRETCH = 2.0
tau, gamma, G = 1.0, 1.0, 7
LOBATTO = -np.cos(np.pi * np.arange(G) / (G-1))
U_NODES = C_STRETCH * np.arctanh(np.clip(LOBATTO, -0.9999, 0.9999))
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_grid = tau * (U1 + U2 + U3)

# Scan α'(α) on a fine grid
alphas = np.linspace(0.05, 1.5, 30)
print(f'{"alpha":>10} {"alpha_prime":>12} {"g=ap-a":>10} {"deficit":>10}')
results = []
for a in alphas:
    P = operator_rank1(a, U_NODES, tau, gamma)
    a_prime = fit_alpha(P, T_grid)
    d, _ = deficit_R2(P, T_grid)
    g = a_prime - a
    print(f'{a:>10.4f} {a_prime:>10.6f} {g:>+10.4f} {d:>10.4e}')
    results.append((a, a_prime, d))

# Find zero-crossings of g
print('\nZero crossings of g(alpha):')
for i in range(len(results)-1):
    a1, ap1, _ = results[i]
    a2, ap2, _ = results[i+1]
    g1 = ap1 - a1; g2 = ap2 - a2
    if g1 * g2 < 0:
        # Linear interp for FP
        a_fp = a1 - g1 * (a2 - a1)/(g2 - g1)
        # Verify
        P = operator_rank1(a_fp, U_NODES, tau, gamma)
        a_prime = fit_alpha(P, T_grid)
        d, _ = deficit_R2(P, T_grid)
        print(f'  ~alpha* = {a_fp:.6f}, verify ap = {a_prime:.6f}, deficit={d:.4e}')

# Bracket-bisect to find FP precisely
print('\nBisection refinement:')
for i in range(len(results)-1):
    a1, ap1, _ = results[i]; a2, ap2, _ = results[i+1]
    g1 = ap1 - a1; g2 = ap2 - a2
    if g1 * g2 < 0:
        lo, hi = (a1, a2) if g1 < 0 else (a2, a1)
        # g(lo) < 0, g(hi) > 0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            P = operator_rank1(mid, U_NODES, tau, gamma)
            g_mid = fit_alpha(P, T_grid) - mid
            if g_mid < 0: lo = mid
            else: hi = mid
        alpha_fp = 0.5 * (lo + hi)
        P = operator_rank1(alpha_fp, U_NODES, tau, gamma)
        a_prime = fit_alpha(P, T_grid); d, _ = deficit_R2(P, T_grid)
        print(f'  alpha* = {alpha_fp:.10f}, residual ap-a = {a_prime - alpha_fp:+.2e}, deficit = {d:.4e}')
