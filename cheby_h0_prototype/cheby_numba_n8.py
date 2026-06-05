"""Same numba operator at N=8 (G=9) — checks per-Φ cost at higher resolution
to assess whether N=10/N=12 is practical with the existing numba+lift pipeline.
"""
import time, numpy as np
from cheby_numba import phi_jit, chebval_jit, chebder_jit, chebroots_jit  # JIT'd kernels

# N=8 setup
N8 = 8
G8 = N8 + 1
C_STRETCH = 2.0
TAU = 1.0
GAMMA = 1.0
NQ = 12

LOBATTO8 = -np.cos(np.pi * np.arange(G8) / N8)
U_NODES8 = C_STRETCH * np.arctanh(np.clip(LOBATTO8, -0.9999, 0.9999))
# Vandermonde at N=8
V8 = np.empty((G8, G8))
for j in range(G8):
    x = LOBATTO8[j]
    V8[j, 0] = 1.0; V8[j, 1] = x
    for k in range(1, G8-1):
        V8[j, k+1] = 2*x*V8[j, k] - V8[j, k-1]
V_INV8 = np.linalg.inv(V8)
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(NQ)

# Build a smooth P field on N=8 grid (sigmoid)
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
U1, U2, U3 = np.meshgrid(U_NODES8, U_NODES8, U_NODES8, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)
P_test = sigmoid(0.5 * T_FIELD)

print(f'N=8 phi test: G={G8}, total cells {G8**3}')
print(f'  P_test range: [{P_test.min():.4f}, {P_test.max():.4f}]')

# Warm up JIT (uses cheby_numba.phi_jit, which is general in G)
print('Triggering JIT...', flush=True)
t = time.time()
P_new = phi_jit(P_test, V_INV8, LOBATTO8, GL_NODES, GL_WEIGHTS, TAU, GAMMA, C_STRETCH, G8, NQ)
t1 = time.time()
print(f'  first call (incl JIT specialization): {t1-t:.2f}s')

# Time second call (steady-state)
t = time.time()
P_new2 = phi_jit(P_test, V_INV8, LOBATTO8, GL_NODES, GL_WEIGHTS, TAU, GAMMA, C_STRETCH, G8, NQ)
t2 = time.time()
print(f'  second call (steady):                 {t2-t:.3f}s')

# Determinism
print(f'  determinism: ||diff||_inf = {float(np.max(np.abs(P_new - P_new2))):.3e}')
print(f'  P_new range: [{P_new.min():.4f}, {P_new.max():.4f}]')

# Compare with N=6 timing
print(f'\nComparison:')
print(f'  N=6 (G=7, 343 cells):  numba ~0.27s per Φ')
print(f'  N=8 (G=9, 729 cells):  numba {t2-t:.3f}s per Φ')
factor = (t2-t)/0.27
print(f'  N=6 → N=8 slowdown:   {factor:.1f}× (theoretical: ~(9/7)^3 * (9/7)^2 = ~4.4× operator cost)')
print(f'\nNewton at N=8: ~150 symmetric DOFs × ~{t2-t:.1f}s = ~{150*(t2-t):.0f}s per outer iter')
print(f'  vs N=6: ~40 sym DOFs × 0.27s = ~11s per iter')
