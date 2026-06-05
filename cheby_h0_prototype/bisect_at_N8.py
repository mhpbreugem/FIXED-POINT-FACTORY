"""Test bisection operator at N=8 (would be 1.23s with chebroots)."""
import os, sys, time, math
import numpy as np
from numba import njit
sys.path.insert(0, '/tmp/cheby_h0')

# Build N=8 setup manually
N8 = 8; G8 = N8 + 1; C_STRETCH = 2.0; TAU = 1.0; GAMMA = 1.0; NQ = 12
LOBATTO8 = -np.cos(np.pi * np.arange(G8) / N8)
U_NODES8 = C_STRETCH * np.arctanh(np.clip(LOBATTO8, -0.9999, 0.9999))
V8 = np.empty((G8, G8))
for j in range(G8):
    x = LOBATTO8[j]
    V8[j, 0] = 1.0; V8[j, 1] = x
    for k in range(1, G8-1):
        V8[j, k+1] = 2*x*V8[j, k] - V8[j, k-1]
V_INV8 = np.linalg.inv(V8)
GL_N, GL_W = np.polynomial.legendre.leggauss(NQ)

from cheby_numba import phi_jit
from cheby_numba_bisect import phi_jit_bisect

U1, U2, U3 = np.meshgrid(U_NODES8, U_NODES8, U_NODES8, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))
P_in = sg(0.5*T)

print('=== N=8 (G=9, 729 cells) timing ===\n')
print('JIT warmup...', flush=True)
_ = phi_jit(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
_ = phi_jit_bisect(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)

print(f'\n{"chebroots(s)":>15} {"bisection(s)":>15} {"speedup":>10} {"max diff":>14}')
for trial in range(5):
    t0 = time.time()
    P_cr = phi_jit(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
    t_cr = time.time() - t0
    t0 = time.time()
    P_bi = phi_jit_bisect(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
    t_bi = time.time() - t0
    diff = float(np.max(np.abs(P_cr - P_bi)))
    print(f'{t_cr:>15.4f} {t_bi:>15.4f} {t_cr/t_bi:>10.2f}x {diff:>14.3e}')
