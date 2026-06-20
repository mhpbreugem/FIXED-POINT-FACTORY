"""Test tabulated operator at N=8."""
import os, sys, time, math
import numpy as np
from numba import njit
sys.path.insert(0, '/tmp/cheby_h0')

N8 = 8; G8 = N8 + 1; C_STRETCH = 2.0; TAU = 1.0; GAMMA = 1.0; NQ = 12
LOBATTO8 = -np.cos(np.pi * np.arange(G8) / N8)
U_NODES8 = C_STRETCH * np.arctanh(np.clip(LOBATTO8, -0.9999, 0.9999))
V8 = np.empty((G8, G8))
for j in range(G8):
    x = LOBATTO8[j]; V8[j, 0] = 1.0; V8[j, 1] = x
    for k in range(1, G8-1):
        V8[j, k+1] = 2*x*V8[j, k] - V8[j, k-1]
V_INV8 = np.linalg.inv(V8)
GL_N, GL_W = np.polynomial.legendre.leggauss(NQ)

from cheby_numba import phi_jit
from cheby_numba_bisect import phi_jit_bisect
from cheby_numba_tab import phi_jit_tab, make_p_grid

U1, U2, U3 = np.meshgrid(U_NODES8, U_NODES8, U_NODES8, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))
P_in = sg(0.5*T)
p_grid_21 = make_p_grid(21)
p_grid_31 = make_p_grid(31)

print('JIT warmup...', flush=True)
_ = phi_jit(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
_ = phi_jit_bisect(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
_ = phi_jit_tab(P_in, V_INV8, LOBATTO8, U_NODES8, p_grid_21, GL_N, GL_W,
                  TAU, GAMMA, C_STRETCH, G8, NQ)

print(f'\nN=8 (G=9, 729 cells) timings:')
print(f'{"chebroots":>12} {"bisect":>12} {"tab(G_p=21)":>14} {"tab(G_p=31)":>14}')
for trial in range(5):
    t0 = time.time()
    P_cr = phi_jit(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
    t_cr = time.time() - t0
    t0 = time.time()
    P_bi = phi_jit_bisect(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, GAMMA, C_STRETCH, G8, NQ)
    t_bi = time.time() - t0
    t0 = time.time()
    P_tb = phi_jit_tab(P_in, V_INV8, LOBATTO8, U_NODES8, p_grid_21, GL_N, GL_W,
                          TAU, GAMMA, C_STRETCH, G8, NQ)
    t_tb21 = time.time() - t0
    t0 = time.time()
    P_tb31 = phi_jit_tab(P_in, V_INV8, LOBATTO8, U_NODES8, p_grid_31, GL_N, GL_W,
                            TAU, GAMMA, C_STRETCH, G8, NQ)
    t_tb31 = time.time() - t0
    print(f'{t_cr:>12.4f} {t_bi:>12.4f} {t_tb21:>14.4f} {t_tb31:>14.4f}')

print(f'\nTotal speedups (chebroots / tab):')
print(f'  G_p=21:  {t_cr/t_tb21:.1f}x')
print(f'  G_p=31:  {t_cr/t_tb31:.1f}x')
print(f'\nVerify outputs match chebroots:')
print(f'  bisect:       max|P_bi - P_cr| = {float(np.max(np.abs(P_cr - P_bi))):.3e}')
print(f'  tab(G_p=21):  max|P_tb - P_cr| = {float(np.max(np.abs(P_cr - P_tb))):.3e}')
print(f'  tab(G_p=31):  max|P_tb31 - P_cr| = {float(np.max(np.abs(P_cr - P_tb31))):.3e}')

# Also test γ dependence with the tabulated operator (should NOT be γ-invariant)
print(f'\n=== Test γ-dependence with TRUE Chebyshev operator (NOT rank-1) ===')
print(f'P_in = sigmoid(0.5*T) at tau=1')
for gamma in [0.1, 0.5, 1.0, 2.0, 10.0]:
    P_cr = phi_jit(P_in, V_INV8, LOBATTO8, GL_N, GL_W, TAU, gamma, C_STRETCH, G8, NQ)
    P_tb = phi_jit_tab(P_in, V_INV8, LOBATTO8, U_NODES8, p_grid_21, GL_N, GL_W,
                          TAU, gamma, C_STRETCH, G8, NQ)
    L = np.log(np.clip(P_cr, 1e-15, 1-1e-15)/(1-np.clip(P_cr, 1e-15, 1-1e-15))).ravel()
    Tf = T.ravel()
    slope = float(np.sum(L*Tf)/np.sum(Tf**2))
    print(f'  gamma={gamma:>5.1f}: chebroots slope={slope:.4f}, '
          f'P_cr range=[{P_cr.min():.4f}, {P_cr.max():.4f}], '
          f'max|cr-tab|={float(np.max(np.abs(P_cr-P_tb))):.2e}')
