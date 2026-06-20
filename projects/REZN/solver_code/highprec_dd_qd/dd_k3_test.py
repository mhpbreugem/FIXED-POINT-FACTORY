"""Cross-validate DD K=3 operator against float64 Lin-CDF Richardson."""
import os, sys, time
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40

import dd_k3_ops as DK
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


G = 7
G_p = 17    # smaller than paper's 121 for DD speed
NQK = 8     # smaller than paper's 16 for DD speed
HS_R2 = (0.5, 0.25)
HS_R4 = (0.5, 0.4, 0.3, 0.2)
TAU = 1.0
GAMMA = 1.0

u_grid = make_cdf_uniform_grid(G)
p_grid = make_p_grid(G_p)
gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)

# Build cold P (sigmoid)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
P0 = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))

print(f"--- DD K=3 operator self-test (G={G}, G_p={G_p}, NQK={NQK}) ---")
print(f"tau={TAU}, gamma={GAMMA}, hs={HS_R2}")

# Float64 reference
P_ref = phi_lin_richardson(P0, u_grid, hs=HS_R2, gamma=GAMMA, tau=TAU,
                              G_p=G_p, NQK=NQK, p_grid=p_grid)
print(f"\nFloat64 Phi: |F|_cold={float(np.max(np.abs(P_ref - P0))):.3e}")

# DD operator (hi part = P0, lo = 0)
P_H = P0.copy().astype(np.float64); P_L = np.zeros_like(P0)
th, tl = split(mp.mpf(TAU))
gh, gl = split(mp.mpf(GAMMA))
w_arr = DK.richardson_weights(HS_R2)

# JIT warmup
print("\nDD JIT warmup..."); t0 = time.time()
P_dd_H, P_dd_L = DK.phi_dd(P_H, P_L, u_grid, p_grid, gl_u, gl_du,
                                 th, tl, gh, gl, HS_R2, w_arr)
print(f"  done in {time.time()-t0:.1f}s")

# Time after warm
t0 = time.time()
for _ in range(2):
    P_dd_H, P_dd_L = DK.phi_dd(P_H, P_L, u_grid, p_grid, gl_u, gl_du,
                                     th, tl, gh, gl, HS_R2, w_arr)
print(f"DD per-Phi (warm): {(time.time()-t0)/2*1000:.0f}ms")

# Compare hi vs float64
P_dd = P_dd_H + P_dd_L
print(f"\n|P_dd - P_ref|_inf = {float(np.max(np.abs(P_dd - P_ref))):.3e}"
        f"  (should be ~ 1e-15 to 1e-13, dominated by float64 reference precision)")

# Compute F = Phi(P) - P in DD
F_H, F_L = np.empty_like(P_H), np.empty_like(P_L)
for i in range(G):
    for j in range(G):
        for k in range(G):
            a, b = DK.D.dd_add(P_dd_H[i,j,k], P_dd_L[i,j,k],
                                  -P_H[i,j,k], -P_L[i,j,k])
            F_H[i,j,k] = a; F_L[i,j,k] = b
absF = np.abs(F_H + F_L)
print(f"|F|_inf (DD) at cold P: {float(absF.max()):.3e}  (cold start: should be ~1e-1)")
