"""Quick V3 test: apply Phi_CARA_v3 once to P_FR at G=10 and check
max|Phi(P_FR) - P_FR| matches float64 reference (~ 0.17).
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint; flint.ctx.prec = 150
from flint import arb

from phi_sigma_delta import set_boundary as set_bd_f64
from phi_sigma_delta_cara import phi_sigmadelta_cara
from flint_sd_v3 import (mp, phi_sigdelta_v3, set_boundary as set_bd_v3, to_np)

G = 10
G_FULL = G + 2
INNER_LO, INNER_HI = 1, G + 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0

dxi = 2.0 / (G + 1)
xi_inner = np.linspace(-1+dxi, 1-dxi, G)
xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])

# Build P_FR
u_full = np.zeros(G_FULL); S_full = np.zeros(G_FULL)
for i, x in enumerate(xi_full):
    if abs(x) < 1 - 1e-15:
        u_full[i] = TOT_u * math.atanh(x); S_full[i] = TOT_S * math.atanh(x)
    else:
        u_full[i] = math.copysign(1e10, x); S_full[i] = math.copysign(1e10, x)
P_FR = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    for j in range(G_FULL):
        for k in range(G_FULL):
            S = u_full[i] + S_full[j]
            if S > 50: P_FR[i,j,k] = 1
            elif S < -50: P_FR[i,j,k] = 0
            else: P_FR[i,j,k] = 1/(1+math.exp(-TAU*S))
P_FR = set_bd_f64(P_FR, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)

# float64 ref
print("float64 Phi_CARA(P_FR)...")
t = time.time()
Pf64 = phi_sigmadelta_cara(P_FR, xi_full, xi_full, xi_full, TOT_u, TOT_S, TOT_d, TAU, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
Pf64 = set_bd_f64(Pf64, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)
print(f"  {time.time()-t:.0f}s, max|delta|={np.max(np.abs((Pf64-P_FR)[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])):.4e}")

# flint v3
xi_mp = [mp(float(x)) for x in xi_full]
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU); gamma = mp(0.1); W_mp = mp(W)
P_v3 = [[[mp(float(P_FR[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P_v3 = set_bd_v3(P_v3)
print("flint v3 Phi_CARA(P_FR)...")
t = time.time()
P_v3_one = phi_sigdelta_v3(P_v3, xi_mp, xi_mp, xi_mp, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                            tau, gamma, W_mp, INNER_LO, INNER_HI, clearing='cara')
P_v3_one = set_bd_v3(P_v3_one)
Pv3np = to_np(P_v3_one)
print(f"  {time.time()-t:.0f}s, max|delta|={np.max(np.abs((Pv3np-P_FR)[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])):.4e}")

# compare to float64
diff_flint_f64 = (Pv3np - Pf64)[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI]
print(f"  max|v3 - float64| = {np.max(np.abs(diff_flint_f64)):.4e}")
print(f"  expected: small (flint v3 should match float64 to discrete-op precision)")
