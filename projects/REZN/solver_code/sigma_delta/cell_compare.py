"""Cell-by-cell comparison of float64 phi_sigma_delta_cara vs flint port.

Setup the SAME P_FR at G_FULL=12 (G_inner=10), apply Phi_CARA in BOTH,
report per-cell diff. Identify the systematic discrepancy."""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

# float64 reference
from phi_sigma_delta import set_boundary as set_bd_f64, fsig as fsig_f64
from phi_sigma_delta_cara import phi_sigmadelta_cara

# flint
import flint; flint.ctx.prec = 200
from flint import arb
from flint_sd_G15_crra_v2 import (mp, fsig, phi_sigdelta as phi_flint,
                                    set_boundary as set_bd_flint,
                                    interp_along_Sigma as interp_flint,
                                    evidence_agent1 as ev1_flint,
                                    evidence_agent_oblique as ev_obl_flint)
import flint_sd_G15_crra_v2 as M

# small grid
G = 10
G_FULL = G + 2
INNER_LO, INNER_HI = 1, G + 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0
M.G_FULL = G_FULL; M.INNER_LO = INNER_LO; M.INNER_HI = INNER_HI

dxi = 2.0 / (G + 1)
xi_inner = np.linspace(-1+dxi, 1-dxi, G)
xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])

# Build P_FR over the full padded cube
u_full = np.zeros(G_FULL); S_full = np.zeros(G_FULL); d_full = np.zeros(G_FULL)
for i, x in enumerate(xi_full):
    if abs(x) < 1 - 1e-15:
        u_full[i] = TOT_u * math.atanh(x)
        S_full[i] = TOT_S * math.atanh(x)
        d_full[i] = TOT_d * math.atanh(x)
    else:
        u_full[i] = math.copysign(1e10, x)
        S_full[i] = math.copysign(1e10, x)
        d_full[i] = math.copysign(1e10, x)

P_FR_f64 = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    for j in range(G_FULL):
        for k in range(G_FULL):
            S = u_full[i] + S_full[j]   # FR is δ-flat
            if S > 50: P_FR_f64[i,j,k] = 1
            elif S < -50: P_FR_f64[i,j,k] = 0
            else: P_FR_f64[i,j,k] = 1/(1+math.exp(-TAU*S))

P_FR_f64 = set_bd_f64(P_FR_f64, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)

# float64 step
print("Running float64 Phi_CARA(P_FR)...", flush=True)
t = time.time()
P_one_f64 = phi_sigmadelta_cara(P_FR_f64, xi_full, xi_full, xi_full,
                                 TOT_u, TOT_S, TOT_d, TAU, W,
                                 INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
P_one_f64 = set_bd_f64(P_one_f64, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)
print(f"  done {time.time()-t:.0f}s", flush=True)
diff_f64 = (P_one_f64 - P_FR_f64)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f"  float64 max|Phi(P_FR)-P_FR| = {np.max(np.abs(diff_f64)):.3e}", flush=True)

# flint setup
xi_u_mp = [mp(float(x)) for x in xi_full]
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU); gamma_unused = mp(0.1); W_mp = mp(W)

P_flint = [[[mp(float(P_FR_f64[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P_flint = set_bd_flint(P_flint)

print("Running flint Phi_CARA(P_FR) at G=10, dps=200...", flush=True)
t = time.time()
P_one_flint = phi_flint(P_flint, xi_u_mp, xi_u_mp, xi_u_mp, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                        tau, gamma_unused, W_mp, clearing='cara')
P_one_flint = set_bd_flint(P_one_flint)
print(f"  done {time.time()-t:.0f}s", flush=True)
P_flint_np = np.array([[[float(P_one_flint[i][j][k]) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)])
diff_flint = (P_flint_np - P_FR_f64)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f"  flint max|Phi(P_FR)-P_FR| = {np.max(np.abs(diff_flint)):.3e}", flush=True)
print(f"  diff (flint - float64) max = {np.max(np.abs(P_one_flint_inner := P_flint_np[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI]) - P_one_f64[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI]):.3e}")
diff_per_cell = P_flint_np[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI] - P_one_f64[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI]
i_max, j_max, k_max = np.unravel_index(np.argmax(np.abs(diff_per_cell)), diff_per_cell.shape)
print(f"  worst cell (rel to float64): (i,j,k)=({i_max},{j_max},{k_max})  "
      f"flint={P_flint_np[INNER_LO+i_max,INNER_LO+j_max,INNER_LO+k_max]:.6f}  "
      f"float64={P_one_f64[INNER_LO+i_max,INNER_LO+j_max,INNER_LO+k_max]:.6f}")
print(f"    u_1={u_full[INNER_LO+i_max]:.3f}, Σ={S_full[INNER_LO+j_max]:.3f}, δ={d_full[INNER_LO+k_max]:.3f}")
print(f"  worst cell (rel to P_FR): max|flint - FR| at unraveled idx")
i_fr, j_fr, k_fr = np.unravel_index(np.argmax(np.abs(diff_flint)), diff_flint.shape)
print(f"  flint vs FR worst: (i,j,k)=({i_fr},{j_fr},{k_fr})  "
      f"flint={P_flint_np[INNER_LO+i_fr,INNER_LO+j_fr,INNER_LO+k_fr]:.6f}  "
      f"FR={P_FR_f64[INNER_LO+i_fr,INNER_LO+j_fr,INNER_LO+k_fr]:.6f}")
print(f"    u_1={u_full[INNER_LO+i_fr]:.3f}, Σ={S_full[INNER_LO+j_fr]:.3f}, δ={d_full[INNER_LO+k_fr]:.3f}")

# save for plotting
import json
np.save(os.path.join(HERE, 'cell_compare_diff_flint.npy'), diff_flint)
np.save(os.path.join(HERE, 'cell_compare_diff_f64.npy'), diff_f64)
np.save(os.path.join(HERE, 'cell_compare_P_flint.npy'), P_flint_np[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])
np.save(os.path.join(HERE, 'cell_compare_P_f64.npy'), P_one_f64[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])
print('\nsaved npys')
