"""STRICT contour Φ in float64, h=0 hardwired, K=3 symmetric.

Mirror of the gmpy2 strict-contour Φ but with plain Python float64
(via dd_phi_contour from earlier work, just the float64 path).

Compares to gmpy2 200-dec result at 20 iters.
"""
import os, sys, time, math
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code')
import numpy as np
import json
from contour_KN_sym import SymGrid, sym_phi, sym_to_full, full_to_sym

G = 11
INNER_LO, INNER_HI = 2, 9
G_INNER = INNER_HI - INNER_LO
UMAX = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
MAX_ITER = 20

u_full = np.linspace(-UMAX, UMAX, G)
sg = SymGrid.build(G=G, K=3)

# FR ansatz
U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')
P_FR_full = 1.0 / (1.0 + np.exp(-TAU * (U1+U2+U3)))
P_sorted = full_to_sym(P_FR_full, sg)

# Weights for ferr (inner-block based)
def f_inf_inner(P_new_full, P_old_full):
    return np.max(np.abs(P_new_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
                          - P_old_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]))

LOG = '/tmp/strict_f64.log'
open(LOG, 'w').close()
def lg(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

lg(f'STRICT-CONTOUR float64 Picard, h=0 hardwired, G={G}, γ={GAMMA}, MAX_ITER={MAX_ITER}')

P_full = sym_to_full(P_sorted, sg)
snapshots = [P_full.copy()]
ferr_t = [0.0]

t0 = time.time()
for it in range(1, MAX_ITER+1):
    P_new_sorted = sym_phi(P_sorted, sg, u_full, TAU, GAMMA, W)
    P_new_full = sym_to_full(P_new_sorted, sg)
    res = f_inf_inner(P_new_full, P_full)
    P_full = P_new_full
    P_sorted = P_new_sorted
    snapshots.append(P_full.copy())
    ferr_t.append(res)
    lg(f'  iter {it:2d}: ferr={res:.3e}')

lg(f'Total {time.time()-t0:.1f}s')

np.save('/tmp/strict_f64_snapshots.npy', np.array(snapshots))
with open('/tmp/strict_f64_ferr.json', 'w') as f:
    json.dump({'ferr': ferr_t, 'MAX_ITER': MAX_ITER, 'GAMMA': GAMMA, 'G_FULL': G,
                'mode': 'strict-contour float64'}, f, indent=2)
lg('saved snapshots')
