"""T0b: validate the symmetric SMOOTH-kernel halo operator (the certified
emin15/lowtau family) at K=3.

1. Operator identity: my sym smooth phi == reznsrc.phi_K3_halo_smooth on a
   random symmetric P (machine precision).
2. Certified fixed point: residual of P_ld_t0.2_g1.035 under my operator.
3. Newton solve from no-learning at G=21: deficit must match certified
   2.5590e-4 (same operator -> should be ~exact).
4. Same at G=15 (the grid used for K=4): quantifies discretisation drift.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from highk_core import SmoothModel
from reznsrc.contour_K3_halo import phi_K3_halo_smooth, init_no_learning_K3

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK'
CERT = 0.0002558973541569669
report = {}

tau, gamma = 0.2, 1.035

# ---------------------------------------------------------------- 1+2
m = SmoothModel(21, 3, tau, gamma)
uf, lo, hi = m.u_full, m.lo, m.hi
tv = np.full(3, tau); gv = np.full(3, gamma); wv = np.full(3, 1.0)

P_full = init_no_learning_K3(uf, tv, gv, wv)
P_ld = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/'
               'dd_k3_overnight/emin15/P_ld_t0.2_g1.035.npy').astype(np.float64)
P_full[lo:hi, lo:hi, lo:hi] = P_ld

# sorted inner vector from the certified cube
x_cert = np.array([P_full[tuple(t)] for t in m.inner_tuples])

# reference phi on the cube
P_ref = phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, wv, m.h)
ref_inner = np.array([P_ref[tuple(t)] for t in m.inner_tuples])

mine = m.phi(x_cert)
op_diff = float(np.max(np.abs(mine - ref_inner)))
resid_cert = float(np.max(np.abs(mine - x_cert)))
print(f"1. operator identity:  max|sym_smooth - phi_K3_halo_smooth| = {op_diff:.3e}")
print(f"2. certified fp residual under sym operator: {resid_cert:.3e}")
report['op_identity'] = op_diff
report['cert_fp_residual'] = resid_cert
assert op_diff < 1e-12

# ---------------------------------------------------------------- 3+4
for G in (21, 15):
    m = SmoothModel(G, 3, tau, gamma)
    t0 = time.time()
    P, Finf, nev, ok = m.solve(f_tol=1e-12, maxiter=80)
    wall = time.time() - t0
    deficit, slope = m.deficit_slope(P)
    rel = deficit / CERT - 1
    print(f"3/4. G={G}: deficit={deficit:.6e}  slope={slope:.5f}  F={Finf:.2e}"
          f"  evals={nev}  conv={ok}  wall={wall:.1f}s  rel vs cert={rel:+.2%}")
    report[f'k3_smooth_G{G}'] = dict(deficit=deficit, slope=slope, Finf=Finf,
                                     rel_err_vs_cert=rel, wall=wall,
                                     converged=ok, n_evals=nev)

json.dump(report, open(f"{OUT}/t0_validation_smooth.json", 'w'), indent=2)
print("wrote t0_validation_smooth.json")
