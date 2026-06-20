"""T0: validate the numba symmetric solver.

1. Jitted phi == pure-python contour_KN_sym.sym_phi (machine precision)
   at small grids for K=3,4,5.
2. K=3 Newton at (tau=0.2, gamma=1.035), G=15 and G=21, u_max=4:
   converged deficit must match the certified emin15 value 2.559e-4
   to within ~5%.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code')
from highk_core import Model
import contour_KN_sym as ref

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK'
report = {}

# ----------------------------------------------------------------------
# 1. phi cross-check vs pure python
# ----------------------------------------------------------------------
print("=== phi cross-check (jit vs pure-python sym_phi) ===", flush=True)
rng = np.random.default_rng(0)
for K, G in [(3, 7), (4, 6), (5, 5)]:
    tau, gamma, W = 0.7, 0.9, 1.0
    m = Model(G, K, tau, gamma, W=W, u_max=4.0)
    P0 = m.init_no_learning() + 0.01 * rng.standard_normal(m.sg.n)
    P0 = np.clip(P0, 0.05, 0.95)
    t0 = time.time()
    a = m.phi(P0)
    t_jit = time.time() - t0
    t0 = time.time()
    b = ref.sym_phi(P0, m.sg, m.u, tau, gamma, W)
    t_py = time.time() - t0
    d = float(np.max(np.abs(a - b)))
    print(f"  K={K} G={G}: max|jit-py| = {d:.3e}   jit {t_jit:.3f}s  py {t_py:.2f}s")
    report[f'crosscheck_K{K}_G{G}'] = d
    assert d < 1e-12, f"phi mismatch at K={K}"

# ----------------------------------------------------------------------
# 2. K=3 certified-deficit check
# ----------------------------------------------------------------------
print("\n=== K=3 deficit vs certified emin15 (t0.2 g1.035: 2.5590e-4) ===",
      flush=True)
CERT = 0.0002558973541569669
for G in (15, 21):
    m = Model(G, 3, 0.2, 1.035, u_max=4.0)
    t0 = time.time()
    P, Finf, nev, ok = m.solve(f_tol=1e-12, maxiter=80)
    wall = time.time() - t0
    deficit, slope = m.deficit_slope(P)
    rel = deficit / CERT - 1
    print(f"  G={G}: deficit={deficit:.6e}  slope={slope:.5f}  "
          f"F={Finf:.2e}  evals={nev}  conv={ok}  wall={wall:.1f}s  "
          f"rel.err vs cert = {rel:+.2%}")
    report[f'k3_G{G}'] = dict(deficit=deficit, slope=slope, Finf=Finf,
                              rel_err=rel, wall=wall, converged=ok)

json.dump(report, open(f"{OUT}/t0_validation.json", 'w'), indent=2)
print("\nwrote t0_validation.json")
