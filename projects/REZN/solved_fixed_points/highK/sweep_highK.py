"""High-K sweep: extend the certified K=3 partial-revelation results to
K=4 and K=5 with the symmetric smooth-kernel halo operator
(h = 0.45*sqrt(du), pad=2 no-learning halo, u_max=4, W=1).

Runs (gamma-continuation within each tau row, ascending gamma):
  A. K=3 G=21  taus {0.1,0.2,0.4,0.6} x 15 gammas   (certified-grade ref)
  B. K=3 G=15  same cells                            (cross-K baseline)
  C. K=4 G=15  same cells                            (main result, T1)
  D. K=3,4,5 G=11, tau=0.2 row, 15 gammas            (K=5 ladder, T2)
  E. K=5 G=15, tau=0.2, gammas {0.1, 1.0, 10.0}      (if budget allows)

f_tol target 1e-10 (most cells land at ~1e-14); honest Finf recorded
per cell.  Deficit = unweighted 1-R^2 of logit(P) on T (emin15 metric).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK')
from highk_core import SmoothModel

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK'
RESULTS = f"{OUT}/results.json"

GAMMAS12 = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 12), 4))
EXTRA = [0.1, 1.0, 10.0]
GAMMAS = sorted(set(GAMMAS12 + EXTRA))
TAUS = [0.10, 0.20, 0.40, 0.60]

results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
T_START = time.time()


def run_row(K, G, tau, gammas, f_tol=1e-10, save_P=False):
    """Solve one tau row with ascending-gamma continuation."""
    x_warm = None
    for gamma in gammas:
        key = f"K{K}_G{G}_t{tau}_g{gamma}"
        if key in results and results[key].get('converged'):
            continue
        t0 = time.time()
        m = SmoothModel(G, K, tau, gamma)
        x, Finf, nev, ok = m.solve(P0=x_warm, f_tol=1e-12, maxiter=80)
        if Finf <= f_tol:
            ok = True
        deficit, slope = m.deficit_slope(x)
        em = m.econ_metrics(x)
        wall = time.time() - t0
        rec = dict(K=K, G=G, tau=tau, gamma=gamma, deficit=deficit,
                   slope=slope, Finf=Finf, n_evals=nev, converged=bool(ok),
                   wall=wall, n_inner=m.n_inner, h=m.h, du=m.du, **em)
        results[key] = rec
        x_warm = x  # continuation
        if save_P:
            np.save(f"{OUT}/P_{key}.npy", x)
        json.dump(results, open(RESULTS, 'w'), indent=1)
        print(f"  {key:<28} deficit={deficit:.4e} slope={slope:.4f} "
              f"F={Finf:.1e} TV={em['TV']:.3f} conv={ok} ({wall:.1f}s)",
              flush=True)


print("=== A. K=3 G=21 reference rows ===", flush=True)
for tau in TAUS:
    run_row(3, 21, tau, GAMMAS)

print("=== B. K=3 G=15 baseline rows ===", flush=True)
for tau in TAUS:
    run_row(3, 15, tau, GAMMAS)

print("=== C. K=4 G=15 main rows (T1) ===", flush=True)
for tau in TAUS:
    run_row(4, 15, tau, GAMMAS, save_P=True)

print(f"[{(time.time()-T_START)/60:.1f} min] === D. G=11 K-ladder, tau=0.2 ===",
      flush=True)
for K in (3, 4, 5):
    run_row(K, 11, 0.20, GAMMAS, save_P=(K == 5))

elapsed = (time.time() - T_START) / 60
print(f"[{elapsed:.1f} min] === E. K=5 G=15 spot cells, tau=0.2 ===", flush=True)
if elapsed < 55:
    run_row(5, 15, 0.20, [0.1, 1.0, 10.0], save_P=True)
else:
    print("  skipped (budget)", flush=True)

print(f"DONE in {(time.time()-T_START)/60:.1f} min; "
      f"{sum(1 for v in results.values() if v['converged'])}/{len(results)} converged",
      flush=True)
