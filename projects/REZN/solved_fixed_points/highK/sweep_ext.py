"""Extension sweep (budget allowed):
  - K=5 G=15 full tau=0.2 row (upgrade from 3 spot cells)
  - K=4 G=21 tau=0.2 row (cross-K at the certified G=21 discretisation)
  - K=5 G=11 rows at tau in {0.1, 0.4, 0.6} (tau-dependence at K=5)
"""
import sys
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK')
import sweep_highK as S  # reuses run_row + results dict (loads results.json)
import time

t0 = time.time()
print("=== EXT 1: K=5 G=15 tau=0.2 full row ===", flush=True)
S.run_row(5, 15, 0.20, S.GAMMAS, save_P=True)

print(f"[{(time.time()-t0)/60:.1f} min] === EXT 2: K=4 G=21 tau=0.2 row ===",
      flush=True)
S.run_row(4, 21, 0.20, S.GAMMAS)

print(f"[{(time.time()-t0)/60:.1f} min] === EXT 3: K=5 G=11 other taus ===",
      flush=True)
for tau in (0.10, 0.40, 0.60):
    S.run_row(5, 11, tau, S.GAMMAS)

print(f"EXT DONE in {(time.time()-t0)/60:.1f} min", flush=True)
