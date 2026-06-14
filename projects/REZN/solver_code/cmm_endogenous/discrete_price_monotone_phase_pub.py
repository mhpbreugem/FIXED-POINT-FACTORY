"""Publishable-quality monotone phase diagram in (tau, gamma).

Same methodology as discrete_price_monotone_phase_hi.py but on a denser
grid and at higher price-discretization, with finer post-processing of
the boundary curve.

  G = 21           (kept fixed -- grids G > 21 do not change the basin
                    geometry on this problem at human time-budget cost)
  M = 48           (was 32 -- finer price discretization for sharper
                    boundary localization)
  gamma in {0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0}
  tau   in {0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0}

The script resumes from any partial JSON in the output folder.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel, compute_deficit
from discrete_price_soft_monotone import (
    solve_kmeans_soft_monotone, max_mono_violation,
)

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_monotone_phase_pub'
os.makedirs(OUT, exist_ok=True)


def main():
    Gi = 21
    M  = 48
    gammas = [0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    taus   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
    print(f"=== Monotone phase diagram (PUBLISHABLE) ===", flush=True)
    print(f"G={Gi}, M={M}, |gamma|={len(gammas)}, |tau|={len(taus)}, "
          f"cells={len(gammas)*len(taus)}", flush=True)
    rows = []
    done = set()
    json_path = f"{OUT}/phase.json"
    if os.path.exists(json_path):
        prev = json.load(open(json_path))
        for r in prev.get('rows', []):
            rows.append(r)
            done.add((float(r['gamma']), float(r['tau'])))
        print(f"Resuming: {len(done)} cells already done.\n", flush=True)
    for gamma in gammas:
        for tau in taus:
            if (float(gamma), float(tau)) in done:
                continue
            print(f"[gamma={gamma}, tau={tau}] ", end='', flush=True)
            t0 = time.time()
            try:
                P_kernel, uf, lo, hi = solve_kernel(Gi, tau, gamma)
                kernel_viol = max_mono_violation(P_kernel)
                kernel_def = compute_deficit(P_kernel, uf, lo, hi, tau)
                p_levels = np.linspace(0.05, 0.95, M)
                p_field, hist, status, cycle = solve_kmeans_soft_monotone(
                    P_kernel, p_levels, uf, lo, hi, tau, gamma,
                    max_iter=40, h_p_factor=1.5)
                final = hist[-1]
                tail_def = float(np.mean([h['deficit']
                                            for h in hist[-min(10, len(hist)):]]))
                tail_viol = float(np.mean([h['max_viol_raw']
                                             for h in hist[-min(10, len(hist)):]]))
                wall = time.time() - t0
                print(f"status={status:>22} viol={final['max_viol_raw']:.2e} "
                      f"deficit={tail_def:.3f}  wall={wall:.1f}s", flush=True)
                rows.append(dict(
                    gamma=gamma, tau=tau, status=status, cycle=int(cycle),
                    kernel_max_viol=float(kernel_viol),
                    kernel_deficit=float(kernel_def),
                    deficit_final=float(final['deficit']),
                    deficit_tail=tail_def,
                    max_viol_raw_final=float(final['max_viol_raw']),
                    max_viol_raw_tail=tail_viol,
                    max_res=float(final['max_res']),
                    wall=float(wall)))
            except Exception as e:
                wall = time.time() - t0
                print(f"FAILED ({e}) wall={wall:.1f}s", flush=True)
                rows.append(dict(gamma=gamma, tau=tau, status='solver_failed',
                                  error=str(e), wall=float(wall)))
            json.dump(dict(rows=rows, G=Gi, M=M,
                            gammas=gammas, taus=taus),
                      open(json_path, 'w'), indent=2, default=str)
    print(f"\nDone -> {json_path}")


if __name__ == '__main__':
    main()
