"""Low-tau certification: extend the emin15 pass to tau in {0.05, 0.10}.
Reuses ld_polish machinery but adapts the warm-start chain (cold start
allowed since tau=0.2 already certified; do gamma-continuation per row)."""
import os, sys, json, time, warnings
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/tmp')
import numpy as np
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
warnings.filterwarnings('ignore')
# Copy ld_ops to /tmp because ld_polish imports it as 'ld_ops'
import shutil
shutil.copy('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd/ld_ops.py', '/tmp/ld_ops.py')
import ld_polish as P  # imports ld_ops, defines chain64, polish_ld, etc.

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
os.makedirs(OUT, exist_ok=True)

TAUS_LOW = [0.05, 0.10]
GAMMAS = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 20), 4))

def main():
    t_start = time.time()
    results = {}
    for tau in TAUS_LOW:
        print(f"\n=== TAU = {tau} ===", flush=True)
        P_cross = None
        for gamma in GAMMAS:
            t0 = time.time()
            key = f"t{tau}_g{gamma}"
            P64, F64 = P.chain64(tau, gamma, P_cross)
            if P64 is None or F64 > 1e-8:
                print(f"  tau={tau} gamma={gamma}: chain64 failed F={F64}", flush=True)
                results[key] = dict(tau=tau, gamma=gamma, status='chain_fail', F64=float(F64))
                continue
            P_cross = P64
            du, uf, lo, hi = P.build_grid(21)
            U1, U2, U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
            T = U1+U2+U3
            P_ld, hist = P.polish_ld(tau, gamma, P64)
            F_ld = hist[-1]
            s, d = P.metrics(P_ld, T)
            wall = time.time() - t0
            verdict = 'ACCEPT' if F_ld <= 1e-15 else 'PROVISIONAL'
            results[key] = dict(tau=tau, gamma=gamma, F64=float(F64),
                                F_ld=float(F_ld), slope=float(s), deficit=float(d),
                                verdict=verdict, wall=float(wall))
            print(f"  tau={tau:5.2f} gamma={gamma:8.4f}  F64={F64:.2e}  "
                  f"F_ld={F_ld:.2e}  d={d:.3e}  {verdict}  ({wall:.0f}s)", flush=True)
            if verdict == 'ACCEPT':
                np.save(f"{OUT}/P_ld_t{tau}_g{gamma}.npy", np.asarray(P_ld, np.float64))
            json.dump(results, open(f"{OUT}/lowtau.json", 'w'), indent=2, default=str)
    n_acc = sum(1 for v in results.values() if v.get('verdict') == 'ACCEPT')
    print(f"\n{n_acc}/{len(results)} ACCEPTED at F_ld <= 1e-15; "
          f"total {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
