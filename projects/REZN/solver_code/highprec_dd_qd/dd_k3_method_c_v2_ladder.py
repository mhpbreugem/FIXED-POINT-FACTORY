"""Method C v2 ladder: walk tau from 0.001 to 1.0 in steps of 0.001.
For each tau, fit the Bayesian-symmetric monotone ansatz and record the
residual F (= structural non-additivity = the 'real' deficit).
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import dd_k3_method_c_v2 as MC2


GAMMA = 100.0
ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2"
OUT_JSON = f"{ROOT}/ladder.json"
COEFS_DIR = f"{ROOT}/coefs"
os.makedirs(COEFS_DIR, exist_ok=True)


def main():
    print(f"=== Method C v2 ladder gamma={GAMMA}, D_SIGMA={MC2.D_SIGMA} ===", flush=True)
    print("JIT warmup...", flush=True); t0 = time.time()
    from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
    u_grid = make_cdf_uniform_grid(MC2.G)
    p_grid = make_p_grid(121)
    p0 = np.zeros(MC2.D_SIGMA+1); p0[0] = 0.5
    MC2.build_P(p0, u_grid)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    taus = [round(0.001 * i, 4) for i in range(1, 1001)]
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    coefs_warm = None
    if results:
        completed = sorted([float(k) for k in results.keys()])
        last = completed[-1] if completed else None
        if last is not None:
            f = f"{COEFS_DIR}/tau{last:.4f}.npy"
            if os.path.exists(f):
                coefs_warm = np.load(f)
                print(f"resuming from tau={last}", flush=True)
                taus = [t for t in taus if t > last + 1e-6]

    t_start = time.time(); n_total = 1000; n_done_seed = 1000 - len(taus)
    for i, tau in enumerate(taus):
        n_done = n_done_seed + i + 1
        t0 = time.time()
        try:
            coefs, P, F, viol, ts = MC2.solve(tau, params0=coefs_warm, gamma=GAMMA,
                                                          max_nfev=100, verbose=False)
        except Exception as e:
            print(f"  [{n_done}] tau={tau} ERROR: {e}", flush=True); continue
        wall = time.time() - t0
        np.save(f"{COEFS_DIR}/tau{tau:.4f}.npy", coefs)
        results[f"{tau:.4f}"] = dict(tau=float(tau), F=float(F),
                                              viol=int(viol), wall=float(wall),
                                              coefs=coefs.tolist(),
                                              P_min=float(P.min()), P_max=float(P.max()))
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        if n_done % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed*(len(taus)-i-1)/max(1, i+1)/60
            print(f"  [{n_done:>4d}/{n_total}] tau={tau:.4f} F={F:.2e} viol={viol} "
                  f"P=[{P.min():.3f},{P.max():.3f}] {wall:.1f}s ETA={eta:.0f}min",
                  flush=True)
        coefs_warm = coefs
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.0f}min",
          flush=True)


if __name__ == "__main__":
    main()
