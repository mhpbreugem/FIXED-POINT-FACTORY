"""Method C v2 (gamma, tau) grid ladder.
10 gammas x 10 taus = 100 cells. Chain warm-start. Record F, slope, deficit,
+ analytic mu lookup at each cell.
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import dd_k3_method_c_v2 as MC2
import dd_k3_mu_analytic as MA
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid


GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
TAUS   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
G = 7

ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_grid"
OUT_JSON = f"{ROOT}/grid.json"
COEFS_DIR = f"{ROOT}/coefs"
os.makedirs(COEFS_DIR, exist_ok=True)


def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, w/ss


def main():
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T = U1 + U2 + U3
    print(f"=== Method C v2 (gamma, tau) grid ===", flush=True)
    print(f"  {len(GAMMAS)} gammas x {len(TAUS)} taus = {len(GAMMAS)*len(TAUS)} cells", flush=True)
    print("JIT warmup...", flush=True); t0 = time.time()
    MC2.build_P(np.zeros(5), u_grid)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    # For each gamma, walk tau (chain warm-start)
    t_start = time.time(); n_done = 0; n_total = len(GAMMAS)*len(TAUS)
    for gamma in GAMMAS:
        coefs_warm = None
        print(f"\n--- gamma={gamma} ---", flush=True)
        for tau in TAUS:
            n_done += 1
            key = f"g{gamma:.4g}_t{tau:.4f}"
            if key in results and results[key].get("F", 1) < 1e-6:
                coefs_warm = np.array(results[key]["coefs"])
                continue
            t0 = time.time()
            try:
                coefs, P, F, viol, ts = MC2.solve(tau, params0=coefs_warm,
                                                            gamma=gamma, max_nfev=200,
                                                            verbose=False)
            except Exception as e:
                print(f"  ERR {key}: {e}", flush=True); continue
            wall = time.time() - t0
            s, dd = fit(P, T)
            np.save(f"{COEFS_DIR}/{key}.npy", coefs)
            results[key] = dict(gamma=float(gamma), tau=float(tau),
                                          F=float(F), viol=int(viol), slope=float(s),
                                          deficit=float(dd),
                                          P_min=float(P.min()), P_max=float(P.max()),
                                          wall=float(wall), coefs=coefs.tolist())
            json.dump(results, open(OUT_JSON, "w"), indent=2)
            tag = "✓" if F < 1e-3 else "M" if F < 1e-1 else "✗"
            print(f"  [{n_done:>3d}/{n_total}] g={gamma:>6.3g} t={tau:.3f} "
                  f"F={F:.2e}[{tag}] s={s:.3f} d={dd:.4f} {wall:.1f}s",
                  flush=True)
            coefs_warm = coefs
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.1f}min",
          flush=True)


if __name__ == "__main__":
    main()
