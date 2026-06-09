"""Full DD K=3 ladder: tau from 0.001 to 1.0 in steps of 0.001 (1000 cells).
Chain warm-start from each previous tau. Auto-save after each cell.
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
import dd_k3_ops as DK
import dd_k3_solver as DKS
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


G = 7
G_p = 121
NQK = 16
HS = (0.5, 0.4, 0.3, 0.2)
GAMMA = 100.0
TARGET_WARM = 1e-12
TARGET_DD = 1e-25

OUT_JSON = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/ladder.json"
FPS_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/fps"
os.makedirs(FPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


def main():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    print(f"=== DD K=3 full ladder gamma={GAMMA}, G={G} ===", flush=True)
    print("JIT warmup...", flush=True); t0 = time.time()
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P0 = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    PL0 = np.zeros_like(P0)
    w_arr = DK.richardson_weights(HS)
    DK.phi_dd(P0, PL0, u_grid, p_grid, gl_u, gl_du, 1.0, 0.0, GAMMA, 0.0, HS, w_arr)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    taus = [round(0.001 * i, 4) for i in range(1, 1001)]
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    P_warm = None
    if results:
        completed = sorted([float(k) for k in results.keys()])
        last = completed[-1] if completed else None
        if last is not None:
            f = f"{FPS_DIR}/tau{last:.4f}.npy"
            if os.path.exists(f):
                P_warm = np.load(f)
                print(f"resuming from tau={last}", flush=True)
                taus = [t for t in taus if t > last + 1e-6]
    n_total = 1000; n_done_seed = 1000 - len(taus)
    t_start = time.time()
    for i, tau in enumerate(taus):
        n_done = n_done_seed + i + 1
        t0 = time.time()
        try:
            P_full, F, P_H, P_L, t_warm, t_nail = DKS.solve_dd_k3(
                GAMMA, tau, P_warm=P_warm, u_grid=u_grid, p_grid=p_grid,
                gl_u=gl_u, gl_du=gl_du, hs=HS,
                target_dd=TARGET_DD, target_warm=TARGET_WARM, verbose=False)
        except Exception as e:
            print(f"  [{n_done}/{n_total}] tau={tau:.4f} ERROR: {e}", flush=True)
            continue
        wall = time.time() - t0
        np.save(f"{FPS_DIR}/tau{tau:.4f}.npy", P_full)
        results[f"{tau:.4f}"] = dict(tau=float(tau), F=float(F), wall=wall,
                                              t_warm=t_warm, t_nail=t_nail)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        tag = "eps" if F < TARGET_DD else "OK" if F < 1e-10 else "FAIL"
        elapsed = time.time() - t_start
        eta = elapsed * (len(taus) - i - 1) / max(1, i+1) / 60
        if n_done % 5 == 0 or F > 1e-15:
            print(f"  [{n_done:>4d}/{n_total}] tau={tau:.4f} F={F:.2e}[{tag}] "
                  f"{wall:.0f}s ETA={eta:.0f}min", flush=True)
        P_warm = P_full.copy()
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.0f}min",
          flush=True)


if __name__ == "__main__":
    main()
