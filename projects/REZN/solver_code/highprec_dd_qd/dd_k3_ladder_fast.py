"""DD K=3 ladder with Jacobian reuse across cells.

Big speedup over per-cell rebuild: build Jacobian once at first cell, reuse
for chord Newton across the whole ladder. Refresh only when chord stalls,
or every REFRESH_EVERY cells.

Each cell: warm-start float64 R4, then 3-8 chord-Newton steps using the
cached Jinv. ~1s per cell instead of ~34s.
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
import scipy.linalg as sla
import dd_k3_ops as DK
import dd_k3_solver as DKS
import dd_ops as DO
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


G = 7
G_p = 121
NQK = 16
HS = (0.5, 0.4, 0.3, 0.2)
GAMMA = 100.0
TARGET_WARM = 1e-12
TARGET_DD = 1e-25
REFRESH_EVERY = 50      # rebuild Jacobian every N cells
MAX_NAIL = 8

ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder"
OUT_JSON = f"{ROOT}/ladder.json"
FPS_DIR = f"{ROOT}/fps"
os.makedirs(FPS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


def build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                       th, tl, gh, gl, hs, w_arr, eps=1e-7):
    n = red.n_red
    F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                      th, tl, gh, gl, hs, w_arr)
    F0 = F0_H + F0_L
    J = np.empty((n, n))
    for k in range(n):
        xp_H = x_H.copy(); xp_H[k] += eps
        Fp_H, Fp_L = DKS.F_dd_red(red, xp_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, w_arr)
        J[:, k] = ((Fp_H + Fp_L) - F0) / eps
    return J


def nail_with_jacobian(red, x_H, x_L, J, u_grid, p_grid, gl_u, gl_du,
                            th, tl, gh, gl, hs, w_arr,
                            target=TARGET_DD, max_iters=MAX_NAIL):
    n = red.n_red
    try: lu, piv = sla.lu_factor(J)
    except Exception:
        F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, w_arr)
        return x_H, x_L, float(np.max(np.abs(F0_H+F0_L))), False
    Jinv = sla.lu_solve((lu, piv), np.eye(n))
    F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                      th, tl, gh, gl, hs, w_arr)
    F_inf = float(np.max(np.abs(F0_H+F0_L)))
    best_F = F_inf
    best_x_H = x_H.copy(); best_x_L = x_L.copy()
    for it in range(max_iters):
        F = F0_H + F0_L
        dx = -Jinv @ F
        for i in range(n):
            aH, aL = DO.dd_add(x_H[i], x_L[i], dx[i], 0.0)
            x_H[i] = aH; x_L[i] = aL
        F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, w_arr)
        F_inf = float(np.max(np.abs(F0_H+F0_L)))
        if F_inf < best_F:
            best_F = F_inf; best_x_H = x_H.copy(); best_x_L = x_L.copy()
        if F_inf < target: return best_x_H, best_x_L, best_F, True
        if it >= 2 and F_inf > 0.9 * best_F: break
    return best_x_H, best_x_L, best_F, best_F < target


def main():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    print(f"=== DD K=3 fast ladder gamma={GAMMA}, G={G} ===", flush=True)
    print("JIT warmup...", flush=True); t0 = time.time()
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P0 = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    PL0 = np.zeros_like(P0)
    w_arr = DK.richardson_weights(HS)
    DK.phi_dd(P0, PL0, u_grid, p_grid, gl_u, gl_du, 1.0, 0.0, GAMMA, 0.0, HS, w_arr)
    print(f"  done {time.time()-t0:.1f}s", flush=True)
    red = DKS.SymRed3(G)
    gh, gl = split(mp.mpf("100"))

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
    J_cached = None; cache_age = 0
    for i, tau in enumerate(taus):
        n_done = n_done_seed + i + 1
        t0 = time.time()
        th, tl = split(mp.mpf(repr(tau)))
        # warm-start float64
        if P_warm is None:
            P_warm = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
        try:
            P_warm_new, F_warm = DKS.solve_warm_f64(P_warm, u_grid, p_grid, HS,
                                                              GAMMA, tau, G_p, NQK,
                                                              target=TARGET_WARM)
        except Exception as e:
            print(f"  [{n_done}] tau={tau} warm ERROR: {e}", flush=True); continue
        t_warm = time.time() - t0
        x_H = red.reduce(P_warm_new).astype(np.float64)
        x_L = np.zeros(red.n_red)
        # nail
        t_n = time.time()
        if J_cached is None or cache_age >= REFRESH_EVERY:
            t_j = time.time()
            J_cached = build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                              th, tl, gh, gl, HS, w_arr)
            cache_age = 0
            t_jac = time.time() - t_j
        else:
            t_jac = 0.0
        x_H, x_L, F, converged = nail_with_jacobian(red, x_H, x_L, J_cached,
                                                                  u_grid, p_grid, gl_u, gl_du,
                                                                  th, tl, gh, gl, HS, w_arr)
        # If not converged, refresh Jacobian and try once more
        if not converged:
            J_cached = build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                              th, tl, gh, gl, HS, w_arr)
            cache_age = 0
            x_H, x_L, F, converged = nail_with_jacobian(red, x_H, x_L, J_cached,
                                                                      u_grid, p_grid, gl_u, gl_du,
                                                                      th, tl, gh, gl, HS, w_arr)
        cache_age += 1
        t_nail = time.time() - t_n
        wall = time.time() - t0
        P_H = red.expand(x_H); P_L = red.expand(x_L)
        P_full = P_H + P_L
        np.save(f"{FPS_DIR}/tau{tau:.4f}.npy", P_full)
        results[f"{tau:.4f}"] = dict(tau=float(tau), F=float(F), wall=wall,
                                              t_warm=t_warm, t_jac=t_jac, t_nail=t_nail,
                                              cache_age=cache_age, refresh=(cache_age==1))
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        tag = "eps" if F < TARGET_DD else "OK" if F < 1e-10 else "FAIL"
        elapsed = time.time() - t_start
        eta = elapsed * (len(taus) - i - 1) / max(1, i+1) / 60
        if n_done % 25 == 0 or F > 1e-15 or t_jac > 5:
            print(f"  [{n_done:>4d}/{n_total}] tau={tau:.4f} F={F:.2e}[{tag}] "
                  f"warm={t_warm:.1f}s jac={t_jac:.1f}s nail={t_nail:.1f}s ETA={eta:.0f}min",
                  flush=True)
        P_warm = P_full
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.0f}min",
          flush=True)


if __name__ == "__main__":
    main()
