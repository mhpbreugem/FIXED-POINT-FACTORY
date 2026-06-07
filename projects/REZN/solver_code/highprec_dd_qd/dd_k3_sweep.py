"""DD K=3 (gamma, tau) sweep with Jacobian reuse across each gamma column.

Architecture:
  - Grid: gamma in {0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000} (10 pts incl center)
           tau   in {0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}
  - Within each gamma column: walk tau from 2.0 -> 0.1 with warm-start chain.
    Compute the FD Jacobian once at tau=2.0; reuse for nail at later tau.
    If nail diverges, refresh Jacobian for the current cell.
  - Across gamma: each column seeded from neighbor (gamma=100 anchor as before).

Per-cell save: P (G^3 float64), P_H/P_L (DD), F, slope, deficit, time.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
from itertools import permutations
import scipy.linalg as sla
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

import dd_k3_ops as DK
import dd_ops as DO
import dd_k3_solver as DKS
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


# --- Grid / params ---
G = 7
G_p = 121
NQK = 16
HS = (0.5, 0.4, 0.3, 0.2)     # R4 Richardson
TARGET_DD = 1e-25
TARGET_WARM = 1e-12
NAIL_MAX = 8

GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
# 9 values — keep 10x10 budget by adding one more
GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
TAUS   = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ANCHOR_GAMMA = 100.0

OUT = "/tmp/dd_k3_sweep.json"
FPS = "/tmp/dd_k3_sweep_fps"
os.makedirs(FPS, exist_ok=True)


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


def deficit_oneToOne(P, T):
    """Nonparametric R^2 deficit grouping logit(P) by unique T values."""
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2 = 1 - float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv == g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2, w/ss


def build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du, th, tl, gh, gl,
                       hs, weights, eps=1e-7):
    n = red.n_red
    F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                      th, tl, gh, gl, hs, weights)
    F0 = F0_H + F0_L
    J = np.empty((n, n))
    for k in range(n):
        xp = x_H.copy(); xp[k] += eps
        Fp_H, Fp_L = DKS.F_dd_red(red, xp, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, weights)
        J[:, k] = ((Fp_H + Fp_L) - F0)/eps
    return J, F0_H, F0_L


def nail_with_jac(red, x_H, x_L, J, u_grid, p_grid, gl_u, gl_du, th, tl, gh, gl,
                       hs, weights, max_iters=NAIL_MAX, target=TARGET_DD):
    n = red.n_red
    try: lu, piv = sla.lu_factor(J)
    except Exception:
        F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, weights)
        return x_H, x_L, float(np.max(np.abs(F0_H + F0_L)))
    Jinv = sla.lu_solve((lu, piv), np.eye(n))
    F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                      th, tl, gh, gl, hs, weights)
    F_inf = float(np.max(np.abs(F0_H + F0_L))); best = F_inf
    best_x_H = x_H.copy(); best_x_L = x_L.copy()
    for it in range(max_iters):
        F = F0_H + F0_L
        dx = -Jinv @ F
        for i in range(n):
            aH, aL = DO.dd_add(x_H[i], x_L[i], dx[i], 0.0)
            x_H[i] = aH; x_L[i] = aL
        F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, hs, weights)
        F_inf = float(np.max(np.abs(F0_H + F0_L)))
        if F_inf < best:
            best = F_inf; best_x_H = x_H.copy(); best_x_L = x_L.copy()
        if F_inf < target: break
        if it >= 2 and F_inf > 0.9*best:
            x_H = best_x_H.copy(); x_L = best_x_L.copy(); break
    return best_x_H, best_x_L, best


def solve_one(gamma, tau, u_grid, p_grid, gl_u, gl_du, P_warm,
                  J_cache=None, refresh_jac=False):
    """Returns (P_full, F, P_H, P_L, J_used)."""
    G = u_grid.size
    red = DKS.SymRed3(G)
    th, tl = split(mp.mpf(repr(tau)))
    gh, gl = split(mp.mpf(repr(gamma)))
    w_arr = DK.richardson_weights(HS)
    # warm-start in float64
    t_w = time.time()
    P_warm, F_warm = DKS.solve_warm_f64(P_warm, u_grid, p_grid, HS, gamma, tau,
                                                  p_grid.size, gl_u.size,
                                                  target=TARGET_WARM)
    t_warm = time.time() - t_w
    x_H = red.reduce(P_warm).astype(np.float64)
    x_L = np.zeros(red.n_red)
    # Jacobian
    t_j = time.time()
    if J_cache is None or refresh_jac:
        J, _, _ = build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                       th, tl, gh, gl, HS, w_arr)
    else:
        J = J_cache
    t_jac = time.time() - t_j
    # nail
    t_n = time.time()
    x_H, x_L, F = nail_with_jac(red, x_H, x_L, J, u_grid, p_grid, gl_u, gl_du,
                                       th, tl, gh, gl, HS, w_arr)
    t_nail = time.time() - t_n
    # If still bad, refresh Jacobian once and retry
    if F > 1e-15 and not refresh_jac:
        t_j = time.time()
        J, _, _ = build_jacobian(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                       th, tl, gh, gl, HS, w_arr)
        t_jac += time.time() - t_j
        x_H, x_L, F = nail_with_jac(red, x_H, x_L, J, u_grid, p_grid, gl_u,
                                           gl_du, th, tl, gh, gl, HS, w_arr)
        t_nail += time.time() - t_n
    P_H = red.expand(x_H); P_L = red.expand(x_L)
    P_full = P_H + P_L
    return P_full, F, P_H, P_L, J, F_warm, t_warm, t_jac, t_nail


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T_full = U1 + U2 + U3
    P_cold = 1.0/(1.0+np.exp(-0.5*T_full))

    # JIT warmup
    print("JIT warmup...", flush=True); t0 = time.time()
    w_arr = DK.richardson_weights(HS); th, tl = split(mp.mpf("1.0"))
    gh, gl = split(mp.mpf("1.0"))
    P_H = P_cold.copy(); P_L = np.zeros_like(P_cold)
    DK.phi_dd(P_H, P_L, u_grid, p_grid, gl_u, gl_du, th, tl, gh, gl, HS, w_arr)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print(f"Resume: {len(results)} cells already saved", flush=True)

    n_total = len(GAMMAS) * len(TAUS); n_done = 0; t_sweep = time.time()
    # Per-tau seed pool: each entry is (gamma_used, P_warm, F).
    seed_pool = {}

    # Order: gammas walked outward from anchor=100 (so each column seeded by
    # the nearest already-solved gamma).
    anchor_i = GAMMAS.index(ANCHOR_GAMMA)
    gamma_order = [ANCHOR_GAMMA]
    for k in range(1, max(len(GAMMAS)-anchor_i, anchor_i+1)):
        if anchor_i + k < len(GAMMAS): gamma_order.append(GAMMAS[anchor_i + k])
        if anchor_i - k >= 0: gamma_order.append(GAMMAS[anchor_i - k])

    for gamma in gamma_order:
        print(f"\n=== Column gamma={gamma} ===", flush=True)
        J_col = None        # Jacobian for this column (computed once at tau=2)
        for tau in TAUS[::-1]:
            n_done += 1
            key = f"g{gamma:.4g}_t{tau:.4f}"
            if key in results and results[key].get("F", 1.0) < 1e-22:
                # reload as seed
                f = f"{FPS}/{key}.npz"
                if os.path.exists(f):
                    d = np.load(f)
                    seed_pool[(gamma, tau)] = (gamma, d["mu_hi"] + d["mu_lo"], results[key]["F"])
                continue
            # Pick warm-start: nearest DD-precise neighbor in same column or same tau.
            P_warm = None; best_d = float("inf")
            for (g2, t2), (gp, P2, F2) in seed_pool.items():
                if F2 > 1e-15: continue
                d = abs(np.log10(g2/gamma)) + 5*abs(t2 - tau)
                if d < best_d: best_d = d; P_warm = P2
            if P_warm is None: P_warm = P_cold.copy()
            # solve
            try:
                P_full, F, P_H, P_L, J_col, F_warm, t_w, t_j, t_n = solve_one(
                    gamma, tau, u_grid, p_grid, gl_u, gl_du, P_warm,
                    J_cache=J_col, refresh_jac=(tau == TAUS[-1]))
            except Exception as e:
                print(f"  [{n_done}/{n_total}] g={gamma} t={tau}: ERROR {e}", flush=True)
                continue
            slope, def_l, def_1to1 = deficit_oneToOne(P_full, T_full)
            np.savez(f"{FPS}/{key}.npz", mu_hi=P_H, mu_lo=P_L, P=P_full,
                       gamma=gamma, tau=tau, F=float(F), slope=slope,
                       deficit_lin=def_l, deficit_oneToOne=def_1to1)
            results[key] = dict(gamma=float(gamma), tau=float(tau), F=float(F),
                                  F_warm=float(F_warm), slope=float(slope),
                                  deficit_lin=float(def_l),
                                  deficit_oneToOne=float(def_1to1),
                                  t_warm=t_w, t_jac=t_j, t_nail=t_n)
            json.dump(results, open(OUT, "w"), indent=2)
            if F < 1e-15:
                seed_pool[(gamma, tau)] = (gamma, P_full.copy(), F)
            elapsed = time.time() - t_sweep
            eta = elapsed*(n_total-n_done)/max(1, n_done)/60
            tag = "eps" if F < 1e-25 else "OK" if F < 1e-10 else "FAIL"
            print(f"  [{n_done}/{n_total}] g={gamma:7.3g} t={tau:.2f} F={F:.2e}[{tag}] "
                  f"warm={t_w:.1f}s jac={t_j:.0f}s nail={t_n:.0f}s "
                  f"slope={slope:.3f} d1to1={def_1to1:.4f} ETA={eta:.0f}min",
                  flush=True)

    n_eps = sum(1 for v in results.values() if v["F"] < 1e-25)
    n_conv = sum(1 for v in results.values() if v["F"] < 1e-10)
    print(f"\n=== DONE ===  {len(results)}/{n_total} in {(time.time()-t_sweep)/60:.0f}min",
          flush=True)
    print(f"  {n_eps} at DD eps; {n_conv} converged", flush=True)


if __name__ == "__main__":
    run()
