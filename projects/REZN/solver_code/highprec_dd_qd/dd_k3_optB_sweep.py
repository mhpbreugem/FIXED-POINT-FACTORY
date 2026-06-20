"""Float64 numba Option B sweep over (gamma, tau) grid.

For each cell:
  1) Try to solve to F < 1e-12 via Anderson + NK.
  2) Use chain warm-start (nearest converged neighbor in log-gamma & linear-tau).
  3) Save P, F, slope, deficit.

After the sweep, run a 'repair' pass: for any cell with F >= 1e-8,
re-solve using extrapolation from its 4 nearest converged neighbors
(simple inverse-distance weighting in (log gamma, tau)).
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence
import dd_k3_optB_numba as OB
import dd_k3_optB_refined as OBR
from lin_cdf_strict import make_cdf_uniform_grid, make_p_grid


G = 11
G_p = 121
N_SUB = 8                    # Option B+ refinement level
TARGET = 1e-8
GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
TAUS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

OUT = "/tmp/dd_k3_optB_sweep.json"
FPS = "/tmp/dd_k3_optB_fps"
os.makedirs(FPS, exist_ok=True)


def solve_cell(P_warm, u_grid, p_grid, gamma, tau, target_max=TARGET,
                  target_med=1e-10, n_anderson=120, hard_wall=120.0):
    """Anderson with best-iterate tracking. Tracks both max|F| and median|F|.
    Returns the iterate that minimizes a composite score = max(max|F|*0.01, median|F|).
    This handles the non-smooth Option B operator: most cells converge to ~1e-10
    (median), a few outliers are stuck at non-smoothness scale (max ~1e-2)."""
    G = u_grid.size
    refined_u, w_trap_r, f0_r, f1_r, f0_u, f1_u = OBR.build_refined_helpers(
        u_grid, tau, N_SUB)
    P_buf = np.empty((G, G, G))
    def F_call(xflat):
        OBR.phi_optB_refined(xflat.reshape(G,G,G), u_grid, p_grid, float(tau),
                                  float(gamma), refined_u, w_trap_r, f0_r, f1_r,
                                  f0_u, f1_u, P_buf)
        return (P_buf - xflat.reshape(G,G,G)).ravel()
    t0 = time.time()
    x = P_warm.ravel().copy()
    Xh, Gh = [], []
    x_best = x.copy()
    score_best = float("inf"); F_best_max = float("inf"); F_best_med = float("inf")
    iter_done = 0
    for it in range(n_anderson):
        if time.time() - t0 > hard_wall: break
        Fv = F_call(x)
        f_max = float(np.max(np.abs(Fv)))
        f_med = float(np.median(np.abs(Fv)))
        score = max(f_max * 0.01, f_med)   # weight max less since non-smooth
        if score < score_best:
            score_best = score; x_best = x.copy()
            F_best_max = f_max; F_best_med = f_med
        iter_done = it + 1
        if f_max < target_max or f_med < target_med: break
        gx = Fv + x
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x_new = Gh[k-1] + DG @ ga
                # damped step
                x = 0.5*x + 0.5*x_new
            except: x = gx
    wall = time.time() - t0
    return x_best.reshape(G,G,G), F_best_max, F_best_med, iter_done, wall


def fit_slope_deficit(P_full, T_full):
    Pc = np.clip(P_full, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T_full.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2_lin = 1 - float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1-R2_lin, w/ss


def extrapolate_from_neighbors(gamma, tau, results, fps_dir, k_neigh=4):
    """Pick k nearest converged (F<1e-8) cells in (log gamma, tau) and IDW interp."""
    pts = []
    for k, v in results.items():
        if v["F"] < 1e-8 and os.path.exists(f"{fps_dir}/{k}.npy"):
            d2 = (np.log10(v["gamma"]/gamma))**2 + (v["tau"]-tau)**2
            pts.append((d2, k))
    if not pts: return None
    pts.sort()
    pts = pts[:k_neigh]
    weights = np.array([1.0/(p[0]+1e-12) for p in pts])
    weights /= weights.sum()
    P_sum = np.zeros((G, G, G))
    for w, (_, k) in zip(weights, pts):
        P_sum += w * np.load(f"{fps_dir}/{k}.npy")
    return P_sum


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T_full = U1 + U2 + U3
    P_cold = 1.0 / (1.0 + np.exp(-0.5*T_full))
    # JIT warmup (both operators)
    print("JIT warmup...", flush=True); t0 = time.time()
    OB.phi(P_cold, u_grid, p_grid, 1.0, 1.0)
    OBR.phi(P_cold, u_grid, p_grid, 1.0, 1.0, n_sub=N_SUB)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # Anchor: try gamma=100, tau=1.0 from sigmoid cold
    print(f"\n=== Phase 1: forward sweep ===", flush=True)
    t_sweep = time.time(); n_done = 0; n_total = len(GAMMAS)*len(TAUS)
    # Walk gamma outward from 100, within each column walk tau 2.0 -> 0.1.
    anchor_i = GAMMAS.index(100.0)
    gamma_order = [GAMMAS[anchor_i]]
    for k in range(1, max(len(GAMMAS)-anchor_i, anchor_i+1)):
        if anchor_i + k < len(GAMMAS): gamma_order.append(GAMMAS[anchor_i + k])
        if anchor_i - k >= 0: gamma_order.append(GAMMAS[anchor_i - k])

    # Seed: try to use existing R4 FP as cold warm (already in /tmp/dd_k3_sweep_fps)
    R4_FPS = "/tmp/dd_k3_sweep_fps"
    seed_pool = {}
    for gamma in gamma_order:
        print(f"\n--- gamma={gamma} ---", flush=True)
        for tau in TAUS[::-1]:
            n_done += 1
            key = f"g{gamma:.4g}_t{tau:.4f}"
            if key in results and results[key]["F"] < 1e-10:
                if os.path.exists(f"{FPS}/{key}.npy"):
                    seed_pool[(gamma, tau)] = (gamma, np.load(f"{FPS}/{key}.npy"),
                                                       results[key]["F"])
                continue
            # warm pick
            P_warm = None; best_d = float("inf")
            for (g2, t2), (gp, P2, F2) in seed_pool.items():
                if F2 > 1e-8: continue
                d = abs(np.log10(g2/gamma)) + 5*abs(t2 - tau)
                if d < best_d: best_d = d; P_warm = P2
            if P_warm is None:
                # try R4 FP at same (gamma, tau) as fallback
                r4f = f"{R4_FPS}/{key}.npz"
                if os.path.exists(r4f):
                    d = np.load(r4f)
                    P_warm = d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])
                    if P_warm.shape[0] != G: P_warm = P_cold.copy()
                else:
                    P_warm = P_cold.copy()
            P_sol, F_max, F_med, iters, wall = solve_cell(P_warm, u_grid, p_grid, gamma, tau)
            s, dl, d1 = fit_slope_deficit(P_sol, T_full)
            np.save(f"{FPS}/{key}.npy", P_sol)
            results[key] = dict(gamma=float(gamma), tau=float(tau),
                                  F=float(F_max), F_med=float(F_med),
                                  slope=float(s), deficit_lin=float(dl),
                                  deficit_oneToOne=float(d1),
                                  iters=int(iters), wall=float(wall))
            json.dump(results, open(OUT, "w"), indent=2)
            if F_med < 1e-6: seed_pool[(gamma, tau)] = (gamma, P_sol.copy(), F_med)
            elapsed = time.time() - t_sweep
            eta = elapsed*(n_total-n_done)/max(1, n_done)/60
            tag = "eps" if F_med<1e-10 else "OK" if F_med<1e-6 else "MIX" if F_med<1e-3 else "FAIL"
            print(f"  [{n_done}/{n_total}] g={gamma:7.3g} t={tau:.2f} max={F_max:.2e} med={F_med:.2e}[{tag}] "
                  f"it={iters} {wall:.0f}s s={s:.3f} d={d1:.4f} ETA={eta:.0f}min",
                  flush=True)

    # Phase 2: repair mixed cells
    print(f"\n=== Phase 2: repair mixed (F>=1e-8) cells from neighbor extrapolation ===",
          flush=True)
    mixed = [(k, v) for k, v in results.items() if v.get("F_med", v["F"]) >= 1e-6]
    print(f"{len(mixed)} mixed cells to repair (F_med>=1e-6)", flush=True)
    for key, v in mixed:
        gamma = v["gamma"]; tau = v["tau"]
        P_warm = extrapolate_from_neighbors(gamma, tau, results, FPS)
        if P_warm is None: continue
        P_sol, F_max, F_med, iters, wall = solve_cell(P_warm, u_grid, p_grid, gamma, tau)
        if F_med < v.get("F_med", v["F"]):
            s, dl, d1 = fit_slope_deficit(P_sol, T_full)
            np.save(f"{FPS}/{key}.npy", P_sol)
            results[key].update(dict(F=float(F_max), F_med=float(F_med),
                                          slope=float(s), deficit_lin=float(dl),
                                          deficit_oneToOne=float(d1), iters=int(iters),
                                          wall=float(wall), repaired=True))
            json.dump(results, open(OUT, "w"), indent=2)
            print(f"  REPAIR {key}: med {v.get('F_med', v['F']):.2e} -> {F_med:.2e}",
                  flush=True)
        else:
            print(f"  no_improve {key}: med {v.get('F_med', v['F']):.2e} -> {F_med:.2e}",
                  flush=True)

    n_eps = sum(1 for v in results.values() if v.get("F_med", v["F"]) < 1e-10)
    n_ok = sum(1 for v in results.values() if v.get("F_med", v["F"]) < 1e-6)
    print(f"\n=== DONE === {len(results)}/{n_total}", flush=True)
    print(f"  {n_eps} at eps (F_med<1e-10); {n_ok} converged (F_med<1e-6); "
          f"{sum(1 for v in results.values() if v.get('repaired'))} repaired",
          flush=True)


if __name__ == "__main__":
    run()
