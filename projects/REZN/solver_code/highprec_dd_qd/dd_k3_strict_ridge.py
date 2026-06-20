"""Denser strict-h=0 (gamma, tau) sweep for the "deficit ridge" study.

This is recommendation #2 from the K=3 overnight study: re-do the deficit
heatmap with the corrected (strict-h=0) operator, so we can tell whether
the spurious ridge produced by the kernel-band R4 operator survives once
the ~0.05-0.6 mu bias is removed.

For each (gamma, tau) in the grid we:
  - Solve the strict-h=0 fixed point starting from the nearest warm-start
    (existing strict FP at tau in {0.2, 1.0}, otherwise the previous tau in the
    column).
  - Solve the kernel-band R4 fixed point at the same cell (warm-start from the
    saved R4 FPs when possible, otherwise from the strict FP).
  - Compute slope and deficit_oneToOne for both.
  - Save a NPZ per cell with P_strict, P_R4, mu tables, residuals, deficit.

Outputs land in:
  projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge/
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

from lin_cdf_strict import (build_mu_table_lin_strict, phi_lin_strict_jit,
                                  make_cdf_uniform_grid, make_p_grid,
                                  make_gl_for_u)
from lin_cdf_richardson import phi_lin_richardson_jit, richardson_weights


# --- Grid / params ---
G = 11
G_p = 121
NQ_STRICT = 16
NQK = 16
HS_R4 = (0.5, 0.4, 0.3, 0.2)
TARGET = 1e-10

GAMMAS = [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
TAUS   = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

OUT_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge"
EXISTING_STRICT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
EXISTING_R4 = "/tmp/dd_k3_sweep_fps"


def solve_strict(P_warm, u_grid, p_grid, gl_u, gl_du, gamma, tau,
                    target=TARGET, n_anderson=80):
    """Anderson + NK on the strict-h=0 operator. Returns best iterate found."""
    G_ = u_grid.size
    def F(xflat):
        Pn = phi_lin_strict_jit(xflat.reshape(G_, G_, G_), u_grid, p_grid,
                                       gl_u, gl_du, tau, gamma, G_, NQ_STRICT)
        return (Pn - xflat.reshape(G_, G_, G_)).ravel()
    x = P_warm.ravel().copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float("inf")
    for it in range(n_anderson):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < target: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1:
            x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except Exception:
                x = gx
    # NK fallback
    if min(Fs) > target:
        try:
            xnk = newton_krylov(F, x_best, f_tol=target, maxiter=20, verbose=False)
            fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G_, G_, G_), fnk, len(Fs)+20
        except NoConvergence as e:
            xnk = e.args[0]; fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G_, G_, G_), fnk, len(Fs)+20
        except Exception:
            pass
    return x_best.reshape(G_, G_, G_), min(Fs), len(Fs)


def solve_R4(P_warm, u_grid, p_grid, gl_u, gl_du, gamma, tau,
                target=TARGET, n_anderson=80):
    """Anderson + NK on the kernel-band R4 operator. Same scheme as strict."""
    G_ = u_grid.size
    hs_arr = np.array(HS_R4, dtype=float)
    w_arr = richardson_weights(HS_R4)
    def F(xflat):
        Pn = phi_lin_richardson_jit(xflat.reshape(G_, G_, G_), u_grid, p_grid,
                                       gl_u, gl_du, tau, gamma, G_, NQK,
                                       hs_arr, w_arr)
        return (Pn - xflat.reshape(G_, G_, G_)).ravel()
    x = P_warm.ravel().copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float("inf")
    for it in range(n_anderson):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < target: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1:
            x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except Exception:
                x = gx
    if min(Fs) > target:
        try:
            xnk = newton_krylov(F, x_best, f_tol=target, maxiter=20, verbose=False)
            fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G_, G_, G_), fnk, len(Fs)+20
        except NoConvergence as e:
            xnk = e.args[0]; fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G_, G_, G_), fnk, len(Fs)+20
        except Exception:
            pass
    return x_best.reshape(G_, G_, G_), min(Fs), len(Fs)


def fit_slope_deficit(P_full, T_full):
    """Return (slope, R2-deficit-vs-linear, nonparametric one-to-one deficit).
    Same definition as dd_k3_sweep.deficit_oneToOne."""
    Pc = np.clip(P_full, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T_full.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    pred = s*Tf + np.mean(L - s*Tf)
    R2_lin = 1 - float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv == g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, 1 - R2_lin, w/ss


def load_existing_strict(gamma, tau):
    """Try to load a saved strict-h=0 FP from the K=3 study (tau in {0.2, 1.0})."""
    key = f"g{int(gamma) if gamma==int(gamma) else gamma:g}_t{tau:.4f}"
    f = os.path.join(EXISTING_STRICT, f"dd_k3_strict_fp_{key}.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["P_strict"].astype(np.float64)
    return None


def load_existing_R4(gamma, tau):
    """Try to load a saved R4 FP from the dd_k3_sweep (tau in {0.2, 1.0})."""
    key = f"g{int(gamma) if gamma==int(gamma) else gamma:g}_t{tau:.4f}"
    f = os.path.join(EXISTING_R4, f"{key}.npz")
    if os.path.exists(f):
        d = np.load(f)
        return d["P"].astype(np.float64)
    return None


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T_full = U1 + U2 + U3
    P_cold = 1.0/(1.0 + np.exp(-0.5*T_full))

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

    # JIT warmup
    print("JIT warmup...", flush=True); t0 = time.time()
    _ = phi_lin_strict_jit(P_cold, u_grid, p_grid, gl_u, gl_du, 1.0, 1.0,
                                G, NQ_STRICT)
    hs_arr = np.array(HS_R4, dtype=float); w_arr = richardson_weights(HS_R4)
    _ = phi_lin_richardson_jit(P_cold, u_grid, p_grid, gl_u, gl_du, 1.0, 1.0,
                                    G, NQK, hs_arr, w_arr)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    results = {}
    # cache warm starts per gamma column
    n_total = len(GAMMAS) * len(TAUS); n_done = 0; t_sweep = time.time()
    for gamma in GAMMAS:
        print(f"\n=== Column gamma={gamma} ===", flush=True)
        # walk tau from 1.0 down to 0.1, warm-starting from prev cell
        last_P_strict = None
        last_P_R4 = None
        for tau in sorted(TAUS, reverse=True):
            n_done += 1
            key = f"g{gamma:g}_t{tau:.4f}"
            # Pick STRICT warm-start: prefer existing saved strict FP, then
            # last solved cell in column, then cold sigmoid.
            P_warm_strict = load_existing_strict(gamma, tau)
            if P_warm_strict is None:
                P_warm_strict = last_P_strict if last_P_strict is not None else P_cold.copy()
            # Pick R4 warm-start similarly
            P_warm_R4 = load_existing_R4(gamma, tau)
            if P_warm_R4 is None:
                P_warm_R4 = last_P_R4 if last_P_R4 is not None else P_cold.copy()

            # Solve strict
            t0 = time.time()
            P_str, F_str, it_str = solve_strict(P_warm_strict, u_grid, p_grid,
                                                       gl_u, gl_du, gamma, tau)
            t_str = time.time() - t0
            # Solve R4
            t0 = time.time()
            P_R4, F_R4, it_R4 = solve_R4(P_warm_R4, u_grid, p_grid, gl_u, gl_du,
                                                  gamma, tau)
            t_R4 = time.time() - t0

            s_str, _, def_str = fit_slope_deficit(P_str, T_full)
            s_R4,  _, def_R4  = fit_slope_deficit(P_R4,  T_full)
            bias_P = float(np.max(np.abs(P_str - P_R4)))

            # build mu tables for diagnostics
            mu_str = build_mu_table_lin_strict(P_str, u_grid, p_grid, gl_u,
                                                       gl_du, tau, G, NQ_STRICT)
            mu_R4_strict = build_mu_table_lin_strict(P_R4, u_grid, p_grid, gl_u,
                                                              gl_du, tau, G, NQ_STRICT)

            np.savez(os.path.join(OUT_DIR, f"strict_ridge_{key}.npz"),
                       P_strict=P_str, P_R4=P_R4, mu_strict=mu_str,
                       mu_strict_at_R4=mu_R4_strict, gamma=gamma, tau=tau,
                       F_strict=F_str, F_R4=F_R4,
                       slope_strict=s_str, slope_R4=s_R4,
                       deficit_strict=def_str, deficit_R4=def_R4,
                       bias_P_inf=bias_P)
            results[key] = dict(gamma=float(gamma), tau=float(tau),
                                  F_strict=float(F_str), F_R4=float(F_R4),
                                  iters_strict=int(it_str), iters_R4=int(it_R4),
                                  wall_strict=t_str, wall_R4=t_R4,
                                  slope_strict=float(s_str), slope_R4=float(s_R4),
                                  deficit_strict=float(def_str),
                                  deficit_R4=float(def_R4),
                                  bias_P_inf=float(bias_P))
            json.dump(results, open(os.path.join(OUT_DIR, "ridge.json"), "w"),
                        indent=2, default=str)
            elapsed = time.time() - t_sweep
            eta = elapsed * (n_total - n_done)/max(1, n_done) / 60
            tag_s = "OK" if F_str < 1e-6 else ("warn" if F_str < 1e-1 else "BAD")
            tag_4 = "OK" if F_R4  < 1e-6 else ("warn" if F_R4  < 1e-1 else "BAD")
            print(f"  [{n_done}/{n_total}] g={gamma:7g} t={tau:.2f} "
                  f"F_str={F_str:.1e}[{tag_s}] F_R4={F_R4:.1e}[{tag_4}] "
                  f"def_str={def_str:.4f} def_R4={def_R4:.4f} "
                  f"bias={bias_P:.3f} t={t_str:.0f}+{t_R4:.0f}s ETA={eta:.0f}m",
                  flush=True)
            last_P_strict = P_str.copy()
            last_P_R4 = P_R4.copy()

    print(f"\nDONE: {len(results)} cells in {(time.time()-t_sweep)/60:.1f}m",
          flush=True)


if __name__ == "__main__":
    run()
