"""Solve the K=3 FP using STRICT-h=0 operator (no kernel) at G=11.

For each (gamma, tau), warm-start from the existing R4-kernel FP and refine via
Anderson + Newton--Krylov on the strict-h=0 operator. Record:
  - strict-h=0 |F| (residual under TRUE h=0 operator)
  - new slope, deficit at the strict-h=0 FP
  - |P_strict - P_R4|_inf (bias of the kernel-band FP vs the true h=0 FP)
  - max |mu_strict - mu_R4| at the strict-h=0 FP
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence
from lin_cdf_strict import (build_mu_table_lin_strict, phi_lin_strict_jit,
                                  phi_lin_strict, make_cdf_uniform_grid, make_p_grid,
                                  make_gl_for_u)
from lin_cdf_kern_tab import build_mu_table_lin_kern, make_p_grid as make_p_grid_k
from lin_cdf_richardson import richardson_weights


G = 11
G_p = 121
NQ_STRICT = 16
NQK_KERN = 16
HS_R4 = (0.5, 0.4, 0.3, 0.2)
TARGET = 1e-10


def solve_strict(P_warm, u_grid, p_grid, gl_u, gl_du, gamma, tau,
                    target=TARGET, n_anderson=80):
    """Anderson + NK on the strict-h=0 operator."""
    G = u_grid.size
    def F(xflat):
        Pn = phi_lin_strict_jit(xflat.reshape(G,G,G), u_grid, p_grid,
                                       gl_u, gl_du, tau, gamma, G, NQ_STRICT)
        return (Pn - xflat.reshape(G,G,G)).ravel()
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
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    # NK fallback
    if min(Fs) > target:
        try:
            xnk = newton_krylov(F, x_best, f_tol=target, maxiter=20, verbose=False)
            fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk, len(Fs)+20
        except NoConvergence as e:
            xnk = e.args[0]; fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk, len(Fs)+20
    return x_best.reshape(G,G,G), min(Fs), len(Fs)


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


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T_full = U1 + U2 + U3
    print("warmup strict-h=0 jit...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    phi_lin_strict_jit(P0, u_grid, p_grid, gl_u, gl_du, 1.0, 1.0, G, NQ_STRICT)
    print(f"  done {time.time()-t0:.1f}s", flush=True)
    fps = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    target_cells = [(g, t) for g in [100.0, 10.0, 1.0, 1000.0, 30.0] for t in [1.0, 0.2]]
    results = {}
    for gamma, tau in target_cells:
        key = f"g{gamma:.4g}_t{tau:.4f}"
        f = f"/tmp/dd_k3_sweep_fps/{key}.npz"
        if not os.path.exists(f):
            print(f"\nSKIP {key} (no R4 FP saved)", flush=True); continue
        d = np.load(f)
        P_R4 = d["P"] if "P" in d.files else (d["mu_hi"] + d["mu_lo"])
        if P_R4.shape[0] != G:
            print(f"\nSKIP {key} (G={P_R4.shape[0]}!={G})", flush=True); continue
        print(f"\n=== {key} (warm from R4 FP) ===", flush=True)
        t0 = time.time()
        P_strict, F_strict, iters = solve_strict(P_R4.astype(np.float64), u_grid,
                                                          p_grid, gl_u, gl_du,
                                                          float(gamma), float(tau))
        wall = time.time() - t0
        print(f"  strict-h=0 solve: F={F_strict:.3e} ({iters} iters, {wall:.0f}s)",
              flush=True)
        # Compute bias of R4 FP vs strict FP
        bias_P = float(np.max(np.abs(P_strict - P_R4)))
        # mu at the strict FP (truth) and at the R4 FP (biased)
        mu_strict_at_strict = build_mu_table_lin_strict(P_strict, u_grid, p_grid,
                                                                gl_u, gl_du, float(tau),
                                                                G, NQ_STRICT)
        mu_strict_at_R4 = build_mu_table_lin_strict(P_R4.astype(np.float64), u_grid,
                                                            p_grid, gl_u, gl_du,
                                                            float(tau), G, NQ_STRICT)
        # also kernel R4 at the strict FP
        gl_u_k, gl_du_k = make_gl_for_u(u_grid[0], u_grid[-1], NQK_KERN)
        mu_R4_at_strict = np.zeros((G_p, G))
        w_R4 = richardson_weights(HS_R4)
        for hi, h in enumerate(HS_R4):
            mu_h = build_mu_table_lin_kern(P_strict, u_grid, p_grid, gl_u_k, gl_du_k,
                                                float(tau), G, NQK_KERN, float(h))
            mu_R4_at_strict += float(w_R4[hi]) * mu_h
        mu_diff_at_strict = float(np.max(np.abs(mu_R4_at_strict - mu_strict_at_strict)))
        # slope/deficit at the strict FP
        s_str, dl_str, d1_str = fit_slope_deficit(P_strict, T_full)
        # slope/deficit at the R4 FP (recompute for sanity)
        s_R4, dl_R4, d1_R4 = fit_slope_deficit(P_R4, T_full)
        print(f"  bias |P_strict - P_R4|_inf = {bias_P:.3e}", flush=True)
        print(f"  |mu_R4(P_strict) - mu_strict(P_strict)|_inf = {mu_diff_at_strict:.3e}",
              flush=True)
        print(f"  slope (R4 vs strict): {s_R4:.4f} vs {s_str:.4f} "
              f"(shift {abs(s_R4-s_str):.4f})", flush=True)
        print(f"  deficit (R4 vs strict): {d1_R4:.4f} vs {d1_str:.4f} "
              f"(shift {abs(d1_R4-d1_str):.4f})", flush=True)
        results[key] = dict(gamma=float(gamma), tau=float(tau),
                              F_strict=float(F_strict), iters=int(iters), wall=wall,
                              bias_P_inf=bias_P,
                              mu_diff_R4_at_strict=mu_diff_at_strict,
                              slope_R4=s_R4, slope_strict=s_str,
                              deficit_R4=d1_R4, deficit_strict=d1_str)
        np.savez(f"/tmp/dd_k3_strict_fp_{key}.npz",
                   P_strict=P_strict, P_R4=P_R4,
                   mu_strict=mu_strict_at_strict, mu_R4_at_strict=mu_R4_at_strict)
        json.dump(results, open("/tmp/dd_k3_strict_solve.json", "w"), indent=2,
                    default=str)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_strict_solve.json",
          flush=True)


if __name__ == "__main__":
    run()
