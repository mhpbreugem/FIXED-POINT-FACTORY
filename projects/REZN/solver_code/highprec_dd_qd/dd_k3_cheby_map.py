"""Cheby-Lobatto convergence map across (gamma, tau).
For each (gamma, tau), solve the FP at G in {11, 15, 21, 31} on Lobatto grid.
Track edge-Cheby coefficient decay (smoothness indicator) and lookup
median error vs strict-h=0. Aim: identify the (gamma, tau) region where
Cheby converges fast (smooth FP) vs slow (kinky FP).
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

import dd_k3_cheby_lobatto as CL
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid


def study_cell(gamma, tau, Gs=(11, 15, 21, 31)):
    p_grid = make_p_grid(121)
    P_prev = None; results = {}
    for G in Gs:
        print(f"  G={G}...", end=" ", flush=True)
        t0 = time.time()
        u_new, U_MAX_new = CL.lobatto_grid(G)
        if P_prev is not None:
            G_prev = P_prev.shape[0]
            u_prev, U_MAX_prev = CL.lobatto_grid(G_prev)
            coefs_prev = CL.cheby_fit_lobatto_3d(P_prev, U_MAX_prev)
            P_init = np.empty((G,G,G))
            for ii in range(G):
                for jj in range(G):
                    for kk in range(G):
                        P_init[ii,jj,kk] = cheb.chebval3d(u_new[ii]/U_MAX_prev,
                                                              u_new[jj]/U_MAX_prev,
                                                              u_new[kk]/U_MAX_prev,
                                                              coefs_prev)
            P_init = np.clip(P_init, 1e-9, 1-1e-9)
        else: P_init = None
        try:
            P, F, u_grid, U_MAX = CL.solve_fp_lobatto(gamma, tau, G, P_init=P_init,
                                                                target=1e-12,
                                                                n_anderson=80)
        except Exception as e:
            print(f"FP FAIL {e}", flush=True)
            results[G] = dict(error=str(e)); continue
        t_fp = time.time() - t0
        if F > 1e-4:
            print(f"FP didn't converge: |F|={F:.2e} ({t_fp:.1f}s)", flush=True)
            results[G] = dict(F_fp=float(F), wall_fp=t_fp, no_lookup=True)
            P_prev = P; continue
        coefs = CL.cheby_fit_lobatto_3d(P, U_MAX)
        # Edge coef: max abs of coefs along the highest mode in each axis
        edge = float(max(np.max(np.abs(coefs[-1, :, :])),
                            np.max(np.abs(coefs[:, -1, :])),
                            np.max(np.abs(coefs[:, :, -1]))))
        # Lookup vs strict-h=0
        t0 = time.time()
        gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, G, 16)
        mu_cheb = np.empty((121, G))
        for k_idx in range(G):
            u_k = u_grid[k_idx]
            for ip in range(121):
                mu_cheb[ip, k_idx] = CL.lookup_analytic_lobatto(
                    coefs, U_MAX, p_grid[ip], u_k, tau, NQK=24)
        t_lookup = time.time() - t0
        d_max = float(np.max(np.abs(mu_cheb - mu_strict)))
        d_med = float(np.median(np.abs(mu_cheb - mu_strict)))
        print(f"|F|={F:.1e} edge={edge:.2e} max|d|={d_max:.2e} med={d_med:.2e} ({t_fp+t_lookup:.0f}s)",
              flush=True)
        results[G] = dict(F_fp=float(F), wall_fp=t_fp,
                              wall_lookup=t_lookup,
                              edge_coef=edge, lookup_max=d_max,
                              lookup_med=d_med)
        P_prev = P
    return results


def main():
    gammas = [1.0, 100.0, 1000.0]
    taus = [0.2, 0.5, 1.0, 2.0]
    Gs = (11, 15, 21, 31)
    all_results = {}
    for tau in taus:
        for gamma in gammas:
            key = f"g{gamma:.4g}_t{tau:.4f}"
            print(f"\n=== {key} ===", flush=True)
            res = study_cell(gamma, tau, Gs=Gs)
            all_results[key] = dict(gamma=gamma, tau=tau, by_G=res)
            json.dump(all_results, open("/tmp/dd_k3_cheby_map.json", "w"),
                        indent=2, default=str)
    print(f"\nDONE -> /tmp/dd_k3_cheby_map.json", flush=True)


if __name__ == "__main__":
    main()
