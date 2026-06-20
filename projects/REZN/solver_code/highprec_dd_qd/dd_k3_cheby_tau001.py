"""Cheby-Lobatto convergence at gamma=100, tau=0.001 (very low tau).
Expect: smooth FP, exponential Cheby convergence.
"""
import os, sys, time
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import minimize
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          solve_fp_lobatto, cheby_slice_2d,
                                          lookup_analytic_lobatto)
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def find_crits(coefs_2d, U_MAX, n_seeds=15):
    deg = coefs_2d.shape[0] - 1
    da = np.zeros((deg, deg+1))
    for j in range(deg+1):
        da[:, j] = cheb.chebder(coefs_2d[:, j], 1) / U_MAX
    db = np.zeros((deg+1, deg))
    for i in range(deg+1):
        db[i, :] = cheb.chebder(coefs_2d[i, :], 1) / U_MAX
    def g2(uv):
        ua, ub = uv; xa, xb = ua/U_MAX, ub/U_MAX
        if abs(xa) > 1 or abs(xb) > 1: return 1e10
        return cheb.chebval2d(xa, xb, da)**2 + cheb.chebval2d(xa, xb, db)**2
    seeds = np.linspace(-U_MAX*0.95, U_MAX*0.95, n_seeds)
    found = []
    for ua0 in seeds:
        for ub0 in seeds:
            try:
                r = minimize(g2, [ua0, ub0], method="Nelder-Mead",
                                options=dict(xatol=1e-8, fatol=1e-12, maxiter=200))
                if r.fun < 1e-3 and abs(r.x[0]) < U_MAX and abs(r.x[1]) < U_MAX:
                    if not any(np.hypot(r.x[0]-c[0], r.x[1]-c[1]) < 0.05 for c in found):
                        found.append((r.x[0], r.x[1], r.fun))
            except: pass
    return found


def main():
    gamma, tau = 100.0, 0.001
    p_grid = make_p_grid(121)
    print(f"\n=== Cheby-Lobatto convergence at gamma={gamma}, tau={tau} ===", flush=True)
    P_prev = None
    for G in [11, 15, 21, 31]:
        print(f"\n--- G={G} ---", flush=True)
        u_grid, U_MAX = lobatto_grid(G)
        if P_prev is not None:
            from numpy.polynomial import chebyshev as cheb_
            G_prev = P_prev.shape[0]
            u_prev, U_MAX_prev = lobatto_grid(G_prev)
            coefs_prev = cheby_fit_lobatto_3d(P_prev, U_MAX_prev)
            P_init = np.empty((G,G,G))
            for ii in range(G):
                for jj in range(G):
                    for kk in range(G):
                        P_init[ii,jj,kk] = cheb_.chebval3d(u_grid[ii]/U_MAX_prev,
                                                                u_grid[jj]/U_MAX_prev,
                                                                u_grid[kk]/U_MAX_prev,
                                                                coefs_prev)
            P_init = np.clip(P_init, 1e-9, 1-1e-9)
        else: P_init = None
        t0 = time.time()
        P, F, u_grid, U_MAX = solve_fp_lobatto(gamma, tau, G, P_init=P_init)
        print(f"  FP solve: |F|={F:.3e} ({time.time()-t0:.0f}s)", flush=True)
        coefs = cheby_fit_lobatto_3d(P, U_MAX)
        ec = edge_coef(coefs)
        print(f"  Cheby edge_coef: {ec:.3e}", flush=True)
        coefs_2d = cheby_slice_2d(coefs, U_MAX, 0.0)
        crits = find_crits(coefs_2d, U_MAX, n_seeds=15)
        print(f"  critical points (slice u_k=0): {len(crits)}", flush=True)
        # Lookup vs strict-h=0
        gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, G, 16)
        t1 = time.time()
        mu_cheb = np.array([[lookup_analytic_lobatto(coefs, U_MAX, p_grid[ip],
                                                                  u_grid[k], tau, NQK=24)
                                      for ip in range(121)] for k in range(G)]).T
        t_lookup = time.time() - t1
        d_max = float(np.max(np.abs(mu_cheb - mu_strict)))
        d_med = float(np.median(np.abs(mu_cheb - mu_strict)))
        print(f"  lookup max|d|={d_max:.3e} med|d|={d_med:.3e} ({t_lookup:.0f}s)",
              flush=True)
        # Check P range
        print(f"  P range: [{P.min():.6f}, {P.max():.6f}]  (sigmoid centered ~0.5)",
              flush=True)
        P_prev = P


if __name__ == "__main__":
    main()
