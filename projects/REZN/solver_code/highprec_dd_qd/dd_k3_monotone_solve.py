"""Solve FP with monotonicity enforced at each iteration.

Algorithm:
  x_{n+1} = Project_monotone(Anderson(Phi)(x_n))

Where Project_monotone is the cumulative-max along each axis (idempotent
sweep). Tests at gamma=100, tau=1, G=21 Lobatto. Reports edge_coef and
lookup quality vs strict-h=0.
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import minimize
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_p_grid
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          cheby_slice_2d, lookup_analytic_lobatto)


def project_monotone(P, n_sweeps=8):
    """Cumulative max along each axis, repeated until idempotent."""
    P_new = P.copy()
    for _ in range(n_sweeps):
        for ax in range(3):
            P_new = np.maximum.accumulate(P_new, axis=ax)
    # Clip
    return np.clip(P_new, 1e-9, 1 - 1e-9)


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def check_monotonicity(P):
    G = P.shape[0]
    violations = {}
    for ax in range(3):
        diffs = np.diff(P, axis=ax)
        violations[f"ax{ax}"] = (int((diffs < 0).sum()), float(diffs.min()))
    return violations


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


def solve_with_monotone_projection(gamma, tau, G, n_iter=80, target=1e-10):
    u_grid, U_MAX = lobatto_grid(G)
    p_grid = make_p_grid(121)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    def F_call(xflat):
        Pn = phi_lin_richardson(xflat.reshape(G,G,G), u_grid,
                                       hs=(0.5,0.4,0.3,0.2),
                                       gamma=gamma, tau=tau,
                                       G_p=121, NQK=16, p_grid=p_grid)
        return (Pn - xflat.reshape(G,G,G)).ravel()
    x = P.ravel().copy()
    Xh, Gh = [], []
    x_best = x.copy(); f_best = float("inf")
    for it in range(n_iter):
        Fv = F_call(x)
        f = float(np.max(np.abs(Fv)))
        if f < f_best: f_best = f; x_best = x.copy()
        # Anderson step
        gx = Fv + x
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x_next = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x_next = Gh[k-1] + DG @ ga
            except: x_next = gx
        # Monotone projection
        P_next = project_monotone(x_next.reshape(G,G,G))
        x = P_next.ravel()
        if it % 10 == 0 or it == n_iter-1:
            P_check = x.reshape(G,G,G)
            viol = check_monotonicity(P_check)
            coefs = cheby_fit_lobatto_3d(P_check, U_MAX)
            ec = edge_coef(coefs)
            print(f"  it{it:3d}: F={f:.3e} viol={viol['ax0'][0]} "
                  f"min_diff={viol['ax0'][1]:.2e} edge={ec:.3e}", flush=True)
        if f < target: break
    return x_best.reshape(G,G,G), f_best, u_grid, U_MAX


def main():
    gamma, tau = 100.0, 1.0
    G = 21
    print(f"=== Monotone-projected FP solve at gamma={gamma}, tau={tau}, G={G} ===",
          flush=True)
    t0 = time.time()
    P, F, u_grid, U_MAX = solve_with_monotone_projection(gamma, tau, G, n_iter=60)
    wall = time.time() - t0
    print(f"\nFinal: F={F:.3e} ({wall:.0f}s)", flush=True)
    viol = check_monotonicity(P)
    print(f"Monotonicity: ax0 viol={viol['ax0'][0]}/8820 min_diff={viol['ax0'][1]:.2e}",
          flush=True)
    coefs = cheby_fit_lobatto_3d(P, U_MAX)
    print(f"Cheby edge_coef: {edge_coef(coefs):.3e}", flush=True)
    coefs_2d = cheby_slice_2d(coefs, U_MAX, 0.0)
    crits = find_crits(coefs_2d, U_MAX, n_seeds=20)
    print(f"Critical points (slice u_k=0): {len(crits)}", flush=True)
    # Lookup vs strict-h=0
    p_grid = make_p_grid(121)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                   tau, G, 16)
    mu_cheb = np.array([[lookup_analytic_lobatto(coefs, U_MAX, p_grid[ip],
                                                              u_grid[k], tau, NQK=24)
                                  for ip in range(121)] for k in range(G)]).T
    print(f"Lookup max|d| vs strict-h=0: {float(np.max(np.abs(mu_cheb - mu_strict))):.3e}",
          flush=True)
    print(f"Lookup median|d|: {float(np.median(np.abs(mu_cheb - mu_strict))):.3e}",
          flush=True)
    # Save
    np.savez("/tmp/dd_k3_monotone_FP.npz", P=P, gamma=gamma, tau=tau, G=G,
                F=F, edge_coef=edge_coef(coefs), n_crits=len(crits))


if __name__ == "__main__":
    main()
