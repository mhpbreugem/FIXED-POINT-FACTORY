"""Test: does enforcing monotonicity remove the critical points and restore
exponential Cheby convergence?

Take the G=21 R4 FP solved on Lobatto, check how monotone it is, apply
cumulative-max projection along each axis, refit Cheby, recompute critical
points + edge coef + lookup quality.
"""
import os, sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          cheby_slice_2d, solve_fp_lobatto,
                                          lookup_analytic_lobatto)
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid

OUT = "/tmp/dd_k3_monotone_test_figs"
os.makedirs(OUT, exist_ok=True)


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def check_monotonicity(P):
    """How non-monotone is P? Return min(dP/du_i) along each axis."""
    G = P.shape[0]
    violations = {}
    for ax in range(3):
        diffs = np.diff(P, axis=ax)
        violations[f"axis{ax}"] = dict(min_diff=float(diffs.min()),
                                              n_violations=int((diffs < 0).sum()),
                                              n_total=int(diffs.size))
    return violations


def project_monotone_cumax(P):
    """Project to monotone via cumulative max along each axis (idempotent,
    converges in few sweeps)."""
    P_new = P.copy()
    for _ in range(5):
        for ax in range(3):
            P_new = np.maximum.accumulate(P_new, axis=ax)
    return P_new


def find_critical_points_slice(coefs_2d, U_MAX, n_seeds=15):
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
        dpa = cheb.chebval2d(xa, xb, da); dpb = cheb.chebval2d(xa, xb, db)
        return dpa*dpa + dpb*dpb
    seeds = np.linspace(-U_MAX*0.95, U_MAX*0.95, n_seeds)
    found = []
    for ua0 in seeds:
        for ub0 in seeds:
            try:
                r = minimize(g2, [ua0, ub0], method="Nelder-Mead",
                                options=dict(xatol=1e-8, fatol=1e-12, maxiter=200))
                if r.fun < 1e-3 and abs(r.x[0]) < U_MAX and abs(r.x[1]) < U_MAX:
                    if not any(np.hypot(r.x[0]-c[0], r.x[1]-c[1]) < 0.05
                                  for c in found):
                        found.append((r.x[0], r.x[1], r.fun))
            except: pass
    return found


def main():
    gamma, tau = 100.0, 1.0
    print(f"=== Monotone-projection test at gamma={gamma}, tau={tau}, G=21 ===",
          flush=True)
    P, F, u_grid, U_MAX = solve_fp_lobatto(gamma, tau, 21)
    print(f"R4 FP: F={F:.2e}", flush=True)

    # Check monotonicity of original FP
    viol = check_monotonicity(P)
    print(f"\nOriginal R4 FP monotonicity:", flush=True)
    for k, v in viol.items():
        print(f"  {k}: min_diff={v['min_diff']:.3e}, violations={v['n_violations']}/{v['n_total']}",
              flush=True)

    # Cheby + critical points + lookup of ORIGINAL
    coefs_orig = cheby_fit_lobatto_3d(P, U_MAX)
    ec_orig = edge_coef(coefs_orig)
    coefs_2d_orig = cheby_slice_2d(coefs_orig, U_MAX, 0.0)
    crits_orig = find_critical_points_slice(coefs_2d_orig, U_MAX, n_seeds=20)
    print(f"\nORIGINAL: edge_coef={ec_orig:.3e}, {len(crits_orig)} critical points",
          flush=True)

    # Project to monotone
    P_mono = project_monotone_cumax(P)
    print(f"\nMonotone projection: max change = {float(np.max(np.abs(P_mono - P))):.3e}",
          flush=True)
    viol_mono = check_monotonicity(P_mono)
    for k, v in viol_mono.items():
        print(f"  {k}: min_diff={v['min_diff']:.3e}, violations={v['n_violations']}/{v['n_total']}",
              flush=True)

    coefs_mono = cheby_fit_lobatto_3d(P_mono, U_MAX)
    ec_mono = edge_coef(coefs_mono)
    coefs_2d_mono = cheby_slice_2d(coefs_mono, U_MAX, 0.0)
    crits_mono = find_critical_points_slice(coefs_2d_mono, U_MAX, n_seeds=20)
    print(f"\nMONOTONE: edge_coef={ec_mono:.3e}, {len(crits_mono)} critical points",
          flush=True)

    # Lookup quality comparison
    print("\nLookup comparison (vs strict-h=0 reference):", flush=True)
    p_grid = make_p_grid(121)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    mu_ref_orig = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, P.shape[0], 16)
    mu_ref_mono = build_mu_table_lin_strict(P_mono, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, P.shape[0], 16)
    mu_cheb_orig = np.array([[lookup_analytic_lobatto(coefs_orig, U_MAX,
                                                                p_grid[ip], u_grid[k],
                                                                tau, NQK=24)
                                          for ip in range(121)] for k in range(P.shape[0])]).T
    mu_cheb_mono = np.array([[lookup_analytic_lobatto(coefs_mono, U_MAX,
                                                                p_grid[ip], u_grid[k],
                                                                tau, NQK=24)
                                          for ip in range(121)] for k in range(P.shape[0])]).T
    print(f"  orig:  max|mu_cheb - mu_strict| = {float(np.max(np.abs(mu_cheb_orig - mu_ref_orig))):.3e}",
          flush=True)
    print(f"  mono:  max|mu_cheb - mu_strict| = {float(np.max(np.abs(mu_cheb_mono - mu_ref_mono))):.3e}",
          flush=True)

    # Plot: critical points before vs after
    n = 200
    u_fine = np.linspace(-U_MAX*0.98, U_MAX*0.98, n)
    def g2_grid(coefs_2d):
        deg = coefs_2d.shape[0] - 1
        da = np.zeros((deg, deg+1))
        for j in range(deg+1):
            da[:, j] = cheb.chebder(coefs_2d[:, j], 1) / U_MAX
        db = np.zeros((deg+1, deg))
        for i in range(deg+1):
            db[i, :] = cheb.chebder(coefs_2d[i, :], 1) / U_MAX
        G2 = np.zeros((n, n))
        for j, ua in enumerate(u_fine):
            for k, ub in enumerate(u_fine):
                xa, xb = ua/U_MAX, ub/U_MAX
                G2[j, k] = (cheb.chebval2d(xa, xb, da)**2
                                + cheb.chebval2d(xa, xb, db)**2)
        return G2
    G2_orig = g2_grid(coefs_2d_orig)
    G2_mono = g2_grid(coefs_2d_mono)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, G2_v, crits, label in [(axs[0], G2_orig, crits_orig, "ORIGINAL"),
                                              (axs[1], G2_mono, crits_mono, "MONOTONE projection")]:
        im = ax.contourf(u_fine, u_fine, np.log10(G2_v + 1e-15).T,
                              levels=30, cmap="viridis", vmin=-15, vmax=2)
        plt.colorbar(im, ax=ax, label=r"$\log_{10}|\nabla P|^2$")
        for ua, ub, _ in crits:
            ax.plot(ua, ub, "rx", ms=10, mew=2)
        ax.set_xlabel("$u_a$"); ax.set_ylabel("$u_b$")
        ax.set_title(f"{label}: {len(crits)} interior critical points")
    plt.suptitle(f"Monotone projection -- $\\gamma{{=}}{gamma}, \\tau{{=}}{tau}, G=21$",
                  fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_orig_vs_mono.png", dpi=140); plt.close()

    # Edge coef summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["ORIGINAL R4", "MONOTONE-projected"]
    edges = [ec_orig, ec_mono]
    ncrits = [len(crits_orig), len(crits_mono)]
    ax.bar(labels, edges, color=["C3", "C2"], alpha=0.7)
    for i, (e, nc) in enumerate(zip(edges, ncrits)):
        ax.text(i, e, f"  edge={e:.2e}\n  {nc} crit pts",
                  ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("max edge Cheby coef")
    ax.set_yscale("log")
    ax.set_title(f"Cheby edge coef: original vs monotone-projected (G=21)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_edge_compare.png", dpi=140); plt.close()
    print(f"\nfigs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
