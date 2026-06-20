"""Method C: monotone-preserving K=3 FP solver.

Architecture:
  1. Represent P via its Cheby tensor coefs C on Lobatto u-grid.
  2. Phi step: analytic-contour lookup from C (Cheby-native), CRRA clear.
  3. Monotone projection: project the resulting P onto the cone {dP/du_i >= 0
     for each axis} via cumulative max + clipping.
  4. Fit new Cheby coefs and iterate.

If the iteration converges to F=0 with the monotone-projected iterate, we
have a TRUE FP that is both monotone and smooth.
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          cheby_slice_2d, lookup_analytic_lobatto)
from cheby_numba import crra_clear_jit
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid


def project_monotone(P, n_sweeps=8):
    P_new = P.copy()
    for _ in range(n_sweeps):
        for ax in range(3):
            P_new = np.maximum.accumulate(P_new, axis=ax)
    return np.clip(P_new, 1e-9, 1.0 - 1e-9)


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def check_mono(P):
    n = 0; m = 0.0
    for ax in range(3):
        diffs = np.diff(P, axis=ax)
        n += int((diffs < 0).sum())
        m = min(m, float(diffs.min()))
    return n, m


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


def phi_cheby_native_smooth(P, u_grid, U_MAX, tau, gamma, NQK=24):
    """One Phi step: fit Cheby to P, analytic lookup at each cube cell,
    CRRA clear. P is on Lobatto u-grid."""
    G = u_grid.size
    coefs = cheby_fit_lobatto_3d(P, U_MAX)
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = cheb.chebval3d(u_grid[i]/U_MAX, u_grid[j]/U_MAX,
                                              u_grid[k]/U_MAX, coefs)
                p_cell = float(min(max(p_cell, 1e-9), 1.0 - 1e-9))
                mu0 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[i], tau, NQK)
                mu1 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[j], tau, NQK)
                mu2 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[k], tau, NQK)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new, coefs


def solve_method_c(gamma, tau, G, n_iter=40, target=1e-10, verbose=True):
    u_grid, U_MAX = lobatto_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P = project_monotone(1.0/(1.0+np.exp(-0.5*(U1+U2+U3))))
    Xh, Gh = [], []
    P_best = P.copy(); F_best = float("inf")
    hist = []
    for it in range(n_iter):
        t0 = time.time()
        P_phi, coefs = phi_cheby_native_smooth(P, u_grid, U_MAX, tau, gamma)
        F_pre = float(np.max(np.abs(P_phi - P)))    # before projection
        # Anderson on (P_phi - P), then projection
        gx = P_phi.ravel()
        Xh.append(P.ravel().copy()); Gh.append(gx.copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: P_next = 0.4*P.ravel() + 0.6*gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                P_next = 0.4*P.ravel() + 0.6*(Gh[k-1] + DG @ ga)
            except: P_next = 0.4*P.ravel() + 0.6*gx
        # Monotone projection
        P_new = project_monotone(P_next.reshape(G,G,G))
        # Measure F at the PROJECTED point (consistent FP definition for the
        # monotone manifold: x = Project(Phi(x)))
        P_phi_check, _ = phi_cheby_native_smooth(P_new, u_grid, U_MAX, tau, gamma)
        P_proj_phi_check = project_monotone(P_phi_check)
        F_mono = float(np.max(np.abs(P_proj_phi_check - P_new)))
        viol, min_diff = check_mono(P_new)
        ec = edge_coef(coefs)
        wall = time.time() - t0
        if verbose and (it < 5 or it % 5 == 4):
            print(f"  it{it+1:2d}: F_phi={F_pre:.3e} F_mono={F_mono:.3e} "
                  f"viol={viol} edge={ec:.3e} ({wall:.0f}s)", flush=True)
        hist.append((it+1, F_pre, F_mono, ec))
        if F_mono < F_best: F_best = F_mono; P_best = P_new.copy()
        if F_mono < target: break
        P = P_new
    return P_best, F_best, u_grid, U_MAX, hist


def main():
    gamma, tau = 100.0, 0.1
    G = 15
    print(f"=== Method C (Cheby-native + monotone projection) at gamma={gamma}, "
          f"tau={tau}, G={G} ===", flush=True)
    P, F, u_grid, U_MAX, hist = solve_method_c(gamma, tau, G, n_iter=25)
    print(f"\nFinal F (monotone-projected FP): {F:.3e}", flush=True)
    viol, mind = check_mono(P)
    print(f"Monotonicity: {viol}/{(G-1)*G*G*3} violations, min_diff={mind:.2e}",
          flush=True)
    coefs = cheby_fit_lobatto_3d(P, U_MAX)
    print(f"Cheby edge: {edge_coef(coefs):.3e}", flush=True)
    coefs_2d = cheby_slice_2d(coefs, U_MAX, 0.0)
    crits = find_crits(coefs_2d, U_MAX, n_seeds=15)
    print(f"Critical points (slice u_k=0): {len(crits)}", flush=True)
    # Lookup vs strict-h=0
    p_grid = make_p_grid(121)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                   tau, G, 16)
    mu_cheb = np.array([[lookup_analytic_lobatto(coefs, U_MAX, p_grid[ip],
                                                              u_grid[k], tau, NQK=24)
                                  for ip in range(121)] for k in range(G)]).T
    print(f"Lookup max|d|: {float(np.max(np.abs(mu_cheb - mu_strict))):.3e}",
          flush=True)
    print(f"Lookup med|d|: {float(np.median(np.abs(mu_cheb - mu_strict))):.3e}",
          flush=True)
    # Save
    np.savez("/tmp/dd_k3_method_c_FP.npz", P=P, gamma=gamma, tau=tau, G=G, F=F,
                edge_coef=edge_coef(coefs), hist=np.array(hist))
    # Plot
    OUT = "/tmp/dd_k3_method_c_figs"
    os.makedirs(OUT, exist_ok=True)
    # Fig 1: history
    hist = np.array(hist)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].semilogy(hist[:, 0], hist[:, 1], "o-", label="$|F_\\Phi|_\\infty$ (Phi residual)")
    axs[0].semilogy(hist[:, 0], hist[:, 2], "s-", label="$|F_{\\rm mono}|$ (after projection)")
    axs[0].set_xlabel("iter"); axs[0].set_ylabel("F")
    axs[0].set_title("Method C convergence"); axs[0].grid(True, alpha=0.3, which="both")
    axs[0].legend()
    axs[1].semilogy(hist[:, 0], hist[:, 3], "o-", color="C2")
    axs[1].set_xlabel("iter"); axs[1].set_ylabel("Cheby edge coef")
    axs[1].set_title("Cheby edge of monotone iterate")
    axs[1].grid(True, alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_method_c_hist.png", dpi=140)
    plt.close()
    # Fig 2: |grad P|^2 on slice with critical points
    n = 200
    u_fine = np.linspace(-U_MAX*0.98, U_MAX*0.98, n)
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
            G2[j, k] = cheb.chebval2d(xa, xb, da)**2 + cheb.chebval2d(xa, xb, db)**2
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.contourf(u_fine, u_fine, np.log10(G2 + 1e-15).T, levels=30, cmap="viridis")
    plt.colorbar(im, ax=ax, label=r"$\log_{10}|\nabla P|^2$")
    for ua, ub, _ in crits:
        ax.plot(ua, ub, "rx", ms=10, mew=2)
    ax.set_xlabel("$u_a$"); ax.set_ylabel("$u_b$")
    ax.set_title(f"Method C FP slice u_k=0, $\\gamma{{=}}{gamma}, \\tau{{=}}{tau}$"
                  f", G={G}: {len(crits)} critical points")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_method_c_grad.png", dpi=140)
    plt.close()
    print(f"figs in {OUT}")


if __name__ == "__main__":
    main()
