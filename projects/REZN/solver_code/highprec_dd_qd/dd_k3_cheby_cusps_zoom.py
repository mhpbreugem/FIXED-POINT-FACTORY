"""Plot the cusps in mu(p, u_k) and align them with critical points of P
(zeros of grad P along the contour). Show that p* = P(critical point) is
exactly where mu has a kink.
"""
import os, sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import minimize, brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          cheby_slice_2d, lookup_analytic_lobatto,
                                          solve_fp_lobatto)

OUT = "/tmp/dd_k3_cheby_cusp_zoom_figs"
os.makedirs(OUT, exist_ok=True)


def find_critical_points_slice(coefs_2d, U_MAX, n_seeds=15):
    """Find local minima of |grad P|^2 on (u_a, u_b) in (-U_MAX, U_MAX)^2."""
    deg = coefs_2d.shape[0] - 1
    # Derivative coefs in u_a and u_b axes
    da = np.zeros((deg, deg+1))
    for j in range(deg+1):
        da[:, j] = cheb.chebder(coefs_2d[:, j], 1) / U_MAX
    db = np.zeros((deg+1, deg))
    for i in range(deg+1):
        db[i, :] = cheb.chebder(coefs_2d[i, :], 1) / U_MAX

    def grad2(uv):
        ua, ub = uv; xa, xb = ua/U_MAX, ub/U_MAX
        if abs(xa) > 1 or abs(xb) > 1: return 1e10
        dpa = cheb.chebval2d(xa, xb, da)
        dpb = cheb.chebval2d(xa, xb, db)
        return dpa*dpa + dpb*dpb

    # Seed minimize from a grid
    seeds = np.linspace(-U_MAX*0.95, U_MAX*0.95, n_seeds)
    found = []
    for ua0 in seeds:
        for ub0 in seeds:
            try:
                r = minimize(grad2, [ua0, ub0], method="Nelder-Mead",
                                options=dict(xatol=1e-8, fatol=1e-12, maxiter=200))
                if r.fun < 1e-3 and abs(r.x[0]) < U_MAX and abs(r.x[1]) < U_MAX:
                    # dedupe
                    if not any(np.hypot(r.x[0]-c[0], r.x[1]-c[1]) < 0.05 for c in found):
                        found.append((r.x[0], r.x[1], r.fun))
            except Exception: pass
    return found


def main():
    # gamma=100, tau=1 (kinky)
    print("solving FP at gamma=100, tau=1 on Lobatto G=21...", flush=True)
    P, F, u_grid, U_MAX = solve_fp_lobatto(100.0, 1.0, 21)
    print(f"  F={F:.2e}", flush=True)
    coefs_3d = cheby_fit_lobatto_3d(P, U_MAX)

    # Slice at u_k = 0
    u_k = 0.0
    coefs_2d = cheby_slice_2d(coefs_3d, U_MAX, u_k)
    print(f"finding critical points of slice u_k=0...", flush=True)
    crits = find_critical_points_slice(coefs_2d, U_MAX, n_seeds=20)
    print(f"  found {len(crits)} critical points", flush=True)
    p_stars = []
    for ua, ub, g2 in crits:
        p_star = cheb.chebval2d(ua/U_MAX, ub/U_MAX, coefs_2d)
        p_stars.append((p_star, ua, ub, g2))
    p_stars.sort()
    for p_star, ua, ub, g2 in p_stars:
        print(f"    crit at (u_a,u_b)=({ua:.3f},{ub:.3f}) -> p*={p_star:.4f} (|grad|^2={g2:.2e})",
              flush=True)

    # Fig 1: |grad P|^2 heatmap on slice u_k=0, with critical points marked
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
    P_slice_vals = np.zeros((n, n))
    for j, ua in enumerate(u_fine):
        for k, ub in enumerate(u_fine):
            xa, xb = ua/U_MAX, ub/U_MAX
            dpa = cheb.chebval2d(xa, xb, da)
            dpb = cheb.chebval2d(xa, xb, db)
            G2[j, k] = dpa*dpa + dpb*dpb
            P_slice_vals[j, k] = cheb.chebval2d(xa, xb, coefs_2d)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axs[0].contourf(u_fine, u_fine, np.log10(G2 + 1e-15).T,
                                  levels=30, cmap="viridis")
    plt.colorbar(im0, ax=axs[0], label=r"$\log_{10}|\nabla P|^2$")
    for p_star, ua, ub, g2 in p_stars:
        axs[0].plot(ua, ub, "rx", ms=12, mew=2.5)
        axs[0].annotate(f"p*={p_star:.3f}", xy=(ua, ub), xytext=(ua+0.15, ub+0.1),
                          color="red", fontsize=8)
    axs[0].set_xlabel("$u_a$"); axs[0].set_ylabel("$u_b$")
    axs[0].set_title(f"slice $u_k=0$: $|\\nabla P|^2$, critical points (red x)")
    # Overlay contours of P at p_star values
    im1 = axs[1].contour(u_fine, u_fine, P_slice_vals.T, levels=20,
                                 cmap="RdBu_r")
    plt.colorbar(im1, ax=axs[1], label="$P(u_k=0, u_a, u_b)$")
    for p_star, ua, ub, g2 in p_stars:
        axs[1].plot(ua, ub, "kx", ms=12, mew=2.5)
        # Draw the special contour
        axs[1].contour(u_fine, u_fine, P_slice_vals.T, levels=[p_star],
                          colors="red", linewidths=2.0)
    axs[1].set_xlabel("$u_a$"); axs[1].set_ylabel("$u_b$")
    axs[1].set_title(f"contour at each p*: passes through critical pt")
    plt.suptitle(r"$\gamma{=}100, \tau{=}1, G{=}21$ FP $P$, slice $u_k{=}0$",
                  fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_grad_critical.png", dpi=140); plt.close()

    # Fig 2: mu(p, u_k=0) with cusps marked
    p_eval = np.linspace(0.001, 0.999, 600)
    mu_vals = np.array([lookup_analytic_lobatto(coefs_3d, U_MAX, p, 0.0, 1.0, NQK=32)
                              for p in p_eval])
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    # Top-left: full mu(p)
    ax = axs[0, 0]
    ax.plot(p_eval, mu_vals, "b-", lw=1.0)
    for p_star, _, _, _ in p_stars:
        ax.axvline(p_star, color="red", lw=0.8, alpha=0.5)
    ax.set_xlabel("$p$"); ax.set_ylabel("$\\mu(p, u_k{=}0)$")
    ax.set_title("Lookup with vertical lines at $p^*$ (critical-point pressures)")
    ax.grid(True, alpha=0.3)
    # Top-right: numerical derivative dmu/dp
    dmu = np.gradient(mu_vals, p_eval)
    ax = axs[0, 1]
    ax.plot(p_eval, dmu, "g-", lw=1.0)
    for p_star, _, _, _ in p_stars:
        ax.axvline(p_star, color="red", lw=0.8, alpha=0.5)
    ax.set_xlabel("$p$"); ax.set_ylabel("$d\\mu / dp$")
    ax.set_title("First derivative -- spikes at $p^*$ are the cusps")
    ax.grid(True, alpha=0.3)
    # Bottom: zoom on the largest cusp
    if p_stars:
        # find the cusp with the biggest derivative spike
        cusp_strengths = []
        for p_star, _, _, _ in p_stars:
            idx = np.argmin(np.abs(p_eval - p_star))
            if 5 < idx < len(p_eval)-5:
                window = dmu[idx-3:idx+3]
                cusp_strengths.append((np.max(np.abs(window)), p_star))
        cusp_strengths.sort(reverse=True)
        for col, (_, p_star) in enumerate(cusp_strengths[:2]):
            mask = (p_eval > p_star - 0.04) & (p_eval < p_star + 0.04)
            ax = axs[1, col]
            ax.plot(p_eval[mask], mu_vals[mask], "b-o", lw=1.5, ms=3)
            ax.axvline(p_star, color="red", lw=1.5, alpha=0.7,
                          label=f"p*={p_star:.4f}")
            ax.set_xlabel("$p$"); ax.set_ylabel("$\\mu(p, u_k{=}0)$")
            ax.set_title(f"Zoom on cusp at p*={p_star:.4f}")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle("Cusps in $\\mu(p, u_k)$ align with critical-point pressures $p^*$",
                  fontsize=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_mu_cusps.png", dpi=140); plt.close()

    print(f"figs in {OUT}: {sorted(os.listdir(OUT))}", flush=True)


if __name__ == "__main__":
    main()
