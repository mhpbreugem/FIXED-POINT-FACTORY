"""Sweep eps to find optimal value per regime; verify the τ=0.2 success.

Also compute lookup μ under the converged smoothed FP and compare to
mu_strict (best-iterate) to check bias.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_strict import phi_lin_strict_jit, make_gl_for_u, build_mu_table_lin_strict
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_strict_smoothed import (phi_smoothed, build_mu_smoothed,
                                          solve_eps_continuation)


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/smooth_strict"


def solve_single_eps(P_warm, u_grid, p_grid, gl_u, gl_du, tau, gamma, eps, n_iter=80, target=1e-14):
    """Pure Anderson at fixed eps."""
    G = u_grid.size; nq = gl_u.size
    x = P_warm.ravel().copy()
    def F(xf): return (phi_smoothed(xf.reshape(G,G,G), u_grid, p_grid,
                                       gl_u, gl_du, tau, gamma, G, nq, eps) - xf.reshape(G,G,G)).ravel()
    Xh, Gh = [], []
    x_best = x.copy(); f_best = float("inf")
    for it in range(n_iter):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv)))
        if f < f_best: f_best = f; x_best = x.copy()
        if f < target: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = 0.5*x + 0.5*gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    return x_best.reshape(G,G,G), f_best


def main():
    G = 11
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(121)
    nq = 16
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)

    print("warmup numba...", flush=True)
    _ = phi_smoothed(np.full((G,G,G), 0.5), u_grid, p_grid, gl_u, gl_du, 1.0, 1.0, G, nq, 1e-2)

    results = {}
    eps_grid = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4]
    for gamma, tau in [(100, 1.0), (100, 0.2), (1000, 1.0), (1000, 0.2)]:
        print(f"\n=== g={gamma}, tau={tau} ===", flush=True)
        fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
        P_warm = fp["P_strict"].astype(np.float64)
        F_warm = float(np.max(np.abs(phi_lin_strict_jit(P_warm, u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq) - P_warm)))
        print(f"  warm-start |F|_orig = {F_warm:.3e}")
        results[f"g{gamma}_t{tau:.4f}"] = dict(gamma=gamma, tau=tau, F_warm=F_warm,
                                                    eps_results=[])
        best_overall_F = F_warm; best_overall_P = P_warm.copy(); best_overall_eps = None
        for eps in eps_grid:
            t0 = time.time()
            P_eps, F_eps = solve_single_eps(P_warm, u_grid, p_grid, gl_u, gl_du, tau, gamma, eps, n_iter=60)
            # Also evaluate the converged P under ORIGINAL strict op
            F_orig = float(np.max(np.abs(phi_lin_strict_jit(P_eps, u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq) - P_eps)))
            wall = time.time() - t0
            print(f"  eps={eps:.0e}: |F|_smoothed = {F_eps:.3e}, |F|_orig = {F_orig:.3e}  ({wall:.1f}s)")
            results[f"g{gamma}_t{tau:.4f}"]["eps_results"].append(
                dict(eps=eps, F_smoothed=F_eps, F_orig=F_orig, wall=wall))
            # Track best by the smoothed |F| if it converged tightly
            if F_eps < best_overall_F and F_eps < 1e-6:
                best_overall_F = F_eps; best_overall_P = P_eps.copy(); best_overall_eps = eps
        # Save best
        results[f"g{gamma}_t{tau:.4f}"]["best_eps"] = best_overall_eps
        results[f"g{gamma}_t{tau:.4f}"]["best_F"] = best_overall_F
        np.savez(f"{OUT}/best_g{gamma}_t{tau:.4f}.npz",
                 P_warm=P_warm, P_best=best_overall_P, best_eps=best_overall_eps,
                 best_F=best_overall_F, F_warm=F_warm)

    json.dump(results, open(f"{OUT}/eps_sweep_results.json", "w"), indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.tab10
    for ic, (tag, r) in enumerate(results.items()):
        eps = [e["eps"] for e in r["eps_results"]]
        Fs = [e["F_smoothed"] for e in r["eps_results"]]
        Fo = [e["F_orig"] for e in r["eps_results"]]
        axes[0].loglog(eps, Fs, '-o', label=tag, color=cmap(ic), ms=6)
        axes[0].axhline(r["F_warm"], color=cmap(ic), ls=':', alpha=0.3)
        axes[1].loglog(eps, Fo, '-s', label=tag, color=cmap(ic), ms=6)
        axes[1].axhline(r["F_warm"], color=cmap(ic), ls=':', alpha=0.3)
    for ax, ylab in zip(axes, ["|F|_inf (smoothed op)", "|F|_inf (original op, no smoothing)"]):
        ax.set_xlabel("eps (regularization)"); ax.set_ylabel(ylab)
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
    axes[0].set_title("Smoothed strict-h=0 convergence (dotted = warm-start |F|)")
    axes[1].set_title("Converged P under ORIGINAL strict-h=0 operator")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/eps_sweep.png", dpi=120)
    plt.close()

    print(f"\nSaved {OUT}/eps_sweep_results.json + figs/eps_sweep.png")


if __name__ == "__main__":
    main()
