"""Method C v2 at tau=0.001: deep solve with max precision + plots."""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import dd_k3_method_c_v2 as MC2
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_richardson import phi_lin_richardson


# Override D_SIGMA temporarily to test more flexibility
TAU = 0.001
GAMMA = 100.0
G = 7
HS = (0.5, 0.4, 0.3, 0.2)
U_MAX = 2.33
OUT = "/tmp/dd_k3_method_c_v2_tau001_figs"
os.makedirs(OUT, exist_ok=True)


def build_P_d(coefs, u_grid, d_sigma):
    from numba import njit
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(MC2.N_QUAD)
    P_out = np.empty((u_grid.size,)*3)
    # Use MC2's numba function via the patched D_SIGMA
    G_loc = u_grid.size
    h_vals = np.empty(G_loc)
    for i in range(G_loc):
        h_vals[i] = MC2._h_at_u(coefs, u_grid[i], d_sigma, U_MAX, gl_nodes, gl_weights)
    for i in range(G_loc):
        for j in range(G_loc):
            for k in range(G_loc):
                L = h_vals[i] + h_vals[j] + h_vals[k]
                if L > 30: P_out[i, j, k] = 1.0
                elif L < -30: P_out[i, j, k] = 0.0
                else: P_out[i, j, k] = 1.0 / (1.0 + np.exp(-L))
    return P_out


def F_residual(coefs, d_sigma, u_grid, p_grid):
    P = build_P_d(coefs, u_grid, d_sigma)
    P_new = phi_lin_richardson(P, u_grid, hs=HS, gamma=GAMMA, tau=TAU,
                                       G_p=p_grid.size, NQK=16, p_grid=p_grid)
    return (P_new - P).ravel()


def main():
    print(f"=== Method C v2 deep solve at tau={TAU} ===")
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    print("JIT warmup", flush=True); t0 = time.time()
    build_P_d(np.zeros(5), u_grid, 4)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    # Test with several d_sigma values
    results = {}
    coefs_warm = None
    for d_sigma in [2, 4, 6, 8]:
        n = d_sigma + 1
        print(f"\n--- d_sigma={d_sigma}, {n} unknowns ---", flush=True)
        if coefs_warm is None:
            c0 = np.zeros(n); c0[0] = 0.05
        else:
            c0 = np.zeros(n)
            c0[:len(coefs_warm)] = coefs_warm    # pad
        t0 = time.time()
        res = least_squares(F_residual, c0,
                                  args=(d_sigma, u_grid, p_grid),
                                  jac="2-point", method="trf",
                                  max_nfev=500, ftol=1e-15, xtol=1e-15,
                                  verbose=0)
        ts = time.time() - t0
        P = build_P_d(res.x, u_grid, d_sigma)
        viol = 0; mind = 0.0
        for ax in range(3):
            dd = np.diff(P, axis=ax)
            viol += int((dd < 0).sum()); mind = min(mind, float(dd.min()))
        F_inf = float(np.max(np.abs(res.fun)))
        print(f"  d_sigma={d_sigma}: F={F_inf:.3e} viol={viol} mind={mind:.3e} "
              f"P=[{P.min():.6f}, {P.max():.6f}] ({ts:.0f}s, nfev={res.nfev})", flush=True)
        results[d_sigma] = dict(coefs=res.x.tolist(), F=F_inf, viol=viol,
                                       P_min=float(P.min()), P_max=float(P.max()),
                                       wall=ts, nfev=res.nfev)
        coefs_warm = res.x
        np.savez(f"/tmp/dd_k3_method_c_v2_tau001_d{d_sigma}.npz",
                    coefs=res.x, P=P, d_sigma=d_sigma)

    # Plot: h(u) and P slice for the best result
    best_d = max(results.keys(), key=lambda k: -results[k]["F"])
    best = results[best_d]
    coefs = np.array(best["coefs"])
    u_fine = np.linspace(-U_MAX, U_MAX, 200)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(MC2.N_QUAD)
    h_vals = np.array([MC2._h_at_u(coefs, u, best_d, U_MAX, gl_nodes, gl_weights)
                              for u in u_fine])
    sigma_vals = np.array([np.sqrt(MC2._sigma2(coefs, u/U_MAX, best_d)) for u in u_fine])

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(u_fine, sigma_vals, "-", lw=1.5, color="C0")
    axs[0].set_xlabel("$u$"); axs[0].set_ylabel(r"$\sigma_*(u)$")
    axs[0].set_title(r"$\sigma_*(u)$ (even polynomial); $h'(u) = \sigma_*(u)^2 \geq 0$")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(u_fine, h_vals, "-", lw=1.5, color="C2")
    axs[1].set_xlabel("$u$"); axs[1].set_ylabel(r"$h(u)$")
    axs[1].set_title(r"$h(u) = \int_0^u \sigma_*^2(s)\,ds$ (odd, monotone increasing)")
    axs[1].grid(True, alpha=0.3); axs[1].axhline(0, color="k", lw=0.5)
    axs[1].axvline(0, color="k", lw=0.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_sigma_h.png", dpi=140); plt.close()

    # P slices
    P = build_P_d(coefs, u_grid, best_d)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for kk in range(3):
        ax = axs[kk]
        slice2 = P[:, :, [0, 3, 6][kk]]
        cs = ax.contourf(u_grid, u_grid, slice2, levels=20, cmap="RdBu_r")
        plt.colorbar(cs, ax=ax)
        ax.contour(u_grid, u_grid, slice2, levels=[0.5], colors="k", linewidths=1.0)
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
        ax.set_title(f"$u_3 = {u_grid[[0, 3, 6][kk]]:.2f}$")
    plt.suptitle(f"P(u1, u2, u3) at tau={TAU}, gamma={GAMMA}: smooth monotone fit (d_sigma={best_d}, F={best['F']:.2e})", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_P_slices.png", dpi=140); plt.close()

    # Monotonicity check
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_idx, ax in enumerate(axs):
        dd = np.diff(P, axis=ax_idx)
        ax.imshow(dd[:, :, 3], cmap="viridis", aspect="auto")
        ax.set_title(f"$dP/du_{ax_idx+1}$ slice (always > 0?)")
        plt.colorbar(ax.images[0], ax=ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_grad.png", dpi=140); plt.close()

    json.dump(results, open(f"/tmp/dd_k3_method_c_v2_tau001_results.json", "w"), indent=2)
    print(f"\nfigs in {OUT}")


if __name__ == "__main__":
    main()
