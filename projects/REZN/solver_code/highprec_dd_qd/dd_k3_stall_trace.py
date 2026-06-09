"""Diagnose the strict-h=0 Anderson stall pattern.

Run the existing solver at (g=100, t=1) starting from saved R4 FP; trace
|F|, oscillation amplitude, and where the failures concentrate (which
cube cells have largest |Phi - P|). This tells us if smoothing is the
right fix or if we just need damping/larger NQ.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_strict import phi_lin_strict_jit, make_gl_for_u
from lin_cdf_pchip import make_cdf_uniform_grid
import matplotlib.pyplot as plt


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/smooth_strict"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def make_p_grid(Gp, eps=1e-3):
    lo, hi = np.log(eps/(1-eps)), np.log((1-eps)/eps)
    return 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, Gp)))


def trace_stall(gamma, tau, n_iter=200, nq=16, omega=1.0):
    """Picard iteration on strict-h=0; trace residual."""
    G = 11
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    P = fp["P_strict"].astype(np.float64).copy()
    F_hist = []; oscamp = []; argmax_hist = []
    for it in range(n_iter):
        Pn = phi_lin_strict_jit(P, u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq)
        F = Pn - P
        Finf = float(np.max(np.abs(F)))
        argmax = np.unravel_index(np.argmax(np.abs(F)), F.shape)
        F_hist.append(Finf)
        argmax_hist.append(argmax)
        if it >= 5:
            recent = F_hist[-5:]
            oscamp.append(max(recent) - min(recent))
        else:
            oscamp.append(np.nan)
        # Damped Picard: P_new = (1-omega)*P + omega*Pn
        P = (1.0 - omega) * P + omega * Pn
    return np.array(F_hist), np.array(oscamp), argmax_hist


def main():
    gamma, tau = 100, 1.0
    print(f"=== strict-h=0 stall trace: g={gamma}, tau={tau} ===")
    results = {}
    for omega in [1.0, 0.5, 0.3]:
        for nq in [16, 32]:
            tag = f"omega{omega}_nq{nq}"
            print(f"\n--- {tag} ---", flush=True)
            t0 = time.time()
            F_hist, osc, am = trace_stall(gamma, tau, n_iter=80, nq=nq, omega=omega)
            wall = time.time() - t0
            print(f"  start |F| = {F_hist[0]:.3e}")
            print(f"  min  |F| = {F_hist.min():.3e}")
            print(f"  end  |F| = {F_hist[-1]:.3e}")
            print(f"  osc amp at end = {osc[-1]:.3e}")
            am_set = set(am[-20:])
            print(f"  unique argmax cells in last 20 iters: {len(am_set)}")
            print(f"  wall: {wall:.1f}s")
            results[tag] = dict(omega=omega, nq=nq, F=F_hist.tolist(), osc=osc.tolist(),
                                start_F=float(F_hist[0]), min_F=float(F_hist.min()),
                                end_F=float(F_hist[-1]), wall=wall)

    json.dump(results, open(f"{OUT}/stall_trace.json", "w"), indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for tag, r in results.items():
        axes[0].semilogy(r["F"], label=tag, alpha=0.7)
        axes[1].semilogy(r["osc"], label=tag, alpha=0.7)
    axes[0].set_xlabel("Picard iteration")
    axes[0].set_ylabel("|F|_inf")
    axes[0].set_title("Damped Picard on strict-h=0")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which='both')
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("oscillation amplitude (5-iter range)")
    axes[1].set_title("Stall amplitude")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/stall_trace.png", dpi=120)
    plt.close()
    print(f"\nsaved {OUT}/figs/stall_trace.png + stall_trace.json")


if __name__ == "__main__":
    main()
