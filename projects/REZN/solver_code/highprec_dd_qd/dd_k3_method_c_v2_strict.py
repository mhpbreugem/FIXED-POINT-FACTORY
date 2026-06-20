"""Method C v2 ansatz solved against the STRICT-h=0 operator (no kernel)."""
import os, sys, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import dd_k3_method_c_v2 as MC2
from lin_cdf_strict import phi_lin_strict, make_cdf_uniform_grid as mk_cdf_strict
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u

GAMMA = 100.0
G = 7
U_MAX = 2.33


def F_strict(coefs, tau, d_sigma, u_grid, p_grid, gl_u, gl_du):
    P = MC2.build_P(coefs, u_grid)
    P_new = phi_lin_strict(P, u_grid, gamma=GAMMA, tau=tau, p_grid=p_grid, NQ=16)
    return (P_new - P).ravel()


def solve_strict(tau, d_sigma, coefs_warm=None, verbose=True):
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    if coefs_warm is None:
        c0 = np.zeros(d_sigma + 1); c0[0] = 0.1
    else:
        c0 = coefs_warm
    t0 = time.time()
    res = least_squares(F_strict, c0,
                              args=(tau, d_sigma, u_grid, p_grid, gl_u, gl_du),
                              jac="2-point", method="trf",
                              max_nfev=500, ftol=1e-15, xtol=1e-15,
                              verbose=2 if verbose else 0)
    ts = time.time() - t0
    P = MC2.build_P(res.x, u_grid)
    viol = 0; mind = 0.0
    for ax in range(3):
        dd = np.diff(P, axis=ax)
        viol += int((dd < 0).sum()); mind = min(mind, float(dd.min()))
    F_inf = float(np.max(np.abs(res.fun)))
    if verbose:
        print(f"  Strict-h=0: F={F_inf:.3e} viol={viol} mind={mind:.3e}", flush=True)
        print(f"  P=[{P.min():.6f}, {P.max():.6f}] ({ts:.0f}s, nfev={res.nfev})", flush=True)
    return res.x, P, F_inf, viol, ts


def main():
    print("=== Method C v2 with STRICT-h=0 operator ===")
    print("JIT warmup..."); t0 = time.time()
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    P0 = np.full((G,G,G), 0.5)
    phi_lin_strict(P0, u_grid, gamma=1.0, tau=1.0, p_grid=p_grid, NQ=16)
    MC2.build_P(np.zeros(5), u_grid)
    print(f"  {time.time()-t0:.1f}s")

    # Solve at tau=0.001 (smooth regime, strict should converge nicely)
    print(f"\n--- tau=0.001 ---")
    coefs1, P1, F1, viol1, ts1 = solve_strict(0.001, 4)

    # Compare to R4 result at same tau
    from lin_cdf_richardson import phi_lin_richardson
    # Take the same coefs, compute F under R4 too
    P_at_coefs = MC2.build_P(coefs1, u_grid)
    F_R4 = phi_lin_richardson(P_at_coefs, u_grid, hs=(0.5,0.4,0.3,0.2),
                                       gamma=GAMMA, tau=0.001, G_p=121, NQK=16,
                                       p_grid=p_grid)
    F_R4_inf = float(np.max(np.abs(F_R4 - P_at_coefs)))
    print(f"\nAt same coefs, R4 gives F={F_R4_inf:.3e}")

    # Also compare lookups
    from lin_cdf_strict import build_mu_table_lin_strict
    mu_strict = build_mu_table_lin_strict(P_at_coefs, u_grid, p_grid, gl_u, gl_du,
                                                   0.001, G, 16)
    print(f"strict mu range at this P: [{mu_strict.min():.6f}, {mu_strict.max():.6f}]")


if __name__ == "__main__":
    main()
