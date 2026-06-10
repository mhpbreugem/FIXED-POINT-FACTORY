"""Proper IFT Newton for K=3 strict-h=0: Powell-hybrid (MINPACK HYBRD) with
   Broyden-rank-1 Jacobian updates -- the standard ``Newton via IFT'' solver.

This is much more efficient than scipy.newton_krylov (which rebuilds the
finite-difference Jacobian on every step). HYBRD builds the Jacobian once
via finite differences and updates it via Broyden's secant rule each step,
so the Jacobian becomes more and more accurate as iteration proceeds.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dd_k3_logodds",
    "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd/dd_k3_logodds.py")
_mod = importlib.util.module_from_spec(_spec); sys.modules["dd_k3_logodds"] = _mod
_spec.loader.exec_module(_mod)
make_u_grid_logodds = _mod.make_u_grid_logodds
compute_L_pub_from_P = _mod.compute_L_pub_from_P
make_p_grid_Lpub = _mod.make_p_grid_Lpub
phi_cube_logodds = _mod.phi_cube_logodds

from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from scipy.stats import norm


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_ift"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def _symm(P):
    Psym = (P + np.transpose(P,(0,2,1)) + np.transpose(P,(1,0,2))
            + np.transpose(P,(1,2,0)) + np.transpose(P,(2,0,1))
            + np.transpose(P,(2,1,0))) / 6.0
    return 0.5 * (Psym + (1.0 - Psym[::-1,::-1,::-1]))


def make_F_logit(u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq):
    """F in P-space with bound penalty: penalize out-of-bounds + standard Phi residual."""
    def F(P_flat):
        P_cube = P_flat.reshape(G, G, G)
        # Penalty for out-of-bounds
        oob_lo = np.maximum(0.0, 1e-12 - P_cube)
        oob_hi = np.maximum(0.0, P_cube - (1.0 - 1e-12))
        P_proj = np.clip(P_cube, 1e-12, 1 - 1e-12)
        P_proj = _symm(P_proj)
        mu_table = build_mu_table_lin_strict(P_proj, u_grid, p_grid, gl_u, gl_du, tau, G, nq)
        P_new = phi_cube_logodds(P_proj, mu_table, p_grid, u_grid, gamma, G)
        P_new = _symm(P_new)
        # Residual = (Phi-P at projected point) PLUS a stiff bound penalty
        F = (P_new - P_proj).ravel() + 1000.0 * (oob_lo - oob_hi).ravel()
        return F
    return F


def solve_ift(P_init, gamma, tau, G, G_p=81, nq=16, method="hybr",
                  xi_max=0.85, verbose=False, tag=""):
    u_grid, _ = make_u_grid_logodds(G, tau, xi_max=xi_max)
    P = P_init.copy() if P_init is not None else 1.0/(1.0+np.exp(-tau*(
        np.add.outer(np.add.outer(u_grid, u_grid), u_grid))))
    P = np.clip(P, 1e-12, 1-1e-12)
    Psym = (P + np.transpose(P,(0,2,1)) + np.transpose(P,(1,0,2))
            + np.transpose(P,(1,2,0)) + np.transpose(P,(2,0,1))
            + np.transpose(P,(2,1,0))) / 6.0
    P = 0.5 * (Psym + (1.0 - Psym[::-1,::-1,::-1]))

    p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
    p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)

    sd = 1.0 / np.sqrt(tau)
    f_mix = 0.5*norm.pdf(u_grid, -0.5, sd) + 0.5*norm.pdf(u_grid, +0.5, sd)
    W3 = f_mix[:, None, None] * f_mix[None, :, None] * f_mix[None, None, :]
    W3 /= W3.max()

    F = make_F_logit(u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq)
    t0 = time.time()
    try:
        sol = root(F, P.ravel(), method=method, options=dict(xtol=1e-14, maxfev=2000))
        P_sol = np.clip(_symm(sol.x.reshape(G, G, G)), 1e-12, 1 - 1e-12)
        nfev = sol.nfev
        success = sol.success
        msg = sol.message
    except Exception as e:
        P_sol = P; nfev = 0; success = False; msg = str(e)
    wall = time.time() - t0
    # Final residual in P-space
    mu_table_final = build_mu_table_lin_strict(P_sol, u_grid, p_grid, gl_u, gl_du, tau, G, nq)
    P_phi = phi_cube_logodds(P_sol, mu_table_final, p_grid, u_grid, gamma, G)
    P_phi = _symm(P_phi); P_phi = np.clip(P_phi, 1e-12, 1-1e-12)
    F_final_arr = P_phi - P_sol
    F_inf = float(np.max(np.abs(F_final_arr)))
    F_w = float(np.max(np.abs(F_final_arr) * W3))
    sym_err = float(np.max(np.abs(P_sol + P_sol[::-1,::-1,::-1] - 1)))
    return P_sol, F_inf, F_w, wall, nfev, success, msg, p_grid


def main():
    G = 11; gamma = 100.0
    taus = np.logspace(np.log10(0.001), np.log10(1.2), 50)
    print(f"IFT-Newton (MINPACK HYBRD) tau ladder: G={G}, gamma={gamma}, "
          f"{len(taus)} steps from {taus[0]:.4f} to {taus[-1]:.4f}", flush=True)
    PICARD_DIR = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_ladder"
    print(f"Warm-starting from Picard FPs at {PICARD_DIR}", flush=True)
    print(f"{'idx':>4} {'tau':>10} {'F_inf':>11} {'F_w':>11} {'P_range':>20} "
          f"{'sym':>9} {'nfev':>5} {'wall':>6} {'OK?':>4}", flush=True)
    results = []
    t0 = time.time()
    for i, tau in enumerate(taus):
        picard_file = f"{PICARD_DIR}/fp_t{tau:.5f}.npz"
        P_warm = np.load(picard_file)['P'] if os.path.exists(picard_file) else None
        P, F_inf, F_w, wall, nfev, ok, msg, p_grid = solve_ift(P_warm, gamma, tau, G)
        sym = float(np.max(np.abs(P + P[::-1,::-1,::-1] - 1)))
        flag = "Y" if ok else "n"
        print(f"{i:>4d} {tau:>10.5f} {F_inf:>11.3e} {F_w:>11.3e} "
              f"[{P.min():.4f}, {P.max():.4f}] {sym:>9.2e} {nfev:>5d} {wall:>5.1f}s {flag:>4}",
              flush=True)
        np.savez(f"{OUT}/fp_t{tau:.5f}.npz",
                  P=P, tau=tau, gamma=gamma, G=G, F_inf=F_inf, F_w=F_w,
                  p_grid=p_grid, sym_err=sym, nfev=nfev, wall=wall,
                  converged=ok)
        results.append(dict(idx=i, tau=float(tau), F_inf=float(F_inf), F_w=float(F_w),
                              P_min=float(P.min()), P_max=float(P.max()),
                              sym_err=float(sym), nfev=int(nfev), wall=wall,
                              converged=bool(ok), msg=msg))
    print(f"\nTotal wall: {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(f"{OUT}/ladder.json", "w"), indent=2)

    # Compare with Picard and Newton-Krylov ladders
    import json as _j
    pic = _j.load(open(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_ladder/ladder.json"))
    pic_F = [r['F_w_best'] for r in pic]
    try:
        nk = _j.load(open(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_newton/ladder.json"))
        nk_F = [r['F_w'] for r in nk]
        nk_nfev = [r['nfev'] for r in nk]
    except Exception:
        nk_F = None; nk_nfev = None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].loglog(taus, [r['F_w'] for r in results], '-o', ms=5, label='IFT-Newton (HYBRD)')
    if nk_F: axes[0].loglog(taus, nk_F, '-s', ms=4, alpha=0.7, label='Newton-Krylov')
    axes[0].loglog(taus, pic_F, '-^', ms=4, alpha=0.7, label='Picard')
    axes[0].axhline(1e-15, color='k', ls=':', alpha=0.5, label='machine eps')
    axes[0].set_xlabel('tau'); axes[0].set_ylabel('|F|_w (density-weighted)')
    axes[0].set_title('Residual: IFT-Newton vs Newton-Krylov vs Picard')
    axes[0].legend(); axes[0].grid(alpha=0.3, which='both')

    axes[1].semilogx(taus, [r['P_min'] for r in results], '-o', ms=4, label='P_min')
    axes[1].semilogx(taus, [r['P_max'] for r in results], '-s', ms=4, label='P_max')
    axes[1].axhline(0.5, color='k', ls=':', alpha=0.5)
    axes[1].set_xlabel('tau'); axes[1].set_ylabel('P')
    axes[1].set_title('IFT-Newton P range')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].semilogx(taus, [r['nfev'] for r in results], '-o', ms=5, label='IFT-Newton (HYBRD)')
    if nk_nfev: axes[2].semilogx(taus, nk_nfev, '-s', ms=4, alpha=0.7, label='Newton-Krylov')
    axes[2].set_xlabel('tau'); axes[2].set_ylabel('F-evals per cell')
    axes[2].set_title('Cost per cell')
    axes[2].legend(); axes[2].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/ladder_ift.png", dpi=120)
    plt.close()
    print(f"Saved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
