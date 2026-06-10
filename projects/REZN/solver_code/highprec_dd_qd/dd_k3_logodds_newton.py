"""IFT-based Newton-Krylov solver for K=3 strict-h=0 in logodds coordinates.

Uses scipy.optimize.newton_krylov: GMRES with finite-difference
Jacobian-vector products. Should converge much deeper than Picard at high tau.

Pipeline:
  1. Compute Phi(P) using existing strict-h=0 builder + crra clearing
  2. Define F(P_flat) = (Phi(P) - P).ravel()
  3. newton_krylov(F, P_warm)
  4. Symmetrize the result (sign-flip + permutation)
  5. Save per-cell with live updates

Drop-in replacement for the Picard ladder script.
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
try: from scipy.optimize._nonlin import NoConvergence
except: from scipy.optimize import NoConvergence
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "dd_k3_logodds",
    "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd/dd_k3_logodds.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["dd_k3_logodds"] = _mod
_spec.loader.exec_module(_mod)
make_u_grid_logodds = _mod.make_u_grid_logodds
compute_L_pub_from_P = _mod.compute_L_pub_from_P
make_p_grid_Lpub = _mod.make_p_grid_Lpub
phi_cube_logodds = _mod.phi_cube_logodds

from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from scipy.stats import norm


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_newton"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def make_Phi(u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq):
    """Return callable Phi(P_cube) for the strict-h=0 logodds operator."""
    def Phi(P_flat):
        P_cube = P_flat.reshape(G, G, G)
        P_cube = np.clip(P_cube, 1e-12, 1 - 1e-12)
        mu_table = build_mu_table_lin_strict(P_cube, u_grid, p_grid, gl_u, gl_du, tau, G, nq)
        P_new = phi_cube_logodds(P_cube, mu_table, p_grid, u_grid, gamma, G)
        # Symmetrize to keep Newton on the symmetric manifold
        Psym = (P_new + np.transpose(P_new, (0,2,1)) + np.transpose(P_new, (1,0,2))
                + np.transpose(P_new, (1,2,0)) + np.transpose(P_new, (2,0,1))
                + np.transpose(P_new, (2,1,0))) / 6.0
        P_new = 0.5 * (Psym + (1.0 - Psym[::-1, ::-1, ::-1]))
        return P_new
    return Phi


def solve_newton(P_init, gamma, tau, G, G_p=81, nq=16, maxiter=40, f_tol=1e-12,
                     xi_max=0.85, verbose=True, tag=""):
    u_grid, xi_grid = make_u_grid_logodds(G, tau, xi_max=xi_max)
    P = P_init.copy() if P_init is not None else 1.0/(1.0+np.exp(-tau*(
        np.add.outer(np.add.outer(u_grid, u_grid), u_grid))))
    P = np.clip(P, 1e-12, 1 - 1e-12)
    # Symmetric init
    Psym = (P + np.transpose(P,(0,2,1)) + np.transpose(P,(1,0,2))
            + np.transpose(P,(1,2,0)) + np.transpose(P,(2,0,1))
            + np.transpose(P,(2,1,0))) / 6.0
    P = 0.5 * (Psym + (1.0 - Psym[::-1,::-1,::-1]))

    # Initial L_pub (frozen during Newton)
    p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
    p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)

    sd = 1.0 / np.sqrt(tau)
    f_mix = 0.5*norm.pdf(u_grid, -0.5, sd) + 0.5*norm.pdf(u_grid, +0.5, sd)
    W3 = f_mix[:, None, None] * f_mix[None, :, None] * f_mix[None, None, :]
    W3 /= W3.max()

    Phi = make_Phi(u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq)

    F_history = []
    iter_count = [0]
    def F_resid(P_flat):
        P_cube = P_flat.reshape(G, G, G)
        Pn = Phi(P_flat)
        F = (Pn - P_cube).ravel()
        F_inf = float(np.max(np.abs(F)))
        F_w = float(np.max(np.abs(F.reshape(G, G, G)) * W3))
        F_history.append((F_inf, F_w))
        iter_count[0] += 1
        if verbose and iter_count[0] % 5 == 1:
            print(f"  [{tag}] F-eval {iter_count[0]}: F_inf={F_inf:.3e} F_w={F_w:.3e}", flush=True)
        return F

    t0 = time.time()
    try:
        x_sol = newton_krylov(F_resid, P.ravel(), f_tol=f_tol, maxiter=maxiter,
                                  method='lgmres', line_search='wolfe', verbose=False)
        P_sol = x_sol.reshape(G, G, G)
    except NoConvergence as e:
        P_sol = e.args[0].reshape(G, G, G)
        if verbose: print(f"  [{tag}] NoConvergence; using best iterate", flush=True)
    except Exception as e:
        P_sol = P
        if verbose: print(f"  [{tag}] solver error: {e}", flush=True)

    wall = time.time() - t0
    F_final = np.max(np.abs(Phi(P_sol.ravel()) - P_sol))
    F_w_final = np.max(np.abs(Phi(P_sol.ravel()) - P_sol) * W3)
    sym_err = np.max(np.abs(P_sol + P_sol[::-1,::-1,::-1] - 1))
    if verbose:
        print(f"  [{tag}] DONE in {wall:.1f}s, "
              f"F_inf={F_final:.3e} F_w={F_w_final:.3e} sym={sym_err:.2e} "
              f"({iter_count[0]} F-evals)", flush=True)
    return P_sol, float(F_final), float(F_w_final), wall, iter_count[0], p_grid


def main():
    G = 11; gamma = 100.0
    taus = np.logspace(np.log10(0.001), np.log10(1.2), 50)
    print(f"NEWTON-KRYLOV tau ladder: G={G}, gamma={gamma}, {len(taus)} steps "
          f"from {taus[0]:.4f} to {taus[-1]:.4f}", flush=True)
    # Load Picard FPs as warm-starts (informative branch)
    PICARD_DIR = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_ladder"
    print(f"Warm-starting from Picard FPs at {PICARD_DIR}", flush=True)
    print(f"{'idx':>4} {'tau':>10} {'F_inf':>11} {'F_w':>11} {'P_range':>20} "
          f"{'sym':>9} {'nfev':>5} {'wall':>6}", flush=True)
    results = []
    t0 = time.time()
    for i, tau in enumerate(taus):
        # Pick up Picard FP at this tau
        picard_file = f"{PICARD_DIR}/fp_t{tau:.5f}.npz"
        if os.path.exists(picard_file):
            P_warm = np.load(picard_file)['P']
        else:
            P_warm = None
        P, F_inf, F_w, wall, nfev, p_grid = solve_newton(P_warm, gamma, tau, G,
                                                                maxiter=30, verbose=False)
        sym = np.max(np.abs(P + P[::-1,::-1,::-1] - 1))
        print(f"{i:>4d} {tau:>10.5f} {F_inf:>11.3e} {F_w:>11.3e} "
              f"[{P.min():.4f}, {P.max():.4f}] {sym:>9.2e} {nfev:>5d} {wall:>5.1f}s",
              flush=True)
        np.savez(f"{OUT}/fp_t{tau:.5f}.npz",
                  P=P, tau=tau, gamma=gamma, G=G, F_inf=F_inf, F_w=F_w,
                  p_grid=p_grid, sym_err=sym, nfev=nfev, wall=wall)
        results.append(dict(idx=i, tau=float(tau), F_inf=float(F_inf), F_w=float(F_w),
                              P_min=float(P.min()), P_max=float(P.max()),
                              sym_err=float(sym), nfev=int(nfev), wall=wall))
        P_prev = P
    print(f"\nTotal wall: {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(f"{OUT}/ladder.json", "w"), indent=2)
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    Finfs = [r['F_inf'] for r in results]
    Fws = [r['F_w'] for r in results]
    syms = [r['sym_err'] for r in results]
    Pmins = [r['P_min'] for r in results]
    Pmaxs = [r['P_max'] for r in results]
    nfevs = [r['nfev'] for r in results]
    axes[0].loglog(taus, Finfs, '-o', ms=4, label='F_inf')
    axes[0].loglog(taus, Fws, '-s', ms=4, label='F_w (density-weighted)')
    axes[0].loglog(taus, syms, '-^', ms=3, label='sym_err')
    axes[0].set_xlabel('tau'); axes[0].set_ylabel('|F|')
    axes[0].set_title(f'Newton-Krylov (gamma={gamma}, G={G}): residual')
    axes[0].legend(); axes[0].grid(alpha=0.3, which='both')
    axes[1].semilogx(taus, Pmins, '-o', ms=3, label='P_min')
    axes[1].semilogx(taus, Pmaxs, '-s', ms=3, label='P_max')
    axes[1].axhline(0.5, color='k', ls=':', alpha=0.5)
    axes[1].set_xlabel('tau'); axes[1].set_ylabel('P')
    axes[1].set_title('FP P range vs tau')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].semilogx(taus, nfevs, '-o', ms=4)
    axes[2].set_xlabel('tau'); axes[2].set_ylabel('Newton-Krylov F-evals')
    axes[2].set_title('cost per cell')
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/ladder_newton.png", dpi=120)
    plt.close()
    print(f"Saved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
