"""Tau-ladder for the corrected logodds K=3 solver.

50 log-spaced tau in [0.001, 1.2], gamma=100, G=11. Chain warm-start.
Live per-cell summary: F_best (raw, weighted), wall, n_iter, P_range.
"""
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
# Import after path setup
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
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds_ladder"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def solve_one(P_init, gamma, tau, G, G_p=81, nq=16, n_iter=40, xi_max=0.85, verbose=False):
    u_grid, xi_grid = make_u_grid_logodds(G, tau, xi_max=xi_max)
    sd = 1.0 / np.sqrt(tau)
    f_mix = 0.5*norm.pdf(u_grid, -0.5, sd) + 0.5*norm.pdf(u_grid, +0.5, sd)
    W3 = f_mix[:, None, None] * f_mix[None, :, None] * f_mix[None, None, :]
    W3 /= W3.max()
    P = P_init.copy() if P_init is not None else 1.0/(1.0+np.exp(-tau*(
        np.add.outer(np.add.outer(u_grid, u_grid), u_grid))))
    P = np.clip(P, 1e-12, 1-1e-12)
    p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
    p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)
    Xh, Gh = [], []
    F_best = float('inf'); P_best = P.copy()
    for it in range(n_iter):
        mu_table = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u, gl_du, tau, G, nq)
        P_new = phi_cube_logodds(P, mu_table, p_grid, u_grid, gamma, G)
        diff = np.abs(P_new - P)
        F_raw = float(np.max(diff))
        F_w = float(np.max(diff * W3))
        if F_w < F_best:
            F_best = F_w; P_best = P.copy()
        # Damped Anderson
        Xh.append(P.ravel().copy()); Gh.append(P_new.ravel().copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: P = 0.5*P + 0.5*P_new
        else:
            try:
                DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
                R_k = Gh[k-1] - Xh[k-1]
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x_a = Gh[k-1] + DG @ ga
                P = 0.5*P + 0.5*x_a.reshape(G,G,G)
            except:
                P = 0.5*P + 0.5*P_new
        P = np.clip(P, 1e-12, 1-1e-12)
        # Symmetry projections
        Psym = (P + np.transpose(P,(0,2,1)) + np.transpose(P,(1,0,2))
                + np.transpose(P,(1,2,0)) + np.transpose(P,(2,0,1))
                + np.transpose(P,(2,1,0))) / 6.0
        P = 0.5 * (Psym + (1.0 - Psym[::-1,::-1,::-1]))
        if it == 5:
            p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
            p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
    return P_best, F_best, p_grid


def main():
    G = 11; gamma = 100.0
    taus = np.logspace(np.log10(0.001), np.log10(1.2), 50)
    print(f"Tau ladder: G={G}, gamma={gamma}, {len(taus)} steps from {taus[0]:.4f} to {taus[-1]:.4f}", flush=True)
    print(f"{'idx':>4} {'tau':>10} {'F_w_best':>12} {'P_range':>22} {'sym_err':>10} {'wall':>6}", flush=True)
    results = []
    P_prev = None
    u_prev = None
    t0 = time.time()
    for i, tau in enumerate(taus):
        # If warm-starting, need to interpolate the previous P from u_prev grid
        # to the new u-grid. For simplicity, only warm-start if u-grids match,
        # otherwise restart from sigmoid init.
        u_now, _ = make_u_grid_logodds(G, tau, xi_max=0.85)
        if P_prev is not None and u_prev is not None and np.allclose(u_now, u_prev):
            P_init = P_prev
        elif P_prev is not None:
            # Different grid -- use P_prev values relative to the previous grid's
            # xi (which is uniform), evaluated at the new xi (also uniform with same xi_max).
            # Since xi spacing is identical for fixed G and xi_max, P_prev IS valid as init.
            P_init = P_prev
        else:
            P_init = None
        t1 = time.time()
        P, F_w, p_grid = solve_one(P_init, gamma, tau, G, n_iter=40)
        wall = time.time() - t1
        sym = np.max(np.abs(P + P[::-1,::-1,::-1] - 1))
        print(f"{i:>4d} {tau:>10.5f} {F_w:>12.3e} [{P.min():.4f}, {P.max():.4f}]   {sym:>10.2e} {wall:>5.1f}s", flush=True)
        np.savez(f"{OUT}/fp_t{tau:.5f}.npz",
                  P=P, tau=tau, gamma=gamma, G=G, F_w_best=F_w,
                  p_grid=p_grid, sym_err=sym)
        results.append(dict(idx=i, tau=float(tau), F_w_best=float(F_w),
                              P_min=float(P.min()), P_max=float(P.max()),
                              sym_err=float(sym), wall=wall))
        P_prev = P; u_prev = u_now
    print(f"\nTotal wall: {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(f"{OUT}/ladder.json", "w"), indent=2)
    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Fs = [r['F_w_best'] for r in results]
    syms = [r['sym_err'] for r in results]
    P_mins = [r['P_min'] for r in results]
    P_maxs = [r['P_max'] for r in results]
    axes[0].loglog(taus, Fs, '-o', ms=4, label='F_w_best')
    axes[0].loglog(taus, syms, '-s', ms=3, label='sym_err')
    axes[0].set_xlabel('tau'); axes[0].set_ylabel('|F|')
    axes[0].set_title(f'tau-ladder (gamma=100, G={G}): residual')
    axes[0].legend(); axes[0].grid(alpha=0.3, which='both')
    axes[1].semilogx(taus, P_mins, '-o', ms=3, label='P_min')
    axes[1].semilogx(taus, P_maxs, '-s', ms=3, label='P_max')
    axes[1].axhline(0.5, color='k', ls=':', alpha=0.5)
    axes[1].set_xlabel('tau'); axes[1].set_ylabel('P')
    axes[1].set_title('FP price range vs tau')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/ladder_summary.png", dpi=120)
    plt.close()
    print(f"Saved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
