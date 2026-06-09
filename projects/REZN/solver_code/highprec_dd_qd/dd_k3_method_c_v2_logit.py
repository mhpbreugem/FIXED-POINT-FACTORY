"""Method C v2 with LOGIT-SPACE residual.
Loss: F_L = logit(Phi(P)) - L  where L = h(u1)+h(u2)+h(u3) is the ansatz.
Vs the P-space loss F = Phi(P) - P.

Boundary regions (P near 0 or 1) where small P-changes correspond to large
L-changes are now properly weighted in the loss.
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from scipy.optimize import least_squares
import dd_k3_method_c_v2 as MC2
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u
from lin_cdf_richardson import phi_lin_richardson


GAMMA = 100.0
G = 7
HS = (0.5, 0.4, 0.3, 0.2)


def F_residual_logit(coefs, tau, u_grid, p_grid, gamma=GAMMA):
    """Residual in logit space: logit(Phi(P)) - logit(P)."""
    P = MC2.build_P(coefs, u_grid)
    P_new = phi_lin_richardson(P, u_grid, hs=HS, gamma=gamma, tau=tau,
                                       G_p=p_grid.size, NQK=16, p_grid=p_grid)
    eps = 1e-12
    Pc = np.clip(P, eps, 1-eps); P_new_c = np.clip(P_new, eps, 1-eps)
    L = np.log(Pc/(1-Pc))
    L_new = np.log(P_new_c/(1-P_new_c))
    return (L_new - L).ravel()


def solve_logit(tau, params0=None, gamma=GAMMA, max_nfev=200, verbose=False):
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)
    if params0 is None:
        params0 = np.zeros(MC2.D_SIGMA + 1); params0[0] = 0.5
    res = least_squares(F_residual_logit, params0,
                              args=(tau, u_grid, p_grid),
                              kwargs=dict(gamma=gamma),
                              jac="2-point", method="trf",
                              max_nfev=max_nfev, ftol=1e-14, xtol=1e-14,
                              verbose=2 if verbose else 0)
    P = MC2.build_P(res.x, u_grid)
    viol = 0; mind = 0.0
    for ax in range(3):
        dd = np.diff(P, axis=ax)
        viol += int((dd < 0).sum()); mind = min(mind, float(dd.min()))
    # Also compute P-space F for comparison
    P_new = phi_lin_richardson(P, u_grid, hs=HS, gamma=gamma, tau=tau,
                                       G_p=p_grid.size, NQK=16, p_grid=p_grid)
    F_P = float(np.max(np.abs(P_new - P)))
    F_L = float(np.max(np.abs(res.fun)))
    return res.x, P, F_P, F_L, viol


if __name__ == "__main__":
    # Quick grid sweep: same as before but with L-space loss
    GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    TAUS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
    ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_logit"
    OUT_JSON = f"{ROOT}/grid.json"
    COEFS_DIR = f"{ROOT}/coefs"
    os.makedirs(COEFS_DIR, exist_ok=True)
    print("=== Method C v2 (logit loss) grid ===")
    print("JIT warmup..."); t0 = time.time()
    u_grid = make_cdf_uniform_grid(G); MC2.build_P(np.zeros(5), u_grid)
    print(f"  {time.time()-t0:.1f}s")
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    t_start = time.time(); n_done = 0; n_total = len(GAMMAS)*len(TAUS)
    for gamma in GAMMAS:
        coefs_warm = None
        print(f"\n--- gamma={gamma} ---")
        for tau in TAUS:
            n_done += 1
            key = f"g{gamma:.4g}_t{tau:.4f}"
            t0 = time.time()
            try:
                coefs, P, F_P, F_L, viol = solve_logit(tau, params0=coefs_warm, gamma=gamma)
            except Exception as e:
                print(f"  ERR: {e}"); continue
            wall = time.time() - t0
            np.save(f"{COEFS_DIR}/{key}.npy", coefs)
            results[key] = dict(gamma=float(gamma), tau=float(tau),
                                          F_P=F_P, F_L=F_L, viol=int(viol), wall=wall,
                                          P_min=float(P.min()), P_max=float(P.max()),
                                          coefs=coefs.tolist())
            json.dump(results, open(OUT_JSON, "w"), indent=2)
            print(f"  [{n_done:>3d}/{n_total}] g={gamma:>6.3g} t={tau:.3f} "
                  f"F_P={F_P:.2e} F_L={F_L:.2e} viol={viol} {wall:.1f}s",
                  flush=True)
            coefs_warm = coefs
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.1f}min")
