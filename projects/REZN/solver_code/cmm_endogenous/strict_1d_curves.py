"""1D-curve formulation of strict h=0 (per user's suggestion).

For each (own-signal U on grid, price p_m on grid), one 1D curve in 2D:
   gamma_{i,m}(s) = (u_2(s), u_3(s)),  s in [0, 1]
discretized as N_arc parametric points.  Agents 2, 3 use the SAME curve
family by S_3 symmetry: agent 2's slice at u_2 = U_i, p_m is read off
agent 1's curves at u_1 = U_i, p_m (after the (u_2 <-> u_1) relabeling).

Cross-agent consistency: at the K=3 binary-symmetric case, all three
agents share identical curve families.  So the unknown is ONE family:
  X = { (u_2_{i,m,s}, u_3_{i,m,s}) : i in [G_own], m in [M], s in [N_arc] }
Total size: G_own * M * N_arc * 2  (~2k at G=8, M=8, N=16).

Penalty against P_FR proximity: each curve point (U_i, u_2, u_3) on
surface p_m should have FR-implied price sigma(tau*(U_i + u_2 + u_3))
distance >= threshold from p_m (i.e. each curve point should NOT lie
on the FR level set sigma(tau*(U_i + u_2 + u_3)) = p_m).
"""
import os, sys, time, json
import numpy as np
from scipy.optimize import least_squares
import math
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.signals import f_signal
from reznsrc.demand import clear_crra

TAU, GAMMA = 2.0, 0.01
G_OWN = 8           # grid for own-signal values
N_ARC = 16          # discretization of each curve
M = 8               # number of price levels
U_MAX = 4.0
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/strict_1d_curves'
os.makedirs(OUT, exist_ok=True)


def make_grids():
    u_grid = np.linspace(-U_MAX*0.95, U_MAX*0.95, G_OWN)
    # Price levels symmetric around 0.5 (binary payoff sign-flip)
    p_levels = np.array([0.04, 0.12, 0.25, 0.40, 0.60, 0.75, 0.88, 0.96])
    return u_grid, p_levels


def make_init_curves(u_grid, p_levels, choice='kernel-like'):
    """Initial 1D curve family.

    For each (U_i in u_grid, p_m in p_levels), initialize the curve
    gamma_{i,m}(s) = (u_2(s), u_3(s)) with s = arclength parameter.
    """
    X = np.zeros((G_OWN, len(p_levels), N_ARC, 2))
    s = np.linspace(-1, 1, N_ARC)  # parameter
    for i, U in enumerate(u_grid):
        for m, p in enumerate(p_levels):
            # FR ansatz: logit p = tau*(U + u_2 + u_3) => sum = logit(p)/tau - U
            # parametrize along the anti-diagonal: u_2 = a + s, u_3 = a - s
            # where 2a = sum, so a = (logit(p)/tau - U)/2
            a = (math.log(p/(1-p))/TAU - U) / 2
            # parameter s scaled to cover roughly [-3, 3] in u-space
            ss = s * 3.0
            X[i, m, :, 0] = a + ss   # u_2
            X[i, m, :, 1] = a - ss   # u_3
    if choice == 'kernel-like':
        # Add a curvature bend to make it not pure FR
        for i, U in enumerate(u_grid):
            for m, p in enumerate(p_levels):
                # bend: shift toward 0.5 in price means a should be ~0
                a_FR = (math.log(p/(1-p))/TAU - U) / 2
                bend = 0.3 * (0.5 - p)  # small offset
                X[i, m, :, 0] += bend
                X[i, m, :, 1] -= bend
    elif choice == 'random':
        rng = np.random.default_rng(42)
        X += 0.3 * rng.standard_normal(X.shape)
    return X


def slice_arclen_integrals(X_im, u_grid_index, U, tau):
    """For one (own, price) curve X_im (N_ARC, 2) representing (u_2(s), u_3(s)),
    compute integrals A_0, A_1 = sum f_v(u_2)*f_v(u_3) * |arclength element|."""
    u2 = X_im[:, 0]; u3 = X_im[:, 1]
    # arclength element from finite differences
    du2 = np.gradient(u2); du3 = np.gradient(u3)
    ds = np.sqrt(du2**2 + du3**2)
    # f_v values
    s_norm = math.sqrt(tau/(2*math.pi))
    f0_2 = s_norm * np.exp(-0.5*tau*(u2+0.5)**2)
    f0_3 = s_norm * np.exp(-0.5*tau*(u3+0.5)**2)
    f1_2 = s_norm * np.exp(-0.5*tau*(u2-0.5)**2)
    f1_3 = s_norm * np.exp(-0.5*tau*(u3-0.5)**2)
    A0 = np.sum(f0_2 * f0_3 * ds)
    A1 = np.sum(f1_2 * f1_3 * ds)
    return A0, A1


def residual_at_curve_point(X, u_grid, p_levels, i_own, m, s_idx, tau, gamma):
    """Residual at the (i_own, m, s_idx) curve point.

    The point is at (u_1, u_2, u_3) where u_1 = u_grid[i_own],
                                          (u_2, u_3) = X[i_own, m, s_idx, :].

    Equilibrium condition: price clearing should equal p_levels[m].
    Posteriors:
       mu_1 = bayes from agent-1 slice integrals (THIS curve's integrals)
       mu_2 = bayes from agent-2 slice integrals at u_2 = X[i_own, m, s_idx, 0]
              -> interpolate over u_grid for u_2; same curve family by symmetry
       mu_3 = same for u_3
    """
    u1 = u_grid[i_own]
    u2 = X[i_own, m, s_idx, 0]
    u3 = X[i_own, m, s_idx, 1]
    s_norm = math.sqrt(tau/(2*math.pi))

    # Agent 1 evidence
    A0_1, A1_1 = slice_arclen_integrals(X[i_own, m], i_own, u1, tau)
    f0_1 = s_norm*math.exp(-0.5*tau*(u1+0.5)**2)
    f1_1 = s_norm*math.exp(-0.5*tau*(u1-0.5)**2)
    num1 = f1_1 * A1_1; den1 = f0_1*A0_1 + num1
    mu_1 = max(1e-12, min(1-1e-12, num1/den1)) if den1 > 0 else 0.5

    # For agents 2, 3: need slice integrals at (u_2, p_m), (u_3, p_m)
    # By S_3 symmetry, agent-2 curve at u_2 = U_target is the same family
    # indexed by u_2_target. Interpolate over u_grid to find appropriate slice.
    def agent_evidence(U_target):
        # Find nearest u_grid index
        idx = np.searchsorted(u_grid, U_target)
        idx = min(max(idx, 1), len(u_grid)-1)
        # Linear interpolation between adjacent slice integrals
        f_lo = (u_grid[idx] - U_target) / (u_grid[idx] - u_grid[idx-1] + 1e-30)
        f_lo = max(0, min(1, f_lo))
        A0_lo, A1_lo = slice_arclen_integrals(X[idx-1, m], idx-1, u_grid[idx-1], tau)
        A0_hi, A1_hi = slice_arclen_integrals(X[idx, m], idx, u_grid[idx], tau)
        return f_lo*A0_lo + (1-f_lo)*A0_hi, f_lo*A1_lo + (1-f_lo)*A1_hi

    A0_2, A1_2 = agent_evidence(u2)
    f0_2 = s_norm*math.exp(-0.5*tau*(u2+0.5)**2)
    f1_2 = s_norm*math.exp(-0.5*tau*(u2-0.5)**2)
    num2 = f1_2*A1_2; den2 = f0_2*A0_2 + num2
    mu_2 = max(1e-12, min(1-1e-12, num2/den2)) if den2 > 0 else 0.5

    A0_3, A1_3 = agent_evidence(u3)
    f0_3 = s_norm*math.exp(-0.5*tau*(u3+0.5)**2)
    f1_3 = s_norm*math.exp(-0.5*tau*(u3-0.5)**2)
    num3 = f1_3*A1_3; den3 = f0_3*A0_3 + num3
    mu_3 = max(1e-12, min(1-1e-12, num3/den3)) if den3 > 0 else 0.5

    # CRRA market clearing
    mu_vec = np.array([mu_1, mu_2, mu_3])
    gamma_vec = np.array([gamma]*3); W_vec = np.array([1.0]*3)
    p_clear = clear_crra(mu_vec, gamma_vec, W_vec)
    return float(p_clear - p_levels[m])


def full_residual(X_flat, u_grid, p_levels, tau, gamma):
    """Residual vector over all curve points."""
    X = X_flat.reshape(G_OWN, len(p_levels), N_ARC, 2)
    r = np.zeros(G_OWN * len(p_levels) * N_ARC)
    k = 0
    for i in range(G_OWN):
        for m in range(len(p_levels)):
            for s in range(N_ARC):
                r[k] = residual_at_curve_point(X, u_grid, p_levels, i, m, s, tau, gamma)
                k += 1
    return r


def FR_distance(X_flat, u_grid, p_levels, tau):
    """For each curve point, distance from FR level set (price under FR)."""
    X = X_flat.reshape(G_OWN, len(p_levels), N_ARC, 2)
    dist = np.zeros(G_OWN * len(p_levels) * N_ARC)
    k = 0
    for i in range(G_OWN):
        u1 = u_grid[i]
        for m in range(len(p_levels)):
            p_m = p_levels[m]
            for s in range(N_ARC):
                u2 = X[i, m, s, 0]; u3 = X[i, m, s, 1]
                p_FR_here = 1.0/(1.0 + math.exp(-tau*(u1 + u2 + u3)))
                dist[k] = abs(p_FR_here - p_m)
                k += 1
    return dist


def main():
    print(f"=== 1D-curve strict h=0, (tau={TAU}, gamma={GAMMA}), "
          f"G_own={G_OWN}, M={M}, N_arc={N_ARC}, "
          f"total unknowns = {G_OWN*M*N_ARC*2} ===\n", flush=True)
    u_grid, p_levels = make_grids()
    print(f"u_grid: {u_grid}", flush=True)
    print(f"p_levels: {p_levels}\n", flush=True)

    results = []
    for choice in ['kernel-like', 'random']:
        for threshold in [0.0, 0.05, 0.15]:
            label = f"{choice}_thr{threshold}"
            print(f"--- {label} ---", flush=True)
            X0 = make_init_curves(u_grid, p_levels, choice=choice)
            x0 = X0.flatten()

            def resid(x):
                r_fp = full_residual(x, u_grid, p_levels, TAU, GAMMA)
                if threshold > 0:
                    d_FR = FR_distance(x, u_grid, p_levels, TAU)
                    r_pen = 10.0 * np.maximum(0.0, threshold - d_FR)
                    return np.concatenate([r_fp, r_pen])
                return r_fp

            # Init metrics
            r0 = resid(x0)
            print(f"  init: max|r|={np.max(np.abs(r0)):.3e}, med|r|={np.median(np.abs(r0)):.3e}",
                  flush=True)
            t0 = time.time()
            try:
                res = least_squares(resid, x0, method='trf', max_nfev=2000,
                                     xtol=1e-10, ftol=1e-10, verbose=0)
                xf = res.x; r_final = resid(xf)
                r_fp_final = full_residual(xf, u_grid, p_levels, TAU, GAMMA)
                d_FR_final = FR_distance(xf, u_grid, p_levels, TAU)
                wall = time.time() - t0
                print(f"  FINAL: max|r_FP|={np.max(np.abs(r_fp_final)):.3e} "
                      f"med|r_FP|={np.median(np.abs(r_fp_final)):.3e}  "
                      f"min_dFR={np.min(d_FR_final):.4f}  "
                      f"nfev={res.nfev}  wall={wall:.0f}s", flush=True)
                results.append(dict(
                    label=label, choice=choice, threshold=threshold,
                    init_max=float(np.max(np.abs(r0))),
                    init_med=float(np.median(np.abs(r0))),
                    final_max_fp=float(np.max(np.abs(r_fp_final))),
                    final_med_fp=float(np.median(np.abs(r_fp_final))),
                    final_min_dFR=float(np.min(d_FR_final)),
                    final_med_dFR=float(np.median(d_FR_final)),
                    nfev=int(res.nfev), wall=wall, status=int(res.status)))
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                results.append(dict(label=label, error=str(e)))
            json.dump(results, open(f"{OUT}/results_1d.json", 'w'), indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'label':<28} {'thr':>5} {'init max':>10} {'final max':>10} "
          f"{'init med':>10} {'final med':>10} {'min dFR':>9}", flush=True)
    for r in results:
        if 'error' in r: continue
        print(f"{r['label']:<28} {r['threshold']:>5.2f} {r['init_max']:>10.3e} "
              f"{r['final_max_fp']:>10.3e} {r['init_med']:>10.3e} "
              f"{r['final_med_fp']:>10.3e} {r['final_min_dFR']:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
