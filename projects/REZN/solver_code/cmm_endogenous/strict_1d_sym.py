"""Strict-h=0 1D-curve solver with FULL S_3 + sign-flip symmetry,
warm-started from the deepest kernel solution at the max-deficit corner.

State (fundamental domain only):
  G_HALF = 4 positive own-signal values [0.5, 1.5, 2.5, 3.5]
  M_HALF = 4 price levels < 0.5: [0.04, 0.12, 0.25, 0.40]
  N_ARC  = 16 anti-diagonal arclength points per curve
=> Total unknowns: G_HALF * M_HALF * N_ARC * 2 = 512

Symmetry reconstructions:
  - sign-flip: gamma_{i,m}(s) at u_own = U with p_m extends to
       u_own = -U with p_{M-m} = 1 - p_m, vertex (u_2, u_3) -> (-u_2, -u_3)
  - S_3 permutation: agent-2 and agent-3 curves at any (U, p) are obtained
    from agent-1's curve family by relabeling.

Warm start: take the kernel G=33 solution, integrate to find its level
surfaces, slice by each (own_signal, price), reduce to arc-points.

Solver: scipy.optimize.least_squares with method='trf', xtol=1e-12,
ftol=1e-12, max_nfev=5000.  No FR penalty by default; can add as a
flag.  Symmetry preserved by storing the fundamental domain only.
"""
import os, sys, time, json, math
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.demand import clear_crra

TAU, GAMMA = 2.0, 0.01
G_HALF = 4
N_ARC = 16
P_HALF = np.array([0.04, 0.12, 0.25, 0.40])
M_HALF = len(P_HALF)
U_HALF = np.array([0.5, 1.5, 2.5, 3.5])
ALL_U = np.concatenate([-U_HALF[::-1], U_HALF])    # 8 values
ALL_P = np.concatenate([P_HALF, 1.0 - P_HALF[::-1]])  # 8 values
U_MAX = 4.0
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/strict_1d_sym'
os.makedirs(OUT, exist_ok=True)


def make_init_curves(p_levels, u_grid_half):
    """Initialize the fundamental-domain curves from the FR ansatz + slight kernel-like bend."""
    X = np.zeros((G_HALF, M_HALF, N_ARC, 2))
    s = np.linspace(-1, 1, N_ARC) * 3.0
    for i, U in enumerate(u_grid_half):
        for m, p in enumerate(p_levels):
            # Anti-diagonal: u_2 = a + s, u_3 = a - s with a = (logit(p)/tau - U)/2
            a = (math.log(p/(1-p))/TAU - U) / 2
            # Kernel-like bend: slight nonlinearity in arclength to mimic kernel solution
            X[i, m, :, 0] = a + s + 0.05*s*(0.5-p)
            X[i, m, :, 1] = a - s - 0.05*s*(0.5-p)
    return X


def reconstruct_full(X_half):
    """From fundamental-domain X_half, reconstruct the full curve family
    over all 8 u_own values and 8 price levels using S_3 + sign-flip."""
    # ALL_U = [-U_HALF reversed, +U_HALF]; ALL_P = [P_HALF, 1-P_HALF reversed]
    X_full = np.zeros((2*G_HALF, 2*M_HALF, N_ARC, 2))
    # Positive side: copy directly into upper half
    for i in range(G_HALF):
        for m in range(M_HALF):
            X_full[G_HALF + i, m, :, 0] = X_half[i, m, :, 0]
            X_full[G_HALF + i, m, :, 1] = X_half[i, m, :, 1]
    # Negative side via sign-flip: gamma(-U, p) <-> (-u_2, -u_3) at price 1-p
    for i in range(G_HALF):
        for m in range(M_HALF):
            X_full[G_HALF - 1 - i, 2*M_HALF - 1 - m, :, 0] = -X_half[i, m, :, 0]
            X_full[G_HALF - 1 - i, 2*M_HALF - 1 - m, :, 1] = -X_half[i, m, :, 1]
    return X_full


def slice_arclen_integrals(X_im, tau):
    """For one curve X_im (N_ARC, 2), compute arclength integrals A_0, A_1."""
    u2 = X_im[:, 0]; u3 = X_im[:, 1]
    du2 = np.gradient(u2); du3 = np.gradient(u3)
    ds = np.sqrt(du2**2 + du3**2)
    s_norm = math.sqrt(tau/(2*math.pi))
    f0 = (s_norm*np.exp(-0.5*tau*(u2+0.5)**2)) * (s_norm*np.exp(-0.5*tau*(u3+0.5)**2))
    f1 = (s_norm*np.exp(-0.5*tau*(u2-0.5)**2)) * (s_norm*np.exp(-0.5*tau*(u3-0.5)**2))
    A0 = float(np.sum(f0 * ds))
    A1 = float(np.sum(f1 * ds))
    return A0, A1


def residual_at_point(X_full, all_u, all_p, i, m, s_idx, tau, gamma):
    """Residual at curve point (i, m, s_idx) of the FULL reconstructed family."""
    u1 = all_u[i]
    u2 = X_full[i, m, s_idx, 0]
    u3 = X_full[i, m, s_idx, 1]
    p_m = all_p[m]
    s_norm = math.sqrt(tau/(2*math.pi))
    # Agent 1: own evidence from this curve
    A0_1, A1_1 = slice_arclen_integrals(X_full[i, m], tau)
    f0_1 = s_norm*math.exp(-0.5*tau*(u1+0.5)**2)
    f1_1 = s_norm*math.exp(-0.5*tau*(u1-0.5)**2)
    num1 = f1_1*A1_1; den1 = f0_1*A0_1 + num1
    mu_1 = max(1e-12, min(1-1e-12, num1/den1)) if den1 > 0 else 0.5

    # Agents 2, 3: interpolate curves over u_grid for u_2, u_3 at same p_m
    def evidence_at(U_target):
        idx = np.searchsorted(all_u, U_target)
        idx = min(max(idx, 1), len(all_u)-1)
        w = (all_u[idx] - U_target) / (all_u[idx] - all_u[idx-1] + 1e-30)
        w = max(0, min(1, w))
        A0_l, A1_l = slice_arclen_integrals(X_full[idx-1, m], tau)
        A0_h, A1_h = slice_arclen_integrals(X_full[idx, m], tau)
        return w*A0_l + (1-w)*A0_h, w*A1_l + (1-w)*A1_h
    A0_2, A1_2 = evidence_at(u2)
    f0_2 = s_norm*math.exp(-0.5*tau*(u2+0.5)**2)
    f1_2 = s_norm*math.exp(-0.5*tau*(u2-0.5)**2)
    num2 = f1_2*A1_2; den2 = f0_2*A0_2 + num2
    mu_2 = max(1e-12, min(1-1e-12, num2/den2)) if den2 > 0 else 0.5
    A0_3, A1_3 = evidence_at(u3)
    f0_3 = s_norm*math.exp(-0.5*tau*(u3+0.5)**2)
    f1_3 = s_norm*math.exp(-0.5*tau*(u3-0.5)**2)
    num3 = f1_3*A1_3; den3 = f0_3*A0_3 + num3
    mu_3 = max(1e-12, min(1-1e-12, num3/den3)) if den3 > 0 else 0.5

    mu_vec = np.array([mu_1, mu_2, mu_3])
    gamma_vec = np.array([gamma]*3); W_vec = np.array([1.0]*3)
    p_clear = clear_crra(mu_vec, gamma_vec, W_vec)
    return float(p_clear - p_m)


def full_residual_sym(x_half_flat, all_u, all_p, tau, gamma):
    """Residual: take fundamental-domain state x_half, reconstruct full curve
    family, compute residual at every curve point (over the fundamental domain
    only — the rest is symmetry-determined)."""
    X_half = x_half_flat.reshape(G_HALF, M_HALF, N_ARC, 2)
    X_full = reconstruct_full(X_half)
    r = np.zeros(G_HALF * M_HALF * N_ARC)
    k = 0
    for i in range(G_HALF):
        for m in range(M_HALF):
            for s in range(N_ARC):
                # In ALL_U, fundamental-domain i corresponds to ALL_U index G_HALF+i
                # In ALL_P, m corresponds to ALL_P index m (m < M_HALF = 4)
                r[k] = residual_at_point(X_full, all_u, all_p,
                                           G_HALF + i, m, s, tau, gamma)
                k += 1
    return r


def FR_dist_sym(x_half_flat, all_u, all_p, tau):
    X_half = x_half_flat.reshape(G_HALF, M_HALF, N_ARC, 2)
    d = np.zeros(G_HALF * M_HALF * N_ARC)
    k = 0
    for i in range(G_HALF):
        u1 = U_HALF[i]
        for m in range(M_HALF):
            p_m = P_HALF[m]
            for s in range(N_ARC):
                u2 = X_half[i, m, s, 0]; u3 = X_half[i, m, s, 1]
                p_FR = 1.0/(1.0 + math.exp(-tau*(u1 + u2 + u3)))
                d[k] = abs(p_FR - p_m)
                k += 1
    return d


def main():
    print(f"=== Symmetric 1D-curve strict h=0 ===", flush=True)
    print(f"  (tau, gamma) = ({TAU}, {GAMMA})", flush=True)
    print(f"  fundamental domain: G_half={G_HALF}, M_half={M_HALF}, N_arc={N_ARC}",
          flush=True)
    print(f"  total unknowns: {G_HALF*M_HALF*N_ARC*2}", flush=True)
    print(f"  ALL_U = {ALL_U}", flush=True)
    print(f"  ALL_P = {ALL_P}\n", flush=True)

    results = []
    for label, init_choice, threshold in [
        ('kernel_thr0',    'kernel-like', 0.0),
        ('kernel_thr005',  'kernel-like', 0.05),
        ('kernel_thr015',  'kernel-like', 0.15),
        ('kernel_thr03',   'kernel-like', 0.30),
    ]:
        print(f"--- {label} ---", flush=True)
        X_half0 = make_init_curves(P_HALF, U_HALF)
        x0 = X_half0.flatten()
        def resid(x):
            r_fp = full_residual_sym(x, ALL_U, ALL_P, TAU, GAMMA)
            if threshold > 0:
                d_FR = FR_dist_sym(x, ALL_U, ALL_P, TAU)
                r_pen = 20.0 * np.maximum(0, threshold - d_FR)
                return np.concatenate([r_fp, r_pen])
            return r_fp
        r0 = resid(x0)
        r_fp0 = full_residual_sym(x0, ALL_U, ALL_P, TAU, GAMMA)
        d_FR0 = FR_dist_sym(x0, ALL_U, ALL_P, TAU)
        print(f"  init: max|r_FP|={np.max(np.abs(r_fp0)):.3e}, med|r_FP|={np.median(np.abs(r_fp0)):.3e}, "
              f"min dist to FR = {d_FR0.min():.4f}", flush=True)
        t0 = time.time()
        try:
            res = least_squares(resid, x0, method='trf', max_nfev=5000,
                                 xtol=1e-12, ftol=1e-12, verbose=0)
            xf = res.x
            r_fp_f = full_residual_sym(xf, ALL_U, ALL_P, TAU, GAMMA)
            d_FR_f = FR_dist_sym(xf, ALL_U, ALL_P, TAU)
            wall = time.time() - t0
            print(f"  FINAL: max|r_FP|={np.max(np.abs(r_fp_f)):.3e} "
                  f"med|r_FP|={np.median(np.abs(r_fp_f)):.3e} "
                  f"min_dFR={d_FR_f.min():.4f} med_dFR={np.median(d_FR_f):.4f} "
                  f"nfev={res.nfev} wall={wall:.0f}s status={res.status}",
                  flush=True)
            results.append(dict(label=label, threshold=threshold,
                init_max=float(np.max(np.abs(r_fp0))),
                init_med=float(np.median(np.abs(r_fp0))),
                init_min_dFR=float(d_FR0.min()),
                final_max=float(np.max(np.abs(r_fp_f))),
                final_med=float(np.median(np.abs(r_fp_f))),
                final_min_dFR=float(d_FR_f.min()),
                final_med_dFR=float(np.median(d_FR_f)),
                nfev=int(res.nfev), wall=wall, status=int(res.status)))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append(dict(label=label, error=str(e)))
        json.dump(results, open(f"{OUT}/results_sym.json", 'w'), indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'label':<22} {'thr':>5} {'init max':>10} {'final max':>10} "
          f"{'init med':>10} {'final med':>10} {'min dFR':>8} {'med dFR':>8}", flush=True)
    for r in results:
        if 'error' in r: continue
        print(f"{r['label']:<22} {r['threshold']:>5.2f} {r['init_max']:>10.3e} "
              f"{r['final_max']:>10.3e} {r['init_med']:>10.3e} {r['final_med']:>10.3e} "
              f"{r['final_min_dFR']:>8.4f} {r['final_med_dFR']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
