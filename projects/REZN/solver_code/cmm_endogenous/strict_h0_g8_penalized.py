"""STRICT h=0 search with PENALTY against P_FR.

User's question: maybe a nontrivial strict-h=0 fixed point exists but every
solver drifts to P_FR.  Test by ADDING a barrier penalty that punishes
proximity to P_FR.  The solver then minimizes
    [r_FP ; lambda * max(0, threshold - ||P - P_FR||) ]
For threshold > 0, the solver is forbidden from landing inside a ball around
P_FR.  If a nontrivial fixed point exists OUTSIDE that ball, the solver will
find it (residual r_FP -> 0, penalty 0).  If no such fixed point exists, the
penalty term will remain non-zero (constraint violation) and r_FP stays
non-trivial.

Three penalty levels: threshold in {0.05, 0.15, 0.3}.  At each, multiple
starting points.  G=8 (cheap).  Higher precision: residual eval uses
existing JIT'd path; the penalty is a single scalar per surface.
"""
import os, sys, time, json
import numpy as np
from scipy.optimize import least_squares, newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from cmm_stage4_blockjac import surface_residual, face_normals
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth

TAU, GAMMA = 2.0, 0.01
G_TEST = 8
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/strict_h0_g8_penalized'
os.makedirs(OUT, exist_ok=True)


def make_PFR(uf, tau):
    U1, U2, U3 = np.meshgrid(uf, uf, uf, indexing='ij')
    return 1.0 / (1.0 + np.exp(-tau*(U1+U2+U3)))


def solve_kernel(Gi, tau, gamma):
    pad = 2; UMAX = 4.0; C = 0.45; K = 3
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P0 = P_full[lo:hi, lo:hi, lo:hi].ravel().copy()
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - x.reshape(Gi, Gi, Gi)).ravel()
    try:
        x = newton_krylov(F, P0, f_tol=1e-12, maxiter=80, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    return x.reshape(Gi, Gi, Gi), uf, lo, hi


def build_mesh_with_FR(P_full, uf, lo, hi, n_levels=10):
    """Extract surfaces AND record the corresponding FR positions for the same
    price levels (used for the penalty)."""
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.05, 0.95, n_levels)
    p_levels = np.unique(np.quantile(p_flat, qs))
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    V = []; F = []; p_kept = []
    V_FR = []
    # P_FR full-grid
    P_FR_full = make_PFR(uf, TAU)
    surfs_FR = extract_surfaces_marching_cubes(P_FR_full, uf, p_levels)
    for ip, (s, sFR) in enumerate(zip(surfs, surfs_FR)):
        if s is None or s[0].shape[0] == 0: continue
        if sFR is None or sFR[0].shape[0] == 0: continue
        V.append(s[0].copy()); F.append(s[1].astype(np.int64))
        V_FR.append(sFR[0].copy())
        p_kept.append(p_levels[ip])
    return V, F, V_FR, np.asarray(p_kept)


def fr_distance_per_vertex(V, V_FR):
    """For each vertex of V, the distance to the NEAREST vertex of V_FR.
    Sum -> distance from current configuration to the FR mesh."""
    # Each vertex has a position in R^3 belonging to the same level set in
    # the limit; compare to the closest V_FR vertex
    d = np.array([np.min(np.linalg.norm(V_FR - v[None, :], axis=1)) for v in V])
    return d


def solve_surface_penalized(V0, F_mesh, V_FR, p_m, tau, gamma,
                              threshold=0.15, lam_pen=5.0, anchor_w=1e-4,
                              max_nfev=400):
    """least_squares with FR-distance penalty."""
    N = V0.shape[0]
    x0 = V0.flatten()
    bound_lo = np.full(x0.size, -8.0); bound_hi = np.full(x0.size, 8.0)

    def resid(x):
        V = x.reshape(N, 3)
        r_fp = surface_residual(V, F_mesh, p_m, tau, gamma)
        # FR distance per vertex
        d_FR = fr_distance_per_vertex(V, V_FR)
        # Barrier: penalty term zero when d_FR >= threshold, lam_pen*(threshold-d_FR) when closer
        r_pen = lam_pen * np.maximum(0.0, threshold - d_FR)
        # Anchor (light): prevent runaway
        r_anchor = anchor_w * (x - x0)
        return np.concatenate([r_fp, r_pen, r_anchor])

    try:
        result = least_squares(resid, x0, method='trf', max_nfev=max_nfev,
                                xtol=1e-9, ftol=1e-9, verbose=0,
                                bounds=(bound_lo, bound_hi))
        V_final = result.x.reshape(N, 3)
        r_fp = surface_residual(V_final, F_mesh, p_m, tau, gamma)
        d_FR_final = fr_distance_per_vertex(V_final, V_FR)
        pen_final = lam_pen * np.maximum(0.0, threshold - d_FR_final)
        return V_final, r_fp, d_FR_final, pen_final, result.nfev, result.status
    except Exception as e:
        return V0, surface_residual(V0, F_mesh, p_m, tau, gamma), \
               np.zeros(N), np.zeros(N), 0, -99


def run_penalized(label, P_inner, uf, lo, hi, threshold, lam_pen=5.0, n_levels=10):
    Gi = hi - lo
    P_full = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    V, F_mesh, V_FR, p_levels = build_mesh_with_FR(P_full, uf, lo, hi, n_levels)
    M = len(p_levels)
    if M == 0: return dict(label=label, threshold=threshold, error='no surfaces')

    init_r = []
    init_dFR = []
    for m in range(M):
        init_r.append(surface_residual(V[m], F_mesh[m], p_levels[m], TAU, GAMMA))
        init_dFR.append(fr_distance_per_vertex(V[m], V_FR[m]))
    init_r_all = np.concatenate(init_r)
    init_dFR_all = np.concatenate(init_dFR)
    print(f"  [{label}, thr={threshold:.2f}] M={M}, total_v={sum(v.shape[0] for v in V)}; "
          f"init max|r|={np.max(np.abs(init_r_all)):.3e}, "
          f"med|r|={np.median(np.abs(init_r_all)):.3e}; "
          f"init dist to FR: min={init_dFR_all.min():.3f} med={np.median(init_dFR_all):.3f}",
          flush=True)

    t0 = time.time()
    final_r = []; final_dFR = []; final_pen = []; nfev_total = 0
    for m in range(M):
        V_f, r_f, dFR_f, pen_f, nfev, status = solve_surface_penalized(
            V[m], F_mesh[m], V_FR[m], p_levels[m], TAU, GAMMA,
            threshold=threshold, lam_pen=lam_pen)
        final_r.append(r_f); final_dFR.append(dFR_f); final_pen.append(pen_f)
        nfev_total += nfev
    wall = time.time() - t0
    final_r_all = np.concatenate(final_r)
    final_dFR_all = np.concatenate(final_dFR)
    final_pen_all = np.concatenate(final_pen)
    print(f"    FINAL: max|r|={np.max(np.abs(final_r_all)):.3e} "
          f"med|r|={np.median(np.abs(final_r_all)):.3e}  "
          f"max_pen={np.max(final_pen_all):.3e}  "
          f"min_dFR={final_dFR_all.min():.4f}  "
          f"nfev={nfev_total}  wall={wall:.1f}s", flush=True)
    return dict(label=label, threshold=threshold, M=M, total_v=int(sum(v.shape[0] for v in V)),
                init_max=float(np.max(np.abs(init_r_all))),
                init_med=float(np.median(np.abs(init_r_all))),
                init_min_dFR=float(init_dFR_all.min()),
                init_med_dFR=float(np.median(init_dFR_all)),
                final_max=float(np.max(np.abs(final_r_all))),
                final_med=float(np.median(np.abs(final_r_all))),
                final_min_dFR=float(final_dFR_all.min()),
                final_med_dFR=float(np.median(final_dFR_all)),
                final_max_pen=float(np.max(final_pen_all)),
                final_med_pen=float(np.median(final_pen_all)),
                nfev_total=int(nfev_total), wall=float(wall))


def main():
    Gi = G_TEST
    print(f"=== PENALIZED strict-h=0 search at G={Gi}, "
          f"(tau={TAU}, gamma={GAMMA}) ===\n", flush=True)
    print("Solving kernel at G=8 fresh...", flush=True)
    P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
    P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    dist_k_FR = float(np.max(np.abs(P_kernel - P_FR)))
    print(f"  kernel solved; P_kernel-P_FR sup-distance = {dist_k_FR:.4f}\n",
          flush=True)

    rng = np.random.default_rng(2026)
    starts = [
        ('kernel', P_kernel.copy()),
        ('PFR+N(0.2)', np.clip(P_FR + 0.2*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.5)', np.clip(P_FR + 0.5*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('0.5*ker+0.5*FR', 0.5*P_kernel + 0.5*P_FR),
        ('no-learn', init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA),
                                          np.full(3, 1.0))[lo:hi, lo:hi, lo:hi]),
    ]
    thresholds = [0.05, 0.15, 0.30]
    all_results = []
    for thr in thresholds:
        print(f"\n========== PENALTY THRESHOLD = {thr} ==========", flush=True)
        for label, P_inner in starts:
            try:
                res = run_penalized(label, P_inner, uf, lo, hi, threshold=thr)
                all_results.append(res)
            except Exception as e:
                all_results.append(dict(label=label, threshold=thr, error=str(e)))
                print(f"  ERROR ({label}, thr={thr}): {e}", flush=True)
            json.dump(all_results, open(f"{OUT}/penalized.json", 'w'),
                      indent=2, default=str)

    print("\n\n=== SUMMARY ===", flush=True)
    print(f"{'thr':>5} {'start':<18} {'init max|r|':>12} {'final max|r|':>13} "
          f"{'init med dFR':>13} {'final med dFR':>14} {'max pen':>10}", flush=True)
    for r in all_results:
        if 'error' in r:
            print(f"{r['threshold']:>5} {r['label']:<18}  ERROR: {r['error']}", flush=True)
            continue
        print(f"{r['threshold']:>5.2f} {r['label']:<18} {r['init_max']:>12.3e} "
              f"{r['final_max']:>13.3e} {r['init_med_dFR']:>13.3f} "
              f"{r['final_med_dFR']:>14.3f} {r['final_max_pen']:>10.3e}", flush=True)
    print(f"\nWrote {OUT}/penalized.json", flush=True)


if __name__ == "__main__":
    main()
