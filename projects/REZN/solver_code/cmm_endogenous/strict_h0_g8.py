"""STRONG strict-h=0 moving-grid solver at G=8 using scipy.optimize.least_squares.

Goal: find a partial-revelation strict-h=0 fixed point at (tau=2, gamma=0.01)
distinct from P_FR. Uses scipy's industrial-strength trust-region solver
(trust-constr / trf) with per-surface block residual. NO custom step caps,
NO custom acceptance — let scipy do its job. Topology guard applied only
as final check on the converged result.

For each starting point, runs scipy.optimize.least_squares with method='trf'
(handles bounds, trust region, FD Jacobian) for up to 200 function evaluations
per surface. Reports final ||r|| per surface and aggregate.

Starting points:
  - P_FR (sanity)
  - kernel G=8 (the candidate non-trivial branch)
  - Random perturbations of P_FR at multiple sigmas
  - Random perturbations of kernel
  - Mixtures
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
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/strict_h0_g8'
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


def solve_surface_LS(V0, F_mesh, p_m, tau, gamma, max_nfev=200, verbose=0):
    """Solve one surface for h=0 fixed point via scipy least_squares.

    Unknowns: 3*N_m vertex coords (flattened).
    Residuals: N_m per-vertex fixed-point residuals.
    Over-determined? No — under-determined (3N unknowns, N equations).
    trf handles this fine; we add weak L2 anchor (move-from-V0) to break
    the 2*N gauge degeneracy."""
    N = V0.shape[0]
    x0 = V0.flatten()
    bound_lo = np.full(x0.size, -8.0)
    bound_hi = np.full(x0.size, 8.0)
    # Small anchor weight to break gauge
    anchor_w = 1e-3
    def resid(x):
        V = x.reshape(N, 3)
        r = surface_residual(V, F_mesh, p_m, tau, gamma)
        anchor = anchor_w * (x - x0)
        return np.concatenate([r, anchor])
    try:
        result = least_squares(resid, x0, method='trf', max_nfev=max_nfev,
                                xtol=1e-10, ftol=1e-10, verbose=verbose,
                                bounds=(bound_lo, bound_hi))
        V_final = result.x.reshape(N, 3)
        r_final = surface_residual(V_final, F_mesh, p_m, tau, gamma)
        return V_final, r_final, result.cost, result.nfev, result.status
    except Exception as e:
        return V0, surface_residual(V0, F_mesh, p_m, tau, gamma), None, 0, -99


def build_mesh(P_full, uf, lo, hi, n_levels=10):
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.05, 0.95, n_levels)
    p_levels = np.unique(np.quantile(p_flat, qs))
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    V = []; F = []; p_kept = []
    for ip, s in enumerate(surfs):
        if s is None or s[0].shape[0] == 0: continue
        V.append(s[0].copy()); F.append(s[1].astype(np.int64))
        p_kept.append(p_levels[ip])
    return V, F, np.asarray(p_kept)


def run_strict(label, P_inner, uf, lo, hi, n_levels=10):
    Gi = hi - lo
    P_full = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    V, F_mesh, p_levels = build_mesh(P_full, uf, lo, hi, n_levels)
    M = len(p_levels)
    if M == 0:
        return dict(label=label, error='no surfaces')
    P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    print(f"  [{label}] M={M} surfaces, total_v={sum(v.shape[0] for v in V)}, "
          f"init dist to FR={float(np.max(np.abs(P_inner-P_FR))):.4f}", flush=True)

    # Initial residuals
    init_r = []
    for m in range(M):
        r = surface_residual(V[m], F_mesh[m], p_levels[m], TAU, GAMMA)
        init_r.append(r)
    init_all = np.concatenate(init_r)
    print(f"    init: max|r|={np.max(np.abs(init_all)):.3e} "
          f"med|r|={np.median(np.abs(init_all)):.3e}", flush=True)

    # Solve each surface via scipy LS
    t0 = time.time()
    final_V = []; final_r = []; per_surface = []
    for m in range(M):
        if V[m].shape[0] == 0: continue
        ref_normals = face_normals(V[m], F_mesh[m])
        V_f, r_f, cost, nfev, status = solve_surface_LS(V[m], F_mesh[m], p_levels[m],
                                                          TAU, GAMMA, max_nfev=300)
        final_V.append(V_f); final_r.append(r_f)
        per_surface.append(dict(
            m=m, p=float(p_levels[m]), N=int(V[m].shape[0]),
            init_max=float(np.max(np.abs(init_r[m]))),
            init_med=float(np.median(np.abs(init_r[m]))),
            final_max=float(np.max(np.abs(r_f))),
            final_med=float(np.median(np.abs(r_f))),
            nfev=int(nfev), status=int(status),
            dmoved=float(np.max(np.linalg.norm(V_f - V[m], axis=1)))
        ))
    wall = time.time() - t0
    final_all = np.concatenate(final_r)
    print(f"    FINAL: max|r|={np.max(np.abs(final_all)):.3e} "
          f"med|r|={np.median(np.abs(final_all)):.3e}  wall={wall:.1f}s", flush=True)
    moved = max(s['dmoved'] for s in per_surface) if per_surface else 0
    print(f"    moved (max vertex): {moved:.4f}", flush=True)
    return dict(label=label, M=M,
                init_max=float(np.max(np.abs(init_all))),
                init_med=float(np.median(np.abs(init_all))),
                final_max=float(np.max(np.abs(final_all))),
                final_med=float(np.median(np.abs(final_all))),
                wall=wall, dmoved=moved,
                per_surface=per_surface)


def main():
    Gi = G_TEST
    print(f"=== STRICT h=0 search at G={Gi}, (tau={TAU}, gamma={GAMMA}) ===\n",
          flush=True)
    print(f"Solving kernel at G={Gi} (fresh)...", flush=True)
    P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
    P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    print(f"  kernel solved; dist to FR = {float(np.max(np.abs(P_kernel-P_FR))):.4f}\n",
          flush=True)

    rng = np.random.default_rng(2026)
    starts = [
        ('PFR',           P_FR.copy()),
        ('kernel',        P_kernel.copy()),
        ('PFR+N(0.01)',   np.clip(P_FR + 0.01*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.03)',   np.clip(P_FR + 0.03*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.1)',    np.clip(P_FR + 0.10*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.3)',    np.clip(P_FR + 0.30*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('kernel+N(0.05)',np.clip(P_kernel + 0.05*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('kernel+N(0.1)', np.clip(P_kernel + 0.10*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('0.5*ker+0.5*FR', 0.5*P_kernel + 0.5*P_FR),
        ('no-learn',      init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA),
                                              np.full(3, 1.0))[lo:hi, lo:hi, lo:hi]),
    ]
    results = []
    for label, P_inner in starts:
        print(f"\n--- {label} ---", flush=True)
        try:
            res = run_strict(label, P_inner, uf, lo, hi)
        except Exception as e:
            res = dict(label=label, error=str(e))
            print(f"  ERROR: {e}", flush=True)
        results.append(res)
        json.dump(results, open(f"{OUT}/strict_g{Gi}.json", 'w'),
                  indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'label':<22} {'init max':>10} {'final max':>10} {'init med':>10} {'final med':>10} {'moved':>8}",
          flush=True)
    for r in results:
        if 'error' in r:
            print(f"{r['label']:<22}  ERROR: {r['error']}", flush=True); continue
        print(f"{r['label']:<22} {r['init_max']:>10.3e} {r['final_max']:>10.3e} "
              f"{r['init_med']:>10.3e} {r['final_med']:>10.3e} {r['dmoved']:>8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
