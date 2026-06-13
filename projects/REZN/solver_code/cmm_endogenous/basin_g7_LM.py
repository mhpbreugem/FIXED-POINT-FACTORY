"""Basin-of-attraction at G=7 with the STRONG free-vertex LM solver.

At G=7 the cube has 7^3 = 343 cells; the moving mesh has only ~50-200
vertices total per surface. This makes full LM with finite-difference
Jacobian + topology guard tractable (Stage 4/5 machinery used at G=21
is fast at G=7).

The question: at this small resolution, does ANY starting point converge
to a nontrivial strict-h=0 fixed point? If yes, the discretization at
G=33 was just the problem; if no even at G=7 with the strongest solver
we have, the GS-degeneracy is structural.

Sequence:
  1. Solve kernel at G=7 to get a reference kernel solution (in addition
     to upsampling the existing G=21).
  2. Build moving meshes at G=7 for: P_FR, kernel-G7, perturbations.
  3. Run free-vertex LM (per-surface block-FD Jacobian, Levenberg
     damping, per-vertex edge cap, topology guard) for up to 30 iters
     from each starting point.
  4. Report final residuals + deficits.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from cmm_stage4_blockjac import (surface_residual, surface_jacobian_fd,
                                  per_vertex_min_edge, face_normals,
                                  topology_ok)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass

TAU, GAMMA = 2.0, 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/basin_g7_LM'
os.makedirs(OUT, exist_ok=True)
K = 3; pad = 2; UMAX = 4.0; C = 0.45


def make_PFR(uf, tau):
    U1, U2, U3 = np.meshgrid(uf, uf, uf, indexing='ij')
    return 1.0 / (1.0 + np.exp(-tau*(U1+U2+U3)))


def solve_kernel(Gi, tau, gamma):
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
        x = newton_krylov(F, P0, f_tol=1e-12, maxiter=40, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    return x.reshape(Gi, Gi, Gi), uf, lo, hi


def build_mesh(P_full, uf, lo, hi, n_levels=10):
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.05, 0.95, n_levels)
    p_levels = np.unique(np.quantile(p_flat, qs))
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    V = []; F = []
    p_kept = []
    for ip, s in enumerate(surfs):
        if s is None or s[0].shape[0] == 0: continue
        V.append(s[0].copy()); F.append(s[1].astype(np.int64))
        p_kept.append(p_levels[ip])
    return V, F, np.asarray(p_kept)


def run_LM_g7(label, P_inner, uf, lo, hi, max_iter=40, fd_eps=1e-5, tol=1e-7,
              n_levels=10):
    Gi = hi - lo
    P_full = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    V, F, p_levels = build_mesh(P_full, uf, lo, hi, n_levels)
    M = len(p_levels)
    n_v = np.array([v.shape[0] for v in V])
    total_v = int(n_v.sum())
    if total_v == 0:
        return dict(label=label, error='no surfaces extracted')
    ref_normals = [face_normals(V[m], F[m]) for m in range(M)]
    min_edges = [per_vertex_min_edge(V[m], F[m]) for m in range(M)]
    V0 = [v.copy() for v in V]
    # JIT warmup
    surface_residual(V[0], F[0], p_levels[0], TAU, GAMMA)
    surface_jacobian_fd(V[0], F[0], p_levels[0], TAU, GAMMA, fd_eps)
    lam = 0.1
    history = []
    r_blocks = J_blocks = None
    need_rebuild = True
    Fmax = float('inf'); Fmed = float('inf')
    for it in range(max_iter):
        if need_rebuild:
            r_blocks = []; J_blocks = []
            for m in range(M):
                if n_v[m] == 0:
                    r_blocks.append(np.zeros(0)); J_blocks.append(None); continue
                r_m = surface_residual(V[m], F[m], p_levels[m], TAU, GAMMA)
                J_m = surface_jacobian_fd(V[m], F[m], p_levels[m], TAU, GAMMA, fd_eps)
                r_blocks.append(r_m); J_blocks.append(J_m)
            r_all = np.concatenate(r_blocks)
            Fmax = float(np.max(np.abs(r_all))); Fmed = float(np.median(np.abs(r_all)))
            history.append((Fmax, Fmed, lam))
            need_rebuild = False
        if Fmax < tol:
            break
        # Solve per-block LM, apply per-vertex caps
        V_new = [v.copy() for v in V]
        for m in range(M):
            if J_blocks[m] is None: continue
            Jm = J_blocks[m]; rm = r_blocks[m]
            JtJ = Jm.T @ Jm
            Jtr = Jm.T @ rm
            A = JtJ + lam * np.eye(JtJ.shape[0])
            try:
                step = -np.linalg.solve(A, Jtr)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A, Jtr, rcond=None)[0]
            step3 = step.reshape(-1, 3)
            v_step_norm = np.linalg.norm(step3, axis=1)
            limit = 0.25 * min_edges[m]
            scale = np.minimum(1.0, limit / np.maximum(v_step_norm, 1e-30))
            step3 = step3 * scale[:, None]
            sn = float(np.max(np.linalg.norm(step3, axis=1))) if step3.size else 0
            if sn > 0.05:
                step3 *= (0.05 / sn)
            V_new[m] = V[m] + step3
        # Topology check
        topo_ok = all(V_new[m].size == 0 or topology_ok(V_new[m], F[m], ref_normals[m])
                       for m in range(M))
        if topo_ok:
            r_new = []
            for m in range(M):
                if V_new[m].size == 0: r_new.append(np.zeros(0)); continue
                r_new.append(surface_residual(V_new[m], F[m], p_levels[m], TAU, GAMMA))
            r_new_all = np.concatenate(r_new)
            cost_old = 0.5*np.dot(r_all, r_all); cost_new = 0.5*np.dot(r_new_all, r_new_all)
            accept = cost_new < cost_old
        else:
            accept = False
        if accept:
            V = V_new; need_rebuild = True
            lam *= 0.5
        else:
            lam *= 3.0
        if lam > 1e10:
            break
    # Final state metrics
    r_blocks_f = [surface_residual(V[m], F[m], p_levels[m], TAU, GAMMA) for m in range(M)]
    r_all_f = np.concatenate(r_blocks_f)
    final_max = float(np.max(np.abs(r_all_f))); final_med = float(np.median(np.abs(r_all_f)))
    # Distance moved
    moved = max(float(np.max(np.linalg.norm(V[m] - V0[m], axis=1))) for m in range(M) if V[m].size)
    return dict(label=label, history=history,
                init_max_r=history[0][0], init_med_r=history[0][1],
                final_max_r=final_max, final_med_r=final_med,
                dmoved=moved, n_iter=len(history))


def main():
    Gi = 7
    print(f"=== G={Gi} basin search with full LM solver, (tau={TAU}, gamma={GAMMA}) ===\n",
          flush=True)
    # Solve kernel fresh at G=7
    print("Solving kernel at G=7 (fresh)...", flush=True)
    P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
    P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    print(f"  kernel G=7 solved; dist to FR = {float(np.max(np.abs(P_kernel-P_FR))):.4f}",
          flush=True)
    rng = np.random.default_rng(2026)
    starts = [
        ('PFR', P_FR.copy()),
        ('kernel_G7', P_kernel.copy()),
        ('PFR+N(0.01)', np.clip(P_FR + 0.01*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.05)', np.clip(P_FR + 0.05*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.1)',  np.clip(P_FR + 0.10*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.2)',  np.clip(P_FR + 0.20*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('PFR+N(0.3)',  np.clip(P_FR + 0.30*rng.standard_normal(P_FR.shape), 1e-6, 1-1e-6)),
        ('no-learning', init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA),
                                              np.full(3, 1.0))[lo:hi, lo:hi, lo:hi]),
        ('0.25*kernel+0.75*FR', 0.25*P_kernel + 0.75*P_FR),
        ('0.50*kernel+0.50*FR', 0.50*P_kernel + 0.50*P_FR),
        ('0.75*kernel+0.25*FR', 0.75*P_kernel + 0.25*P_FR),
    ]
    results = []
    for label, P_inner in starts:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        res = run_LM_g7(label, P_inner, uf, lo, hi, max_iter=40)
        res['wall'] = time.time() - t0
        results.append(res)
        if 'error' in res:
            print(f"  ERROR: {res['error']}", flush=True)
        else:
            print(f"  iter={res['n_iter']} init max/med = {res['init_max_r']:.3e}/{res['init_med_r']:.3e} "
                  f"-> final {res['final_max_r']:.3e}/{res['final_med_r']:.3e} "
                  f"moved={res['dmoved']:.3f} wall={res['wall']:.1f}s", flush=True)
        json.dump(results, open(f"{OUT}/basin_g7_LM.json", 'w'), indent=2, default=str)
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'label':<22} {'init max':>10} {'final max':>10} {'init med':>10} {'final med':>10} {'moved':>8}", flush=True)
    for r in results:
        if 'error' in r: continue
        print(f"{r['label']:<22} {r['init_max_r']:>10.3e} {r['final_max_r']:>10.3e} "
              f"{r['init_med_r']:>10.3e} {r['final_med_r']:>10.3e} {r['dmoved']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
