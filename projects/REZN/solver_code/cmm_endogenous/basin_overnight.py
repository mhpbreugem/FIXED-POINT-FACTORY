"""Overnight basin-of-attraction sweep for the strict-h=0 moving-grid solver
at the max-deficit corner (tau=2, gamma=0.01).

PURPOSE: settle whether the strict-h=0 operator admits ANY nontrivial fixed
point distinct from the trivial P_FR. Three lines of attack at multiple G:

  Section A: P_FR sanity at each G — does the solver leave P_FR alone?
  Section B: Kernel warm starts — does any kernel solution land on a non-FR FP?
  Section C: Perturbation analysis — random N(0, sigma) noise around P_FR.
              If a basin exists, large-sigma perturbations should reveal it.
  Section D: Continuation-style mixtures alpha*kernel + (1-alpha)*P_FR.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3

TAU, GAMMA = 2.0, 0.01
DEEP = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/deep_max_deficit'
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/basin_overnight'
os.makedirs(OUT, exist_ok=True)


def make_PFR(uf, tau):
    U1, U2, U3 = np.meshgrid(uf, uf, uf, indexing='ij')
    return 1.0 / (1.0 + np.exp(-tau*(U1+U2+U3)))


def build_mesh(P_full, uf, lo, hi, n_levels=16):
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.05, 0.95, n_levels)
    p_levels = np.quantile(p_flat, qs)
    p_levels = np.unique(p_levels)
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    n_verts = np.array([s[0].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    n_faces = np.array([s[1].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    verts = np.zeros((int(n_verts.sum()), 3))
    faces = np.zeros((int(n_faces.sum()), 3), dtype=np.int64)
    iv = ifc = 0
    for s in surfs:
        if s is None: continue
        v, f = s
        verts[iv:iv+v.shape[0]] = v; faces[ifc:ifc+f.shape[0]] = f.astype(np.int64)
        iv += v.shape[0]; ifc += f.shape[0]
    return verts, faces, n_verts, n_faces, p_levels


def eval_res(verts, n_verts, faces, n_faces, p_levels):
    r = vertex_residual_jit(verts, n_verts, faces, n_faces,
                             np.asarray(p_levels), TAU, GAMMA)
    return r


def picard_step(verts, n_verts, faces, n_faces, p_levels, omega=0.05,
                step_cap_abs=0.03):
    """Damped Picard along surface normals (cheap, stable)."""
    r = eval_res(verts, n_verts, faces, n_faces, p_levels)
    # Per-vertex normal (averaged from incident faces)
    normals = np.zeros_like(verts); counts = np.zeros(verts.shape[0])
    v_off = np.concatenate(([0], np.cumsum(n_verts)))
    f_off = np.concatenate(([0], np.cumsum(n_faces)))
    for m in range(n_verts.size):
        if n_verts[m] == 0: continue
        V = verts[v_off[m]:v_off[m+1]]
        F = faces[f_off[m]:f_off[m+1]]
        for tri in F:
            v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
            n_face = np.cross(v1-v0, v2-v0)
            ln = np.linalg.norm(n_face)
            if ln < 1e-20: continue
            n_face = n_face / ln
            for k in tri:
                normals[v_off[m] + k] += n_face
                counts[v_off[m] + k] += 1
    valid = counts > 0
    normals[valid] /= counts[valid, None]
    nm = np.linalg.norm(normals, axis=1, keepdims=True)
    nm = np.where(nm > 0, nm, 1.0)
    normals = normals / nm
    step_sz = np.clip(omega * r, -step_cap_abs, step_cap_abs)
    return verts + step_sz[:, None] * normals, r


def run_trajectory(label, P_inner, uf, lo, hi, n_iter=30, n_levels=16):
    Gi = hi - lo
    P_full = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    verts0, faces, n_verts, n_faces, p_levels = build_mesh(P_full, uf, lo, hi, n_levels)
    P_FR_inner = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    print(f"    {label}: {len(p_levels)} surfs, {verts0.shape[0]} verts, "
          f"init dist_to_FR={float(np.max(np.abs(P_inner-P_FR_inner))):.3f}",
          flush=True)
    verts = verts0.copy()
    traj = []
    for it in range(n_iter+1):
        r = eval_res(verts, n_verts, faces, n_faces, p_levels)
        max_r = float(np.max(np.abs(r))); med_r = float(np.median(np.abs(r)))
        dmoved = float(np.max(np.linalg.norm(verts - verts0, axis=1)))
        traj.append(dict(it=it, max_r=max_r, med_r=med_r, dmoved=dmoved))
        if max_r < 1e-6: break
        if it < n_iter:
            verts, _ = picard_step(verts, n_verts, faces, n_faces, p_levels)
    return dict(label=label, init_dist_to_FR=float(np.max(np.abs(P_inner-P_FR_inner))),
                traj=traj,
                max_r_init=traj[0]['max_r'], med_r_init=traj[0]['med_r'],
                max_r_final=traj[-1]['max_r'], med_r_final=traj[-1]['med_r'],
                dmoved_final=traj[-1]['dmoved'])


def get_kernel(Gi):
    """Best-available kernel solution at this G, resampled if needed."""
    if Gi == 21:
        return np.load(f"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/P_ld_t{TAU}_g{GAMMA}.npy")
    src = f"{DEEP}/P_t{TAU}_g{GAMMA}_G{Gi}.npy"
    if os.path.exists(src): return np.load(src)
    # Fall back: upsample from G=33
    P33 = np.load(f"{DEEP}/P_t{TAU}_g{GAMMA}_G33.npy")
    idx = np.linspace(0, 32, Gi).astype(int)
    return P33[idx][:, idx][:, :, idx].copy()


def main():
    rng = np.random.default_rng(2026)
    all_results = {}
    for Gi in [15, 21, 25, 33]:
        print(f"\n\n========== G = {Gi} ==========", flush=True)
        du, uf, lo, hi = build_grid(Gi)
        P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
        try:
            P_kernel = get_kernel(Gi)
            if P_kernel.shape[0] != Gi:
                idx = np.linspace(0, P_kernel.shape[0]-1, Gi).astype(int)
                P_kernel = P_kernel[idx][:, idx][:, :, idx]
        except FileNotFoundError:
            P_kernel = None
            print(f"  no kernel at G={Gi}", flush=True)

        runs_G = []

        # A. P_FR sanity
        runs_G.append(run_trajectory('PFR', P_FR.copy(), uf, lo, hi))
        json.dump(runs_G, open(f"{OUT}/G{Gi}.json", 'w'), indent=2, default=str)

        # B. kernel warm start (if available)
        if P_kernel is not None:
            runs_G.append(run_trajectory('kernel', P_kernel.copy(), uf, lo, hi))
            json.dump(runs_G, open(f"{OUT}/G{Gi}.json", 'w'), indent=2, default=str)

        # C. perturbations of P_FR
        for sigma in [0.01, 0.05, 0.1, 0.2]:
            P_pert = P_FR + sigma * rng.standard_normal(P_FR.shape)
            P_pert = np.clip(P_pert, 1e-6, 1-1e-6)
            runs_G.append(run_trajectory(f'PFR+N({sigma})', P_pert, uf, lo, hi))
            json.dump(runs_G, open(f"{OUT}/G{Gi}.json", 'w'), indent=2, default=str)

        # D. mixtures
        if P_kernel is not None:
            for alpha in [0.25, 0.5, 0.75]:
                mix = alpha * P_kernel + (1 - alpha) * P_FR
                runs_G.append(run_trajectory(f'mix_a{alpha}', mix, uf, lo, hi))
                json.dump(runs_G, open(f"{OUT}/G{Gi}.json", 'w'), indent=2, default=str)

        all_results[f'G{Gi}'] = runs_G
        json.dump(all_results, open(f"{OUT}/all_basin.json", 'w'), indent=2, default=str)
        print(f"\n  G={Gi} complete: {len(runs_G)} trajectories", flush=True)

    # Summary
    print("\n\n========== SUMMARY ==========", flush=True)
    print(f"{'G':<5} {'label':<22} {'init max_r':>11} {'final max_r':>12} {'med_init':>10} {'med_final':>10} {'dmoved':>9}", flush=True)
    for gkey, runs in all_results.items():
        for r in runs:
            print(f"{gkey:<5} {r['label']:<22} {r['max_r_init']:>11.3e} {r['max_r_final']:>12.3e} "
                  f"{r['med_r_init']:>10.3e} {r['med_r_final']:>10.3e} {r['dmoved_final']:>9.4f}",
                  flush=True)
    print(f"\nSaved to {OUT}/all_basin.json", flush=True)


if __name__ == "__main__":
    main()
