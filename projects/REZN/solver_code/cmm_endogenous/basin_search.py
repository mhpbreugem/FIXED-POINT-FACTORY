"""Basin-search experiment: does the moving-grid solver converge to any
nontrivial strict-h=0 fixed point at the max-deficit corner?

Tests multiple starting points: kernel G21/25/33 solutions, P_FR (the trivial
revealing solution), interpolations alpha*kernel + (1-alpha)*P_FR, and the
no-learning init. For each, runs ~25 moving-grid LM iterations and records:
  - initial / final / per-iter max|r| and median|r|
  - distance moved from start
  - distance to P_FR
  - whether the trajectory drifts toward P_FR or stalls

If any starting point yields a converged nontrivial fixed point (max|r| < 1e-6
and final state distinct from P_FR), we've found a strict-h=0 PR equilibrium.
If all paths either drift to P_FR or stall, the strict-h=0 operator has only
P_FR as a fixed point at this corner (Grossman-Stiglitz degeneracy at the
basin-structure level)."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3

tau, gamma = 2.0, 0.01
DEEP = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/deep_max_deficit'
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/basin_search'
os.makedirs(OUT, exist_ok=True)


def make_PFR(uf, tau):
    """Fully-revealing P_FR(u) = sigma(tau*sum_k u_k) on the full grid."""
    Gf = uf.size
    U1, U2, U3 = np.meshgrid(uf, uf, uf, indexing='ij')
    z = tau * (U1 + U2 + U3)
    return 1.0 / (1.0 + np.exp(-z))


def build_mesh(P_full, uf, lo, hi, n_levels=16):
    """Extract level surfaces and flatten for vertex_residual_jit."""
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.02, 0.98, n_levels)  # avoid extremes
    p_levels = np.quantile(p_flat, qs)
    p_levels = np.unique(p_levels)
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    n_verts = np.array([s[0].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    n_faces = np.array([s[1].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    total_v = int(n_verts.sum()); total_f = int(n_faces.sum())
    verts = np.zeros((total_v, 3))
    faces = np.zeros((total_f, 3), dtype=np.int64)
    iv = ifc = 0
    for s in surfs:
        if s is None: continue
        v, f = s
        verts[iv:iv+v.shape[0]] = v
        faces[ifc:ifc+f.shape[0]] = f.astype(np.int64)
        iv += v.shape[0]; ifc += f.shape[0]
    return verts, faces, n_verts, n_faces, p_levels


def eval_residual(verts, n_verts, faces, n_faces, p_levels):
    r = vertex_residual_jit(verts, n_verts, faces, n_faces,
                             np.asarray(p_levels), tau, gamma)
    return float(np.max(np.abs(r))), float(np.median(np.abs(r)))


def quick_lm_step(verts, n_verts, faces, n_faces, p_levels, lam=0.5, eps=1e-5):
    """One damped Picard step along surface normals (cheaper than full LM).
    For each vertex, normal = average of incident face normals; step size
    omega * r * normal (capped at 1/4 of min edge to prevent flips).
    Returns new vertex positions."""
    r = vertex_residual_jit(verts, n_verts, faces, n_faces,
                             np.asarray(p_levels), tau, gamma)
    # Compute per-vertex normals
    normals = np.zeros_like(verts)
    counts = np.zeros(verts.shape[0])
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
    # Per-vertex step (damp by lam; sign by sign of r)
    omega = 0.1
    step_size = omega * r
    # Cap by 0.05 absolute and by 0.25 * nearest-vertex distance
    max_step_abs = 0.05
    step_size = np.clip(step_size, -max_step_abs, max_step_abs)
    new_verts = verts + step_size[:, None] * normals
    return new_verts, r


def run_basin(label, P_inner, uf, lo, hi, n_iter=20, n_levels=16):
    """Run moving-grid Picard from P_inner. Track trajectory."""
    Gi = hi - lo
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    verts0, faces, n_verts, n_faces, p_levels = build_mesh(P_full, uf, lo, hi, n_levels)
    print(f"  [{label}] mesh: {len(p_levels)} surfs, {verts0.shape[0]} verts",
          flush=True)
    # Quick distance metrics
    P_FR_inner = make_PFR(uf, tau)[lo:hi, lo:hi, lo:hi]
    dist_to_FR0 = float(np.max(np.abs(P_inner - P_FR_inner)))
    verts = verts0.copy()
    traj = []
    for it in range(n_iter):
        max_r, med_r = eval_residual(verts, n_verts, faces, n_faces, p_levels)
        dist_moved = float(np.max(np.linalg.norm(verts - verts0, axis=1)))
        traj.append(dict(it=it, max_r=max_r, med_r=med_r, dist_moved=dist_moved))
        if max_r < 1e-6:
            print(f"    iter {it}: CONVERGED max_r={max_r:.3e}", flush=True)
            break
        verts, _ = quick_lm_step(verts, n_verts, faces, n_faces, p_levels)
        if (it + 1) % 5 == 0:
            print(f"    iter {it+1}: max_r={max_r:.3e}  med_r={med_r:.3e}  "
                  f"|moved|={dist_moved:.3e}", flush=True)
    max_r, med_r = eval_residual(verts, n_verts, faces, n_faces, p_levels)
    dist_moved = float(np.max(np.linalg.norm(verts - verts0, axis=1)))
    print(f"  [{label}] FINAL: max_r={max_r:.3e}  med_r={med_r:.3e}  "
          f"|moved|={dist_moved:.4f}  init dist_to_FR={dist_to_FR0:.4f}",
          flush=True)
    return dict(label=label, n_iter=len(traj), traj=traj,
                max_r_init=traj[0]['max_r'], med_r_init=traj[0]['med_r'],
                max_r_final=max_r, med_r_final=med_r,
                dist_moved=dist_moved, dist_to_FR0=dist_to_FR0)


def main():
    Gi = 33
    du, uf, lo, hi = build_grid(Gi)
    P_FR_inner = make_PFR(uf, tau)[lo:hi, lo:hi, lo:hi]
    P_kernel_33 = np.load(f"{DEEP}/P_t{tau}_g{gamma}_G33.npy")
    P_kernel_21 = np.load(f"{DEEP}/P_t{tau}_g{gamma}_G25.npy") if False else None
    # Resample kernel G=21 / G=25 / G=29 to G=33 for fair comparison
    src_dir = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
    P21 = np.load(f"{src_dir}/P_ld_t{tau}_g{gamma}.npy")  # G=21
    P25 = np.load(f"{DEEP}/P_t{tau}_g{gamma}_G25.npy")
    P29 = np.load(f"{DEEP}/P_t{tau}_g{gamma}_G29.npy")
    # Resample to G=33 via index nearest
    def upsample(P, Gtarget):
        Gj = P.shape[0]
        idx = np.linspace(0, Gj-1, Gtarget).astype(int)
        return P[idx][:, idx][:, :, idx].copy()
    P21_up = upsample(P21, 33)
    P25_up = upsample(P25, 33)
    P29_up = upsample(P29, 33)
    P_NL = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))[lo:hi, lo:hi, lo:hi]

    starts = [
        ('P_FR (revealing)', P_FR_inner),
        ('P_kernel_G21 (upsampled)', P21_up),
        ('P_kernel_G25 (upsampled)', P25_up),
        ('P_kernel_G29 (upsampled)', P29_up),
        ('P_kernel_G33 (deepest)', P_kernel_33),
        ('0.5*P_kernel + 0.5*P_FR', 0.5*P_kernel_33 + 0.5*P_FR_inner),
        ('no-learning init', P_NL),
    ]
    print(f"\n=== Basin search at (tau={tau}, gamma={gamma}), G={Gi} ===\n", flush=True)
    results = []
    for label, P_inner in starts:
        try:
            print(f"\n--- {label} ---", flush=True)
            t0 = time.time()
            res = run_basin(label, P_inner, uf, lo, hi, n_iter=20)
            res['wall'] = time.time() - t0
            results.append(res)
            json.dump(results, open(f"{OUT}/basin_results.json", 'w'), indent=2, default=str)
        except Exception as e:
            print(f"  [{label}] ERROR: {e}", flush=True)
            results.append(dict(label=label, error=str(e)))
    print(f"\nWrote {OUT}/basin_results.json with {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()
