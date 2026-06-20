"""CMM Stage 3d: damped Picard on moving mesh along surface normals.

Newton-Krylov in Stage 3c found vanishingly small steps because the
per-vertex finite-difference matvec is noisy on a strict-h=0 residual.
Try a simpler dynamic: each vertex moves along its initial-surface
normal by amount proportional to its local residual. This is Picard
iteration in normal space:

  delta_{n+1} = delta_n - omega * r(delta_n)

with omega ~ 0.05 (small to avoid mesh flip). The fixed-topology
moving mesh means the residual is smooth in delta -- the cube version
of this diverged ONLY because re-triangulation was discontinuous.

Reports per-iter max|r|, median|r|, mesh quality, max displacement.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          EMIN15, OUT, C)
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    p_flat = P_inner.ravel()
    qs = np.linspace(0, 1, 33); edges = np.quantile(p_flat, qs)
    p_levels = 0.5*(edges[:-1] + edges[1:])
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    n_verts_arr = np.array([s[0].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    n_faces_arr = np.array([s[1].shape[0] if s else 0 for s in surfs], dtype=np.int64)
    total_v = int(n_verts_arr.sum()); total_f = int(n_faces_arr.sum())
    all_verts0 = np.zeros((total_v, 3))
    all_faces = np.zeros((total_f, 3), dtype=np.int64)
    iv = 0; ifc = 0
    for s in surfs:
        if s is None: continue
        v, f = s
        all_verts0[iv:iv+v.shape[0]] = v
        all_faces[ifc:ifc+f.shape[0]] = f.astype(np.int64)
        iv += v.shape[0]; ifc += f.shape[0]
    print(f"Mesh: {len(p_levels)} surfaces, {total_v} verts, {total_f} faces",
          flush=True)

    # Per-vertex normals from initial mesh
    v_off = np.concatenate(([0], np.cumsum(n_verts_arr)))
    f_off = np.concatenate(([0], np.cumsum(n_faces_arr)))
    normals = np.zeros_like(all_verts0)
    for m in range(len(p_levels)):
        if n_verts_arr[m] == 0: continue
        V = all_verts0[v_off[m]:v_off[m+1]]
        F = all_faces[f_off[m]:f_off[m+1]]
        for tri in F:
            v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
            nor = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(nor)
            if ln == 0: continue
            nor = nor / ln
            normals[v_off[m] + tri[0]] += nor
            normals[v_off[m] + tri[1]] += nor
            normals[v_off[m] + tri[2]] += nor
    nm = np.linalg.norm(normals, axis=1, keepdims=True)
    nm = np.where(nm > 0, nm, 1.0)
    normals = normals / nm

    # Picard iteration
    def F(delta):
        v_new = all_verts0 + delta[:, None] * normals
        return vertex_residual_jit(v_new, n_verts_arr, all_faces,
                                     n_faces_arr, p_levels, tau, gamma)
    delta = np.zeros(total_v)
    r = F(delta)
    print(f"Initial: max|r|={np.max(np.abs(r)):.3e}  med|r|={np.median(np.abs(r)):.3e}",
          flush=True)
    history = [(0.0, float(np.max(np.abs(r))), float(np.median(np.abs(r))))]
    omega = 0.05
    for it in range(30):
        t0 = time.time()
        delta_new = delta - omega * r
        r_new = F(delta_new)
        Fmax = float(np.max(np.abs(r_new)))
        Fmed = float(np.median(np.abs(r_new)))
        wall = time.time() - t0
        if Fmax > 1.5 * np.max(np.abs(r)):
            # diverging: halve omega, revert
            omega *= 0.5
            print(f"  iter {it}: divergence (|r| jumped) -- omega -> {omega:.3f}",
                  flush=True)
            continue
        delta = delta_new; r = r_new
        history.append((float(np.max(np.abs(delta))), Fmax, Fmed))
        print(f"  iter {it}: max|r|={Fmax:.3e}  med|r|={Fmed:.3e}  "
              f"|delta|max={np.max(np.abs(delta)):.3e}  omega={omega:.3f}  ({wall:.1f}s)",
              flush=True)
        if Fmax < 1e-8: break
    np.save(f"{OUT}/stage3d_delta.npy", delta)
    json.dump(dict(history=history, n_iter=len(history)-1,
                    final_max=float(np.max(np.abs(r))),
                    final_median=float(np.median(np.abs(r))),
                    converged=bool(np.max(np.abs(r)) < 1e-8)),
              open(f"{OUT}/stage3d_picard.json", "w"), indent=2)
    print(f"\nFinal: max|r|={np.max(np.abs(r)):.3e}, "
          f"reduction {history[0][1]/np.max(np.abs(r)):.1f}x", flush=True)


if __name__ == "__main__":
    main()
