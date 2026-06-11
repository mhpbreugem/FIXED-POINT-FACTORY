"""CMM Stage 3c: Newton-Krylov on the moving mesh.

State x: all vertex coordinates of all surfaces, flattened (3*total_v floats).
Residual r(x): per-vertex residual r_v = clear_CRRA(mu(v); gamma) - p_m (Stage 3b).
Solve r(x) = 0 via Newton-Krylov.

Constraint: vertices of S_m must stay on {P=p_m} — but we DEFINE P by the
moving mesh itself, so this is automatic once r = 0. No explicit projection.
What can go wrong: vertices may inflate / collapse / triangle-flip, making
the mesh degenerate. We monitor mesh quality (min-edge / max-edge ratios)
and stop if degeneracy appears.

The first Newton step's reduction tells us whether the moving-mesh route
converges. If it does: full Newton to machine eps, then arclength integrals
give the strict h=0 fixed point at (tau, gamma) = (2, 0.098). If the residual
contracts substantially in 1-2 steps, the strict h=0 deficit can be computed
and compared with the immortal anchor.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          EMIN15, OUT, C)
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass


class MeshState:
    def __init__(self, all_verts0, n_verts_arr, faces_concat, n_faces_arr,
                 p_levels, tau, gamma):
        self.all_verts0 = all_verts0.copy()  # initial positions (45690, 3)
        self.n_verts_arr = n_verts_arr
        self.faces_concat = faces_concat
        self.n_faces_arr = n_faces_arr
        self.p_levels = p_levels
        self.tau = tau; self.gamma = gamma
        self.shape = all_verts0.shape

    def residual_flat(self, x):
        v = x.reshape(self.shape)
        return vertex_residual_jit(v, self.n_verts_arr, self.faces_concat,
                                     self.n_faces_arr, self.p_levels,
                                     self.tau, self.gamma)


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
    all_verts = np.zeros((total_v, 3))
    all_faces = np.zeros((total_f, 3), dtype=np.int64)
    iv = 0; ifc = 0
    for s in surfs:
        if s is None: continue
        v, f = s
        all_verts[iv:iv+v.shape[0]] = v
        all_faces[ifc:ifc+f.shape[0]] = f.astype(np.int64)
        iv += v.shape[0]; ifc += f.shape[0]
    print(f"Mesh: {len(p_levels)} surfaces, {total_v} verts, {total_f} faces",
          flush=True)

    state = MeshState(all_verts, n_verts_arr, all_faces, n_faces_arr,
                       p_levels, tau, gamma)

    # Strategy: Newton-Krylov on (total_v,) per-surface residuals would
    # be naturally 3*total_v free variables vs total_v equations -- under-
    # determined. The natural choice is to constrain each vertex to move
    # PERPENDICULARLY to the surface normal, parametrized by a scalar
    # displacement per vertex. State dimension: total_v scalars. The
    # vertex update is: v_new = v + delta * normal.
    # Compute surface normals from initial mesh.
    print("Computing initial mesh normals (one-time, per-vertex average "
          "of incident face normals)...", flush=True)
    t0 = time.time()
    normals = np.zeros_like(all_verts)
    # Per surface, accumulate face normals to vertices
    v_off = np.concatenate(([0], np.cumsum(n_verts_arr)))
    f_off = np.concatenate(([0], np.cumsum(n_faces_arr)))
    for m in range(len(p_levels)):
        if n_verts_arr[m] == 0: continue
        V = all_verts[v_off[m]:v_off[m+1]]
        F = all_faces[f_off[m]:f_off[m+1]]   # already local indices
        for tri in F:
            v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln == 0: continue
            n = n / ln
            normals[v_off[m] + tri[0]] += n
            normals[v_off[m] + tri[1]] += n
            normals[v_off[m] + tri[2]] += n
    # normalize
    nm = np.linalg.norm(normals, axis=1, keepdims=True)
    nm = np.where(nm > 0, nm, 1.0)
    normals = normals / nm
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    def F_of_delta(delta):
        """Residual as a function of per-vertex normal displacements."""
        v_new = all_verts + delta[:, None] * normals
        return state.residual_flat(v_new.ravel())

    r0 = F_of_delta(np.zeros(total_v))
    print(f"Initial residual: max {np.max(np.abs(r0)):.3e}  "
          f"median {np.median(np.abs(r0)):.3e}", flush=True)

    print("\n=== Newton-Krylov ===", flush=True)
    history = [float(np.max(np.abs(r0)))]
    delta = np.zeros(total_v)
    for it in range(8):
        t0 = time.time()
        try:
            delta = newton_krylov(F_of_delta, delta, f_tol=1e-3 * history[-1],
                                    maxiter=5, verbose=False, method='gmres')
        except NoConvergence as e:
            delta = np.asarray(e.args[0], dtype=float)
        wall = time.time() - t0
        r = F_of_delta(delta)
        Fmax = float(np.max(np.abs(r)))
        history.append(Fmax)
        # mesh degeneracy probe
        v_new = all_verts + delta[:, None] * normals
        # measure edge-length distribution per surface
        edge_min = np.inf; edge_max = 0
        for m in range(len(p_levels)):
            if n_verts_arr[m] == 0: continue
            V = v_new[v_off[m]:v_off[m+1]]
            F = all_faces[f_off[m]:f_off[m+1]]
            for tri in F[::max(1, len(F)//200)]:  # sample
                for (a,b) in ((0,1),(1,2),(2,0)):
                    L = np.linalg.norm(V[tri[a]] - V[tri[b]])
                    if L < edge_min: edge_min = L
                    if L > edge_max: edge_max = L
        ratio = edge_max / max(edge_min, 1e-12)
        print(f"  iter {it}: max|r|={Fmax:.3e}  edge ratio max/min={ratio:.1f}  "
              f"|delta|_max={np.max(np.abs(delta)):.3e}  ({wall:.0f}s)",
              flush=True)
        if history[-1] / history[-2] > 0.95 and it >= 2:
            print("  Stagnated -- stopping.", flush=True); break
        if ratio > 1000:
            print("  Mesh degenerate -- stopping.", flush=True); break

    np.save(f"{OUT}/stage3c_delta.npy", delta)
    v_final = all_verts + delta[:, None] * normals
    np.save(f"{OUT}/stage3c_verts.npy", v_final)
    json.dump(dict(F_history=history, n_iter=len(history)-1,
                    final_max_r=float(history[-1]),
                    reduction=float(history[0]/history[-1])),
              open(f"{OUT}/stage3c_newton.json", "w"), indent=2)
    print(f"\nResidual history: {[f'{x:.2e}' for x in history]}", flush=True)
    print(f"Reduction: {history[0]/history[-1]:.1f}x", flush=True)
    print(f"\nsaved stage3c_delta.npy, stage3c_verts.npy, stage3c_newton.json",
          flush=True)


if __name__ == "__main__":
    main()
