"""CMM Stage 3e: vertices move FREELY in 3D under Levenberg-Marquardt.

State: x = all_verts.ravel()  -- 3*total_v free coords.
Residual: N_v scalars (clear_CRRA at each vertex - p_m).
Solver: scipy.optimize.least_squares with method='lm' or 'trf'.

The 2-parameter tangential gauge (vertex sliding along its surface)
is absorbed by LM's trust region: any zero-residual config is fine.

For 22734-vertex problem, jac='2-point' would do 22734*3=68k FD calls
per Newton step (too slow even at 0.6s/eval). Use 'jacobian-free' aka
matrix-free Krylov via scipy.optimize._lsq.trf with sparse Jacobian.

Practical compromise here: use trf with JAC sparsity hint (only nearby
vertices on the SAME surface affect each residual through the slice
integrals -- so the Jacobian is block-sparse per surface). With this
sparsity, trf computes the Jacobian column-by-column but skips zero
columns.

For first attempt: cap the per-surface vertex count by downsampling
the mesh, just to show CONVERGENCE before scaling up."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          EMIN15, OUT, C)
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    # Use fewer price levels for tractability
    p_flat = P_inner.ravel()
    qs = np.linspace(0, 1, 17); edges = np.quantile(p_flat, qs)
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
    print(f"Mesh: {len(p_levels)} surfaces, {total_v} verts, {total_f} faces "
          f"(state dim = 3 * {total_v} = {3*total_v})", flush=True)

    # Build Jacobian sparsity: each surface block (n_m verts, n_m equations,
    # 3*n_m unknowns) is dense within; OFF-block entries are zero (different
    # surfaces don't share residuals).
    v_off = np.concatenate(([0], np.cumsum(n_verts_arr)))
    print("Building Jacobian sparsity pattern...", flush=True)
    jac_sparsity = lil_matrix((total_v, 3*total_v), dtype=bool)
    for m in range(len(p_levels)):
        if n_verts_arr[m] == 0: continue
        a, b = v_off[m], v_off[m+1]
        jac_sparsity[a:b, 3*a:3*b] = True
    jac_sparsity = jac_sparsity.tocsr()
    print(f"  done (nnz = {jac_sparsity.nnz}, "
          f"density = {jac_sparsity.nnz/(total_v*3*total_v)*100:.2f}%)", flush=True)

    def F(x):
        v = x.reshape(total_v, 3)
        return vertex_residual_jit(v, n_verts_arr, all_faces,
                                     n_faces_arr, p_levels, tau, gamma)

    x0 = all_verts0.ravel()
    r0 = F(x0)
    print(f"Initial: max|r|={np.max(np.abs(r0)):.3e}  "
          f"med|r|={np.median(np.abs(r0)):.3e}", flush=True)

    print("\n=== Levenberg-Marquardt (trf, sparse Jacobian) ===", flush=True)
    t0 = time.time()
    # Aggressive bounds: each vertex coord stays within [-UMAX, UMAX]
    lb = np.full(3*total_v, -4.0); ub = np.full(3*total_v, 4.0)
    res = least_squares(F, x0, method='trf', jac='2-point',
                          jac_sparsity=jac_sparsity, bounds=(lb, ub),
                          xtol=1e-8, ftol=1e-8, max_nfev=20, verbose=2)
    wall = time.time() - t0
    print(f"\nLM done ({wall:.0f}s)", flush=True)
    print(f"  nfev = {res.nfev}, njev = {res.njev}", flush=True)
    print(f"  status = {res.status}, message = {res.message}", flush=True)
    r_final = res.fun
    print(f"  Final max|r|={np.max(np.abs(r_final)):.3e}  "
          f"med|r|={np.median(np.abs(r_final)):.3e}  "
          f"||r||={np.linalg.norm(r_final):.3e}", flush=True)
    np.save(f"{OUT}/stage3e_verts.npy", res.x.reshape(total_v, 3))
    json.dump(dict(initial_max=float(np.max(np.abs(r0))),
                    initial_med=float(np.median(np.abs(r0))),
                    final_max=float(np.max(np.abs(r_final))),
                    final_med=float(np.median(np.abs(r_final))),
                    nfev=int(res.nfev), wall=float(wall),
                    n_surfaces=len(p_levels), total_v=int(total_v)),
              open(f"{OUT}/stage3e_lm.json", "w"), indent=2)
    print(f"\nsaved stage3e_verts.npy, stage3e_lm.json", flush=True)


if __name__ == "__main__":
    main()
