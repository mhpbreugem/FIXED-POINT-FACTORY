"""CMM Stage 4: full free-vertex moving-mesh solver.

Stage 3e tried scipy least_squares with column-by-column FD: each Jacobian
column required a full-mesh Phi eval (~0.6s), so a single LM iteration
took tens of minutes. Stage 4 fixes this:

  1. PER-SURFACE residual function (JIT). Only re-evaluates the slice
     integrals on the perturbed surface, not the whole mesh. Cost per
     surface residual = O(N_m * N_face_m).

  2. PER-SURFACE block Jacobian via local FD. Vertex perturbations
     affect only their own surface's residuals (the mathematical fact:
     surfaces don't share residuals), so the global Jacobian is exactly
     block-diagonal by surface. Build the per-surface block by FD
     perturbing each of 3*N_m coords; cost = O(N_m^2).

  3. Sparse block-diagonal global Jacobian. Solve the LM normal
     equation (J^T J + lambda diag(J^T J)) dx = -J^T r via sparse
     direct solver (scipy spsolve).

  4. TRUST-REGION LM with rho-based step accept/reject. Initial
     lambda from max diag(J^T J), classical update rule.

  5. TOPOLOGY GUARD: after each accepted step, check signed face
     volumes for inversion. Halt if any face flips.

  6. (Off by default) Laplacian mesh smoothing for tangential gauge.
     We do NOT enable it for the initial run -- LM finds A solution
     in the 2*N_v tangential gauge automatically.

Target: converge max|r| from 0.32 to <1e-6 in <50 LM iters on the
(tau=2, gamma=0.098) anchor at moderate resolution. Then compute the
deficit and compare with the deep-ladder continuum estimate 0.268 +/- 0.010.
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          EMIN15, OUT, C)
from cmm_stage3b_jit import (slice_arclen_jit, f_signal_jit, clear_crra_jit)
from reznsrc.contour_K3_halo import init_no_learning_K3


# ---------------- Per-surface residual ----------------

@njit(cache=True, parallel=True, fastmath=False)
def surface_residual(V_m, F_m, p_m, tau, gamma):
    """Residual at each vertex of ONE surface (length N_m)."""
    N_m = V_m.shape[0]
    r = np.empty(N_m)
    for i in prange(N_m):
        U0 = V_m[i, 0]; U1 = V_m[i, 1]; U2 = V_m[i, 2]
        A00, A01 = slice_arclen_jit(V_m, F_m, 0, U0, tau)
        A10, A11 = slice_arclen_jit(V_m, F_m, 1, U1, tau)
        A20, A21 = slice_arclen_jit(V_m, F_m, 2, U2, tau)
        mu = np.empty(3)
        for k in range(3):
            if k == 0: U = U0; A0_ = A00; A1_ = A01
            elif k == 1: U = U1; A0_ = A10; A1_ = A11
            else: U = U2; A0_ = A20; A1_ = A21
            f0 = f_signal_jit(U, 0, tau); f1 = f_signal_jit(U, 1, tau)
            num = f1*A1_; den = f0*A0_ + num
            if den <= 0: mu[k] = 0.5
            else:
                val = num/den
                if val < 1e-12: mu[k] = 1e-12
                elif val > 1 - 1e-12: mu[k] = 1 - 1e-12
                else: mu[k] = val
        p_clear = clear_crra_jit(mu, gamma)
        r[i] = p_clear - p_m
    return r


# ---------------- Per-surface block Jacobian via FD ----------------

@njit(cache=True, parallel=True, fastmath=False)
def surface_jacobian_fd(V_m, F_m, p_m, tau, gamma, eps):
    """Forward-FD Jacobian of surface_residual wrt V_m, shape (N_m, 3*N_m)."""
    N_m = V_m.shape[0]
    r0 = surface_residual(V_m, F_m, p_m, tau, gamma)
    J = np.zeros((N_m, 3*N_m))
    # parallelize over columns (3*N_m), each does ONE perturbed evaluation
    for col in prange(3*N_m):
        v_idx = col // 3; k_axis = col % 3
        V_pert = V_m.copy()
        V_pert[v_idx, k_axis] += eps
        # inline per-column residual
        for i in range(N_m):
            U0 = V_pert[i, 0]; U1 = V_pert[i, 1]; U2 = V_pert[i, 2]
            A00, A01 = slice_arclen_jit(V_pert, F_m, 0, U0, tau)
            A10, A11 = slice_arclen_jit(V_pert, F_m, 1, U1, tau)
            A20, A21 = slice_arclen_jit(V_pert, F_m, 2, U2, tau)
            mu = np.empty(3)
            for k in range(3):
                if k == 0: U = U0; A0_ = A00; A1_ = A01
                elif k == 1: U = U1; A0_ = A10; A1_ = A11
                else: U = U2; A0_ = A20; A1_ = A21
                f0 = f_signal_jit(U, 0, tau); f1 = f_signal_jit(U, 1, tau)
                num = f1*A1_; den = f0*A0_ + num
                if den <= 0: mu[k] = 0.5
                else:
                    val = num/den
                    if val < 1e-12: mu[k] = 1e-12
                    elif val > 1 - 1e-12: mu[k] = 1 - 1e-12
                    else: mu[k] = val
            p_clear = clear_crra_jit(mu, gamma)
            J[i, col] = (p_clear - p_m - r0[i]) / eps
    return J


# ---------------- Topology guard ----------------

@njit(cache=True)
def topology_ok(V_m, F_m, ref_normals):
    """Return True if all face normals are within 90deg of reference normals."""
    for f in range(F_m.shape[0]):
        i0, i1, i2 = F_m[f, 0], F_m[f, 1], F_m[f, 2]
        v0 = V_m[i0]; v1 = V_m[i1]; v2 = V_m[i2]
        a = v1 - v0; b = v2 - v0
        n = np.cross(a, b)
        if np.dot(n, ref_normals[f]) <= 0.0:
            return False
    return True


def face_normals(V_m, F_m):
    """Per-face raw normal (not normalized) for reference."""
    out = np.zeros((F_m.shape[0], 3))
    for f in range(F_m.shape[0]):
        i0, i1, i2 = F_m[f, 0], F_m[f, 1], F_m[f, 2]
        a = V_m[i1] - V_m[i0]; b = V_m[i2] - V_m[i0]
        out[f] = np.cross(a, b)
    return out


# ---------------- Solver ----------------

def per_vertex_min_edge(V_m, F_m):
    """For each vertex, the minimum length of any incident edge."""
    N_m = V_m.shape[0]
    min_edge = np.full(N_m, np.inf)
    for f in range(F_m.shape[0]):
        i0, i1, i2 = F_m[f, 0], F_m[f, 1], F_m[f, 2]
        for (a, b) in [(i0, i1), (i1, i2), (i2, i0)]:
            d = float(np.linalg.norm(V_m[a] - V_m[b]))
            if d < min_edge[a]: min_edge[a] = d
            if d < min_edge[b]: min_edge[b] = d
    return min_edge


def solve_cmm(P_full, uf, lo, hi, tau, gamma, p_levels,
               max_iter=50, fd_eps=1e-5, tol=1e-7, verbose=True,
               lam0=10.0):
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    M = len(p_levels)
    V = [s[0].copy() if s else np.zeros((0,3)) for s in surfs]
    F = [s[1].astype(np.int64) if s else np.zeros((0,3), np.int64) for s in surfs]
    ref_normals = [face_normals(V[m], F[m]) if V[m].size else None for m in range(M)]
    min_edges = [per_vertex_min_edge(V[m], F[m]) if V[m].size else None
                  for m in range(M)]
    n_v = np.array([v.shape[0] for v in V])
    total_v = int(n_v.sum())
    v_off = np.concatenate(([0], np.cumsum(n_v)))
    if verbose:
        print(f"Mesh: {M} surfaces, total {total_v} verts, "
              f"max per surface {int(n_v.max())}", flush=True)
    # Warm JIT
    surface_residual(V[0], F[0], p_levels[0], tau, gamma)
    surface_jacobian_fd(V[0], F[0], p_levels[0], tau, gamma, fd_eps)

    # LM trust-region damping (start with large lam => small steps; LM expands as it can)
    lam = lam0
    history = []
    need_rebuild = True
    r_blocks = None; J_blocks = None; r = None
    Fmax = Fmed = float('inf')
    for it in range(max_iter):
        if need_rebuild:
            t0 = time.time()
            r_blocks = []; J_blocks = []
            for m in range(M):
                if V[m].size == 0:
                    r_blocks.append(np.zeros(0)); J_blocks.append(None); continue
                r_m = surface_residual(V[m], F[m], p_levels[m], tau, gamma)
                J_m = surface_jacobian_fd(V[m], F[m], p_levels[m], tau, gamma, fd_eps)
                r_blocks.append(r_m); J_blocks.append(J_m)
            r = np.concatenate(r_blocks)
            t_eval = time.time() - t0
            Fmax = float(np.max(np.abs(r))); Fmed = float(np.median(np.abs(r)))
            need_rebuild = False
            if verbose:
                print(f"  iter {it}: max|r|={Fmax:.3e} med|r|={Fmed:.3e} "
                      f"lam={lam:.2e} build={t_eval:.0f}s", flush=True)
            history.append((Fmax, Fmed, lam))
            if Fmax < tol: break
        else:
            if verbose:
                print(f"  iter {it}: retry lam={lam:.2e}", flush=True)
        # Solve per-surface LM normal equations (Levenberg uniform damping)
        dx = np.zeros(3*total_v)
        for m in range(M):
            if J_blocks[m] is None: continue
            Jm = J_blocks[m]; rm = r_blocks[m]
            JtJ = Jm.T @ Jm
            Jtr = Jm.T @ rm
            # Uniform Levenberg damping: lam * I_(3*N_m). Caps step in
            # tangential gauge directions where diag(JtJ) is ~0.
            n_col = JtJ.shape[0]
            A = JtJ + lam * np.eye(n_col)
            try:
                step = -np.linalg.solve(A, Jtr)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A, Jtr, rcond=None)[0]
            # Per-vertex local cap: each vertex's |step| <= min_edge/4
            step3 = step.reshape(-1, 3)
            v_step_norm = np.linalg.norm(step3, axis=1)  # (N_m,)
            limit = 0.25 * min_edges[m]
            scale = np.minimum(1.0, limit / np.maximum(v_step_norm, 1e-30))
            step3 = step3 * scale[:, None]
            # Global cap for extra safety
            sn = float(np.max(np.linalg.norm(step3, axis=1)))
            if sn > 0.02:
                step3 = step3 * (0.02 / sn)
            dx[3*v_off[m]:3*v_off[m+1]] = step3.ravel()
        # try step
        V_new = [v.copy() for v in V]
        for m in range(M):
            if n_v[m] == 0: continue
            V_new[m] = V[m] + dx[3*v_off[m]:3*v_off[m+1]].reshape(-1, 3)
        # topology check
        topo_ok = all(V_new[m].size == 0 or topology_ok(V_new[m], F[m], ref_normals[m])
                       for m in range(M))
        # evaluate new residual
        if topo_ok:
            r_new_blocks = [surface_residual(V_new[m], F[m], p_levels[m], tau, gamma)
                             if V_new[m].size else np.zeros(0) for m in range(M)]
            r_new = np.concatenate(r_new_blocks)
            cost_old = 0.5*np.dot(r, r); cost_new = 0.5*np.dot(r_new, r_new)
            predicted = 0.0
            for m in range(M):
                if J_blocks[m] is None: continue
                step = dx[3*v_off[m]:3*v_off[m+1]]
                predicted += 0.5*np.dot(r_blocks[m], r_blocks[m]) \
                              - 0.5*np.dot(r_blocks[m] + J_blocks[m]@step,
                                            r_blocks[m] + J_blocks[m]@step)
            actual = cost_old - cost_new
            rho = actual / max(predicted, 1e-30)
            accept = rho > 0.0 and cost_new < cost_old
        else:
            accept = False; rho = -1.0; r_new = r
        if accept:
            V = V_new
            need_rebuild = True
            if rho > 0.75: lam *= 0.3
            elif rho < 0.25: lam *= 2.0
        else:
            lam *= 4.0
            # need_rebuild stays False -- reuse the Jacobian at the same V
        if verbose:
            tag = 'ok' if topo_ok else 'TOPO_FLIP'
            print(f"    step rho={rho:+.2f} "
                  f"accepted={accept}  {tag}",
                  flush=True)
        if lam > 1e12:
            print("    lambda too large, stagnation", flush=True); break
    return dict(V=V, F=F, p_levels=p_levels, history=history,
                final_max=Fmax, final_median=Fmed,
                converged=(Fmax < tol))


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    p_flat = P_inner.ravel()
    qs = np.linspace(0, 1, 17); edges = np.quantile(p_flat, qs)
    p_levels = 0.5*(edges[:-1] + edges[1:])

    t_start = time.time()
    out = solve_cmm(P_full, uf, lo, hi, tau, gamma, p_levels,
                     max_iter=30, fd_eps=1e-5, tol=1e-7, verbose=True,
                     lam0=0.1)
    wall = time.time() - t_start
    print(f"\nTotal {wall/60:.1f} min", flush=True)
    print(f"Converged: {out['converged']}  final max|r|={out['final_max']:.3e}",
          flush=True)
    # Save state
    for m, (Vm, Fm) in enumerate(zip(out['V'], out['F'])):
        if Vm.size:
            np.savez(f"{OUT}/stage4_surf_m{m:02d}.npz", V=Vm, F=Fm,
                     p=out['p_levels'][m])
    json.dump(dict(history=out['history'],
                    final_max=float(out['final_max']),
                    final_median=float(out['final_median']),
                    converged=bool(out['converged']),
                    wall_min=float(wall/60),
                    tau=tau, gamma=gamma, n_surfaces=len(p_levels),
                    n_verts_total=int(sum(v.shape[0] for v in out['V']))),
              open(f"{OUT}/stage4_lm.json", "w"), indent=2)
    print(f"\nsaved stage4_lm.json and per-surface npz files", flush=True)


if __name__ == "__main__":
    main()
