"""CMM Stage 5: SPARSE finite-difference Jacobian for the free-vertex
moving-mesh solver.

Stage 4's bottleneck: surface_jacobian_fd builds a DENSE (N_m, 3*N_m)
block by perturbing each of the 3*N_m coords and recomputing the FULL
per-surface residual -> O(N_m^2 * N_face) per surface, ~500 s for the
full 16-surface / 11.6k-vertex Jacobian.

Stage 5 exploits locality. When vertex w moves, only the ~6 faces
incident to w change shape. Residual r_i depends on the faces crossed
by vertex i's three slice planes {u_k = v_i[k]}. Hence

    dr_i/dv_w != 0  only if  (i == w)  or
        faces(incident to w)  intersects  faces(crossed by i's slices).

Algorithm:
  1. vertex->faces adjacency (CSR, static topology).
  2. face->rows: for each face f, the rows i whose slice plane (any of
     3 axes) can cross f (straddle test with a safety margin >> eps).
     Union over a vertex's incident faces (+ the vertex's own row)
     gives the column sparsity pattern, stored CSR per vertex.
  3. For column (w, axis): perturb v_w[axis] by eps. For each affected
     row i, rebuild its three slice integrals CHEAPLY:
       - if i == w and k == axis: the plane itself moved -> full slice
         recompute on the perturbed mesh (one row per column, O(N_face));
       - else: A_pert = A_base + sum over faces incident to w of
         (contrib_pert - contrib_base): O(#incident faces) ~ 6.
     Entry = (r_pert(i) - r0(i)) / eps.
  4. numba njit + prange over columns; values written into per-column
     segments of preallocated CSC arrays -> scipy.sparse.csc_matrix.
  5. LM step per surface: (J^T J + lam I) dx = -J^T r with sparse J
     (scipy spsolve). SAME safeguards as stage 4: per-vertex step cap
     0.25*min incident edge, global cap 0.02, topology guard against
     face-normal flips, Levenberg uniform damping, rho-based accept
     with Jacobian reuse on rejection.
  6. Validation: on one small surface, compare against stage 4's dense
     surface_jacobian_fd; require max abs difference < 1e-8.

Outputs: stage5_lm.json, stage5_surf_m*.npz, RESULTS_stage5.md under
the cmm_endogenous results dir (never touches stage4_* files).
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
from cmm_stage4_blockjac import (surface_residual, surface_jacobian_fd,
                                  topology_ok, face_normals,
                                  per_vertex_min_edge)
from reznsrc.contour_K3_halo import init_no_learning_K3


# ---------------- per-face slice contribution (mirror of slice_arclen_jit body) ----------------

@njit(cache=True, fastmath=False)
def face_contrib(verts, faces, f, axis, U, tau):
    """Contribution of ONE face to the (A0, A1) slice-arclength integrals
    for the plane {u_axis = U}. Exact replica of the per-face body of
    slice_arclen_jit (so base + delta reproduces the full sum)."""
    other0 = 1 if axis == 0 else 0
    other1 = 2 if axis < 2 else 1
    i0 = faces[f, 0]; i1 = faces[f, 1]; i2 = faces[f, 2]
    z0 = verts[i0, axis] - U
    z1 = verts[i1, axis] - U
    z2 = verts[i2, axis] - U
    s0 = 0 if z0 == 0 else (1 if z0 > 0 else -1)
    s1 = 0 if z1 == 0 else (1 if z1 > 0 else -1)
    s2 = 0 if z2 == 0 else (1 if z2 > 0 else -1)
    if s0 == s1 == s2 and s0 != 0:
        return 0.0, 0.0
    pts_a = np.empty(2)
    pts_b = np.empty(2)
    np_ = 0
    # edge (i0, i1)
    if z0 == 0 and np_ < 2:
        pts_a[np_] = verts[i0, other0]; pts_b[np_] = verts[i0, other1]; np_ += 1
    elif z0*z1 < 0 and np_ < 2:
        t = z0/(z0 - z1)
        pts_a[np_] = verts[i0, other0] + t*(verts[i1, other0] - verts[i0, other0])
        pts_b[np_] = verts[i0, other1] + t*(verts[i1, other1] - verts[i0, other1])
        np_ += 1
    # edge (i1, i2)
    if z1 == 0 and np_ < 2:
        dup = False
        if np_ > 0:
            if abs(pts_a[np_-1] - verts[i1, other0]) < 1e-12 and abs(pts_b[np_-1] - verts[i1, other1]) < 1e-12:
                dup = True
        if not dup:
            pts_a[np_] = verts[i1, other0]; pts_b[np_] = verts[i1, other1]; np_ += 1
    elif z1*z2 < 0 and np_ < 2:
        t = z1/(z1 - z2)
        pts_a[np_] = verts[i1, other0] + t*(verts[i2, other0] - verts[i1, other0])
        pts_b[np_] = verts[i1, other1] + t*(verts[i2, other1] - verts[i1, other1])
        np_ += 1
    # edge (i2, i0)
    if z2 == 0 and np_ < 2:
        dup = False
        if np_ > 0:
            if abs(pts_a[np_-1] - verts[i2, other0]) < 1e-12 and abs(pts_b[np_-1] - verts[i2, other1]) < 1e-12:
                dup = True
        if not dup:
            pts_a[np_] = verts[i2, other0]; pts_b[np_] = verts[i2, other1]; np_ += 1
    elif z2*z0 < 0 and np_ < 2:
        t = z2/(z2 - z0)
        pts_a[np_] = verts[i2, other0] + t*(verts[i0, other0] - verts[i2, other0])
        pts_b[np_] = verts[i2, other1] + t*(verts[i0, other1] - verts[i2, other1])
        np_ += 1
    if np_ < 2:
        return 0.0, 0.0
    da = pts_a[1] - pts_a[0]; db = pts_b[1] - pts_b[0]
    L = np.sqrt(da*da + db*db)
    if L == 0:
        return 0.0, 0.0
    sqrt_t = np.sqrt(tau/(2*np.pi))
    a0 = 0.0; a1 = 0.0
    for s in (0.5 - 0.5/np.sqrt(3.0), 0.5 + 0.5/np.sqrt(3.0)):
        a_s = pts_a[0] + s*da
        b_s = pts_b[0] + s*db
        d0a = a_s + 0.5; d0b = b_s + 0.5
        d1a = a_s - 0.5; d1b = b_s - 0.5
        f0 = sqrt_t*np.exp(-0.5*tau*d0a*d0a) * sqrt_t*np.exp(-0.5*tau*d0b*d0b)
        f1 = sqrt_t*np.exp(-0.5*tau*d1a*d1a) * sqrt_t*np.exp(-0.5*tau*d1b*d1b)
        a0 += 0.5 * f0 * L
        a1 += 0.5 * f1 * L
    return a0, a1


@njit(cache=True, fastmath=False)
def residual_from_A(U0, U1, U2, A0r, A1r, tau, gamma, p_m):
    """Residual given slice integrals A0r[k], A1r[k] (same algebra as
    surface_residual in stage 4)."""
    mu = np.empty(3)
    for k in range(3):
        if k == 0: U = U0
        elif k == 1: U = U1
        else: U = U2
        f0 = f_signal_jit(U, 0, tau); f1 = f_signal_jit(U, 1, tau)
        num = f1*A1r[k]; den = f0*A0r[k] + num
        if den <= 0:
            mu[k] = 0.5
        else:
            val = num/den
            if val < 1e-12: mu[k] = 1e-12
            elif val > 1 - 1e-12: mu[k] = 1 - 1e-12
            else: mu[k] = val
    return clear_crra_jit(mu, gamma) - p_m


@njit(cache=True, parallel=True, fastmath=False)
def surface_residual_and_base(V_m, F_m, p_m, tau, gamma):
    """Residual r0 plus the base slice integrals A0b, A1b (N, 3)."""
    N_m = V_m.shape[0]
    r0 = np.empty(N_m)
    A0b = np.empty((N_m, 3))
    A1b = np.empty((N_m, 3))
    for i in prange(N_m):
        for k in range(3):
            a0, a1 = slice_arclen_jit(V_m, F_m, k, V_m[i, k], tau)
            A0b[i, k] = a0; A1b[i, k] = a1
        r0[i] = residual_from_A(V_m[i, 0], V_m[i, 1], V_m[i, 2],
                                 A0b[i], A1b[i], tau, gamma, p_m)
    return r0, A0b, A1b


# ---------------- sparsity pattern ----------------

@njit(cache=True)
def build_vertex_faces(F_m, N_m):
    """CSR vertex -> incident faces."""
    Nf = F_m.shape[0]
    counts = np.zeros(N_m, np.int64)
    for f in range(Nf):
        for j in range(3):
            counts[F_m[f, j]] += 1
    indptr = np.zeros(N_m + 1, np.int64)
    for v in range(N_m):
        indptr[v+1] = indptr[v] + counts[v]
    idx = np.empty(indptr[N_m], np.int64)
    fill = indptr[:N_m].copy()
    for f in range(Nf):
        for j in range(3):
            v = F_m[f, j]
            idx[fill[v]] = f
            fill[v] += 1
    return indptr, idx


@njit(cache=True, parallel=True)
def _face_rows_count(V_m, F_m, margin):
    """counts[f] = #rows i whose slice plane (some axis) can cross face f."""
    Nf = F_m.shape[0]; N_m = V_m.shape[0]
    counts = np.zeros(Nf, np.int64)
    for f in prange(Nf):
        mn = np.empty(3); mx = np.empty(3)
        for k in range(3):
            a = V_m[F_m[f, 0], k]; b = V_m[F_m[f, 1], k]; c = V_m[F_m[f, 2], k]
            lo = a; hi = a
            if b < lo: lo = b
            if b > hi: hi = b
            if c < lo: lo = c
            if c > hi: hi = c
            mn[k] = lo - margin; mx[k] = hi + margin
        cnt = 0
        for i in range(N_m):
            for k in range(3):
                if V_m[i, k] >= mn[k] and V_m[i, k] <= mx[k]:
                    cnt += 1
                    break
        counts[f] = cnt
    return counts


@njit(cache=True, parallel=True)
def _face_rows_fill(V_m, F_m, margin, indptr, rows):
    Nf = F_m.shape[0]; N_m = V_m.shape[0]
    for f in prange(Nf):
        mn = np.empty(3); mx = np.empty(3)
        for k in range(3):
            a = V_m[F_m[f, 0], k]; b = V_m[F_m[f, 1], k]; c = V_m[F_m[f, 2], k]
            lo = a; hi = a
            if b < lo: lo = b
            if b > hi: hi = b
            if c < lo: lo = c
            if c > hi: hi = c
            mn[k] = lo - margin; mx[k] = hi + margin
        pos = indptr[f]
        for i in range(N_m):
            for k in range(3):
                if V_m[i, k] >= mn[k] and V_m[i, k] <= mx[k]:
                    rows[pos] = i
                    pos += 1
                    break


def build_face_rows(V_m, F_m, margin):
    counts = _face_rows_count(V_m, F_m, margin)
    indptr = np.zeros(F_m.shape[0] + 1, np.int64)
    indptr[1:] = np.cumsum(counts)
    rows = np.empty(indptr[-1], np.int64)
    _face_rows_fill(V_m, F_m, margin, indptr, rows)
    return indptr, rows


@njit(cache=True, parallel=True)
def _vertex_rows_count(N_m, vf_indptr, vf_idx, fr_indptr, fr_rows):
    counts = np.zeros(N_m, np.int64)
    for w in prange(N_m):
        mark = np.zeros(N_m, np.uint8)
        mark[w] = 1
        c = 1
        for t in range(vf_indptr[w], vf_indptr[w+1]):
            f = vf_idx[t]
            for s in range(fr_indptr[f], fr_indptr[f+1]):
                i = fr_rows[s]
                if mark[i] == 0:
                    mark[i] = 1
                    c += 1
        counts[w] = c
    return counts


@njit(cache=True, parallel=True)
def _vertex_rows_fill(N_m, vf_indptr, vf_idx, fr_indptr, fr_rows, indptr, out_rows):
    for w in prange(N_m):
        mark = np.zeros(N_m, np.uint8)
        mark[w] = 1
        for t in range(vf_indptr[w], vf_indptr[w+1]):
            f = vf_idx[t]
            for s in range(fr_indptr[f], fr_indptr[f+1]):
                mark[fr_rows[s]] = 1
        pos = indptr[w]
        for i in range(N_m):
            if mark[i] == 1:
                out_rows[pos] = i
                pos += 1


def build_vertex_rows(N_m, vf_indptr, vf_idx, fr_indptr, fr_rows):
    counts = _vertex_rows_count(N_m, vf_indptr, vf_idx, fr_indptr, fr_rows)
    indptr = np.zeros(N_m + 1, np.int64)
    indptr[1:] = np.cumsum(counts)
    rows = np.empty(indptr[-1], np.int64)
    _vertex_rows_fill(N_m, vf_indptr, vf_idx, fr_indptr, fr_rows, indptr, rows)
    return indptr, rows


# ---------------- sparse FD Jacobian fill ----------------

@njit(cache=True, parallel=True, fastmath=False)
def fill_sparse_cols(V_m, F_m, p_m, tau, gamma, eps,
                     vf_indptr, vf_idx, vr_indptr, vr_rows,
                     col_indptr, r0, A0b, A1b):
    """Per-column FD restricted to affected rows; returns CSC (rows, vals)."""
    N_m = V_m.shape[0]
    ncol = 3 * N_m
    nnz = col_indptr[ncol]
    rows_out = np.empty(nnz, np.int64)
    vals = np.empty(nnz)
    for col in prange(ncol):
        w = col // 3
        a = col - 3*w
        Vp = V_m.copy()
        Vp[w, a] += eps
        base = col_indptr[col]
        A0p = np.empty(3); A1p = np.empty(3)
        nr = vr_indptr[w+1] - vr_indptr[w]
        for jj in range(nr):
            i = vr_rows[vr_indptr[w] + jj]
            for k in range(3):
                U = Vp[i, k]
                if i == w and k == a:
                    # the slice plane itself moved: full recompute
                    a0, a1 = slice_arclen_jit(Vp, F_m, k, U, tau)
                    A0p[k] = a0; A1p[k] = a1
                else:
                    d0 = 0.0; d1 = 0.0
                    for t in range(vf_indptr[w], vf_indptr[w+1]):
                        ff = vf_idx[t]
                        c0b, c1b = face_contrib(V_m, F_m, ff, k, U, tau)
                        c0p, c1p = face_contrib(Vp, F_m, ff, k, U, tau)
                        d0 += c0p - c0b
                        d1 += c1p - c1b
                    A0p[k] = A0b[i, k] + d0
                    A1p[k] = A1b[i, k] + d1
            rp = residual_from_A(Vp[i, 0], Vp[i, 1], Vp[i, 2],
                                  A0p, A1p, tau, gamma, p_m)
            rows_out[base + jj] = i
            vals[base + jj] = (rp - r0[i]) / eps
    return rows_out, vals


def build_surface_sparse_jacobian(V_m, F_m, p_m, tau, gamma, eps,
                                   vf_indptr, vf_idx, margin):
    """Returns (r0, J_sparse_csc) for one surface."""
    N_m = V_m.shape[0]
    r0, A0b, A1b = surface_residual_and_base(V_m, F_m, p_m, tau, gamma)
    fr_indptr, fr_rows = build_face_rows(V_m, F_m, margin)
    vr_indptr, vr_rows = build_vertex_rows(N_m, vf_indptr, vf_idx,
                                            fr_indptr, fr_rows)
    counts_v = np.diff(vr_indptr)
    col_counts = np.repeat(counts_v, 3)
    col_indptr = np.zeros(3*N_m + 1, np.int64)
    col_indptr[1:] = np.cumsum(col_counts)
    rows_out, vals = fill_sparse_cols(V_m, F_m, p_m, tau, gamma, eps,
                                       vf_indptr, vf_idx, vr_indptr, vr_rows,
                                       col_indptr, r0, A0b, A1b)
    J = sp.csc_matrix((vals, rows_out, col_indptr), shape=(N_m, 3*N_m))
    return r0, J


# ---------------- solver ----------------

def solve_cmm_sparse(P_full, uf, tau, gamma, p_levels,
                      max_iter=60, fd_eps=1e-5, tol=1e-7, lam0=0.1,
                      wall_budget_s=90*60, stall_limit=8, verbose=True):
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    M = len(p_levels)
    V = [s[0].copy() if s else np.zeros((0, 3)) for s in surfs]
    F = [s[1].astype(np.int64) if s else np.zeros((0, 3), np.int64) for s in surfs]
    ref_normals = [face_normals(V[m], F[m]) if V[m].size else None for m in range(M)]
    min_edges = [per_vertex_min_edge(V[m], F[m]) if V[m].size else None
                  for m in range(M)]
    vf_adj = [build_vertex_faces(F[m], V[m].shape[0]) if V[m].size else None
               for m in range(M)]
    n_v = np.array([v.shape[0] for v in V])
    total_v = int(n_v.sum())
    v_off = np.concatenate(([0], np.cumsum(n_v)))
    margin = max(1e-4, 10*fd_eps)
    if verbose:
        print(f"Mesh: {M} surfaces, total {total_v} verts, "
              f"max per surface {int(n_v.max())}", flush=True)

    lam = lam0
    history = []
    need_rebuild = True
    r_blocks = None; J_blocks = None; r = None
    Fmax = Fmed = float('inf')
    best_cost = float('inf')
    stall = 0
    stop_reason = 'max_iter'
    t_solve0 = time.time()
    jac_build_times = []
    for it in range(max_iter):
        t_it0 = time.time()
        if time.time() - t_solve0 > wall_budget_s:
            stop_reason = 'wall_budget'
            print("  wall budget exceeded, stopping", flush=True)
            break
        if need_rebuild:
            t0 = time.time()
            r_blocks = []; J_blocks = []
            for m in range(M):
                if V[m].size == 0:
                    r_blocks.append(np.zeros(0)); J_blocks.append(None); continue
                r_m, J_m = build_surface_sparse_jacobian(
                    V[m], F[m], p_levels[m], tau, gamma, fd_eps,
                    vf_adj[m][0], vf_adj[m][1], margin)
                r_blocks.append(r_m); J_blocks.append(J_m)
            r = np.concatenate(r_blocks)
            t_build = time.time() - t0
            jac_build_times.append(t_build)
            Fmax = float(np.max(np.abs(r))); Fmed = float(np.median(np.abs(r)))
            need_rebuild = False
            if verbose:
                print(f"  iter {it}: max|r|={Fmax:.3e} med|r|={Fmed:.3e} "
                      f"lam={lam:.2e} jac_build={t_build:.1f}s", flush=True)
            if Fmax < tol:
                history.append(dict(it=it, max_r=Fmax, med_r=Fmed, lam=lam,
                                     accepted=True, wall=time.time()-t_it0))
                stop_reason = 'converged'
                break
        else:
            if verbose:
                print(f"  iter {it}: retry lam={lam:.2e}", flush=True)
        # Per-surface LM normal equations with SPARSE J
        dx = np.zeros(3*total_v)
        for m in range(M):
            if J_blocks[m] is None: continue
            Jm = J_blocks[m]; rm = r_blocks[m]
            n_col = Jm.shape[1]
            JtJ = (Jm.T @ Jm).tocsc()
            Jtr = Jm.T @ rm
            A = JtJ + lam * sp.identity(n_col, format='csc')
            try:
                step = -spla.spsolve(A, Jtr)
            except Exception:
                step = -np.linalg.lstsq(A.toarray(), Jtr, rcond=None)[0]
            step3 = step.reshape(-1, 3)
            v_step_norm = np.linalg.norm(step3, axis=1)
            limit = 0.25 * min_edges[m]
            scale = np.minimum(1.0, limit / np.maximum(v_step_norm, 1e-30))
            step3 = step3 * scale[:, None]
            sn = float(np.max(np.linalg.norm(step3, axis=1)))
            if sn > 0.02:
                step3 = step3 * (0.02 / sn)
            dx[3*v_off[m]:3*v_off[m+1]] = step3.ravel()
        # try step
        V_new = [v.copy() for v in V]
        for m in range(M):
            if n_v[m] == 0: continue
            V_new[m] = V[m] + dx[3*v_off[m]:3*v_off[m+1]].reshape(-1, 3)
        topo_ok = all(V_new[m].size == 0 or topology_ok(V_new[m], F[m], ref_normals[m])
                       for m in range(M))
        if topo_ok:
            r_new_blocks = [surface_residual(V_new[m], F[m], p_levels[m], tau, gamma)
                             if V_new[m].size else np.zeros(0) for m in range(M)]
            r_new = np.concatenate(r_new_blocks)
            cost_old = 0.5*np.dot(r, r); cost_new = 0.5*np.dot(r_new, r_new)
            predicted = 0.0
            for m in range(M):
                if J_blocks[m] is None: continue
                step = dx[3*v_off[m]:3*v_off[m+1]]
                lin = r_blocks[m] + J_blocks[m] @ step
                predicted += 0.5*np.dot(r_blocks[m], r_blocks[m]) - 0.5*np.dot(lin, lin)
            actual = cost_old - cost_new
            rho = actual / max(predicted, 1e-30)
            accept = rho > 0.0 and cost_new < cost_old
        else:
            accept = False; rho = -1.0; cost_new = float('inf')
        if accept:
            V = V_new
            need_rebuild = True
            if rho > 0.75: lam *= 0.3
            elif rho < 0.25: lam *= 2.0
        else:
            lam *= 4.0
        # stall accounting (cost reduction over consecutive iterations)
        cost_now = 0.5*float(np.dot(r, r)) if not accept else float(cost_new)
        if cost_now < best_cost - 1e-16:
            best_cost = cost_now
            stall = 0
        else:
            stall += 1
        t_it = time.time() - t_it0
        history.append(dict(it=it, max_r=Fmax, med_r=Fmed, lam=float(lam),
                             accepted=bool(accept), rho=float(rho),
                             topo_ok=bool(topo_ok), wall=float(t_it)))
        if verbose:
            tag = 'ok' if topo_ok else 'TOPO_FLIP'
            print(f"    step rho={rho:+.2f} accepted={accept} {tag} "
                  f"iter_wall={t_it:.1f}s", flush=True)
        if stall >= stall_limit:
            stop_reason = 'stalled'
            print(f"    no cost reduction over {stall_limit} consecutive iters, "
                  f"stopping", flush=True)
            break
        if lam > 1e12:
            stop_reason = 'lambda_blowup'
            print("    lambda too large, stagnation", flush=True)
            break
    # final residual at current V
    r_final_blocks = [surface_residual(V[m], F[m], p_levels[m], tau, gamma)
                       if V[m].size else np.zeros(0) for m in range(M)]
    r_final = np.concatenate(r_final_blocks)
    Fmax = float(np.max(np.abs(r_final))); Fmed = float(np.median(np.abs(r_final)))
    return dict(V=V, F=F, p_levels=p_levels, history=history,
                r_blocks=r_final_blocks,
                final_max=Fmax, final_median=Fmed,
                converged=(Fmax < tol), stop_reason=stop_reason,
                jac_build_times=jac_build_times)


# ---------------- validation ----------------

def validate_against_dense(V_m, F_m, p_m, tau, gamma, eps, margin):
    """Compare sparse vs stage-4 dense FD Jacobian on one surface."""
    N_m = V_m.shape[0]
    # warm both JITs on a no-op-sized call first (they're warmed by caller)
    t0 = time.time()
    J_dense = surface_jacobian_fd(V_m, F_m, p_m, tau, gamma, eps)
    t_dense = time.time() - t0
    vf_indptr, vf_idx = build_vertex_faces(F_m, N_m)
    t0 = time.time()
    r0, J_sparse = build_surface_sparse_jacobian(V_m, F_m, p_m, tau, gamma, eps,
                                                  vf_indptr, vf_idx, margin)
    t_sparse = time.time() - t0
    diff = float(np.max(np.abs(J_sparse.toarray() - J_dense)))
    nnz_frac = J_sparse.nnz / float(N_m * 3 * N_m)
    return diff, t_dense, t_sparse, nnz_frac, J_sparse.nnz


# ---------------- diagnostics ----------------

def residual_diagnostics(V, F, p_levels, r_blocks):
    """Where does the residual concentrate? Per-surface stats + boundary
    proximity binning + worst vertices."""
    per_surface = []
    worst = []
    for m in range(len(p_levels)):
        rm = r_blocks[m]
        if rm.size == 0:
            per_surface.append(dict(m=m, p=float(p_levels[m]), n=0))
            continue
        am = np.abs(rm)
        bmax = np.max(np.abs(V[m]), axis=1)  # max-abs coord per vertex
        per_surface.append(dict(
            m=m, p=float(p_levels[m]), n=int(rm.size),
            max_r=float(am.max()), med_r=float(np.median(am)),
            p90_r=float(np.percentile(am, 90)),
            argmax_vertex=[float(x) for x in V[m][int(np.argmax(am))]],
            mean_r_core=float(am[bmax <= 2.0].mean()) if np.any(bmax <= 2.0) else None,
            mean_r_mid=float(am[(bmax > 2.0) & (bmax <= 3.2)].mean())
                if np.any((bmax > 2.0) & (bmax <= 3.2)) else None,
            mean_r_boundary=float(am[bmax > 3.2].mean()) if np.any(bmax > 3.2) else None,
        ))
        order = np.argsort(am)[::-1][:5]
        for i in order:
            worst.append(dict(m=m, i=int(i), r=float(rm[i]),
                               v=[float(x) for x in V[m][i]],
                               bmax=float(bmax[i])))
    worst.sort(key=lambda d: -abs(d['r']))
    return per_surface, worst[:20]


# ---------------- main ----------------

def main(validate_only=False):
    os.makedirs(OUT, exist_ok=True)
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

    fd_eps = 1e-5
    margin = max(1e-4, 10*fd_eps)

    # ----- VALIDATION on the smallest surface -----
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    sizes = [(s[0].shape[0] if s else 0) for s in surfs]
    m_small = int(np.argmin([n if n > 0 else 10**9 for n in sizes]))
    Vs = surfs[m_small][0].copy(); Fs = surfs[m_small][1].astype(np.int64)
    print(f"Validation surface m={m_small}: N={Vs.shape[0]} verts, "
          f"{Fs.shape[0]} faces, p={p_levels[m_small]:.4f}", flush=True)
    print("Warming JIT...", flush=True)
    t0 = time.time()
    surface_residual(Vs, Fs, p_levels[m_small], tau, gamma)
    vfp, vfi = build_vertex_faces(Fs, Vs.shape[0])
    # warm sparse path on the real surface (also serves as compile)
    build_surface_sparse_jacobian(Vs, Fs, p_levels[m_small], tau, gamma,
                                   fd_eps, vfp, vfi, margin)
    surface_jacobian_fd(Vs, Fs, p_levels[m_small], tau, gamma, fd_eps)
    print(f"  JIT warm: {time.time()-t0:.0f}s", flush=True)

    diff, t_dense, t_sparse, nnz_frac, nnz = validate_against_dense(
        Vs, Fs, p_levels[m_small], tau, gamma, fd_eps, margin)
    print(f"VALIDATION: max|J_sparse - J_dense| = {diff:.3e} "
          f"({'PASS' if diff < 1e-8 else 'FAIL'} vs 1e-8)", flush=True)
    print(f"  dense build:  {t_dense:.2f}s   sparse build: {t_sparse:.2f}s "
          f"(x{t_dense/max(t_sparse,1e-9):.0f})  nnz={nnz} "
          f"({100*nnz_frac:.1f}% dense)", flush=True)
    # dense full-problem extrapolation: cost ~ N^2 * N_face per surface
    nv_all = np.array(sizes, float)
    nf_all = np.array([(s[1].shape[0] if s else 0) for s in surfs], float)
    c_val = nv_all[m_small]**2 * nf_all[m_small]
    dense_full_est = t_dense * float(np.sum(nv_all**2 * nf_all) / c_val)
    print(f"  dense FULL-Jacobian estimate: {dense_full_est:.0f}s "
          f"(stage4 observed ~500s)", flush=True)
    val_info = dict(surface=m_small, n_verts=int(Vs.shape[0]),
                    max_abs_diff=diff, dense_s=t_dense, sparse_s=t_sparse,
                    nnz=int(nnz), nnz_frac=nnz_frac,
                    dense_full_estimate_s=dense_full_est)
    if validate_only:
        return val_info
    if diff >= 1e-8:
        print("VALIDATION FAILED -- aborting solve", flush=True)
        json.dump(dict(validation=val_info, aborted=True),
                  open(f"{OUT}/stage5_lm.json", "w"), indent=2)
        return val_info

    # ----- SOLVE -----
    t_start = time.time()
    out = solve_cmm_sparse(P_full, uf, tau, gamma, p_levels,
                            max_iter=60, fd_eps=fd_eps, tol=1e-7, lam0=0.1,
                            wall_budget_s=90*60, stall_limit=8, verbose=True)
    wall = time.time() - t_start
    print(f"\nTotal {wall/60:.1f} min  stop={out['stop_reason']}", flush=True)
    print(f"Converged: {out['converged']}  final max|r|={out['final_max']:.3e} "
          f"med|r|={out['final_median']:.3e}", flush=True)

    per_surface, worst = residual_diagnostics(out['V'], out['F'],
                                               out['p_levels'], out['r_blocks'])

    init_max = out['history'][0]['max_r'] if out['history'] else float('nan')
    substantially_reduced = out['final_max'] < 0.5*init_max
    if out['converged'] or substantially_reduced:
        for m, (Vm, Fm) in enumerate(zip(out['V'], out['F'])):
            if Vm.size:
                np.savez(f"{OUT}/stage5_surf_m{m:02d}.npz", V=Vm, F=Fm,
                         p=out['p_levels'][m])
        print("saved stage5_surf_m*.npz", flush=True)

    json.dump(dict(validation=val_info,
                    history=out['history'],
                    final_max=float(out['final_max']),
                    final_median=float(out['final_median']),
                    converged=bool(out['converged']),
                    stop_reason=out['stop_reason'],
                    wall_min=float(wall/60),
                    jac_build_times_s=[float(t) for t in out['jac_build_times']],
                    tau=tau, gamma=gamma, n_surfaces=len(p_levels),
                    n_verts_total=int(sum(v.shape[0] for v in out['V'])),
                    per_surface=per_surface, worst_vertices=worst),
              open(f"{OUT}/stage5_lm.json", "w"), indent=2)
    print(f"saved {OUT}/stage5_lm.json", flush=True)

    # ----- RESULTS markdown -----
    lines = ["# CMM Stage 5: sparse-FD free-vertex LM results", ""]
    lines.append(f"(tau, gamma) = ({tau}, {gamma}), G={Gi}, "
                 f"{len(p_levels)} surfaces, "
                 f"{int(sum(v.shape[0] for v in out['V']))} vertices total.")
    lines.append("")
    lines.append("## Jacobian validation and speed")
    lines.append(f"- max|J_sparse - J_dense| on surface m={m_small} "
                 f"(N={Vs.shape[0]}): **{diff:.3e}** (< 1e-8 required)")
    lines.append(f"- dense FD build (that surface): {t_dense:.2f}s; "
                 f"sparse: {t_sparse:.2f}s")
    lines.append(f"- full 16-surface Jacobian: sparse "
                 f"{np.median(out['jac_build_times']):.1f}s median per build "
                 f"(stage4 dense ~500s; dense extrapolation {dense_full_est:.0f}s)")
    lines.append(f"- sparsity: {100*nnz_frac:.1f}% of dense on validation surface")
    lines.append("")
    lines.append("## Residual trajectory")
    lines.append("| iter | max|r| | med|r| | lam | accepted | wall (s) |")
    lines.append("|---|---|---|---|---|---|")
    for h in out['history']:
        lines.append(f"| {h['it']} | {h['max_r']:.3e} | {h['med_r']:.3e} | "
                     f"{h['lam']:.1e} | {h.get('accepted','-')} | "
                     f"{h.get('wall', 0):.1f} |")
    lines.append("")
    lines.append(f"Stop reason: **{out['stop_reason']}**; "
                 f"final max|r| = {out['final_max']:.3e}, "
                 f"median = {out['final_median']:.3e}; wall {wall/60:.1f} min.")
    lines.append("")
    lines.append("## Where the residual concentrates")
    lines.append("| m | p_m | n | max|r| | med|r| | mean|r| core(<=2) | mid(2-3.2) | boundary(>3.2) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in per_surface:
        if s.get('n', 0) == 0: continue
        fmt = lambda x: ('-' if x is None else f"{x:.2e}")
        lines.append(f"| {s['m']} | {s['p']:.4f} | {s['n']} | {s['max_r']:.2e} | "
                     f"{s['med_r']:.2e} | {fmt(s['mean_r_core'])} | "
                     f"{fmt(s['mean_r_mid'])} | {fmt(s['mean_r_boundary'])} |")
    lines.append("")
    lines.append("Worst 20 vertices (surface, |r|, position, max-abs coord):")
    for wv in worst:
        lines.append(f"- m={wv['m']} r={wv['r']:+.3e} v=({wv['v'][0]:+.2f},"
                     f"{wv['v'][1]:+.2f},{wv['v'][2]:+.2f}) bmax={wv['bmax']:.2f}")
    open(f"{OUT}/RESULTS_stage5.md", "w").write("\n".join(lines) + "\n")
    print(f"saved {OUT}/RESULTS_stage5.md", flush=True)
    return out


if __name__ == "__main__":
    main(validate_only=('--validate-only' in sys.argv))
