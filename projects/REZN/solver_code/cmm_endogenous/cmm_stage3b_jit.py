"""CMM Stage 3b: JIT'd moving-mesh residual + first Newton attempt.

Speedup: replace the Python slice_mesh + stitch + arclength code with
numba kernels. Each face becomes a simple sign-test on 3 vertex z-values
and (when crossed) interpolated coords; arclength integration is a
single ufunc-friendly summation.

Cost target: <30 s per full-residual evaluation at G=21 (~45k vertices).
Then Newton-Krylov with finite-difference matvec for the first Newton
step; report whether the residual contracts.
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          EMIN15, OUT, C)
from reznsrc.contour_K3_halo import init_no_learning_K3


@njit(cache=True, fastmath=False)
def f_signal_jit(u, v, tau):
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return np.sqrt(tau/(2*np.pi)) * np.exp(-0.5*tau*d*d)


@njit(cache=True, fastmath=False)
def clear_crra_jit(mu, gamma):
    """Bisection CRRA market clearing."""
    EPS_PRICE = 1e-12
    K = mu.shape[0]
    a = EPS_PRICE; b = 1 - EPS_PRICE
    def excess(p):
        s = 0.0
        for k in range(K):
            lz = (np.log(mu[k]/(1-mu[k])) - np.log(p/(1-p))) / gamma
            if lz >= 0:
                e = np.exp(-lz)
                s += (1 - e) / ((1 - p)*e + p)
            else:
                e = np.exp(lz)
                s += (e - 1) / ((1 - p) + p*e)
        return s
    fa = excess(a); fb = excess(b)
    if fa <= 0: return a
    if fb >= 0: return b
    for _ in range(60):
        c = 0.5*(a + b); fc = excess(c)
        if fc >= 0: a = c; fa = fc
        else: b = c; fb = fc
        if (b - a) < 1e-14: break
    return 0.5*(a + b)


@njit(cache=True, fastmath=False)
def slice_arclen_jit(verts, faces, axis, U, tau):
    """For a fixed mesh (verts, faces), slice by {u_axis = U}; return
    (A_0, A_1) arclength integrals of f_v(other coords) on the polyline.
    Doesn't need explicit stitch -- each crossing-triangle contributes
    one segment, and the arclength integral on each segment is closed
    form: ∫_0^1 f_v(p0 + t*(p1-p0)) * |p1-p0| dt approximated by
    midpoint rule + endpoint correction (2-point Gauss). Sums over all
    crossing triangles."""
    other0 = 1 if axis == 0 else 0
    other1 = 2 if axis < 2 else 1
    A0 = 0.0; A1 = 0.0
    N_face = faces.shape[0]
    sqrt_t = np.sqrt(tau/(2*np.pi))
    for f in range(N_face):
        i0 = faces[f, 0]; i1 = faces[f, 1]; i2 = faces[f, 2]
        z0 = verts[i0, axis] - U
        z1 = verts[i1, axis] - U
        z2 = verts[i2, axis] - U
        s0 = 0 if z0 == 0 else (1 if z0 > 0 else -1)
        s1 = 0 if z1 == 0 else (1 if z1 > 0 else -1)
        s2 = 0 if z2 == 0 else (1 if z2 > 0 else -1)
        if s0 == s1 == s2 and s0 != 0: continue   # all same side
        # find two intersection points
        pts_a = np.empty(2)  # other0 coords
        pts_b = np.empty(2)  # other1 coords
        np_ = 0
        # check each of 3 edges
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
            # avoid duplicating
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
        if np_ < 2: continue
        # segment endpoints (pts_a[0], pts_b[0]) -> (pts_a[1], pts_b[1])
        da = pts_a[1] - pts_a[0]; db = pts_b[1] - pts_b[0]
        L = np.sqrt(da*da + db*db)
        if L == 0: continue
        # 2-point Gauss-Legendre on [0,1] -> nodes 0.5 +- 0.5/sqrt(3)
        for s in (0.5 - 0.5/np.sqrt(3.0), 0.5 + 0.5/np.sqrt(3.0)):
            a_s = pts_a[0] + s*da
            b_s = pts_b[0] + s*db
            d0a = a_s + 0.5; d0b = b_s + 0.5
            d1a = a_s - 0.5; d1b = b_s - 0.5
            f0 = sqrt_t*np.exp(-0.5*tau*d0a*d0a) * sqrt_t*np.exp(-0.5*tau*d0b*d0b)
            f1 = sqrt_t*np.exp(-0.5*tau*d1a*d1a) * sqrt_t*np.exp(-0.5*tau*d1b*d1b)
            A0 += 0.5 * f0 * L
            A1 += 0.5 * f1 * L
    return A0, A1


@njit(cache=True, fastmath=False, parallel=True)
def vertex_residual_jit(all_verts_concat, n_verts_arr, faces_concat,
                         n_faces_arr, p_levels, tau, gamma):
    """Per-vertex residual r_v = clear_CRRA(mu(v)) - p_m for all
    vertices of all surfaces. Returns flat array of residuals."""
    total_v = 0
    for n in n_verts_arr: total_v += n
    r = np.empty(total_v)
    # cumulative offsets
    v_off = np.zeros(n_verts_arr.size + 1, dtype=np.int64)
    f_off = np.zeros(n_faces_arr.size + 1, dtype=np.int64)
    for i in range(n_verts_arr.size):
        v_off[i+1] = v_off[i] + n_verts_arr[i]
    for i in range(n_faces_arr.size):
        f_off[i+1] = f_off[i] + n_faces_arr[i]
    for m in prange(n_verts_arr.size):
        if n_verts_arr[m] == 0: continue
        p_m = p_levels[m]
        verts_m = all_verts_concat[v_off[m]:v_off[m+1]]
        faces_m = faces_concat[f_off[m]:f_off[m+1]]
        for vi in range(n_verts_arr[m]):
            U0 = verts_m[vi, 0]; U1 = verts_m[vi, 1]; U2 = verts_m[vi, 2]
            A0_0, A1_0 = slice_arclen_jit(verts_m, faces_m, 0, U0, tau)
            A0_1, A1_1 = slice_arclen_jit(verts_m, faces_m, 1, U1, tau)
            A0_2, A1_2 = slice_arclen_jit(verts_m, faces_m, 2, U2, tau)
            mu = np.empty(3)
            for k in range(3):
                if k == 0: U = U0; A0 = A0_0; A1 = A1_0
                elif k == 1: U = U1; A0 = A0_1; A1 = A1_1
                else: U = U2; A0 = A0_2; A1 = A1_2
                f0 = f_signal_jit(U, 0, tau); f1 = f_signal_jit(U, 1, tau)
                num = f1*A1; den = f0*A0 + num
                if den <= 0: mu[k] = 0.5
                else:
                    val = num/den
                    if val < 1e-12: mu[k] = 1e-12
                    elif val > 1 - 1e-12: mu[k] = 1 - 1e-12
                    else: mu[k] = val
            p_clear = clear_crra_jit(mu, gamma)
            r[v_off[m] + vi] = p_clear - p_m
    return r


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
    # flatten
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
    print(f"Mesh: {len(p_levels)} surfaces, {total_v} vertices, {total_f} faces",
          flush=True)
    # warm up JIT (compile)
    print("Compiling JIT (first call)...", flush=True)
    t0 = time.time()
    r = vertex_residual_jit(all_verts, n_verts_arr, all_faces, n_faces_arr,
                              p_levels, tau, gamma)
    print(f"  compile + first eval: {time.time()-t0:.0f}s", flush=True)
    # second eval = pure runtime
    t0 = time.time()
    r = vertex_residual_jit(all_verts, n_verts_arr, all_faces, n_faces_arr,
                              p_levels, tau, gamma)
    wall = time.time() - t0
    print(f"  steady-state eval:    {wall:.1f}s", flush=True)
    print(f"  max|r|: {np.max(np.abs(r)):.3e}", flush=True)
    print(f"  median|r|: {np.median(np.abs(r)):.3e}", flush=True)
    print(f"  p90|r|: {np.percentile(np.abs(r), 90):.3e}", flush=True)
    json.dump(dict(n_surfaces=len(p_levels), total_v=int(total_v),
                    total_f=int(total_f),
                    max_r=float(np.max(np.abs(r))),
                    median_r=float(np.median(np.abs(r))),
                    p90_r=float(np.percentile(np.abs(r), 90)),
                    eval_s=float(wall)),
              open(f"{OUT}/stage3b_jit.json", "w"), indent=2)
    print("\nsaved stage3b_jit.json", flush=True)


if __name__ == "__main__":
    main()
