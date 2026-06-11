"""CMM Stage 3 - Moving-mesh prototype: vertex positions as primary DOFs.

Stage 2 negative result: cube-based CMM diverged under iteration because
marching_cubes re-triangulates discontinuously when cube values move,
breaking smoothness of Phi_CMM in P. Stage 3 fix: pin the topology of
each level surface mesh at iteration 0, evolve only the vertex POSITIONS
under Newton.

Architecture:
  - Initial mesh: extract surfaces of the kernel FP P* by marching cubes
    ONCE. Get M = len(p_levels) surfaces, each with N_m verts and F_m
    fixed faces.
  - State vector x: stack of all vertex coordinates (3*sum(N_m) floats).
  - Residual: for each vertex v of S_m,
        r_v = clear_CRRA(mu_0(v), mu_1(v), mu_2(v); gamma) - p_m
    where mu_k(v) is the Bayes posterior built from arclength integrals
    A_v(p_m, u_k=v.coord[k]) over slices of the CURRENT meshes.
  - Slice extraction stays as before (the topology of the slice CHANGES
    smoothly with vertex positions as long as no triangle inverts, which
    we monitor).
  - Newton-Krylov on x.

The smoothness gain: arclength integrals over a fixed-topology mesh are
analytic functions of vertex coordinates (line-integrals over piecewise-
linear curves with smooth dependence on endpoints), so Phi is smooth.

Validation gate: at iteration 0, with x = initial mesh from P*, the
residual r should equal the Stage-1c |CMM-P*| field (~0.32 max,
~6e-3 median). Newton should reduce this. Whether it converges to
machine eps will tell us if the moving-mesh route works.

This file: Stage 3a -- just write the residual function and check its
value at x_0 matches the Stage-1c diagnostic. No Newton yet."""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          slice_mesh_by_hyperplane, stitch_segments,
                          EMIN15, OUT, C, UMAX, pad)
from cmm_stage2_iterate import arclen_int_pair
from reznsrc.contour_K3_halo import init_no_learning_K3
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear


class MovingMesh:
    """Fixed-topology mesh of M level surfaces. Vertex positions evolve."""

    def __init__(self, P_full, uf, p_levels, tau, gamma, lo, hi):
        self.uf = uf; self.tau = tau; self.gamma = gamma
        self.lo = lo; self.hi = hi
        self.p_levels = np.asarray(p_levels)
        surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
        # Store positions and faces per surface; faces are FIXED.
        self.verts = []   # list of (N_m, 3) float arrays
        self.faces = []   # list of (F_m, 3) int arrays
        self.n_verts = []
        for s in surfs:
            if s is None:
                self.verts.append(np.zeros((0, 3))); self.faces.append(np.zeros((0,3), int))
                self.n_verts.append(0)
            else:
                v, f = s
                self.verts.append(v.copy()); self.faces.append(f.copy())
                self.n_verts.append(len(v))
        self.total_verts = int(sum(self.n_verts))
        # vertex slices into x
        offs = [0]
        for n in self.n_verts: offs.append(offs[-1] + n)
        self.offs = offs

    def state(self):
        """Flatten vertices to (3*total_verts,) state vector."""
        return np.concatenate([v.ravel() for v in self.verts])

    def set_state(self, x):
        for m, n in enumerate(self.n_verts):
            if n == 0: continue
            self.verts[m] = x[3*self.offs[m]:3*self.offs[m+1]].reshape(n, 3)

    def evidence(self, m_surf, k_own, U):
        """Arclength integrals A_0, A_1 over S_{p_m} sliced by {u_k=U}."""
        if self.n_verts[m_surf] == 0: return 0.0, 0.0
        segs = slice_mesh_by_hyperplane(self.verts[m_surf],
                                          self.faces[m_surf], k_own, U)
        if not segs: return 0.0, 0.0
        chains = stitch_segments(segs)
        return arclen_int_pair(chains, self.tau)

    def residual(self):
        """Vertex residual: clear_CRRA(mu) - p_m at every vertex of every
        surface. Returns (total_verts,) ravel of scalars."""
        r = np.empty(self.total_verts)
        idx = 0
        for m, n in enumerate(self.n_verts):
            if n == 0: continue
            p_m = self.p_levels[m]
            V = self.verts[m]  # (n, 3) physical coords
            for v in V:
                mu = np.empty(3)
                for k_own in range(3):
                    U = float(v[k_own])
                    A0, A1 = self.evidence(m, k_own, U)
                    f0 = _fsig(U, 0, self.tau); f1 = _fsig(U, 1, self.tau)
                    num = f1*A1; den = f0*A0 + num
                    if den <= 0: mu[k_own] = 0.5
                    else: mu[k_own] = max(min(num/den, 1-1e-12), 1e-12)
                p_clear = _clear(mu, np.full(3, self.gamma), np.full(3, 1.0))
                r[idx] = p_clear - p_m
                idx += 1
        return r


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    P_inner_kernel = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner_kernel
    # price levels: 64 quantile-binned
    p_flat = P_inner_kernel.ravel()
    qs = np.linspace(0, 1, 65); edges = np.quantile(p_flat, qs)
    p_levels = 0.5*(edges[:-1] + edges[1:])

    t0 = time.time()
    mesh = MovingMesh(P_full, uf, p_levels, tau, gamma, lo, hi)
    print(f"Mesh built: {len(p_levels)} surfaces, total {mesh.total_verts} vertices  "
          f"({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    r = mesh.residual()
    wall = time.time() - t0
    print(f"Initial residual at kernel FP:", flush=True)
    print(f"  max |r|: {np.max(np.abs(r)):.3e}", flush=True)
    print(f"  median |r|: {np.median(np.abs(r)):.3e}", flush=True)
    print(f"  p90 |r|: {np.percentile(np.abs(r), 90):.3e}", flush=True)
    print(f"  wall: {wall:.0f}s", flush=True)
    json.dump(dict(
        n_surfaces=len(p_levels), total_verts=int(mesh.total_verts),
        max_r=float(np.max(np.abs(r))),
        median_r=float(np.median(np.abs(r))),
        p90_r=float(np.percentile(np.abs(r), 90)),
        wall=float(wall), tau=tau, gamma=gamma, Gi=Gi),
        open(f"{OUT}/stage3a_initial_residual.json", "w"), indent=2)
    print("\nsaved stage3a_initial_residual.json", flush=True)


if __name__ == "__main__":
    main()
