"""Stage 3a (fast): SAMPLE 500 vertices to validate the moving-mesh
residual math. Full 45k-vertex evaluation is too slow without a JIT'd
slice; that's the point of Stage 3b. This just answers: does the
per-vertex residual at iter 0 land in the expected range (max ~0.32,
median ~6e-3) that Stage 1c reported on the cube?"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, EMIN15, OUT, C)
from cmm_stage3a_moving_mesh import MovingMesh
from reznsrc.contour_K3_halo import init_no_learning_K3

Gi = 21
du, uf, lo, hi = build_grid(Gi)
tau, gamma = 2.0, 0.0980
P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
P_full[lo:hi, lo:hi, lo:hi] = P_inner
p_flat = P_inner.ravel()
qs = np.linspace(0, 1, 33); edges = np.quantile(p_flat, qs)
p_levels = 0.5*(edges[:-1] + edges[1:])

mesh = MovingMesh(P_full, uf, p_levels, tau, gamma, lo, hi)
print(f"Mesh: {len(p_levels)} surfaces, {mesh.total_verts} vertices total",
      flush=True)
# Sample 500 random vertices across all surfaces
rng = np.random.default_rng(0)
N_sample = 500
all_idx = []  # (m, vertex_idx_in_m)
for m, n in enumerate(mesh.n_verts):
    for vi in range(n):
        all_idx.append((m, vi))
picks = rng.choice(len(all_idx), size=N_sample, replace=False)
samples = [all_idx[i] for i in picks]

t0 = time.time()
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear
r_sample = np.empty(N_sample)
for k, (m, vi) in enumerate(samples):
    v = mesh.verts[m][vi]
    p_m = mesh.p_levels[m]
    mu = np.empty(3)
    for k_own in range(3):
        U = float(v[k_own])
        A0, A1 = mesh.evidence(m, k_own, U)
        f0 = _fsig(U, 0, tau); f1 = _fsig(U, 1, tau)
        num = f1*A1; den = f0*A0 + num
        if den <= 0: mu[k_own] = 0.5
        else: mu[k_own] = max(min(num/den, 1-1e-12), 1e-12)
    p_clear = _clear(mu, np.full(3, gamma), np.full(3, 1.0))
    r_sample[k] = p_clear - p_m
    if (k+1) % 100 == 0:
        print(f"  ... {k+1}/{N_sample}  elapsed {time.time()-t0:.0f}s",
              flush=True)
wall = time.time() - t0
print(f"\nSample of {N_sample} vertices, wall {wall:.0f}s "
      f"({wall/N_sample*1000:.0f} ms/vertex)", flush=True)
print(f"  max |r|: {np.max(np.abs(r_sample)):.3e}", flush=True)
print(f"  median |r|: {np.median(np.abs(r_sample)):.3e}", flush=True)
print(f"  p90 |r|: {np.percentile(np.abs(r_sample), 90):.3e}", flush=True)
json.dump(dict(n_sample=N_sample, max=float(np.max(np.abs(r_sample))),
                median=float(np.median(np.abs(r_sample))),
                p90=float(np.percentile(np.abs(r_sample), 90)),
                wall=float(wall), n_surfaces=len(p_levels),
                total_verts=int(mesh.total_verts)),
          open(f"{OUT}/stage3a_sample.json", "w"), indent=2)
print(f"\nsaved {OUT}/stage3a_sample.json", flush=True)
