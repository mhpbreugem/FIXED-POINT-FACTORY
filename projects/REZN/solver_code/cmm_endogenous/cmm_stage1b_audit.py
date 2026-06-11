"""Stage 1b: audit the outlier cell (8,12,9) and improve the slice
extractor. Then re-run a broader 20-cell validation."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          slice_mesh_by_hyperplane, stitch_segments,
                          arclength_int_f, phi_cmm_pointwise,
                          EMIN15, OUT, C, UMAX, pad)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear


def cmm_evidence_v2(P_full, uf, p_target, k_own, U_own, tau, surf=None):
    """Strict h=0 evidence with explicit chain reporting (debugging)."""
    if surf is None:
        surf = extract_surfaces_marching_cubes(P_full, uf, [p_target])[0]
    if surf is None: return 0.0, 0.0, []
    verts, faces = surf
    segs = slice_mesh_by_hyperplane(verts, faces, k_own, U_own)
    chains = stitch_segments(segs)
    A0 = 0.0; A1 = 0.0
    for chain in chains:
        A0 += arclength_int_f(chain, lambda u: _fsig(u, 0, tau),
                                       lambda u: _fsig(u, 0, tau))
        A1 += arclength_int_f(chain, lambda u: _fsig(u, 1, tau),
                                       lambda u: _fsig(u, 1, tau))
    return A0, A1, chains


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    h = C * np.sqrt(du)
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    Pn_kernel = phi_K3_halo_smooth(P_full, uf, lo, hi, np.full(3, tau),
                                    np.full(3, gamma), np.full(3, 1.0), h)

    # Audit (8,12,9): P_star
    ki, mi, li = 8, 12, 9
    i, j, l = lo + ki, lo + mi, lo + li
    p_target = float(P_inner[ki, mi, li])
    print(f"=== audit cell ({ki},{mi},{li}) ===")
    print(f"p_target = {p_target:.6f}")
    surf = extract_surfaces_marching_cubes(P_full, uf, [p_target])[0]
    verts, faces = surf
    print(f"Surface S_p: {len(verts)} verts, {len(faces)} faces")
    for k_own, idx in [(0, i), (1, j), (2, l)]:
        U = uf[idx]
        A0, A1, chains = cmm_evidence_v2(P_full, uf, p_target, k_own, U, tau, surf=surf)
        print(f"  agent {k_own}, own_u={U:.3f}: {len(chains)} chains, "
              f"total len={sum(len(c) for c in chains)}, A0={A0:.4f}, A1={A1:.4f}")
        for ci, c in enumerate(chains):
            arc = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
            print(f"    chain {ci}: {len(c)} pts, arclen={arc:.3f}, "
                  f"endpoints={c[0].tolist()} -> {c[-1].tolist()}")

    # Broader sweep: 50 cells, time + diff against kernel
    print(f"\n=== broader 50-cell sweep ===")
    rng = np.random.default_rng(42)
    test_cells = [(int(rng.integers(0, Gi)), int(rng.integers(0, Gi)),
                   int(rng.integers(0, Gi))) for _ in range(50)]
    out = []
    t0 = time.time()
    for (ki, mi, li) in test_cells:
        p_orig = float(P_inner[ki, mi, li])
        p_kernel = float(Pn_kernel[lo + ki, lo + mi, lo + li])
        p_cmm = phi_cmm_pointwise(P_full, uf, lo, hi, tau, gamma, ki, mi, li)
        out.append(dict(cell=[ki,mi,li], P_star=p_orig, Phi_kernel=p_kernel,
                        Phi_cmm=p_cmm,
                        diff_cmm_kernel=abs(p_cmm-p_kernel),
                        diff_cmm_star=abs(p_cmm-p_orig)))
    wall = time.time() - t0
    diffs = np.array([r['diff_cmm_kernel'] for r in out])
    print(f"50 cells in {wall:.0f}s ({wall/50*1000:.0f} ms/cell)")
    print(f"|CMM - kernel|: median {np.median(diffs):.3e}, "
          f"p90 {np.percentile(diffs, 90):.3e}, max {diffs.max():.3e}")
    diffs_star = np.array([r['diff_cmm_star'] for r in out])
    print(f"|CMM - P*|:     median {np.median(diffs_star):.3e}, "
          f"p90 {np.percentile(diffs_star, 90):.3e}, max {diffs_star.max():.3e}")
    json.dump(dict(audit_cell=[8,12,9], sweep=out,
                    diff_kernel_median=float(np.median(diffs)),
                    diff_kernel_p90=float(np.percentile(diffs, 90)),
                    diff_kernel_max=float(diffs.max()),
                    diff_star_median=float(np.median(diffs_star)),
                    diff_star_max=float(diffs_star.max()),
                    h_kernel=float(h),
                    cells_n=len(out), wall_s=wall),
              open(f"{OUT}/stage1b_audit.json", "w"), indent=2)
    print(f"\nSaved -> {OUT}/stage1b_audit.json")


if __name__ == "__main__":
    main()
