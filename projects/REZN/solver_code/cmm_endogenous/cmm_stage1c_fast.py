"""Stage 1c: a fast batched CMM Phi evaluation.

Key speedup: surfaces are cached by price level. We bin the M = G^3
target prices into N_bin distinct levels (or use the unique sorted set
with rounding tolerance), extract each surface ONCE per Phi evaluation,
then iterate cells reusing the cached surface.

Goal: full G=21 Phi in < 60 s, so a Newton iteration is feasible.
Then a final 1000-cell validation on the audit fixed point shows the
P-distribution of |CMM - kernel|.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          slice_mesh_by_hyperplane, stitch_segments,
                          arclength_int_f, EMIN15, OUT, C, UMAX, pad)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear


def bin_prices(P_inner, n_bins=64):
    """Bin all inner cell prices into n_bins distinct level values.
    Returns (bin_centers, cell_to_bin_idx)."""
    p_flat = P_inner.ravel()
    # use quantile binning so each bin has similar number of cells
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p_flat, qs)
    edges[0] -= 1e-9; edges[-1] += 1e-9
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.searchsorted(edges, p_flat, side='right') - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return centers, idx.reshape(P_inner.shape)


def cmm_phi_fast(P_full, uf, lo, hi, tau, gamma, n_bins=64):
    """Batched CMM Phi: extract surfaces at the binned price levels,
    then iterate cells reusing the surfaces. Returns the inner Phi
    array shape (Gi, Gi, Gi)."""
    Gi = hi - lo
    P_inner = P_full[lo:hi, lo:hi, lo:hi]
    centers, bin_idx = bin_prices(P_inner, n_bins=n_bins)
    surfs = extract_surfaces_marching_cubes(P_full, uf, centers)
    Pn = np.empty_like(P_inner)
    for ki in range(Gi):
        for mi in range(Gi):
            for li in range(Gi):
                b = int(bin_idx[ki, mi, li])
                surf = surfs[b]
                if surf is None:
                    Pn[ki, mi, li] = P_inner[ki, mi, li]
                    continue
                verts, faces = surf
                mu = np.empty(3)
                for k_own, idx_glob in [(0, lo+ki), (1, lo+mi), (2, lo+li)]:
                    U = uf[idx_glob]
                    segs = slice_mesh_by_hyperplane(verts, faces, k_own, U)
                    chains = stitch_segments(segs)
                    A0 = 0.0; A1 = 0.0
                    for chain in chains:
                        A0 += arclength_int_f(chain, lambda u: _fsig(u, 0, tau),
                                                      lambda u: _fsig(u, 0, tau))
                        A1 += arclength_int_f(chain, lambda u: _fsig(u, 1, tau),
                                                      lambda u: _fsig(u, 1, tau))
                    f0 = _fsig(U, 0, tau); f1 = _fsig(U, 1, tau)
                    num = f1 * A1; den = f0 * A0 + num
                    if den <= 0: mu[k_own] = 0.5
                    else:
                        m = num / den
                        mu[k_own] = max(min(m, 1 - 1e-12), 1e-12)
                Pn[ki, mi, li] = _clear(mu, np.full(3, gamma), np.full(3, 1.0))
    return Pn


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
    F_kernel = float(np.max(np.abs(Pn_kernel[lo:hi, lo:hi, lo:hi] - P_inner)))
    print(f"Kernel Phi: F = {F_kernel:.3e}")

    for nb in [32, 64, 128]:
        t0 = time.time()
        Pn = cmm_phi_fast(P_full, uf, lo, hi, tau, gamma, n_bins=nb)
        wall = time.time() - t0
        diff = Pn - Pn_kernel[lo:hi, lo:hi, lo:hi]
        F_cmm_kernel = float(np.max(np.abs(diff)))
        F_cmm_star   = float(np.max(np.abs(Pn - P_inner)))
        print(f"  n_bins={nb:3d}: full-cube {wall:5.1f}s  "
              f"|CMM-kernel|_inf={F_cmm_kernel:.3e}  "
              f"|CMM-P*|_inf={F_cmm_star:.3e}  "
              f"median {float(np.median(np.abs(diff))):.3e}")
        np.save(f"{OUT}/Pn_cmm_nbins{nb}.npy", Pn)
    json.dump(dict(F_kernel=F_kernel,
                    h=float(h),
                    Gi=Gi,
                    tau=tau, gamma=gamma),
              open(f"{OUT}/stage1c_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
