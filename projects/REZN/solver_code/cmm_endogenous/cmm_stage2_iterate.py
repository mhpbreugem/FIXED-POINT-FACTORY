"""CMM Stage 2: a fast CMM Phi + fixed-point iteration / Newton-Krylov
toward a strict h=0 fixed point.

Speed strategy: invert the loops. For each (price bin m, axis k, own-signal
U_idx), compute ONE arclength integral set A_v(m, k, U). Then assemble Phi
by looking up + Bayes + CRRA clear per cell. This collapses ~27000 cell-
slice operations into ~8000 (axis, U, bin) operations, and removes the
slice / stitch work from the per-cell inner loop.

Newton strategy: Newton-Krylov on the cube values P[i,j,l] using Phi_CMM
as the operator. Warm start: the emin15-certified longdouble fixed point
of the KERNEL operator. The question being tested: does the strict h=0
operator converge from the kernel FP starting point, and to what deficit?
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          slice_mesh_by_hyperplane, stitch_segments, EMIN15,
                          OUT, C, UMAX, pad)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear


def arclen_int_pair(chains, tau):
    """Arclength integrals A_0 = sum f_0(u_j)f_0(u_l) dsigma and similarly
    A_1 over the polylines. Returns (A_0, A_1)."""
    A0 = 0.0; A1 = 0.0
    for c in chains:
        if len(c) < 2: continue
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)
        u_j, u_l = c[:, 0], c[:, 1]
        # f0(u) = sqrt(tau/2pi) exp(-tau (u+0.5)^2 /2)
        # f1(u) = sqrt(tau/2pi) exp(-tau (u-0.5)^2 /2)
        s = np.sqrt(tau/(2*np.pi))
        f0_j = s*np.exp(-0.5*tau*(u_j+0.5)**2)
        f0_l = s*np.exp(-0.5*tau*(u_l+0.5)**2)
        f1_j = s*np.exp(-0.5*tau*(u_j-0.5)**2)
        f1_l = s*np.exp(-0.5*tau*(u_l-0.5)**2)
        v0 = f0_j*f0_l; v1 = f1_j*f1_l
        A0 += float(0.5*np.sum(d*(v0[:-1]+v0[1:])))
        A1 += float(0.5*np.sum(d*(v1[:-1]+v1[1:])))
    return A0, A1


def cmm_phi_inverted(P_full, uf, lo, hi, tau, gamma, n_bins=128):
    """Fast CMM Phi: pre-compute (A_0, A_1)(bin, axis, U_idx) ONCE,
    then assemble per-cell value."""
    Gi = hi - lo
    P_inner = P_full[lo:hi, lo:hi, lo:hi]
    p_flat = P_inner.ravel()
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(p_flat, qs); edges[0] -= 1e-9; edges[-1] += 1e-9
    centers = 0.5*(edges[:-1] + edges[1:])
    bin_idx_flat = np.clip(np.searchsorted(edges, p_flat, side='right') - 1, 0, n_bins-1)
    bin_idx = bin_idx_flat.reshape(Gi, Gi, Gi)
    surfs = extract_surfaces_marching_cubes(P_full, uf, centers)

    A0 = np.zeros((n_bins, 3, Gi))
    A1 = np.zeros((n_bins, 3, Gi))
    for m, surf in enumerate(surfs):
        if surf is None: continue
        verts, faces = surf
        for k_own in range(3):
            for U_idx in range(Gi):
                U = uf[lo + U_idx]
                segs = slice_mesh_by_hyperplane(verts, faces, k_own, U)
                if not segs:
                    continue
                chains = stitch_segments(segs)
                a0, a1 = arclen_int_pair(chains, tau)
                A0[m, k_own, U_idx] = a0
                A1[m, k_own, U_idx] = a1
    # Assemble Phi vectorized over cells
    Pn = np.empty_like(P_inner)
    # Per-cell loop only for the (Bayes + clear) step, which is fast
    f0_u = np.array([_fsig(uf[lo + i], 0, tau) for i in range(Gi)])
    f1_u = np.array([_fsig(uf[lo + i], 1, tau) for i in range(Gi)])
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                m = int(bin_idx[i, j, l])
                # agent 0 own-signal = i
                num0 = f1_u[i]*A1[m, 0, i]; den0 = f0_u[i]*A0[m, 0, i] + num0
                mu0 = num0/den0 if den0 > 0 else 0.5
                num1 = f1_u[j]*A1[m, 1, j]; den1 = f0_u[j]*A0[m, 1, j] + num1
                mu1 = num1/den1 if den1 > 0 else 0.5
                num2 = f1_u[l]*A1[m, 2, l]; den2 = f0_u[l]*A0[m, 2, l] + num2
                mu2 = num2/den2 if den2 > 0 else 0.5
                mu_vec = np.array([max(min(mu0, 1-1e-12), 1e-12),
                                    max(min(mu1, 1-1e-12), 1e-12),
                                    max(min(mu2, 1-1e-12), 1e-12)])
                Pn[i,j,l] = _clear(mu_vec, np.full(3, gamma), np.full(3, 1.0))
    return Pn


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    h = C * np.sqrt(du)
    P_inner_kernel = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner_kernel

    print(f"=== timing fast CMM Phi ===", flush=True)
    for nb in [32, 64, 128]:
        t0 = time.time()
        Pn = cmm_phi_inverted(P_full, uf, lo, hi, tau, gamma, n_bins=nb)
        wall = time.time() - t0
        Pn_kernel = phi_K3_halo_smooth(P_full, uf, lo, hi, np.full(3, tau),
                                         np.full(3, gamma), np.full(3, 1.0), h)
        diff = float(np.max(np.abs(Pn - Pn_kernel[lo:hi, lo:hi, lo:hi])))
        print(f"  n_bins={nb:3d}: {wall:5.1f}s  |CMM-kernel|_inf={diff:.3e}",
              flush=True)
    np.save(f"{OUT}/Pn_cmm_fast_nbins128.npy", Pn)

    # ---- Iteration test ----
    print("\n=== iteration from kernel FP ===", flush=True)
    P_inner = P_inner_kernel.copy()
    for it in range(8):
        P_full[lo:hi, lo:hi, lo:hi] = P_inner
        t0 = time.time()
        Pn = cmm_phi_inverted(P_full, uf, lo, hi, tau, gamma, n_bins=128)
        wall = time.time() - t0
        res = float(np.max(np.abs(Pn - P_inner)))
        # damped update
        omega = 0.5
        P_inner = (1 - omega)*P_inner + omega*Pn
        print(f"  iter {it}: res_inf={res:.3e}  ({wall:.0f}s)", flush=True)
    np.save(f"{OUT}/P_cmm_after_8iter.npy", P_inner)
    print("\nsaved P_cmm_after_8iter.npy")


if __name__ == "__main__":
    main()
