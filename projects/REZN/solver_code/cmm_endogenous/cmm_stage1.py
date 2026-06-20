"""Co-Area Mesh Method (CMM), Stage 1: strict h=0 Phi evaluation.

Idea (user's): rather than sampling a price surface on a fixed (u1,u2,u3)
cube and using a kernel K_h to approximate the level-set delta, place
the GRID POINTS THEMSELVES on level sets {P=p_m}. The integration over
{P=p_m} becomes arclength integration along a 1D curve — no kernel, no
bandwidth, h=0 by construction. The unknowns become the curve VERTEX
POSITIONS rather than cube VALUES.

Stage 1 deliverable: forward operator evaluation only. Given a converged
cube fixed point P*, extract level surfaces by marching cubes, then
compute Phi(P*) via the CMM machinery. Validate against the kernel-
smooth operator's residual at the same P*. Success = the CMM residual
is no worse than the kernel residual (we are h=0, but the surface is
only as accurate as the input P*).

Stage 2 (later): make vertex positions the Newton unknowns and converge
the moving mesh to a strict h=0 fixed point. Stage 3: validate against
the immortal anchor.

Geometry: surface S_m = {(u1,u2,u3) in R^3 : P(u) = p_m}. Slice by
{u_k=U} (own-signal hyperplane) -> 1D curve C in (u_j,u_l). The slice
of a 2D triangle mesh by a plane is closed-form (linear interpolation
along the 3 edges of each crossing triangle, gives 0 or 2 intersection
points -> a line segment per crossing triangle). Stitch segments into
a piecewise-linear curve. Arclength integral over that curve gives
A_v = ∫ f_v(u_j) f_v(u_l) dsigma at h=0.

For Stage 1, we use the cube P* itself + skimage marching_cubes to get
the surfaces. This is NOT yet "endogenous" — the surfaces are derived
from a cube. But it isolates the slice-integration math from the
mesh-evolution math, which is what we need to validate first.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.signals import f_signal as _fsig
from reznsrc.demand import clear_crra as _clear

K = 3; pad = 2; UMAX = 4.0; C = 0.45
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cmm_endogenous'


def build_grid(Gi):
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def extract_surfaces_marching_cubes(P_full, uf, p_levels):
    """For each target price p_m, extract S_m = {P = p_m} as a triangle
    mesh. Uses skimage.measure.marching_cubes, returns list of (verts,
    faces) tuples in (u1,u2,u3) physical coords (not voxel indices)."""
    from skimage import measure
    Gf = P_full.shape[0]
    out = []
    for p in p_levels:
        try:
            verts_vox, faces, *_ = measure.marching_cubes(P_full, level=p)
        except Exception as e:
            out.append(None); continue
        # voxel coords are float indices into uf; convert to physical u
        verts = np.empty_like(verts_vox)
        for d in range(3):
            verts[:, d] = np.interp(verts_vox[:, d], np.arange(Gf), uf)
        out.append((verts, faces))
    return out


def slice_mesh_by_hyperplane(verts, faces, axis, U):
    """Intersect the 2D triangle mesh with the hyperplane {u_axis = U}.
    Returns a list of (P_a, P_b) line segments in the 2D other-coord
    plane (orthogonal axes), suitable for arclength integration of any
    function f(other_coords).

    For each triangle, classify vertices as below/on/above the
    hyperplane. A triangle with vertices straddling the plane is cut
    along two of its edges, yielding one line segment.
    """
    z = verts[:, axis] - U
    other = [d for d in range(3) if d != axis]
    segments = []
    for tri in faces:
        zt = z[tri]
        s = np.sign(zt)
        # If all same sign and none zero, no intersection.
        nz = np.sum(s == 0)
        np_ = np.sum(s > 0); nm = np.sum(s < 0)
        if (np_ == 3 or nm == 3):
            continue
        # Find the two edges that are crossed.
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            zi, zj = zt[i], zt[j]
            if zi == 0 and zj == 0:
                # whole edge lies on the plane
                pts.append(verts[tri[i]][other])
                pts.append(verts[tri[j]][other])
            elif zi == 0:
                pts.append(verts[tri[i]][other])
            elif zi * zj < 0:
                t = zi / (zi - zj)
                pts.append((1 - t) * verts[tri[i]][other] + t * verts[tri[j]][other])
        if len(pts) >= 2:
            # take the first two distinct points
            P = np.array(pts)
            # dedup
            keep = [P[0]]
            for p in P[1:]:
                if all(np.linalg.norm(p - q) > 1e-12 for q in keep):
                    keep.append(p)
                if len(keep) == 2: break
            if len(keep) == 2:
                segments.append(np.stack(keep))
    return segments


def stitch_segments(segs, tol=1e-9):
    """Stitch line segments into ordered polylines by greedy chaining.
    Returns list of polylines (each (n,2) array of points)."""
    if not segs: return []
    rem = [s.copy() for s in segs]
    chains = []
    while rem:
        chain = [rem[0][0], rem[0][1]]
        rem.pop(0)
        # try extending forward
        extended = True
        while extended:
            extended = False
            for k, s in enumerate(rem):
                if np.linalg.norm(s[0] - chain[-1]) < tol:
                    chain.append(s[1]); rem.pop(k); extended = True; break
                if np.linalg.norm(s[1] - chain[-1]) < tol:
                    chain.append(s[0]); rem.pop(k); extended = True; break
        # try extending backward
        extended = True
        while extended:
            extended = False
            for k, s in enumerate(rem):
                if np.linalg.norm(s[0] - chain[0]) < tol:
                    chain.insert(0, s[1]); rem.pop(k); extended = True; break
                if np.linalg.norm(s[1] - chain[0]) < tol:
                    chain.insert(0, s[0]); rem.pop(k); extended = True; break
        chains.append(np.array(chain))
    return chains


def arclength_int_f(chain, fa, fb):
    """∫_chain f_a(coord_a) * f_b(coord_b) dsigma using trapezoid on
    arclength. chain: (n,2) ordered points; fa, fb: callable on coord."""
    if len(chain) < 2: return 0.0
    n = len(chain)
    seg_lens = np.linalg.norm(np.diff(chain, axis=0), axis=1)  # (n-1,)
    # arclength accumulator
    s = np.concatenate(([0.0], np.cumsum(seg_lens)))
    vals = np.array([fa(p[0]) * fb(p[1]) for p in chain])
    # trapezoid
    return float(0.5 * np.sum(seg_lens * (vals[:-1] + vals[1:])))


def cmm_evidence(P_full, uf, p_target, k_own, U_own, tau):
    """Strict h=0 evidence integrals A_0, A_1 for agent k_own at
    own-signal U_own, target price p_target. Returns (A_0, A_1).

    Note: extracts the surface S_{p_target} on the fly from P_full
    (Stage 1 — derived mesh). Slow but correct."""
    surf = extract_surfaces_marching_cubes(P_full, uf, [p_target])[0]
    if surf is None:
        return 0.0, 0.0
    verts, faces = surf
    segs = slice_mesh_by_hyperplane(verts, faces, k_own, U_own)
    chains = stitch_segments(segs)
    A0 = 0.0; A1 = 0.0
    for chain in chains:
        A0 += arclength_int_f(chain, lambda u: _fsig(u, 0, tau),
                                       lambda u: _fsig(u, 0, tau))
        A1 += arclength_int_f(chain, lambda u: _fsig(u, 1, tau),
                                       lambda u: _fsig(u, 1, tau))
    return A0, A1


def phi_cmm_pointwise(P_full, uf, lo, hi, tau, gamma, k_inner, m_inner, l_inner):
    """One cell's Phi update via CMM. Returns the new P at this inner
    cell. Slow (re-extracts surface each call) — Stage 1 reference impl."""
    i, j, l = lo + k_inner, lo + m_inner, lo + l_inner
    p = P_full[i, j, l]
    mu = np.empty(3)
    for k_own, idx in [(0, i), (1, j), (2, l)]:
        U = uf[idx]
        A0, A1 = cmm_evidence(P_full, uf, p, k_own, U, tau)
        f0 = _fsig(U, 0, tau); f1 = _fsig(U, 1, tau)
        num = f1 * A1; den = f0 * A0 + num
        if den <= 0: mu[k_own] = 0.5
        else:
            m = num / den
            mu[k_own] = max(min(m, 1 - 1e-12), 1e-12)
    return _clear(mu, np.full(3, gamma), np.full(3, 1.0))


def main():
    """Stage 1 validation: load the certified emin15 fixed point at
    (tau=2, gamma=0.098, G=21), evaluate Phi at one inner cell via:
      (a) the kernel-smooth operator (h = 0.45*sqrt(du))
      (b) the strict h=0 CMM (this script)
    Compare. If CMM gives a result close to the cell's own price (i.e.
    near machine eps residual since P* is a fixed point of the kernel
    operator only up to O(h)), we have evidence the slice-integration
    math is correct."""
    os.makedirs(OUT, exist_ok=True)
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    h = C * np.sqrt(du)

    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner

    # Kernel-smooth Phi
    t0 = time.time()
    Pn_kernel = phi_K3_halo_smooth(P_full, uf, lo, hi, np.full(3, tau),
                                    np.full(3, gamma), np.full(3, 1.0), h)
    t_kernel = time.time() - t0
    F_kernel = float(np.max(np.abs(Pn_kernel[lo:hi, lo:hi, lo:hi] - P_inner)))
    print(f"Kernel Phi: F = {F_kernel:.3e} ({t_kernel:.2f}s)")

    # CMM at a few representative cells (don't sweep the whole cube — too slow at Stage 1)
    test_cells = [(10, 10, 10), (5, 10, 15), (8, 12, 9), (15, 5, 5)]
    cmm_results = []
    for (ki, mi, li) in test_cells:
        p_orig = float(P_inner[ki, mi, li])
        t0 = time.time()
        p_cmm = phi_cmm_pointwise(P_full, uf, lo, hi, tau, gamma, ki, mi, li)
        t = time.time() - t0
        p_kernel = float(Pn_kernel[lo + ki, lo + mi, lo + li])
        print(f"  cell ({ki},{mi},{li}): P*={p_orig:.6f}  "
              f"Phi_kernel={p_kernel:.6f}  Phi_cmm={p_cmm:.6f}  "
              f"|cmm-kernel|={abs(p_cmm-p_kernel):.3e}  ({t:.1f}s)")
        cmm_results.append(dict(cell=[ki,mi,li], P_star=p_orig,
                                Phi_kernel=p_kernel, Phi_cmm=p_cmm,
                                diff_cmm_kernel=abs(p_cmm-p_kernel),
                                diff_cmm_star=abs(p_cmm-p_orig), wall=t))
    json.dump(cmm_results, open(f"{OUT}/stage1_pointwise.json", "w"), indent=2)
    print(f"\nSaved -> {OUT}/stage1_pointwise.json")


if __name__ == "__main__":
    main()
