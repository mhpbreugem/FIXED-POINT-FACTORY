"""Test the weak-fixed-point bridge proposal.

The conjecture:
   The kernel-h->0 limit P* is NOT a fixed point of the pointwise
   strict-h=0 operator Phi_0, but IS a fixed point of the WEAK
   (surface-integrated) operator:
       Phi_0^weak(P)(p) := average over {P=p} of clear(mu(u; P)),
   weighted by the prior density w(u) = 0.5*(prod_f0 + prod_f1).

Two empirical tests:
   A. Compute the per-surface integrated residual at the kernel warm start.
      If it is MUCH SMALLER than the pointwise max residual (~0.28),
      the weak FP hypothesis is supported.

   B. Run the moving-mesh solver minimizing the integrated residual +
      a Tikhonov anchor on vertex positions.  If the solver converges
      to a small integrated residual with nontrivial deficit (different
      from P_FR), we have a CONSTRUCTIVE strict-h=0 weak-FP.

This is the bridge between the kernel-h->0 limit and a strict-h=0
formulation: they are the same object under the weak/integrated reading.
"""
import os, sys, time, json
import math
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from cmm_stage3b_jit import vertex_residual_jit
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.demand import clear_crra
from scipy.optimize import newton_krylov, least_squares
try: from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

TAU, GAMMA = 2.0, 0.01
G = 8
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/weak_fp_test'
os.makedirs(OUT, exist_ok=True)


def make_PFR(uf, tau):
    U1, U2, U3 = np.meshgrid(uf, uf, uf, indexing='ij')
    return 1.0 / (1.0 + np.exp(-tau*(U1+U2+U3)))


def solve_kernel(Gi, tau, gamma):
    pad = 2; UMAX = 4.0; C = 0.45; K = 3
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P0 = P_full[lo:hi, lo:hi, lo:hi].ravel().copy()
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - x.reshape(Gi, Gi, Gi)).ravel()
    try:
        x = newton_krylov(F, P0, f_tol=1e-12, maxiter=80, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    return x.reshape(Gi, Gi, Gi), uf, lo, hi


def build_mesh(P_full, uf, lo, hi, n_levels=10):
    p_flat = P_full[lo:hi, lo:hi, lo:hi].ravel()
    qs = np.linspace(0.05, 0.95, n_levels)
    p_levels = np.unique(np.quantile(p_flat, qs))
    surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
    V = []; F = []; p_kept = []
    for ip, s in enumerate(surfs):
        if s is None or s[0].shape[0] == 0: continue
        V.append(s[0].copy()); F.append(s[1].astype(np.int64))
        p_kept.append(p_levels[ip])
    return V, F, np.asarray(p_kept)


def f_signal(u, v, tau):
    s = math.sqrt(tau/(2*math.pi))
    mean = 0.5 if v == 1 else -0.5
    return s*math.exp(-0.5*tau*(u-mean)**2)


def w_density(u):
    """Prior density on signals (mixture of f_0, f_1 product)."""
    p0 = f_signal(u[0], 0, TAU)*f_signal(u[1], 0, TAU)*f_signal(u[2], 0, TAU)
    p1 = f_signal(u[0], 1, TAU)*f_signal(u[1], 1, TAU)*f_signal(u[2], 1, TAU)
    return 0.5*(p0 + p1)


def vertex_pointwise_residual(V_all_concat, n_verts_arr, F_concat, n_faces_arr, p_levels):
    """Per-vertex pointwise residual = p_clear - p_m. Reuses Stage3b JIT path."""
    r = vertex_residual_jit(V_all_concat, n_verts_arr, F_concat, n_faces_arr,
                              np.asarray(p_levels), TAU, GAMMA)
    return r


def face_areas(V, F):
    """Per-face area (1/2 |cross|)."""
    A = np.zeros(F.shape[0])
    for i, tri in enumerate(F):
        v0, v1, v2 = V[tri[0]], V[tri[1]], V[tri[2]]
        A[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return A


def vertex_areas(V, F):
    """Per-vertex barycentric area (sum / 3 of incident face areas)."""
    fa = face_areas(V, F)
    va = np.zeros(V.shape[0])
    for i, tri in enumerate(F):
        for k in tri:
            va[k] += fa[i] / 3
    return va


def per_surface_weighted_residual(V, F, r_local, p_m):
    """Integrated residual over a surface: sum_v r_v * w(v) * A(v),
    normalized by sum_v w(v) * A(v).  Returns the average and the
    weighted sum (= unnormalized integral)."""
    if V.shape[0] == 0: return 0.0, 0.0
    va = vertex_areas(V, F)
    w = np.array([w_density(V[i]) for i in range(V.shape[0])])
    wA = w * va
    num = float(np.sum(r_local * wA))
    den = float(np.sum(wA))
    return (num/den if den > 0 else 0.0), num


def test_A_kernel_warmstart(P_full, uf, lo, hi):
    """Test A: does the kernel solution satisfy the weak FP eqn?"""
    V_list, F_list, p_levels = build_mesh(P_full, uf, lo, hi, n_levels=10)
    M = len(p_levels)

    # Build concatenated arrays for the JIT residual
    n_v = np.array([v.shape[0] for v in V_list])
    n_f = np.array([f.shape[0] for f in F_list])
    V_concat = np.concatenate(V_list)
    F_concat = np.concatenate(F_list)

    # Pointwise per-vertex residual
    r_all = vertex_pointwise_residual(V_concat, n_v, F_concat, n_f, p_levels)

    # Split per surface and compute per-surface integrated residual
    v_off = np.concatenate(([0], np.cumsum(n_v)))
    print(f"\n--- Test A: weak FP residual at kernel solution ---", flush=True)
    print(f"{'surf':>5} {'p_m':>7} {'max|r_pt|':>11} {'med|r_pt|':>11} "
          f"{'avg(w*r)/avg(w)':>17} {'normalized':>11}", flush=True)
    per_surf = []
    for m in range(M):
        r_loc = r_all[v_off[m]:v_off[m+1]]
        max_r = float(np.max(np.abs(r_loc)))
        med_r = float(np.median(np.abs(r_loc)))
        avg_r, wsum = per_surface_weighted_residual(V_list[m], F_list[m], r_loc, p_levels[m])
        # also normalized by surface area
        va = vertex_areas(V_list[m], F_list[m])
        norm_r = float(np.sum(r_loc * va) / np.sum(va)) if np.sum(va) > 0 else 0
        per_surf.append(dict(p_m=float(p_levels[m]), max_r=max_r, med_r=med_r,
                              avg_w_r=float(avg_r), norm_r=norm_r,
                              n_v=int(n_v[m])))
        print(f"{m:>5} {p_levels[m]:>7.3f} {max_r:>11.3e} {med_r:>11.3e} "
              f"{avg_r:>17.3e} {norm_r:>11.3e}", flush=True)
    # Summary: pointwise sup vs weak sup
    sup_pt = max(s['max_r'] for s in per_surf)
    sup_weak = max(abs(s['avg_w_r']) for s in per_surf)
    print(f"\n  sup pointwise residual:    {sup_pt:.4e}", flush=True)
    print(f"  sup weak (integrated) residual: {sup_weak:.4e}", flush=True)
    print(f"  ratio sup_weak / sup_pt:   {sup_weak/sup_pt:.4f}", flush=True)
    print(f"\n  If sup_weak << sup_pt:  WEAK FP hypothesis SUPPORTED.", flush=True)
    print(f"  If sup_weak ~ sup_pt:   NOT a weak FP either.", flush=True)
    return dict(per_surface=per_surf, sup_pointwise=sup_pt,
                sup_weak=sup_weak, ratio=sup_weak/sup_pt if sup_pt > 0 else 0)


def main():
    print(f"=== Weak fixed-point test at (tau={TAU}, gamma={GAMMA}), G={G} ===\n",
          flush=True)
    print("Solving kernel at G=8...", flush=True)
    P_kernel, uf, lo, hi = solve_kernel(G, TAU, GAMMA)
    P_FR = make_PFR(uf, TAU)[lo:hi, lo:hi, lo:hi]
    print(f"  kernel solved; ||P_kernel - P_FR||_inf = {np.max(np.abs(P_kernel - P_FR)):.4f}",
          flush=True)

    # Build kernel P_full
    P_full = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_kernel

    # TEST A: pointwise vs weak residual at the kernel solution
    result_kernel = test_A_kernel_warmstart(P_full, uf, lo, hi)

    # TEST B: same at P_FR (should be zero for both pointwise and weak)
    P_full_FR = init_no_learning_K3(uf, np.full(3, TAU), np.full(3, GAMMA), np.full(3, 1.0))
    P_full_FR[lo:hi, lo:hi, lo:hi] = P_FR
    print(f"\n\n=== Test A2: P_FR for comparison ===", flush=True)
    result_FR = test_A_kernel_warmstart(P_full_FR, uf, lo, hi)

    json.dump(dict(kernel_test=result_kernel, FR_test=result_FR,
                    tau=TAU, gamma=GAMMA, G=G),
              open(f"{OUT}/weak_fp_test.json", 'w'), indent=2, default=str)
    print(f"\n\nSaved {OUT}/weak_fp_test.json", flush=True)

    # Verdict
    print(f"\n\n=== VERDICT ===", flush=True)
    print(f"Kernel: sup pointwise = {result_kernel['sup_pointwise']:.3e}, "
          f"sup weak = {result_kernel['sup_weak']:.3e}, "
          f"ratio = {result_kernel['ratio']:.4f}", flush=True)
    print(f"P_FR:   sup pointwise = {result_FR['sup_pointwise']:.3e}, "
          f"sup weak = {result_FR['sup_weak']:.3e}, "
          f"ratio = {result_FR['ratio']:.4f}", flush=True)
    if result_kernel['ratio'] < 0.1:
        print(f"\n  >>> Kernel is APPROXIMATELY a weak FP (ratio < 0.1)", flush=True)
        print(f"      Bridge is empirically supported.", flush=True)
    else:
        print(f"\n  >>> Kernel is NOT a weak FP either (ratio >= 0.1)", flush=True)
        print(f"      Weak-FP formulation does NOT bridge the kernel limit", flush=True)
        print(f"      to a strict-h=0 fixed point.  Need a different bridge.", flush=True)


if __name__ == "__main__":
    main()
