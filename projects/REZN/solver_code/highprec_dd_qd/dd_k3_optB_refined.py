"""Option B+ (refined): densify (u_a, u_b) via interpolation before
building empirical CDF. Resample until the gap between sorted P values
is below a threshold, then average upper/lower step-CDF as the smooth
density.

Algorithm:
  1. Start with cube cell samples (G^2 per slice).
  2. Bilinear-interpolate P onto a refined uniform grid in (u_a, u_b)
     with refinement factor n_sub (so G_refined = (G-1)*n_sub + 1).
  3. Compute f_v at refined points.
  4. Sort by P, accumulate cumulative weighted F_v.
  5. Build sandwich: F_lo (step from below) and F_hi (step from above);
     average gives a smooth estimator. Or PCHIP through sorted (P, F) as before.
  6. Differentiate for density a_v(p), then mu via Bayes ratio.

The bandwidth from upper-lower sandwich tightens monotonically as n_sub grows.
"""
import os, sys, time
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit, prange
from cheby_numba import f_signal_jit, crra_clear_jit
from dd_k3_optB_numba import _pchip_slopes, _pchip_deriv_eval, _interp_mu


@njit(cache=True)
def _bilin_slice(P_slice, u_grid, refined_u, refined_P):
    """Bilinear interpolation of P_slice onto refined uniform grid."""
    G = u_grid.size; Gr = refined_u.size
    for ia in range(Gr):
        ua = refined_u[ia]
        if ua <= u_grid[0]: ig = 0; wa = 0.0
        elif ua >= u_grid[G-1]: ig = G-2; wa = 1.0
        else:
            lo = 0; hi = G-1
            while hi - lo > 1:
                mid = (lo+hi)//2
                if u_grid[mid] <= ua: lo = mid
                else: hi = mid
            ig = lo; wa = (ua - u_grid[lo])/(u_grid[hi]-u_grid[lo])
        for ib in range(Gr):
            ub = refined_u[ib]
            if ub <= u_grid[0]: jg = 0; wb = 0.0
            elif ub >= u_grid[G-1]: jg = G-2; wb = 1.0
            else:
                lo = 0; hi = G-1
                while hi - lo > 1:
                    mid = (lo+hi)//2
                    if u_grid[mid] <= ub: lo = mid
                    else: hi = mid
                jg = lo; wb = (ub - u_grid[lo])/(u_grid[hi]-u_grid[lo])
            refined_P[ia, ib] = ((1-wa)*(1-wb)*P_slice[ig,   jg  ]
                                    + wa  *(1-wb)*P_slice[ig+1, jg  ]
                                    + (1-wa)*wb*P_slice[ig,   jg+1]
                                    + wa  *wb*P_slice[ig+1, jg+1])


@njit(cache=True)
def build_mu_optB_refined(P_vals, u_grid, p_grid, tau, refined_u, w_trap_r,
                              f0_r, f1_r, f0_u, f1_u, mu_table):
    """Option B+ mu-table: bilinear-refine each slice, sort, CDF, PCHIP density.
    refined_u: precomputed refined u grid (size Gr)
    w_trap_r: trapezoidal weights on refined grid
    f0_r, f1_r: f_v at refined u grid
    f0_u, f1_u: f_v at original u grid (for f_v(u_k))"""
    G = u_grid.size; G_p = p_grid.size; Gr = refined_u.size
    GrSq = Gr * Gr
    P_arr = np.empty(GrSq); w_arr = np.empty(GrSq)
    f0_arr = np.empty(GrSq); f1_arr = np.empty(GrSq)
    refined_P = np.empty((Gr, Gr))
    Pu = np.empty(GrSq); F0u = np.empty(GrSq); F1u = np.empty(GrSq)
    m0 = np.empty(GrSq); m1 = np.empty(GrSq)
    for k_node in range(G):
        # Extract slice and bilinear-refine
        slice_P = np.empty((G, G))
        for i in range(G):
            for j in range(G):
                slice_P[i, j] = P_vals[k_node, i, j]
        _bilin_slice(slice_P, u_grid, refined_u, refined_P)
        s = 0
        for ia in range(Gr):
            for ib in range(Gr):
                P_arr[s] = refined_P[ia, ib]
                w_arr[s] = w_trap_r[ia] * w_trap_r[ib]
                f0_arr[s] = f0_r[ia] * f0_r[ib]
                f1_arr[s] = f1_r[ia] * f1_r[ib]
                s += 1
        # Sort + cumulative + unique
        order = np.argsort(P_arr)
        # BC-augmented: prepend (P=0, F_v=0) and append (P=1, F_v=total)
        Pu[0] = 0.0; F0u[0] = 0.0; F1u[0] = 0.0
        n_u = 1; cum0 = 0.0; cum1 = 0.0
        prev_P = -1.0
        for s in range(GrSq):
            ii = order[s]
            cum0 += w_arr[ii] * f0_arr[ii]
            cum1 += w_arr[ii] * f1_arr[ii]
            P_cur = P_arr[ii]
            if P_cur > prev_P + 1e-15:
                # ensure not crowding the lower BC at P=0
                if P_cur > 1e-15:
                    Pu[n_u] = P_cur; F0u[n_u] = cum0; F1u[n_u] = cum1
                    n_u += 1
                else:
                    F0u[0] = cum0; F1u[0] = cum1   # absorb into BC
                prev_P = P_cur
            else:
                F0u[n_u-1] = cum0; F1u[n_u-1] = cum1
        # Append upper BC at P=1 with total cumulative
        if Pu[n_u-1] < 1.0 - 1e-15:
            Pu[n_u] = 1.0; F0u[n_u] = cum0; F1u[n_u] = cum1
            n_u += 1
        else:
            F0u[n_u-1] = cum0; F1u[n_u-1] = cum1
        if n_u < 4:
            for ip in range(G_p): mu_table[ip, k_node] = 0.5
            continue
        _pchip_slopes(Pu, F0u, n_u, m0)
        _pchip_slopes(Pu, F1u, n_u, m1)
        f0k = f0_u[k_node]; f1k = f1_u[k_node]
        for ip in range(G_p):
            p = p_grid[ip]
            a0 = _pchip_deriv_eval(Pu, F0u, m0, n_u, p)
            a1 = _pchip_deriv_eval(Pu, F1u, m1, n_u, p)
            den = f0k*a0 + f1k*a1
            if den > 1e-300:
                m = f1k*a1 / den
                if m < 1e-9: m = 1e-9
                elif m > 1.0-1e-9: m = 1.0-1e-9
                mu_table[ip, k_node] = m
            else:
                mu_table[ip, k_node] = 0.5


def build_refined_helpers(u_grid, tau, n_sub):
    """Build refined u-grid, trap weights, f_v on refined."""
    G = u_grid.size
    Gr = (G-1) * n_sub + 1
    refined_u = np.empty(Gr)
    idx = 0
    for i in range(G-1):
        for k in range(n_sub):
            t = k / n_sub
            refined_u[idx] = (1-t)*u_grid[i] + t*u_grid[i+1]; idx += 1
    refined_u[idx] = u_grid[-1]
    du = np.diff(refined_u)
    w_trap = np.empty(Gr); w_trap[0] = 0.5*du[0]; w_trap[-1] = 0.5*du[-1]
    w_trap[1:-1] = 0.5*(du[:-1] + du[1:])
    f0_r = np.array([f_signal_jit(u, 0, tau) for u in refined_u])
    f1_r = np.array([f_signal_jit(u, 1, tau) for u in refined_u])
    f0_u = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
    f1_u = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
    return refined_u, w_trap, f0_r, f1_r, f0_u, f1_u


@njit(cache=True, parallel=True)
def phi_optB_refined(P_vals, u_grid, p_grid, tau, gamma, refined_u, w_trap_r,
                          f0_r, f1_r, f0_u, f1_u, P_new):
    G = u_grid.size; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    build_mu_optB_refined(P_vals, u_grid, p_grid, tau, refined_u, w_trap_r,
                              f0_r, f1_r, f0_u, f1_u, mu_table)
    eps = 1e-9
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                pc = P_vals[i, j, k]
                if pc < eps: pc = eps
                elif pc > 1.0-eps: pc = 1.0-eps
                mu0 = _interp_mu(mu_table, pc, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, pc, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, pc, p_grid, k, G_p)
                if mu0 < eps: mu0 = eps
                elif mu0 > 1.0-eps: mu0 = 1.0-eps
                if mu1 < eps: mu1 = eps
                elif mu1 > 1.0-eps: mu1 = 1.0-eps
                if mu2 < eps: mu2 = eps
                elif mu2 > 1.0-eps: mu2 = 1.0-eps
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)


def phi(P, u_grid, p_grid, tau, gamma, n_sub=4):
    refined_u, w_trap_r, f0_r, f1_r, f0_u, f1_u = build_refined_helpers(
        u_grid, tau, n_sub)
    P_new = np.empty_like(P)
    phi_optB_refined(P, u_grid, p_grid, float(tau), float(gamma),
                          refined_u, w_trap_r, f0_r, f1_r, f0_u, f1_u, P_new)
    return P_new


# ============================ Test convergence vs strict-h=0 ============================
if __name__ == "__main__":
    import glob
    from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                      make_p_grid, make_gl_for_u)
    G = 11; G_p = 121
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    print("JIT warmup...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    phi(P0, u_grid, p_grid, 1.0, 1.0, n_sub=2)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    print(f"\nRefinement convergence test (n_sub=1,2,4,8,16) at saved R4 FPs vs strict-h=0",
          flush=True)
    print(f"{'cell':30s} {'n_sub':>6s} {'max|d|':>10s} {'med|d|':>10s} {'Gr':>5s} {'wall':>6s}",
          flush=True)
    for fp in fp_files[:4]:
        d = np.load(fp)
        if d["mu_hi"].shape[0] != G: continue
        P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
        gamma = float(d["gamma"]); tau = float(d["tau"])
        label = f"g{gamma:.4g}_t{tau:.4f}"
        # Strict-h=0 reference
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u, gl_du,
                                                       tau, G, 16)
        for n_sub in [1, 2, 4, 8, 16]:
            refined_u, w_trap_r, f0_r, f1_r, f0_u, f1_u = build_refined_helpers(
                u_grid, tau, n_sub)
            Gr = refined_u.size
            mu_buf = np.empty((G_p, G))
            t0 = time.time()
            for _ in range(3):
                build_mu_optB_refined(P, u_grid, p_grid, tau, refined_u, w_trap_r,
                                          f0_r, f1_r, f0_u, f1_u, mu_buf)
            wall = (time.time()-t0)/3
            d_max = float(np.max(np.abs(mu_buf - mu_strict)))
            d_med = float(np.median(np.abs(mu_buf - mu_strict)))
            print(f"{label:30s} {n_sub:>6d} {d_max:>10.3e} {d_med:>10.3e} {Gr:>5d} {wall*1000:>5.0f}ms",
                  flush=True)
        print()
