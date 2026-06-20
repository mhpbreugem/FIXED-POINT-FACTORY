"""Method C: monotone smooth parameterization (numba-fied).
P = sigmoid(L), L = base + triple integral of rho^2.
Monotone in each axis by construction.
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
from itertools import combinations_with_replacement, permutations
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u
from lin_cdf_richardson import phi_lin_richardson


G = 7
D_RHO = 3       # polynomial degree of rho per axis (low for speed)
U_MAX = 2.33
N_QUAD = 8      # GL quad per axis for the triple integral
GAMMA = 100.0


def symmetric_indices(d):
    return list(combinations_with_replacement(range(d+1), 3))


def expand_sym_coefs(c_sym, d):
    C = np.zeros((d+1, d+1, d+1))
    for i, t in enumerate(symmetric_indices(d)):
        v = c_sym[i]
        for p in set(permutations(t)):
            C[p[0], p[1], p[2]] = v
    return C


@njit(cache=True)
def _chebval3d_inplace(C, xi1, xi2, xi3, deg):
    """Evaluate Cheby tensor C of shape (deg+1, deg+1, deg+1) at (xi1, xi2, xi3)."""
    T1 = np.empty(deg+1); T2 = np.empty(deg+1); T3 = np.empty(deg+1)
    T1[0] = 1.0; T2[0] = 1.0; T3[0] = 1.0
    if deg >= 1:
        T1[1] = xi1; T2[1] = xi2; T3[1] = xi3
        for i in range(1, deg):
            T1[i+1] = 2*xi1*T1[i] - T1[i-1]
            T2[i+1] = 2*xi2*T2[i] - T2[i-1]
            T3[i+1] = 2*xi3*T3[i] - T3[i-1]
    s = 0.0
    for i in range(deg+1):
        for j in range(deg+1):
            for k in range(deg+1):
                s += C[i, j, k] * T1[i] * T2[j] * T3[k]
    return s


@njit(cache=True, parallel=True)
def _build_P_numba(C, base, u_grid, gl_nodes, gl_weights, deg, U, P_out):
    """For each cube cell (i, j, k): integrate rho^2 over (-U, u_i) x ... x (-U, u_k).
    L = base + integral; P = sigmoid(L)."""
    G_loc = u_grid.size
    n_q = gl_nodes.size
    for i in prange(G_loc):
        u_i = u_grid[i]
        h1 = (u_i - (-U)) * 0.5; m1 = ((-U) + u_i) * 0.5
        for j in range(G_loc):
            u_j = u_grid[j]
            h2 = (u_j - (-U)) * 0.5; m2 = ((-U) + u_j) * 0.5
            for k in range(G_loc):
                u_k = u_grid[k]
                h3 = (u_k - (-U)) * 0.5; m3 = ((-U) + u_k) * 0.5
                integral = 0.0
                for a in range(n_q):
                    s1 = m1 + h1 * gl_nodes[a]
                    w1 = h1 * gl_weights[a]
                    xi1 = s1 / U
                    for b in range(n_q):
                        s2 = m2 + h2 * gl_nodes[b]
                        w2 = h2 * gl_weights[b]
                        xi2 = s2 / U
                        for c in range(n_q):
                            s3 = m3 + h3 * gl_nodes[c]
                            w3 = h3 * gl_weights[c]
                            xi3 = s3 / U
                            r = _chebval3d_inplace(C, xi1, xi2, xi3, deg)
                            integral += w1 * w2 * w3 * r * r
                L = base + integral
                if L > 30: P_out[i, j, k] = 1.0
                elif L < -30: P_out[i, j, k] = 0.0
                else: P_out[i, j, k] = 1.0 / (1.0 + np.exp(-L))


def build_P_from_rho(c_sym, base, u_grid, deg=D_RHO):
    C = expand_sym_coefs(c_sym, deg)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(N_QUAD)
    P_out = np.empty((u_grid.size,)*3)
    _build_P_numba(C, base, u_grid, gl_nodes, gl_weights, deg, U_MAX, P_out)
    return P_out


def F_residual(params, tau, u_grid, p_grid, hs=(0.5, 0.4, 0.3, 0.2),
                  d_rho=D_RHO, gamma=GAMMA):
    n_sym = len(symmetric_indices(d_rho))
    c_sym = params[:n_sym]
    base = params[n_sym]
    P = build_P_from_rho(c_sym, base, u_grid, deg=d_rho)
    P_new = phi_lin_richardson(P, u_grid, hs=hs, gamma=gamma, tau=tau,
                                       G_p=p_grid.size, NQK=16, p_grid=p_grid)
    return (P_new - P).ravel()


if __name__ == "__main__":
    tau = 1.0
    print(f"=== Method C smooth-monotone, gamma={GAMMA}, tau={tau} ===")
    n_sym = len(symmetric_indices(D_RHO))
    print(f"d_rho={D_RHO}, # sym coefs = {n_sym}, total unknowns = {n_sym+1}")
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(121)

    print("JIT warmup...", flush=True); t0 = time.time()
    P_test = build_P_from_rho(np.array([0.1] + [0.0]*(n_sym-1)), 0.0, u_grid)
    print(f"  done {time.time()-t0:.1f}s, P range [{P_test.min():.3f}, {P_test.max():.3f}]",
          flush=True)
    t0 = time.time()
    P_test = build_P_from_rho(np.array([0.1] + [0.0]*(n_sym-1)), 0.0, u_grid)
    t_pbuild = time.time() - t0
    print(f"  build_P: {t_pbuild*1000:.0f}ms per call", flush=True)

    params0 = np.zeros(n_sym + 1)
    params0[0] = 0.1
    print("\nInitial F:")
    t0 = time.time()
    F0 = F_residual(params0, tau, u_grid, p_grid)
    print(f"  |F0|_inf={np.max(np.abs(F0)):.3e} ({time.time()-t0:.1f}s)", flush=True)

    print("\nRunning TRF...", flush=True)
    t0 = time.time()
    res = least_squares(F_residual, params0,
                              args=(tau, u_grid, p_grid),
                              jac="2-point", method="trf",
                              max_nfev=200, ftol=1e-12, xtol=1e-12,
                              verbose=2)
    print(f"  done in {time.time()-t0:.0f}s, status={res.status}, "
          f"|F|_inf={np.max(np.abs(res.fun)):.3e}", flush=True)

    c_sym = res.x[:n_sym]; base = res.x[-1]
    P = build_P_from_rho(c_sym, base, u_grid)
    print(f"\nFinal P range: [{P.min():.6f}, {P.max():.6f}]")
    for ax in range(3):
        dd = np.diff(P, axis=ax)
        print(f"  axis{ax}: min dP/du = {dd.min():.3e}, # violations = {int((dd<0).sum())}")
    np.savez(f"/tmp/dd_k3_method_c_smooth_tau{tau}_d{D_RHO}.npz",
                params=res.x, P=P, F=float(np.max(np.abs(res.fun))),
                tau=tau, gamma=GAMMA, d_rho=D_RHO)
