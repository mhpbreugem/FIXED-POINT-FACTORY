"""Cheby-h=0 lookup via analytic 1D root-finding on a fitted Chebyshev polynomial.

Plan:
  1. Take a saved FP P (cube on CDF-uniform grid).
  2. Project to a Chebyshev tensor polynomial by least-squares fit on Lobatto-like
     evaluation (we have arbitrary nodes -> use chebvander 3D fit).
  3. For each (p, u_k) lookup:
     - The 2D slice P(u_k, u_a, u_b) is a Cheby polynomial.
     - For each Gauss-Legendre u_a node, find roots in u_b of P_slice(u_a, u_b) = p
       via companion matrix of the 1D Cheby in u_b.
     - At each root, evaluate analytic dP/du_b = derivative of the Cheby in u_b.
     - Integral: sum f_v(u_a) f_v(u_b_root) / |dP/du_b| * w_a (GL weight on u_a).
  4. mu = f_1(u_k) A_1 / (f_0(u_k) A_0 + f_1(u_k) A_1).

Compare to strict-h=0 (POU + piecewise-linear roots) at the saved FPs.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)
from cheby_numba import f_signal_jit


G = 11
G_p = 121


def cheby_fit_3d(P_cube, u_grid, deg=None):
    """Least-squares fit of a 3D Cheby tensor polynomial to P_cube on u_grid.
    Returns (coefs[(deg+1, deg+1, deg+1)], u_max) where the polynomial is
    P(u1, u2, u3) = sum_{ijk} coefs[i,j,k] T_i(u1/u_max) T_j(u2/u_max) T_k(u3/u_max).
    """
    G = u_grid.size
    if deg is None: deg = G - 1
    u_max = max(abs(u_grid[0]), abs(u_grid[-1])) * 1.0
    xi = u_grid / u_max
    # Build per-axis Cheby Vandermonde (G x (deg+1))
    V = cheb.chebvander(xi, deg)
    # Fit by 3 successive 1D LS solves (per axis)
    # V_full = V_xi (kron) V_xi (kron) V_xi
    # We can do tensor solve: contract axis by axis.
    # axis 0: solve V @ A_0 = P along first axis for each (j, k)
    A0 = np.linalg.lstsq(V, P_cube.reshape(G, G*G), rcond=None)[0]
    A0 = A0.reshape(deg+1, G, G)
    # axis 1
    A0_t = A0.transpose(1, 0, 2)            # (G, deg+1, G)
    A1 = np.linalg.lstsq(V, A0_t.reshape(G, (deg+1)*G), rcond=None)[0]
    A1 = A1.reshape(deg+1, deg+1, G).transpose(1, 0, 2)
    # axis 2
    A1_t = A1.transpose(2, 0, 1)            # (G, deg+1, deg+1)
    A2 = np.linalg.lstsq(V, A1_t.reshape(G, (deg+1)*(deg+1)), rcond=None)[0]
    A2 = A2.reshape(deg+1, deg+1, deg+1).transpose(1, 2, 0)
    return A2, u_max


def cheby_eval_3d(coefs, u_max, u1, u2, u3):
    """Evaluate 3D Cheby polynomial at (u1, u2, u3)."""
    return cheb.chebval3d(u1/u_max, u2/u_max, u3/u_max, coefs)


def cheby_slice_2d(coefs, u_max, u_k):
    """Return 2D Cheby coefs for P(u_k, u_a, u_b) as a polynomial in (u_a, u_b)."""
    deg = coefs.shape[0] - 1
    xi_k = u_k / u_max
    # T_i(xi_k) for i=0..deg
    T_xi_k = np.zeros(deg+1)
    T_xi_k[0] = 1.0
    if deg >= 1: T_xi_k[1] = xi_k
    for i in range(1, deg):
        T_xi_k[i+1] = 2*xi_k*T_xi_k[i] - T_xi_k[i-1]
    # Sum_i T_i(xi_k) coefs[i, :, :]
    coefs_2d = np.einsum("i,ijk->jk", T_xi_k, coefs)
    return coefs_2d   # (deg+1, deg+1) in (u_a, u_b)


def cheby_deriv_2d_axis_b(coefs_2d, u_max):
    """Return 2D Cheby coefs of d/du_b of the input (a, b) coefs."""
    deg_b = coefs_2d.shape[1] - 1
    out = np.empty((coefs_2d.shape[0], deg_b))
    # Per-row chebder along axis 1
    for i in range(coefs_2d.shape[0]):
        out[i, :] = cheb.chebder(coefs_2d[i, :], 1) / u_max
    return out


def cheby_eval_1d_at_xi_a(coefs_2d, xi_a):
    """Evaluate 2D Cheby coefs at xi_a (along axis 0). Returns 1D Cheby coefs in u_b."""
    deg = coefs_2d.shape[0] - 1
    T_a = np.zeros(deg+1)
    T_a[0] = 1.0
    if deg >= 1: T_a[1] = xi_a
    for i in range(1, deg):
        T_a[i+1] = 2*xi_a*T_a[i] - T_a[i-1]
    return T_a @ coefs_2d                # (deg_b+1,) coefs in u_b


def find_real_roots_in_interval(coefs_1d, lo, hi):
    """All real roots of 1D Cheby polynomial in (lo, hi) via companion matrix."""
    # Use np.polynomial.chebyshev.chebroots
    try:
        roots = cheb.chebroots(coefs_1d)
    except Exception:
        return np.array([])
    real = roots[np.abs(roots.imag) < 1e-8].real
    in_range = real[(real >= lo) & (real <= hi)]
    return np.sort(in_range)


def lookup_analytic_cheby(coefs_3d, u_max, p_target, u_k, tau, NQK=24):
    """Analytic strict-h=0 lookup mu(p, u_k) via Cheby root finding.
    Use GL quadrature in u_a; find roots in u_b via companion matrix; integrate.
    """
    coefs_2d = cheby_slice_2d(coefs_3d, u_max, u_k)   # (deg+1, deg+1) in u_a, u_b
    deg = coefs_2d.shape[0] - 1
    # GL nodes for u_a on [-u_max, u_max]
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    u_a_nodes = u_max * n_xi
    du_a_w = u_max * w_xi
    f0a = np.array([f_signal_jit(u, 0, tau) for u in u_a_nodes])
    f1a = np.array([f_signal_jit(u, 1, tau) for u in u_a_nodes])
    A0 = 0.0; A1 = 0.0
    for q in range(NQK):
        u_a = u_a_nodes[q]
        xi_a = u_a / u_max
        if abs(xi_a) > 1.0: continue
        # 1D coefs in u_b for fixed u_a, polynomial: P(u_k, u_a, u_b) - p
        coefs_b = cheby_eval_1d_at_xi_a(coefs_2d, xi_a).copy()
        coefs_b[0] -= p_target
        # Real roots in (-u_max, u_max) -- xi_b in (-1, 1)
        roots_xi = find_real_roots_in_interval(coefs_b, -1.0, 1.0)
        if len(roots_xi) == 0: continue
        # Derivative coefs in u_b
        coefs_b_deriv = cheb.chebder(coefs_b, 1)
        for xi_b in roots_xi:
            u_b = xi_b * u_max
            f0b = f_signal_jit(u_b, 0, tau)
            f1b = f_signal_jit(u_b, 1, tau)
            dPdb_xi = cheb.chebval(xi_b, coefs_b_deriv)   # d/d xi_b
            dPdb_u = dPdb_xi / u_max                          # chain rule: dxi/du = 1/u_max
            if abs(dPdb_u) < 1e-300: continue
            wt = du_a_w[q] / abs(dPdb_u)
            A0 += wt * f0a[q] * f0b
            A1 += wt * f1a[q] * f1b
    f0k = f_signal_jit(u_k, 0, tau)
    f1k = f_signal_jit(u_k, 1, tau)
    den = f0k * A0 + f1k * A1
    if den > 1e-300: return f1k * A1 / den
    return 0.5


def build_mu_table_analytic_cheby(P_cube, u_grid, p_grid, tau, NQK=24, deg=None):
    coefs, u_max = cheby_fit_3d(P_cube, u_grid, deg=deg)
    G = u_grid.size; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    for k in range(G):
        u_k = u_grid[k]
        for ip in range(G_p):
            mu_table[ip, k] = lookup_analytic_cheby(coefs, u_max, p_grid[ip],
                                                              u_k, tau, NQK=NQK)
    return mu_table, coefs


def cheby_coef_diagnostic(coefs):
    """Return decay rate of Cheby coefficients along each axis."""
    deg = coefs.shape[0] - 1
    max_axis = []
    for ax in range(3):
        # max |coef| at each axis-slice along ax
        axes = list(range(3)); axes.remove(ax)
        max_abs = np.max(np.abs(coefs), axis=tuple(axes))
        max_axis.append(max_abs)
    return max_axis


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    print(f"--- Cheby h=0 via analytic 1D root-find ---", flush=True)
    results = {}
    for fp in fp_files[:16]:
        d = np.load(fp)
        if d["mu_hi"].shape[0] != G: continue
        P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
        gamma = float(d["gamma"]); tau = float(d["tau"])
        key = os.path.basename(fp).replace(".npz", "")
        print(f"\n=== {key} ===", flush=True)
        # Strict-h=0 reference (POU + linear)
        t0 = time.time()
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, G, 16)
        t_strict = time.time() - t0
        cell = dict(gamma=gamma, tau=tau, t_strict=t_strict)
        # Try several Cheby degs to see convergence
        for deg in [G-1, G, G+2, G+4]:
            if deg >= G + 5: continue
            try:
                t0 = time.time()
                mu_cheb, coefs = build_mu_table_analytic_cheby(
                    P, u_grid, p_grid, tau, NQK=24, deg=deg)
                t_cheb = time.time() - t0
                d_max = float(np.max(np.abs(mu_cheb - mu_strict)))
                d_med = float(np.median(np.abs(mu_cheb - mu_strict)))
                # Diagnostic: largest coef in last layer (convergence indicator)
                edge_coef = float(np.max(np.abs(coefs[-1, :, :])) + \
                                       np.max(np.abs(coefs[:, -1, :])) + \
                                       np.max(np.abs(coefs[:, :, -1])))
                print(f"  deg={deg:2d}: max|d|={d_max:.3e} med|d|={d_med:.3e} "
                      f"edge_coef={edge_coef:.3e} ({t_cheb:.1f}s)", flush=True)
                cell[f"deg={deg}"] = dict(max=d_max, med=d_med,
                                                edge_coef=edge_coef, wall=t_cheb)
            except Exception as e:
                print(f"  deg={deg}: FAILED {e}", flush=True)
                cell[f"deg={deg}"] = dict(error=str(e))
        results[key] = cell
        json.dump(results, open("/tmp/dd_k3_cheby_h0.json", "w"), indent=2,
                    default=str)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_cheby_h0.json", flush=True)


if __name__ == "__main__":
    run()
