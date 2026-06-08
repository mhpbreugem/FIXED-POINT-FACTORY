"""Re-solve K=3 FP at Chebyshev Lobatto u-grid (so Cheby fit is exact),
then evaluate analytic lookup via 1D root finding. Test convergence at
increasing G.
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_p_grid, make_gl_for_u
from lin_cdf_strict import build_mu_table_lin_strict
from cheby_numba import f_signal_jit


def lobatto_grid(G, U_MAX=2.33):
    """Cheby Lobatto nodes scaled to [-U_MAX, U_MAX]. Strictly inside the
    practical signal range."""
    xi = -np.cos(np.pi * np.arange(G) / (G - 1))
    return xi * U_MAX, U_MAX


def cheby_fit_lobatto_3d(P_cube, U_MAX):
    """Exact interpolation: Cheby coefs from values at Lobatto nodes."""
    G = P_cube.shape[0]
    # 1D Cheby coefs from Lobatto values: use chebinterpolate (FFT-style)
    # Build transform matrix
    deg = G - 1
    xi = -np.cos(np.pi * np.arange(G) / (G - 1))
    V = cheb.chebvander(xi, deg)
    # Interpolation: c = V^{-1} y
    Vinv = np.linalg.inv(V)
    # Apply axis by axis
    A0 = np.tensordot(Vinv, P_cube, axes=([1], [0]))   # (deg+1, G, G)
    A1 = np.tensordot(Vinv, A0, axes=([1], [1])).transpose(1, 0, 2)
    A2 = np.tensordot(Vinv, A1, axes=([1], [2])).transpose(1, 2, 0)
    return A2


def cheby_slice_2d(coefs, U_MAX, u_k):
    deg = coefs.shape[0] - 1
    xi_k = u_k / U_MAX
    T_xi_k = np.zeros(deg+1)
    T_xi_k[0] = 1.0
    if deg >= 1: T_xi_k[1] = xi_k
    for i in range(1, deg):
        T_xi_k[i+1] = 2*xi_k*T_xi_k[i] - T_xi_k[i-1]
    return np.einsum("i,ijk->jk", T_xi_k, coefs)


def cheby_eval_1d_at_xi_a(coefs_2d, xi_a):
    deg = coefs_2d.shape[0] - 1
    T_a = np.zeros(deg+1)
    T_a[0] = 1.0
    if deg >= 1: T_a[1] = xi_a
    for i in range(1, deg):
        T_a[i+1] = 2*xi_a*T_a[i] - T_a[i-1]
    return T_a @ coefs_2d


def find_real_roots_in_unit(coefs_1d):
    try:
        roots = cheb.chebroots(coefs_1d)
    except Exception: return np.array([])
    real = roots[np.abs(roots.imag) < 1e-8].real
    return np.sort(real[(real >= -1.0) & (real <= 1.0)])


def lookup_analytic_lobatto(coefs_3d, U_MAX, p_target, u_k, tau, NQK=24):
    coefs_2d = cheby_slice_2d(coefs_3d, U_MAX, u_k)
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    u_a_nodes = U_MAX * n_xi
    du_a_w = U_MAX * w_xi
    A0 = 0.0; A1 = 0.0
    for q in range(NQK):
        u_a = u_a_nodes[q]; xi_a = u_a / U_MAX
        if abs(xi_a) > 1.0: continue
        coefs_b = cheby_eval_1d_at_xi_a(coefs_2d, xi_a).copy()
        coefs_b[0] -= p_target
        roots_xi = find_real_roots_in_unit(coefs_b)
        if len(roots_xi) == 0: continue
        coefs_b_deriv = cheb.chebder(coefs_b, 1)
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        for xi_b in roots_xi:
            u_b = xi_b * U_MAX
            dPdb_xi = cheb.chebval(xi_b, coefs_b_deriv)
            dPdb_u = dPdb_xi / U_MAX
            if abs(dPdb_u) < 1e-300: continue
            wt = du_a_w[q] / abs(dPdb_u)
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
    den = f0k * A0 + f1k * A1
    return f1k * A1 / den if den > 1e-300 else 0.5


def solve_fp_lobatto(gamma, tau, G, hs=(0.5, 0.4, 0.3, 0.2), P_init=None,
                          n_anderson=80, target=1e-12):
    """Solve FP with Lin-CDF R4 operator on Lobatto u-grid."""
    u_grid, U_MAX = lobatto_grid(G)
    p_grid = make_p_grid(121)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T = U1 + U2 + U3
    if P_init is None: P_init = 1.0/(1.0+np.exp(-0.5*T))
    def F(xflat):
        Pn = phi_lin_richardson(xflat.reshape(G,G,G), u_grid, hs=hs,
                                       gamma=gamma, tau=tau, G_p=121, NQK=16,
                                       p_grid=p_grid)
        return (Pn - xflat.reshape(G,G,G)).ravel()
    x = P_init.ravel().copy()
    Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float("inf")
    for it in range(n_anderson):
        Fv = F(x); gx = Fv + x
        f = float(np.max(np.abs(Fv))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < target: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: x = gx
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                x = Gh[k-1] + DG @ ga
            except: x = gx
    if min(Fs) > target:
        try:
            xnk = newton_krylov(F, x_best, f_tol=target, maxiter=20, verbose=False)
            fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk, u_grid, U_MAX
        except NoConvergence as e:
            xnk = e.args[0]; fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk, u_grid, U_MAX
    return x_best.reshape(G,G,G), min(Fs), u_grid, U_MAX


def run():
    gamma, tau = 100.0, 1.0
    # Solve at successive G on Lobatto grid
    p_grid = make_p_grid(121)
    print(f"\n=== Cheby-Lobatto convergence study at gamma={gamma}, tau={tau} ===",
          flush=True)
    P_prev = None
    for G in [11, 15, 21]:
        print(f"\n--- G={G} ---", flush=True)
        t0 = time.time()
        if P_prev is not None:
            # Project to new Lobatto grid via Cheby interp from previous solution
            u_new, U_MAX_new = lobatto_grid(G)
            G_prev = P_prev.shape[0]
            U_MAX_prev = max(abs(lobatto_grid(G_prev)[0][0]),
                                abs(lobatto_grid(G_prev)[0][-1]))
            coefs_prev = cheby_fit_lobatto_3d(P_prev, U_MAX_prev)
            P_init = np.empty((G,G,G))
            for ii in range(G):
                for jj in range(G):
                    for kk in range(G):
                        P_init[ii,jj,kk] = cheb.chebval3d(u_new[ii]/U_MAX_prev,
                                                              u_new[jj]/U_MAX_prev,
                                                              u_new[kk]/U_MAX_prev,
                                                              coefs_prev)
            P_init = np.clip(P_init, 1e-9, 1-1e-9)
        else: P_init = None
        P, F, u_grid, U_MAX = solve_fp_lobatto(gamma, tau, G, P_init=P_init)
        wall = time.time() - t0
        print(f"  FP solve: |F|={F:.3e} ({wall:.1f}s)", flush=True)
        # Fit Cheby (exact at Lobatto)
        coefs = cheby_fit_lobatto_3d(P, U_MAX)
        # Coef decay diagnostic
        coef_abs = np.abs(coefs)
        edge_coefs = []
        for ax in range(3):
            axes = list(range(3)); axes.remove(ax)
            edge_coefs.append(float(np.max(coef_abs.max(axis=tuple(axes))[-3:])))
        print(f"  Cheby fit: top 3 edge coefs ~ {edge_coefs}", flush=True)
        # Lookup vs strict-h=0 on CDF-uniform-equivalent test grid
        # Use Lobatto u-grid for strict comparison too
        from lin_cdf_strict import make_gl_for_u
        gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], 16)
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, G, 16)
        # Cheby analytic lookup
        t0 = time.time()
        mu_cheb = np.empty((121, G))
        for k_idx in range(G):
            u_k = u_grid[k_idx]
            for ip in range(121):
                mu_cheb[ip, k_idx] = lookup_analytic_lobatto(coefs, U_MAX,
                                                                       p_grid[ip],
                                                                       u_k, tau, NQK=24)
        t_lookup = time.time() - t0
        d_max = float(np.max(np.abs(mu_cheb - mu_strict)))
        d_med = float(np.median(np.abs(mu_cheb - mu_strict)))
        print(f"  lookup: max|d|={d_max:.3e} med|d|={d_med:.3e} ({t_lookup:.1f}s)",
              flush=True)
        P_prev = P


if __name__ == "__main__":
    run()
