"""DD K=3 solver:
  Stage 1: float64 Lin-CDF R4 Anderson/NK to F < 1e-12 (warm start)
  Stage 2: build float64 finite-difference Jacobian over the symmetric-reduced
           unknowns of the 3D P tensor
  Stage 3: DD chord Newton (float64 LU on DD residual) to F < 1e-25
"""
import os, sys, time
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
from itertools import combinations_with_replacement, permutations
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence
import scipy.linalg as sla
import dd_k3_ops as DK
import dd_ops as DO
from lin_cdf_richardson import phi_lin_richardson
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


# ---- Symmetric reduction: P(u_i, u_j, u_k) is fully symmetric in (i,j,k)
class SymRed3:
    def __init__(self, G):
        self.G = G
        triples = list(combinations_with_replacement(range(G), 3))
        self.triples = triples
        self.n_red = len(triples)
    def reduce(self, P):
        out = np.empty(self.n_red)
        for i, t in enumerate(self.triples): out[i] = P[t[0], t[1], t[2]]
        return out
    def expand(self, vec, dtype=float):
        G = self.G
        P = np.empty((G, G, G), dtype=dtype)
        for i, t in enumerate(self.triples):
            v = vec[i]
            for p in set(permutations(t)):
                P[p[0], p[1], p[2]] = v
        return P


def solve_warm_f64(P0, u_grid, p_grid, hs, gamma, tau, G_p, NQK,
                       target=1e-12, n_anderson=60):
    """Float64 Anderson/NK warm-start to F < target."""
    G = u_grid.size
    def F(xflat):
        Pn = phi_lin_richardson(xflat.reshape(G,G,G), u_grid, hs=hs,
                                     gamma=gamma, tau=tau, G_p=G_p, NQK=NQK,
                                     p_grid=p_grid)
        return (Pn - xflat.reshape(G,G,G)).ravel()
    x = P0.ravel().copy(); Xh, Gh = [], []; Fs = []
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
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk
        except NoConvergence as e:
            xnk = e.args[0]; fnk = float(np.max(np.abs(F(xnk))))
            if fnk < min(Fs): return xnk.reshape(G,G,G), fnk
    return x_best.reshape(G,G,G), min(Fs)


def F_dd_red(red, x_red_H, x_red_L, u_grid, p_grid, gl_u, gl_du,
                 th, tl, gh, gl, hs, weights):
    """Reduced DD residual: F_red = reduce(Phi(expand(x))) - x."""
    P_H = red.expand(x_red_H); P_L = red.expand(x_red_L)
    PnH, PnL = DK.phi_dd(P_H, P_L, u_grid, p_grid, gl_u, gl_du,
                                th, tl, gh, gl, hs, weights)
    n = red.n_red
    F_H = np.empty(n); F_L = np.empty(n)
    for i, t in enumerate(red.triples):
        ah, al = DO.dd_add(PnH[t[0], t[1], t[2]], PnL[t[0], t[1], t[2]],
                                -x_red_H[i], -x_red_L[i])
        F_H[i] = ah; F_L[i] = al
    return F_H, F_L


def nail_dd(P_warm, u_grid, p_grid, gl_u, gl_du, hs, gamma, tau,
                target_dd=1e-25, max_iters=15, verbose=False):
    """Chord-Newton DD nail on the symmetric-reduced unknowns."""
    G = u_grid.size; red = SymRed3(G); n = red.n_red
    th, tl = split(mp.mpf(repr(tau)))
    gh, gl = split(mp.mpf(repr(gamma)))
    w_arr = DK.richardson_weights(hs)
    x_H = red.reduce(P_warm).astype(np.float64); x_L = np.zeros(n)

    # FD Jacobian in float64 (one Phi eval per perturbed unknown)
    if verbose: print(f"    building {n}x{n} FD Jacobian...", flush=True)
    eps = 1e-7
    F0_H, F0_L = F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                th, tl, gh, gl, hs, w_arr)
    F0 = F0_H + F0_L
    J = np.empty((n, n))
    for k in range(n):
        xp_H = x_H.copy(); xp_H[k] += eps
        Fp_H, Fp_L = F_dd_red(red, xp_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                    th, tl, gh, gl, hs, w_arr)
        J[:, k] = ((Fp_H + Fp_L) - F0) / eps
    try: lu, piv = sla.lu_factor(J)
    except Exception:
        red0 = SymRed3(G); return P_warm, float(np.max(np.abs(F0))), red0.expand(x_H), red0.expand(x_L)
    Jinv = sla.lu_solve((lu, piv), np.eye(n))

    # Chord Newton in DD
    F_inf = float(np.max(np.abs(F0))); best_F = F_inf
    best_x_H = x_H.copy(); best_x_L = x_L.copy()
    for it in range(max_iters):
        # dx = - Jinv @ F   (float64 LU on DD F preserves DD because dx still tracks F)
        F = F0_H + F0_L
        dx = -Jinv @ F
        # x_new = x + dx in DD (dx is float64 but add it as DD)
        for i in range(n):
            aH, aL = DO.dd_add(x_H[i], x_L[i], dx[i], 0.0)
            x_H[i] = aH; x_L[i] = aL
        F0_H, F0_L = F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                    th, tl, gh, gl, hs, w_arr)
        F_inf = float(np.max(np.abs(F0_H + F0_L)))
        if verbose:
            print(f"    nail it{it+1}: |F|={F_inf:.3e}", flush=True)
        if F_inf < best_F:
            best_F = F_inf; best_x_H = x_H.copy(); best_x_L = x_L.copy()
        if F_inf < target_dd: break
        if it >= 3 and F_inf > 0.9*best_F: break    # stagnation
    P_H = red.expand(best_x_H); P_L = red.expand(best_x_L)
    return P_H + P_L, best_F, P_H, P_L


def solve_dd_k3(gamma, tau, P_warm=None, u_grid=None, p_grid=None, gl_u=None,
                  gl_du=None, hs=(0.5, 0.4, 0.3, 0.2), target_dd=1e-25,
                  target_warm=1e-12, verbose=False):
    """Full DD K=3 solve at (gamma, tau). Returns (P_full, F_best, P_H, P_L)."""
    G = u_grid.size; G_p = p_grid.size; NQK = gl_u.size
    if P_warm is None:
        U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
        P_warm = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    # warm-start in float64
    if verbose: print(f"  warm float64...", flush=True)
    t0 = time.time()
    P_warm, F_warm = solve_warm_f64(P_warm, u_grid, p_grid, hs, gamma, tau,
                                              G_p, NQK, target=target_warm)
    t_warm = time.time() - t0
    if verbose: print(f"    F_warm={F_warm:.3e} ({t_warm:.1f}s)", flush=True)
    # DD nail
    t0 = time.time()
    P_full, F_nail, P_H, P_L = nail_dd(P_warm, u_grid, p_grid, gl_u, gl_du,
                                                 hs, gamma, tau,
                                                 target_dd=target_dd, verbose=verbose)
    t_nail = time.time() - t0
    if verbose: print(f"  nail DD: F={F_nail:.3e} ({t_nail:.1f}s)", flush=True)
    return P_full, F_nail, P_H, P_L, t_warm, t_nail


if __name__ == "__main__":
    G = 7; G_p = 121; NQK = 16
    HS = (0.5, 0.4, 0.3, 0.2)
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    print(f"--- DD K=3 solver self-test (G={G}, G_p={G_p}, NQK={NQK}, R4) ---")
    print(f"sym-reduced unknowns: {SymRed3(G).n_red}")
    P, F, P_H, P_L, tw, tn = solve_dd_k3(1.0, 1.0, u_grid=u_grid, p_grid=p_grid,
                                                   gl_u=gl_u, gl_du=gl_du, hs=HS,
                                                   verbose=True)
    print(f"\n=> tau=1, gamma=1: F={F:.3e}, warm {tw:.1f}s, nail {tn:.1f}s")
