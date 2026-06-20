"""CMM Stage 6: graph/height-function formulation (user's 1D-per-agent idea).

Level surface S_m = { t = H_m(w) },  t = (u1+u2+u3)/sqrt(3), w = transverse.
Slice curves are global graphs over the anti-diagonal coordinate; recovered
from H_m by monotone scalar root-solves (IFT-smooth). No mesh, no topology.

Stage 6a: build H_m from the kernel FP, implement the residual, validate
its values against the Stage 3b mesh residual statistics, then run LM on
the (much smaller) H state.

Coordinates:
  e_t = (1,1,1)/sqrt(3); e_a = (1,-1,0)/sqrt(2); e_b = (1,1,-2)/sqrt(6)
  u = a*e_a + b*e_b + t*e_t
Slice for agent k at own-signal X, price p_m, anti-diag position s:
  the two free coords are (c+s, c-s) in the (j,l) plane; t = (X+2c)/sqrt(3)
  root-solve in c:  (X + 2c)/sqrt(3) = H_m(w(y(c,s)))  (monotone in c)
"""
import os, sys, time, json
import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, EMIN15, OUT
from reznsrc.contour_K3_halo import init_no_learning_K3
from reznsrc.demand import clear_crra as _clear

SQ3 = np.sqrt(3.0); SQ2 = np.sqrt(2.0); SQ6 = np.sqrt(6.0)
E_T = np.array([1.0, 1.0, 1.0]) / SQ3
E_A = np.array([1.0, -1.0, 0.0]) / SQ2
E_B = np.array([1.0, 1.0, -2.0]) / SQ6


def uvw_to_u(a, b, t):
    """(a,b,t) -> (u1,u2,u3). Arrays broadcast."""
    return (np.multiply.outer(a, E_A) + np.multiply.outer(b, E_B)
            + np.multiply.outer(t, E_T))


def u_to_abt(u):
    a = u @ E_A; b = u @ E_B; t = u @ E_T
    return a, b, t


class GraphState:
    """M height functions H_m on a square (a,b) grid + spline interpolants."""

    def __init__(self, a_grid, b_grid, H):  # H: (M, Na, Nb)
        self.a_grid = a_grid; self.b_grid = b_grid
        self.H = H
        self.M = H.shape[0]
        self._splines = None

    def splines(self):
        if self._splines is None:
            self._splines = [RectBivariateSpline(self.a_grid, self.b_grid,
                                                   self.H[m], kx=3, ky=3)
                              for m in range(self.M)]
        return self._splines

    def invalidate(self):
        self._splines = None


def build_initial_H(P_full, uf, lo, hi, p_levels, a_grid, b_grid, t_range):
    """Root-solve the kernel FP cube along the diagonal for each (a,b)."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((uf, uf, uf), P_full,
                                      bounds_error=False, fill_value=None)
    Na, Nb = a_grid.size, b_grid.size
    M = len(p_levels)
    H = np.zeros((M, Na, Nb))
    t_lo, t_hi = t_range
    ts = np.linspace(t_lo, t_hi, 200)
    for ia, a in enumerate(a_grid):
        for ib, b in enumerate(b_grid):
            pts = uvw_to_u(a, b, ts)        # (200, 3)
            vals = interp(pts)               # (200,)
            # P monotone in t along diagonal: invert by interp
            v, idx = np.unique(vals, return_index=True)
            tt = ts[idx]
            for m, p in enumerate(p_levels):
                H[m, ia, ib] = np.interp(p, v, tt)
    return H


def slice_curve(spl_m, X, k_own, s_arr, c_init, n_newton=12):
    """Solve (X + 2c)/sqrt(3) = H_m(w(y(c,s))) for c at each s (vector).

    y has y_k = X; the other two coords (in cyclic order j,l) are
    (c+s, c-s). Returns c_arr (nan where no root in range).
    Smooth: scalar monotone Newton with bisection fallback.
    """
    # build u from (k_own, X, c, s):
    def build_u(c, s):
        u = np.empty(c.shape + (3,))
        jl = [(1, 2), (2, 0), (0, 1)][k_own]
        u[..., k_own] = X
        u[..., jl[0]] = c + s
        u[..., jl[1]] = c - s
        return u

    c = np.full(s_arr.shape, c_init, dtype=float)
    for it in range(n_newton):
        u = build_u(c, s_arr)
        a, b, t = u_to_abt(u)
        Hv = spl_m.ev(a, b)
        F = (X + 2*c)/SQ3 - Hv
        # dF/dc: 2/sqrt(3) - dH/da * da/dc - dH/db * db/dc
        da_dc = (E_A[[1, 2, 0][k_own]] + E_A[[2, 0, 1][k_own]])
        db_dc = (E_B[[1, 2, 0][k_own]] + E_B[[2, 0, 1][k_own]])
        # NOTE: with the cyclic (j,l) convention above, j=[1,2,0][k], l=[2,0,1][k]
        Ha = spl_m.ev(a, b, dx=1); Hb = spl_m.ev(a, b, dy=1)
        dF = 2.0/SQ3 - Ha*da_dc - Hb*db_dc
        dF = np.where(np.abs(dF) < 0.2, np.sign(dF)*0.2, dF)  # guard
        step = F / dF
        step = np.clip(step, -0.5, 0.5)
        c = c - step
        if np.max(np.abs(F)) < 1e-12:
            break
    return c


def f_pair(u_j, u_l, v, tau):
    s = np.sqrt(tau/(2*np.pi))
    mean = 0.5 if v == 1 else -0.5
    return (s*np.exp(-0.5*tau*(u_j-mean)**2)) * (s*np.exp(-0.5*tau*(u_l-mean)**2))


def residual(state, p_levels, tau, gamma, vert_ab, S_MAX=5.0, N_S=61):
    """Residual at each surface's (a,b) vertex set.

    vert_ab: list of (a_pts, b_pts) per surface (flattened vertex sets).
    Returns list of residual arrays."""
    spls = state.splines()
    s_arr = np.linspace(-S_MAX, S_MAX, N_S)
    res = []
    for m in range(state.M):
        p_m = p_levels[m]
        spl = spls[m]
        a_pts, b_pts = vert_ab[m]
        t_pts = spl.ev(a_pts, b_pts)
        U3 = uvw_to_u(a_pts, b_pts, np.zeros_like(a_pts)) \
              + np.multiply.outer(t_pts, E_T)   # (Nv, 3)
        r_m = np.empty(a_pts.size)
        for iv in range(a_pts.size):
            u_vert = U3[iv]
            mu = np.empty(3)
            for k in range(3):
                X = u_vert[k]
                jl = [(1, 2), (2, 0), (0, 1)][k]
                c0 = 0.5*(u_vert[jl[0]] + u_vert[jl[1]])
                c = slice_curve(spl, X, k, s_arr, c0)
                u_j = c + s_arr; u_l = c - s_arr
                # arclength element: dy/ds = (c'+1, c'-1); use FD on c
                dc = np.gradient(c, s_arr)
                dsig = np.sqrt((dc + 1)**2 + (dc - 1)**2)
                f0v = f_pair(u_j, u_l, 0, tau); f1v = f_pair(u_j, u_l, 1, tau)
                A0 = np.trapz(f0v*dsig, s_arr)
                A1 = np.trapz(f1v*dsig, s_arr)
                sg = np.sqrt(tau/(2*np.pi))
                f0o = sg*np.exp(-0.5*tau*(X+0.5)**2)
                f1o = sg*np.exp(-0.5*tau*(X-0.5)**2)
                num = f1o*A1; den = f0o*A0 + num
                mu[k] = min(max(num/den if den > 0 else 0.5, 1e-12), 1-1e-12)
            r_m[iv] = _clear(mu, np.full(3, gamma), np.full(3, 1.0)) - p_m
        res.append(r_m)
    return res


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    tau, gamma = 2.0, 0.0980
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    p_flat = P_inner.ravel()
    qs = np.linspace(0, 1, 17); edges = np.quantile(p_flat, qs)
    p_levels = 0.5*(edges[:-1] + edges[1:])

    # transverse grid: |a|,|b| up to 4.5 (the cube corner is at |w| ~ 5.6
    # but density there is negligible)
    a_grid = np.linspace(-4.5, 4.5, 25)
    b_grid = np.linspace(-4.5, 4.5, 25)
    t0 = time.time()
    H = build_initial_H(P_full, uf, lo, hi, p_levels, a_grid, b_grid,
                         t_range=(-7.0, 7.0))
    print(f"Initial H built: {H.shape} ({time.time()-t0:.0f}s)", flush=True)
    state = GraphState(a_grid, b_grid, H)

    # vertex set per surface: interior transverse points (margin from edge)
    A2, B2 = np.meshgrid(a_grid[2:-2], b_grid[2:-2], indexing='ij')
    vert = [(A2.ravel(), B2.ravel()) for _ in range(len(p_levels))]

    t0 = time.time()
    res = residual(state, p_levels, tau, gamma, vert)
    r_all = np.concatenate(res)
    wall = time.time() - t0
    print(f"Initial residual ({r_all.size} vertices): "
          f"max|r|={np.max(np.abs(r_all)):.3e} "
          f"med|r|={np.median(np.abs(r_all)):.3e} ({wall:.0f}s)", flush=True)
    per_m = [(float(p_levels[m]), float(np.max(np.abs(res[m]))),
               float(np.median(np.abs(res[m])))) for m in range(len(p_levels))]
    for p_m, mx, md in per_m:
        print(f"  p={p_m:.3f}: max={mx:.3e} med={md:.3e}", flush=True)
    json.dump(dict(max=float(np.max(np.abs(r_all))),
                    med=float(np.median(np.abs(r_all))),
                    per_surface=per_m, wall=wall,
                    n_vert=int(r_all.size), state_dim=int(H.size)),
              open(f"{OUT}/stage6a_initial.json", "w"), indent=2)
    np.save(f"{OUT}/stage6a_H0.npy", H)
    print("saved stage6a_initial.json / stage6a_H0.npy", flush=True)


if __name__ == "__main__":
    main()
