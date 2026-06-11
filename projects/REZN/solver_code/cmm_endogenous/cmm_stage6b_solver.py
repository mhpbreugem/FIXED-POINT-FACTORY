"""CMM Stage 6b: graph/height-function formulation — convergent solver.

Level surface S_m = { t = H_m(a,b) },  t = (u1+u2+u3)/sqrt(3),
(a,b) transverse coords:  e_a = (1,-1,0)/sqrt2, e_b = (1,1,-2)/sqrt6.
Unknowns: height arrays H_m on a fixed (a,b) grid. No mesh, no topology.

Differences vs the stage-6a sketch (cmm_stage6_graph.py):
  * np.trapz -> uniform-step trapezoid (np.trapezoid semantics) inside numba.
  * dF/dc RE-DERIVED (see below); the sketch's cyclic form E_A[j]+E_A[l]
    is algebraically equal to -E_A[k] (components of e_a, e_b sum to 0),
    so it was correct but opaque; we use the clean closed form.
  * c'(s) analytic by implicit differentiation (no np.gradient).
  * H interpolation: local C1 bicubic (Catmull-Rom) instead of a global
    RectBivariateSpline -> numba-friendly, perturbation of one node has
    local support, linear extrapolation with clamped tilt outside grid.
  * full LM solver with block-(in fact fully-)independent surfaces.

DERIVATION (slice-curve root equation and its derivatives)
----------------------------------------------------------
Agent k, own-signal X, surface m. Other two coords (cyclic j=(k+1)%3,
l=(k+2)%3) parametrized by the anti-diagonal coordinate s:
    u_k = X,  u_j = c + s,  u_l = c - s.
Then  t = e_t.u = (X + 2c)/sqrt3,  and using  sum_i e_a[i] = sum_i e_b[i] = 0
(so e_a[j]+e_a[l] = -e_a[k], idem e_b):
    a(c,s) = e_a.u = e_a[k] X + (e_a[j]+e_a[l]) c + (e_a[j]-e_a[l]) s
           = e_a[k] (X - c) + DAS_k s,        DAS_k := e_a[j]-e_a[l]
    b(c,s) = e_b[k] (X - c) + DBS_k s,        DBS_k := e_b[j]-e_b[l]
Root equation:  F(c;s) = (X + 2c)/sqrt3 - H_m(a(c,s), b(c,s)) = 0.
    dF/dc = 2/sqrt3 - H_a * (da/dc) - H_b * (db/dc)
          = 2/sqrt3 + H_a e_a[k] + H_b e_b[k]            (da/dc = -e_a[k])
With |H_a|,|H_b| clamped to TILT=0.6 outside the grid and
sqrt(e_a[k]^2 + e_b[k]^2) = sqrt(2/3) for every k, dF/dc >= 2/sqrt3 -
0.6*(|e_a[k]|+|e_b[k]|) > 0.2 in the extrapolation region, so F is
globally monotone in c there and a bracket always exists.

Analytic c'(s) by implicit differentiation of F(c(s), s) = 0:
    dF/ds = -(H_a DAS_k + H_b DBS_k)
    c'(s) = -F_s / F_c = (H_a DAS_k + H_b DBS_k)
                         / (2/sqrt3 + H_a e_a[k] + H_b e_b[k])
Arclength element of the curve (u_j, u_l) = (c+s, c-s):
    dsigma = sqrt((c'+1)^2 + (c'-1)^2) ds.
Evidence:  A_v = int f_v(u_j) f_v(u_l) dsigma  (uniform-s trapezoid);
the Gaussian normalisation constants cancel in the posterior ratio
    mu_k = f1(X) A1 / (f0(X) A0 + f1(X) A1)
so plain exponentials are used. Residual r = clear_CRRA(mu; gamma) - p_m.
"""
import os, sys, time, json, argparse
import numpy as np
import numba
from numba import njit, prange

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, EMIN15, OUT
from reznsrc.contour_K3_halo import init_no_learning_K3
from reznsrc.demand import clear_crra as clear_crra_ref   # cross-check only

SQ3 = np.sqrt(3.0); SQ2 = np.sqrt(2.0); SQ6 = np.sqrt(6.0)
E_T = np.array([1.0, 1.0, 1.0]) / SQ3
E_A = np.array([1.0, -1.0, 0.0]) / SQ2
E_B = np.array([1.0, 1.0, -2.0]) / SQ6
# per-agent slice constants (j=(k+1)%3, l=(k+2)%3)
DAS = np.array([E_A[1] - E_A[2], E_A[2] - E_A[0], E_A[0] - E_A[1]])
DBS = np.array([E_B[1] - E_B[2], E_B[2] - E_B[0], E_B[0] - E_B[1]])
TILT = 0.6        # max |grad H| outside the grid (root uniqueness guard)
FC_FLOOR = 0.05   # floor on dF/dc when forming c' (vertical-tangent guard)


def uvw_to_u(a, b, t):
    return (np.multiply.outer(a, E_A) + np.multiply.outer(b, E_B)
            + np.multiply.outer(t, E_T))


# ===================== numba kernels =====================

@njit(cache=True, fastmath=False)
def build_pad(H):
    """Pad H by one linearly-extrapolated ghost node on each side
    (Catmull-Rom stencil support for the edge cells)."""
    Na, Nb = H.shape
    Hp = np.empty((Na + 2, Nb + 2))
    Hp[1:Na+1, 1:Nb+1] = H
    Hp[0, 1:Nb+1] = 2.0*H[0, :] - H[1, :]
    Hp[Na+1, 1:Nb+1] = 2.0*H[Na-1, :] - H[Na-2, :]
    for i in range(Na + 2):
        Hp[i, 0] = 2.0*Hp[i, 1] - Hp[i, 2]
        Hp[i, Nb+1] = 2.0*Hp[i, Nb] - Hp[i, Nb-1]
    return Hp


@njit(cache=True, fastmath=False, inline='always')
def _heval_in(Hp, a0, da, Na, b0, db, Nb, a, b):
    """C1 bicubic (Catmull-Rom) value + gradient, a,b INSIDE the grid box."""
    fa = (a - a0) / da
    ia = int(np.floor(fa))
    if ia < 0: ia = 0
    if ia > Na - 2: ia = Na - 2
    xa = fa - ia
    fb = (b - b0) / db
    ib = int(np.floor(fb))
    if ib < 0: ib = 0
    if ib > Nb - 2: ib = Nb - 2
    xb = fb - ib
    # Catmull-Rom weights and d/dx
    x = xa; x2 = x*x; x3 = x2*x
    wa0 = 0.5*(-x3 + 2*x2 - x); wa1 = 0.5*(3*x3 - 5*x2 + 2)
    wa2 = 0.5*(-3*x3 + 4*x2 + x); wa3 = 0.5*(x3 - x2)
    da0 = 0.5*(-3*x2 + 4*x - 1); da1 = 0.5*(9*x2 - 10*x)
    da2 = 0.5*(-9*x2 + 8*x + 1); da3 = 0.5*(3*x2 - 2*x)
    x = xb; x2 = x*x; x3 = x2*x
    wb0 = 0.5*(-x3 + 2*x2 - x); wb1 = 0.5*(3*x3 - 5*x2 + 2)
    wb2 = 0.5*(-3*x3 + 4*x2 + x); wb3 = 0.5*(x3 - x2)
    db0 = 0.5*(-3*x2 + 4*x - 1); db1 = 0.5*(9*x2 - 10*x)
    db2 = 0.5*(-9*x2 + 8*x + 1); db3 = 0.5*(3*x2 - 2*x)
    H = 0.0; Ha = 0.0; Hb = 0.0
    for p in range(4):
        if p == 0: wa = wa0; dwa = da0
        elif p == 1: wa = wa1; dwa = da1
        elif p == 2: wa = wa2; dwa = da2
        else: wa = wa3; dwa = da3
        v0 = Hp[ia + p, ib]; v1 = Hp[ia + p, ib + 1]
        v2 = Hp[ia + p, ib + 2]; v3 = Hp[ia + p, ib + 3]
        rowv = wb0*v0 + wb1*v1 + wb2*v2 + wb3*v3
        rowd = db0*v0 + db1*v1 + db2*v2 + db3*v3
        H += wa*rowv; Ha += dwa*rowv; Hb += wa*rowd
    return H, Ha/da, Hb/db


@njit(cache=True, fastmath=False, inline='always')
def heval(Hp, a0, da, Na, b0, db, Nb, a, b):
    """Value + gradient with linear, tilt-clamped extrapolation outside."""
    amax = a0 + (Na - 1)*da
    bmax = b0 + (Nb - 1)*db
    ac = a; bc = b
    if ac < a0: ac = a0
    elif ac > amax: ac = amax
    if bc < b0: bc = b0
    elif bc > bmax: bc = bmax
    H, Ha, Hb = _heval_in(Hp, a0, da, Na, b0, db, Nb, ac, bc)
    if ac != a or bc != b:
        if Ha > TILT: Ha = TILT
        elif Ha < -TILT: Ha = -TILT
        if Hb > TILT: Hb = TILT
        elif Hb < -TILT: Hb = -TILT
        H += Ha*(a - ac) + Hb*(b - bc)
    return H, Ha, Hb


@njit(cache=True, fastmath=False)
def _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, c):
    aa = eak*(X - c) + dask*s
    bb = ebk*(X - c) + dbsk*s
    H, Ha, Hb = heval(Hp, a0, da, Na, b0, db, Nb, aa, bb)
    return (X + 2.0*c)/SQ3 - H, Ha, Hb


@njit(cache=True, fastmath=False)
def solve_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s,
            c_init, max_newton):
    """Newton w/ clipped steps + guarded slope; bisection fallback.
    Returns (c, Ha, Hb, F, ok)."""
    c = c_init
    F = 0.0; Ha = 0.0; Hb = 0.0
    for it in range(max_newton):
        F, Ha, Hb = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk,
                             dask, dbsk, s, c)
        if abs(F) < 1e-13:
            return c, Ha, Hb, F, True
        Fc = 2.0/SQ3 + Ha*eak + Hb*ebk
        if Fc < 0.2: Fc = 0.2
        step = F / Fc
        if step > 0.5: step = 0.5
        elif step < -0.5: step = -0.5
        c -= step
    # final check after last step
    F, Ha, Hb = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk,
                         dask, dbsk, s, c)
    if abs(F) < 1e-13:
        return c, Ha, Hb, F, True
    # ---- bisection fallback (F is increasing in c globally thanks to
    # the clamped-tilt extrapolation; inside the grid we just bracket) --
    lo = c - 1.0; hi = c + 1.0
    Flo, _, _ = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, lo)
    Fhi, _, _ = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, hi)
    width = 1.0
    while (Flo > 0.0 or Fhi < 0.0) and width < 64.0:
        width *= 2.0
        lo = c - width; hi = c + width
        Flo, _, _ = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, lo)
        Fhi, _, _ = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, hi)
    if Flo > 0.0 or Fhi < 0.0:
        return c, Ha, Hb, F, False
    for it in range(100):
        mid = 0.5*(lo + hi)
        Fm, Ha, Hb = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, mid)
        if Fm >= 0.0: hi = mid
        else: lo = mid
        if hi - lo < 1e-15:
            break
    c = 0.5*(lo + hi)
    F, Ha, Hb = _F_of_c(Hp, a0, da, Na, b0, db, Nb, X, eak, ebk, dask, dbsk, s, c)
    return c, Ha, Hb, F, abs(F) < 1e-10


@njit(cache=True, fastmath=False)
def clear_crra_jit(mu, gamma):
    """Bisection CRRA clearing (W=1, common gamma) — mirrors
    reznsrc.demand.clear_crra; cross-validated in validate_clear()."""
    EPSP = 1e-12
    K = mu.shape[0]
    a = EPSP; b = 1.0 - EPSP
    fa = 0.0; fb = 0.0
    for k in range(K):
        za = (np.log(mu[k]/(1-mu[k])) - np.log(a/(1-a))) / gamma
        if za >= 0: e = np.exp(-za); fa += (1-e)/((1-a)*e + a)
        else: e = np.exp(za); fa += (e-1)/((1-a) + a*e)
        zb = (np.log(mu[k]/(1-mu[k])) - np.log(b/(1-b))) / gamma
        if zb >= 0: e = np.exp(-zb); fb += (1-e)/((1-b)*e + b)
        else: e = np.exp(zb); fb += (e-1)/((1-b) + b*e)
    if fa <= 0: return a
    if fb >= 0: return b
    for _ in range(60):
        cmid = 0.5*(a + b)
        fc = 0.0
        for k in range(K):
            z = (np.log(mu[k]/(1-mu[k])) - np.log(cmid/(1-cmid))) / gamma
            if z >= 0: e = np.exp(-z); fc += (1-e)/((1-cmid)*e + cmid)
            else: e = np.exp(z); fc += (e-1)/((1-cmid) + cmid*e)
        if fc >= 0: a = cmid
        else: b = cmid
        if (b - a) < 1e-14: break
    return 0.5*(a + b)


@njit(cache=True, fastmath=False)
def surface_residual_jit(H, a0, da, Na, b0, db, Nb, p_m, tau, gamma,
                          va, vb, s_arr, c_io, use_warm, max_newton):
    """Residual at each vertex (va[i], vb[i]) of surface {t=H}.
    c_io (Nv,3,Ns): in: warm-start roots if use_warm; out: solved roots.
    Returns (r, n_fail)."""
    Nv = va.size; Ns = s_arr.size
    ds = s_arr[1] - s_arr[0]
    r = np.empty(Nv)
    n_fail = 0
    Hp = build_pad(H)
    JL0 = np.array([1, 2, 0]); JL1 = np.array([2, 0, 1])
    eav = np.array([E_A[0], E_A[1], E_A[2]])
    ebv = np.array([E_B[0], E_B[1], E_B[2]])
    dasv = np.array([DAS[0], DAS[1], DAS[2]])
    dbsv = np.array([DBS[0], DBS[1], DBS[2]])
    etv = np.array([E_T[0], E_T[1], E_T[2]])
    for v in range(Nv):
        av = va[v]; bv = vb[v]
        tv, _, _ = heval(Hp, a0, da, Na, b0, db, Nb, av, bv)
        u0 = av*eav[0] + bv*ebv[0] + tv*etv[0]
        u1 = av*eav[1] + bv*ebv[1] + tv*etv[1]
        u2 = av*eav[2] + bv*ebv[2] + tv*etv[2]
        uu = np.empty(3); uu[0] = u0; uu[1] = u1; uu[2] = u2
        mu = np.empty(3)
        for k in range(3):
            X = uu[k]
            uj = uu[JL0[k]]; ul = uu[JL1[k]]
            c0v = 0.5*(uj + ul)
            eak = eav[k]; ebk = ebv[k]
            dask = dasv[k]; dbsk = dbsv[k]
            A0 = 0.0; A1 = 0.0
            for i_s in range(Ns):
                s = s_arr[i_s]
                ci = c_io[v, k, i_s] if use_warm else c0v
                c, Ha, Hb, F, ok = solve_c(Hp, a0, da, Na, b0, db, Nb, X,
                                            eak, ebk, dask, dbsk, s, ci,
                                            max_newton)
                if not ok:
                    n_fail += 1
                c_io[v, k, i_s] = c
                Fc = 2.0/SQ3 + Ha*eak + Hb*ebk
                if Fc < FC_FLOOR: Fc = FC_FLOOR
                cp = (Ha*dask + Hb*dbsk) / Fc          # analytic c'(s)
                dsig = np.sqrt((cp + 1.0)**2 + (cp - 1.0)**2)
                ujp = c + s; ulp = c - s
                f0 = np.exp(-0.5*tau*((ujp + 0.5)**2 + (ulp + 0.5)**2))
                f1 = np.exp(-0.5*tau*((ujp - 0.5)**2 + (ulp - 0.5)**2))
                w = ds if (0 < i_s < Ns - 1) else 0.5*ds   # trapezoid
                A0 += w*f0*dsig
                A1 += w*f1*dsig
            f0X = np.exp(-0.5*tau*(X + 0.5)**2)
            f1X = np.exp(-0.5*tau*(X - 0.5)**2)
            num = f1X*A1; den = f0X*A0 + num
            if den <= 0.0:
                mu[k] = 0.5
            else:
                val = num/den
                if val < 1e-12: val = 1e-12
                elif val > 1.0 - 1e-12: val = 1.0 - 1e-12
                mu[k] = val
        r[v] = clear_crra_jit(mu, gamma) - p_m
    return r, n_fail


@njit(cache=True, fastmath=False, parallel=True)
def surface_jacobian_jit(H, a0, da, Na, b0, db, Nb, p_m, tau, gamma,
                          va, vb, s_arr, c_base, r0, eps):
    """Dense FD Jacobian dr/dH (Nv x Na*Nb), warm-started Newton (the
    perturbed roots are within ~eps of the base roots -> 3 iterations)."""
    n = Na*Nb
    Nv = va.size
    J = np.empty((Nv, n))
    for col in prange(n):
        i = col // Nb; jn = col % Nb
        Hpert = H.copy()
        Hpert[i, jn] += eps
        cw = c_base.copy()
        r1, _ = surface_residual_jit(Hpert, a0, da, Na, b0, db, Nb, p_m,
                                      tau, gamma, va, vb, s_arr, cw,
                                      True, 6)
        for v in range(Nv):
            J[v, col] = (r1[v] - r0[v]) / eps
    return J


@njit(cache=True, fastmath=False)
def reconstruct_logitP_jit(heights, logit_p, t_pts):
    """heights (M,) monotone-increasing surface heights at one (a,b);
    piecewise-linear interp of logit p in t, end-slope extrapolation."""
    M = heights.size
    out = np.empty(t_pts.size)
    for i in range(t_pts.size):
        t = t_pts[i]
        if t <= heights[0]:
            sl = (logit_p[1] - logit_p[0]) / max(heights[1] - heights[0], 1e-9)
            out[i] = logit_p[0] + sl*(t - heights[0])
        elif t >= heights[M-1]:
            sl = (logit_p[M-1] - logit_p[M-2]) / max(heights[M-1] - heights[M-2], 1e-9)
            out[i] = logit_p[M-1] + sl*(t - heights[M-1])
        else:
            m = 0
            while heights[m+1] < t:
                m += 1
            frac = (t - heights[m]) / max(heights[m+1] - heights[m], 1e-9)
            out[i] = logit_p[m] + frac*(logit_p[m+1] - logit_p[m])
    return out


# ===================== python layer =====================

def load_P_full(Gi=21, tau=2.0, gamma=0.098):
    du, uf, lo, hi = build_grid(Gi)
    P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma),
                                  np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    return du, uf, lo, hi, P_inner, P_full


def make_levels(P_inner, M):
    qs = np.linspace(0, 1, M + 1)
    edges = np.quantile(P_inner.ravel(), qs)
    return 0.5*(edges[:-1] + edges[1:])


def build_initial_H(P_full, uf, p_levels, a_grid, b_grid,
                    t_range=(-7.0, 7.0), n_t=281):
    """Root-solve P(u(a,b,t)) = p_m along the diagonal at each (a,b)."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((uf, uf, uf), P_full,
                                      bounds_error=False, fill_value=None)
    Na, Nb = a_grid.size, b_grid.size
    M = len(p_levels)
    H = np.zeros((M, Na, Nb))
    ts = np.linspace(t_range[0], t_range[1], n_t)
    for ia, a in enumerate(a_grid):
        for ib, b in enumerate(b_grid):
            pp = (a*E_A + b*E_B)[None, :] + np.multiply.outer(ts, E_T)
            vals = interp(pp)
            vals = np.maximum.accumulate(vals)              # monotone guard
            vals = vals + 1e-12*np.arange(ts.size)          # strict
            H[:, ia, ib] = np.interp(p_levels, vals, ts)
    return H


class Problem:
    def __init__(self, M, Na, half_width, n_vert_margin, tau, gamma,
                 s_max=4.0, n_s=81):
        self.tau, self.gamma = tau, gamma
        self.M = M
        self.a_grid = np.linspace(-half_width, half_width, Na)
        self.b_grid = np.linspace(-half_width, half_width, Na)
        self.Na = Na; self.Nb = Na
        self.a0 = self.a_grid[0]; self.da = self.a_grid[1] - self.a_grid[0]
        self.b0 = self.b_grid[0]; self.db = self.da
        mg = n_vert_margin
        A2, B2 = np.meshgrid(self.a_grid[mg:-mg], self.b_grid[mg:-mg],
                              indexing='ij')
        self.va = np.ascontiguousarray(A2.ravel())
        self.vb = np.ascontiguousarray(B2.ravel())
        self.s_arr = np.linspace(-s_max, s_max, n_s)

    def residual(self, H_m, p_m, c_io=None, use_warm=False, max_newton=40):
        if c_io is None:
            c_io = np.zeros((self.va.size, 3, self.s_arr.size))
            use_warm = False
        r, nf = surface_residual_jit(H_m, self.a0, self.da, self.Na,
                                      self.b0, self.db, self.Nb, p_m,
                                      self.tau, self.gamma, self.va,
                                      self.vb, self.s_arr, c_io,
                                      use_warm, max_newton)
        return r, c_io, nf

    def jacobian(self, H_m, p_m, c_base, r0, eps=1e-6):
        return surface_jacobian_jit(H_m, self.a0, self.da, self.Na,
                                     self.b0, self.db, self.Nb, p_m,
                                     self.tau, self.gamma, self.va,
                                     self.vb, self.s_arr, c_base, r0, eps)


def lm_solve_surface(pb, H0, p_m, tag, max_iter=80, tol=1e-9,
                     lam0=1e-3, log=print):
    """LM on one surface's heights. Returns (H, r, traj)."""
    H = H0.copy()
    r, c_base, nf = pb.residual(H, p_m)
    cost = 0.5*float(r @ r)
    lam = lam0
    traj = [dict(it=0, max_r=float(np.max(np.abs(r))),
                 med_r=float(np.median(np.abs(r))), cost=cost, lam=lam,
                 nfail=int(nf))]
    n = pb.Na*pb.Nb
    I = np.eye(n)
    it = 0
    while it < max_iter and np.max(np.abs(r)) > tol:
        it += 1
        t0 = time.time()
        J = pb.jacobian(H, p_m, c_base, r)
        JtJ = J.T @ J
        Jtr = J.T @ r
        D = np.diag(JtJ).copy()
        Dfl = max(float(D.max())*1e-12, 1e-14)
        accepted = False
        for trial in range(25):
            A = JtJ + lam*np.diag(D + Dfl)
            try:
                d = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            Hn = H + d.reshape(pb.Na, pb.Nb)
            cw = c_base.copy()
            rn, cw, nf = pb.residual(Hn, p_m, cw, use_warm=True,
                                      max_newton=40)
            costn = 0.5*float(rn @ rn)
            pred = -float(d @ Jtr) - 0.5*float(d @ (JtJ @ d))
            rho = (cost - costn)/pred if pred > 0 else -1.0
            if nf == 0 and costn < cost:
                H = Hn; r = rn; c_base = cw; cost = costn
                lam = max(lam/3.0, 1e-12)
                accepted = True
                break
            lam *= 4.0
            if lam > 1e13:
                break
        traj.append(dict(it=it, max_r=float(np.max(np.abs(r))),
                         med_r=float(np.median(np.abs(r))), cost=cost,
                         lam=lam, accepted=accepted, nfail=int(nf),
                         wall=time.time() - t0))
        log(f"    [{tag}] it={it:3d} max|r|={traj[-1]['max_r']:.3e} "
            f"med|r|={traj[-1]['med_r']:.3e} lam={lam:.1e} "
            f"acc={accepted} ({traj[-1]['wall']:.1f}s)")
        if not accepted:
            log(f"    [{tag}] LM stuck (lam={lam:.1e}) — stop")
            break
    return H, r, traj


def signflip_diag(Hs, p_levels):
    """max_m max_(a,b) |H_m(-w) + H_{M-1-m}(w)|; index-flip works because
    the grids are symmetric. Also p-level asymmetry."""
    M = Hs.shape[0]
    dH = max(float(np.max(np.abs(Hs[m][::-1, ::-1] + Hs[M-1-m])))
             for m in range(M))
    dp = max(abs(float(p_levels[m] + p_levels[M-1-m] - 1.0)) for m in range(M))
    return dH, dp


def deficit_from_H(Hs, p_levels, pb, uf, lo, hi):
    """Reconstruct logit P on the inner 21^3 cube from the height family
    (piecewise-linear in t through the (H_m, logit p_m) ladder), then
    UNWEIGHTED 1-R^2 of logit P on u1+u2+u3."""
    ui = uf[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    pts = np.stack([U1.ravel(), U2.ravel(), U3.ravel()], axis=1)
    a = pts @ E_A; b = pts @ E_B; t = pts @ E_T
    M = Hs.shape[0]
    logit_p = np.log(p_levels) - np.log1p(-p_levels)
    # heights of all surfaces at all (a,b)
    hts = np.empty((pts.shape[0], M))
    for m in range(M):
        Hp = build_pad(np.ascontiguousarray(Hs[m]))
        hv = np.empty(pts.shape[0])
        for i in range(pts.shape[0]):
            hv[i], _, _ = heval(Hp, pb.a0, pb.da, pb.Na, pb.b0, pb.db,
                                 pb.Nb, a[i], b[i])
        hts[:, m] = hv
    # enforce monotone ladder pointwise (diagnostic count)
    n_viol = int(np.sum(np.diff(hts, axis=1) <= 0))
    hts = np.maximum.accumulate(hts + 1e-12*np.arange(M)[None, :], axis=1)
    L = np.empty(pts.shape[0])
    for i in range(pts.shape[0]):
        L[i] = reconstruct_logitP_jit(np.ascontiguousarray(hts[i]),
                                       logit_p, np.array([t[i]]))[0]
    T = pts.sum(axis=1)
    d = unweighted_deficit(L, T)
    return d, n_viol, L.reshape(ui.size, ui.size, ui.size)


def unweighted_deficit(L, T):
    Lm = L - L.mean(); Tm = T - T.mean()
    r2 = (Lm @ Tm)**2 / ((Lm @ Lm)*(Tm @ Tm))
    return float(1.0 - r2)


def validate_clear(gamma, n=200, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        mu = rng.uniform(1e-6, 1 - 1e-6, 3)
        p1 = clear_crra_jit(mu, gamma)
        p2 = clear_crra_ref(mu, np.full(3, gamma), np.full(3, 1.0))
        worst = max(worst, abs(p1 - p2))
    return worst


# ===================== driver tasks =====================

def setup_problem(M, Na, margin, tau, gamma, P_full, uf, P_inner):
    pb = Problem(M, Na, half_width=4.5, n_vert_margin=margin,
                 tau=tau, gamma=gamma)
    p_levels = make_levels(P_inner, M)
    H0 = build_initial_H(P_full, uf, p_levels, pb.a_grid, pb.b_grid)
    return pb, p_levels, H0


def t1_validate(pb, p_levels, H0, P_full, uf, seed=42):
    """5 random vertices: root accuracy + curve consistency vs cube P."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((uf, uf, uf), P_full,
                                      bounds_error=False, fill_value=None)
    rng = np.random.default_rng(seed)
    out = []
    for trial in range(5):
        # sample until the lifted vertex lies inside the (padded) cube so
        # the P-consistency check is meaningful
        for _ in range(200):
            m = int(rng.integers(0, pb.M))
            iv = int(rng.integers(0, pb.va.size))
            k = int(rng.integers(0, 3))
            H_m = np.ascontiguousarray(H0[m])
            Hp = build_pad(H_m)
            av, bv = pb.va[iv], pb.vb[iv]
            tv, _, _ = heval(Hp, pb.a0, pb.da, pb.Na, pb.b0, pb.db,
                              pb.Nb, av, bv)
            u = av*E_A + bv*E_B + tv*E_T
            if np.all((u >= uf[2]) & (u <= uf[-3])):
                break
        X = u[k]
        jl0 = [1, 2, 0][k]; jl1 = [2, 0, 1][k]
        c0 = 0.5*(u[jl0] + u[jl1])
        maxF = 0.0
        ys = []
        for s in pb.s_arr:
            c, Ha, Hb, F, ok = solve_c(Hp, pb.a0, pb.da, pb.Na, pb.b0,
                                        pb.db, pb.Nb, X, E_A[k], E_B[k],
                                        DAS[k], DBS[k], s, c0, 40)
            maxF = max(maxF, abs(F))
            y = np.empty(3); y[k] = X; y[jl0] = c + s; y[jl1] = c - s
            ys.append(y)
        ys = np.array(ys)
        inside = np.all((ys >= uf[0]) & (ys <= uf[-1]), axis=1)
        if inside.sum() > 0:
            dP = np.abs(interp(ys[inside]) - p_levels[m])
            dmx, dmd = float(dP.max()), float(np.median(dP))
        else:
            dmx, dmd = float('nan'), float('nan')
        out.append(dict(m=m, vertex=[float(av), float(bv)], k=k,
                        max_rootF=float(maxF),
                        n_inside=int(inside.sum()),
                        dP_max=dmx, dP_med=dmd))
    return out


def t2_initial_residual(pb, p_levels, H0, log=print):
    t0 = time.time()
    res = []
    per = []
    for m in range(pb.M):
        r, _, nf = pb.residual(np.ascontiguousarray(H0[m]), p_levels[m])
        res.append(r)
        per.append(dict(p=float(p_levels[m]), max=float(np.max(np.abs(r))),
                        med=float(np.median(np.abs(r))), nfail=int(nf)))
        log(f"  p={p_levels[m]:.3f}: max|r|={per[-1]['max']:.3e} "
            f"med|r|={per[-1]['med']:.3e} nfail={nf}")
    r_all = np.concatenate(res)
    stats = dict(max=float(np.max(np.abs(r_all))),
                 med=float(np.median(np.abs(r_all))),
                 p90=float(np.percentile(np.abs(r_all), 90)),
                 n_vert=int(r_all.size), wall=time.time() - t0,
                 per_surface=per)
    return stats


def run(args):
    os.makedirs(OUT, exist_ok=True)
    tau, gamma = 2.0, 0.098
    du, uf, lo, hi, P_inner, P_full = load_P_full(21, tau, gamma)
    results = {}
    ckpt_path = f"{OUT}/stage6b_results.json"
    if os.path.exists(ckpt_path) and not args.fresh:
        results = json.load(open(ckpt_path))

    def save():
        json.dump(results, open(ckpt_path, "w"), indent=2)

    # reference: unweighted deficit of the warm-start cube
    Lp = np.log(np.clip(P_inner, 1e-12, 1-1e-12))
    Lp = Lp - np.log1p(-np.clip(P_inner, 1e-12, 1-1e-12))
    ui = uf[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    d_kernel = unweighted_deficit(Lp.ravel(), (U1 + U2 + U3).ravel())
    results['deficit_kernel_cube_unweighted'] = d_kernel
    print(f"warm-start cube unweighted deficit = {d_kernel:.4f} "
          f"(target kernel value 0.2796)")

    results['clear_crosscheck_maxdiff'] = validate_clear(gamma)
    print(f"clear_crra cross-check max diff = "
          f"{results['clear_crosscheck_maxdiff']:.2e}")

    # ---------------- coarse problem ----------------
    M, Na, margin = args.M, args.Na, args.margin
    print(f"Setting up: M={M}, H grid {Na}x{Na}, vertex margin {margin}")
    t0 = time.time()
    pb, p_levels, H0 = setup_problem(M, Na, margin, tau, gamma, P_full,
                                      uf, P_inner)
    print(f"  initial H built ({time.time()-t0:.0f}s); "
          f"vertices/surface = {pb.va.size}, state/surface = {Na*Na}")
    dH, dp = signflip_diag(H0, p_levels)
    print(f"  sign-flip diag (initial): max|H_m(-w)+H_(M-1-m)(w)| = {dH:.3e}, "
          f"p-level asym = {dp:.3e}")
    results['config'] = dict(M=M, Na=Na, margin=margin,
                             n_vert=int(pb.va.size),
                             p_levels=[float(p) for p in p_levels],
                             s_max=float(pb.s_arr[-1]), n_s=int(pb.s_arr.size))
    results['signflip_initial'] = dict(dH=dH, dp=dp)

    # T1
    print("T1: slice-root validation at initial H ...")
    results['T1'] = t1_validate(pb, p_levels, H0, P_full, uf)
    for d in results['T1']:
        print(f"  m={d['m']} vert=({d['vertex'][0]:+.2f},{d['vertex'][1]:+.2f}) "
              f"k={d['k']}: max|F|={d['max_rootF']:.2e} "
              f"|P-p_m| max={d['dP_max']:.2e} med={d['dP_med']:.2e} "
              f"({d['n_inside']}/{pb.s_arr.size} pts in cube)")
    save()

    # T2
    print("T2: initial residual over all vertices ...")
    results['T2'] = t2_initial_residual(pb, p_levels, H0)
    s = results['T2']
    gate = 0.06 <= s['med'] <= 0.54
    results['T2']['gate_V1'] = bool(gate)
    print(f"  TOTAL: max={s['max']:.3e} med={s['med']:.3e} p90={s['p90']:.3e} "
          f"({s['wall']:.1f}s, {s['n_vert']} vertices)  "
          f"gate V1 (med in [0.06,0.54] vs mesh 0.18): "
          f"{'PASS' if gate else 'FAIL'}")
    save()
    if args.t12_only:
        return

    # T3: LM per surface (surfaces are independent)
    print("T3: LM solve per surface ...")
    Hs = H0.copy()
    results['T3'] = {}
    finals = []
    for m in range(pb.M):
        t0 = time.time()
        Hm, rm, traj = lm_solve_surface(pb, np.ascontiguousarray(H0[m]),
                                         p_levels[m], tag=f"m={m}",
                                         max_iter=args.max_iter,
                                         tol=args.tol)
        Hs[m] = Hm
        finals.append(rm)
        results['T3'][f"m{m}"] = dict(p=float(p_levels[m]), traj=traj,
                                       final_max=float(np.max(np.abs(rm))),
                                       final_med=float(np.median(np.abs(rm))),
                                       wall=time.time() - t0)
        print(f"  surface m={m} (p={p_levels[m]:.3f}): "
              f"final max|r|={results['T3'][f'm{m}']['final_max']:.3e} "
              f"({results['T3'][f'm{m}']['wall']:.0f}s)")
        np.save(f"{OUT}/stage6b_H.npy", Hs)
        save()
    r_all = np.concatenate(finals)
    results['final'] = dict(max=float(np.max(np.abs(r_all))),
                            med=float(np.median(np.abs(r_all))),
                            p90=float(np.percentile(np.abs(r_all), 90)))
    converged = results['final']['max'] < 1e-6
    results['final']['converged_1e-6'] = bool(converged)
    print(f"FINAL: max|r|={results['final']['max']:.3e} "
          f"med|r|={results['final']['med']:.3e} converged(<1e-6)={converged}")
    dH, dp = signflip_diag(Hs, p_levels)
    results['signflip_final'] = dict(dH=dH, dp=dp)
    print(f"  sign-flip diag (final): {dH:.3e}")
    save()

    # T4: deficit
    print("T4: deficit from reconstructed P ...")
    d0, nv0, _ = deficit_from_H(H0, p_levels, pb, uf, lo, hi)
    dF_, nvF, Lrec = deficit_from_H(Hs, p_levels, pb, uf, lo, hi)
    results['T4'] = dict(deficit_initialH=d0, deficit_finalH=dF_,
                         ladder_violations_initial=nv0,
                         ladder_violations_final=nvF,
                         d_inf=0.268, d_inf_err=0.010,
                         d_kernel_G21=0.2796)
    print(f"  deficit(reconstr, initial H) = {d0:.4f}  [reconstruction check "
          f"vs cube {d_kernel:.4f}]")
    print(f"  deficit(reconstr, final H)   = {dF_:.4f}  "
          f"[d_inf = 0.268 +/- 0.010, kernel G=21 = 0.2796]")
    save()
    np.save(f"{OUT}/stage6b_H.npy", Hs)
    print("done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=8)
    ap.add_argument('--Na', type=int, default=15)
    ap.add_argument('--margin', type=int, default=1)
    ap.add_argument('--max-iter', type=int, default=80)
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--t12-only', action='store_true')
    ap.add_argument('--fresh', action='store_true')
    run(ap.parse_args())


if __name__ == "__main__":
    main()
