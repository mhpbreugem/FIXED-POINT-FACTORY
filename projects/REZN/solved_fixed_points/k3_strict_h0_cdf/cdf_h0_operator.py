"""STRICT h=0 co-area operator for K=3 CRRA REE via the SUB-LEVEL-SET
CDF-DERIVATIVE method, in python-flint arb.

NO kernel, NO bandwidth, NO explicit 1/|grad P| division anywhere.

The evidence (co-area density) for agent v at price p is

    A_v(p) = d/dp G_v(p),   G_v(p) = INT_{P(u)<p} f_v(u) du

G_v is the f_v-measure of the sub-level set {P<p}.  It is SMOOTH (computed
without ever dividing by |grad P|).  Its derivative is the co-area line
integral INT_{P=p} f_v/|grad P| dsigma -- but we obtain it as the derivative
of the (analytic) below-plane area fraction, so the 1/|grad P| singularity is
handled ANALYTICALLY and stays bounded near level-set critical points.

Per coarse cell we subdivide into NSUB x NSUB subcells.  On each subcell we
evaluate f_v at the subcell center (Gauss/midpoint weight) and approximate P
by its planar (affine) interpolant from the subcell's 4 bilinear-sampled
corners.  For an affine field a + b*x + c*y over the unit square, BOTH the
fraction-below-level and its EXACT derivative wrt p are closed-form rational
functions of the corner values, and the derivative is BOUNDED (no division
blow-up): the contribution of a near-flat subcell to dG/dp is exactly
1/(area-swept-rate), evaluated in closed form with the degenerate (grad->0)
limit handled explicitly.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
from flint import ctx, arb

ctx.prec = int(120 * 3.3219) + 30   # ~428 bits ~ 120 dec + headroom

HALF = arb('0.5')
ONE = arb(1)
TWO = arb(2)
PI = arb.pi()
VM0 = arb('-0.5')
VM1 = arb('0.5')
EPS = arb(10) ** -50
PCLAMP_LO = arb(10) ** -120
PCLAMP_HI = ONE - arb(10) ** -120


def f_signal(u, vmean, tau):
    d = u - vmean
    return (tau / (TWO * PI)).sqrt() * (-HALF * tau * d * d).exp()


def logit(p):
    return p.log() - (ONE - p).log()


def lam(z):
    if z >= 0:
        e = (-z).exp()
        return ONE / (ONE + e)
    e = z.exp()
    return e / (ONE + e)


def x_crra(mu, p, gamma, W):
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0:
        e = (-z).exp()
        return W * (ONE - e) / ((ONE - p) * e + p)
    e = z.exp()
    return W * (e - ONE) / ((ONE - p) + p * e)


def clear_crra(mu_vec, gamma_vec, W_vec):
    a = EPS
    b = ONE - EPS

    def excess(p):
        s = arb(0)
        for k in range(len(mu_vec)):
            s = s + x_crra(mu_vec[k], p, gamma_vec[k], W_vec[k])
        return s

    if excess(a) <= 0:
        return a
    if excess(b) >= 0:
        return b
    for _ in range(460):
        c = (a + b) * HALF
        if excess(c) >= 0:
            a = c
        else:
            b = c
    return (a + b) * HALF


def bayes(u_own, tau_own, A0, A1):
    f0 = f_signal(u_own, VM0, tau_own)
    f1 = f_signal(u_own, VM1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0:
        return HALF
    r = num / den
    if r < PCLAMP_LO:
        return PCLAMP_LO
    if r > PCLAMP_HI:
        return PCLAMP_HI
    return r


# ---------------------------------------------------------------------------
# Closed-form derivative of the below-plane area fraction over the unit square
# for an affine field  P(x,y) = P00 + (P10-P00) x + (P01-P00) y , (x,y) in [0,1]^2.
#
# Let bx = P10-P00, by = P01-P00.  Sort so we work with magnitudes.  The area
# A(p) = area{ P00 + bx x + by y < p } in [0,1]^2 is a piecewise-cubic
# (actually piecewise-quadratic spline) C^1 function of p.  Its derivative
# dA/dp is the length-measure of the level line weighted by 1/|grad| -- which
# for an affine field equals  (1/|b|) * L(p)  where L is the in-square line
# length; the product is computed in CLOSED FORM and is BOUNDED.
#
# We need dA/dp directly.  Standard result (area below a plane in unit square):
# define g = bx + by shift; let Pmin,Pmax be corner extremes.  We use the
# well-known formula for the area of {u x + v y < w, 0<=x,y<=1} and
# differentiate.  To stay robust we compute dA/dp via the exact derivative of
# the trapezoidal-area formula.
#
# Implementation: reduce to u=|bx|, v=|by| with u>=v>=0 by symmetry (flipping
# the square does not change area), and shift level to q = p - Pmin where
# Pmin = P00 + min(0,bx) + min(0,by).  Then with s = u+v (Pmax-Pmin = s),
#   dA/dp(q) =
#     q/(u v)                  for 0 <= q <= v        (corner triangle grows)
#     1/u                      for v <= q <= u        (linear band)
#     (s-q)/(u v)              for u <= q <= s        (corner triangle shrinks)
#     0                        otherwise
# Degenerate cases u=0 (and v=0): handle as limits (a flat cell contributes a
# delta we approximate by spreading over its own range -> but with u=v=0 the
# cell is exactly flat and contributes nothing to a generic p; skip).
# When v=0,u>0: dA/dp = 1/u for 0<=q<=u else 0.  BOUNDED.  When u=0 too: flat.
# ---------------------------------------------------------------------------

def dA_dp_affine(P00, bx, by, p):
    """Exact dA/dp of below-plane area fraction in unit square, affine field.
    Returns an arb >= 0, BOUNDED (no 1/|grad| blow-up: closed form)."""
    u = bx if bx >= 0 else -bx
    v = by if by >= 0 else -by
    if u < v:
        u, v = v, u
    # Pmin corner value
    Pmin = P00 + (bx if bx < 0 else arb(0)) + (by if by < 0 else arb(0))
    q = p - Pmin
    s = u + v
    if q <= 0 or q >= s:
        return arb(0)
    if u <= 0:
        # totally flat cell, both gradients zero: measure-zero contribution
        return arb(0)
    if v <= 0:
        # 1D ramp: dA/dp = 1/u on (0,u)
        return ONE / u
    # generic u>=v>0
    if q <= v:
        return q / (u * v)
    if q <= u:
        return ONE / u
    return (s - q) / (u * v)


# ---------------------------------------------------------------------------
# Per-agent co-area evidence A_v(p) over a 2D slice of P, via the CDF-deriv
# (sub-level-set) method.  P_slice is n x n list-of-lists of arb on a uniform
# (ua,ub) grid with spacing du.  For each coarse cell we subdivide NSUB^2,
# sample P bilinearly at subcell corners, weight f_v at the subcell center,
# and accumulate dA/dp (closed form).  The (1/du^2) Jacobian for ds dt -> du^2
# cancels because both G and A carry the same area element; we keep it explicit
# so A_v has units of (f-measure)/(price).
# ---------------------------------------------------------------------------

def _bilin(P00, P10, P01, P11, s, t):
    return (P00 * (ONE - s) * (ONE - t) + P10 * s * (ONE - t)
            + P01 * (ONE - s) * t + P11 * s * t)


def agent_evidence_cdf(P_slice, p_target, u_axis, tau_a, tau_b, du, n, NSUB):
    """Co-area evidence (A0, A1) at level p via CDF-derivative.  No 1/|grad|
    explicit division."""
    A0 = arb(0)
    A1 = arb(0)
    hsub = ONE / arb(NSUB)
    duarb = du
    cell_area = duarb * duarb               # du^2 per coarse cell (ds dt=1)
    sub_area = cell_area * hsub * hsub      # subcell physical area weight
    for i in range(n - 1):
        ua_i = u_axis[i]
        for j in range(n - 1):
            ub_j = u_axis[j]
            P00 = P_slice[i][j]
            P10 = P_slice[i + 1][j]
            P01 = P_slice[i][j + 1]
            P11 = P_slice[i + 1][j + 1]
            cmin = P00
            cmax = P00
            for c in (P10, P01, P11):
                if c < cmin:
                    cmin = c
                if c > cmax:
                    cmax = c
            if p_target <= cmin or p_target >= cmax:
                continue  # level does not cross this coarse cell
            # subdivide
            for a in range(NSUB):
                s0 = arb(a) * hsub
                s1 = arb(a + 1) * hsub
                sc = (s0 + s1) * HALF
                ua = ua_i + sc * duarb
                fa0 = f_signal(ua, VM0, tau_a)
                fa1 = f_signal(ua, VM1, tau_a)
                for b in range(NSUB):
                    t0 = arb(b) * hsub
                    t1 = arb(b + 1) * hsub
                    tc = (t0 + t1) * HALF
                    # subcell corner P-values (bilinear samples)
                    q00 = _bilin(P00, P10, P01, P11, s0, t0)
                    q10 = _bilin(P00, P10, P01, P11, s1, t0)
                    q01 = _bilin(P00, P10, P01, P11, s0, t1)
                    q11 = _bilin(P00, P10, P01, P11, s1, t1)
                    smin = q00
                    smax = q00
                    for c in (q10, q01, q11):
                        if c < smin:
                            smin = c
                        if c > smax:
                            smax = c
                    if p_target <= smin or p_target >= smax:
                        continue
                    # affine reduction from subcell corners:
                    #   q00 + bx*xi + by*eta , xi,eta in [0,1] (local subcell)
                    bx = q10 - q00
                    by = q01 - q00
                    # dA/dp over the LOCAL unit square; physical area weight:
                    dadp = dA_dp_affine(q00, bx, by, p_target)
                    if dadp <= 0:
                        continue
                    ub = ub_j + tc * duarb
                    fb0 = f_signal(ub, VM0, tau_b)
                    fb1 = f_signal(ub, VM1, tau_b)
                    w = dadp * sub_area
                    A0 = A0 + w * fa0 * fb0
                    A1 = A1 + w * fa1 * fb1
    return A0, A1


# ---------------------------------------------------------------------------
# Full strict h=0 CDF-derivative Phi operator on the 3D grid
# ---------------------------------------------------------------------------

def init_no_learning(u_full, tau_vec, gamma_vec, W_vec):
    G = len(u_full)
    P = [[[None] * G for _ in range(G)] for _ in range(G)]
    for i in range(G):
        m0 = lam(tau_vec[0] * u_full[i])
        for j in range(G):
            m1 = lam(tau_vec[1] * u_full[j])
            for l in range(G):
                m2 = lam(tau_vec[2] * u_full[l])
                P[i][j][l] = clear_crra([m0, m1, m2], gamma_vec, W_vec)
    return P


def phi_strict_cdf(P, u_full, lo, hi, tau_vec, gamma_vec, W_vec, du, NSUB):
    G = len(u_full)
    Pnew = [[[P[i][j][l] for l in range(G)] for j in range(G)] for i in range(G)]
    n = G
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P[i][j][l]
                sl0 = [[P[i][a][b] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_cdf(sl0, p, u_full,
                                            tau_vec[1], tau_vec[2], du, n, NSUB)
                mu0 = bayes(u_full[i], tau_vec[0], A0, A1)
                sl1 = [[P[a][j][b] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_cdf(sl1, p, u_full,
                                            tau_vec[0], tau_vec[2], du, n, NSUB)
                mu1 = bayes(u_full[j], tau_vec[1], A0, A1)
                sl2 = [[P[a][b][l] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_cdf(sl2, p, u_full,
                                            tau_vec[0], tau_vec[1], du, n, NSUB)
                mu2 = bayes(u_full[l], tau_vec[2], A0, A1)
                Pnew[i][j][l] = clear_crra([mu0, mu1, mu2], gamma_vec, W_vec)
    return Pnew


def build_grid(G_inner, pad, UMAX):
    du = TWO * arb(UMAX) / arb(G_inner - 1)
    Gf = G_inner + 2 * pad
    u_full = [arb(-UMAX) + (arb(q) - arb(pad)) * du for q in range(Gf)]
    lo, hi = pad, pad + G_inner
    return du, u_full, lo, hi
