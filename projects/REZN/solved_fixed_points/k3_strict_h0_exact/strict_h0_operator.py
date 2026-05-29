"""STRICTLY h=0 co-area level-set operator for K=3 CRRA REE, in python-flint arb.

NO kernel, NO bandwidth, NO smoothing parameter anywhere. The level set {P=p}
is tracked as a continuous curve via MARCHING SQUARES on bilinearly-interpolated
P, and the co-area weight 1/|grad P| is exact. Evidence

    A_v(p) = int_{P=p} f_v(u_a) f_v(u_b) / |grad P| dsigma

is computed by summing exact contour-segment integrals (3-pt Gauss-Legendre per
segment, bilinear gradient at the midpoint). All arithmetic in flint arb.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
from flint import ctx, arb

# ~130 decimal precision + headroom
ctx.prec = int(130 * 3.3219) + 30   # ~462 bits

# ---- constants as arb ----
HALF = arb('0.5')
ONE = arb(1)
TWO = arb(2)
PI = arb.pi()
VM0 = arb('-0.5')   # signal mean under v=0
VM1 = arb('0.5')    # signal mean under v=1
EPS = arb(10) ** -50            # keep probs strictly inside (0,1) for clearing band edges
PCLAMP_LO = arb(10) ** -60
PCLAMP_HI = ONE - arb(10) ** -60

# 3-point Gauss-Legendre nodes/weights on [-1,1]
_g = (arb(3) / arb(5)).sqrt()
GL_NODES = [-_g, arb(0), _g]
GL_W = [arb(5) / arb(9), arb(8) / arb(9), arb(5) / arb(9)]


def f_signal(u, vmean, tau):
    """Gaussian signal density sqrt(tau/2pi) exp(-0.5 tau (u-vmean)^2)."""
    d = u - vmean
    return (tau / (TWO * PI)).sqrt() * (-HALF * tau * d * d).exp()


def logit(p):
    return p.log() - (ONE - p).log()


def x_crra(mu, p, gamma, W):
    """CRRA demand, rearranged for stability (matches reference_operator)."""
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0:
        e = (-z).exp()
        return W * (ONE - e) / ((ONE - p) * e + p)
    e = z.exp()
    return W * (e - ONE) / ((ONE - p) + p * e)


def clear_crra(mu_vec, gamma_vec, W_vec):
    """Market clearing price via flint bisection. Excess demand decreasing in p."""
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
    # ~ (462/1) bits of bisection -> need ~ 470 steps for 1e-130; use 480
    for _ in range(480):
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
# Marching-squares co-area contour integration over a 2D P-slice.
#
# Slice P is indexed [a, b] where signal of "other agent A" runs along axis a
# (value ua = u_axis[a]) and agent B along axis b (ub = u_axis[b]).  Within a
# cell with corners
#     P00=P[i,j]   P10=P[i+1,j]   P01=P[i,j+1]   P11=P[i+1,j+1]
# we use bilinear interpolation in local coords (s,t) in [0,1]^2 mapping to
# ua = ua_i + s*du,  ub = ub_j + t*du:
#     P(s,t) = P00(1-s)(1-t) + P10 s(1-t) + P01(1-s)t + P11 s t
# The marching-squares contour {P=p} crosses cell edges; we integrate
#     f_a(ua) f_b(ub) / |grad P| along each segment via 3-pt Gauss-Legendre.
# |grad P| is in (ua,ub) units: dP/dua = (1/du) dP/ds, dP/dub=(1/du) dP/dt.
# ---------------------------------------------------------------------------

def _interp_edge(pa, pb, p):
    """Fraction along an edge where P goes pa->pb crossing level p. In [0,1]."""
    d = pb - pa
    # callers only invoke when (pa-p) and (pb-p) straddle 0, so d != 0
    return (p - pa) / d


def agent_evidence_strict(P_slice, p_target, u_axis, tau_a, tau_b, du, n):
    """Exact co-area evidence A_v(p) for one agent via marching squares.

    P_slice: list-of-lists n x n of arb. u_axis: list of n arb signal coords.
    Returns (A0, A1).
    """
    A0 = arb(0)
    A1 = arb(0)
    invdu = ONE / du
    for i in range(n - 1):
        ua_i = u_axis[i]
        for j in range(n - 1):
            ub_j = u_axis[j]
            P00 = P_slice[i][j]
            P10 = P_slice[i + 1][j]
            P01 = P_slice[i][j + 1]
            P11 = P_slice[i + 1][j + 1]
            # corner sign relative to level p (>=0 means corner value >= p)
            c00 = P00 >= p_target
            c10 = P10 >= p_target
            c01 = P01 >= p_target
            c11 = P11 >= p_target
            case = (8 if c00 else 0) | (4 if c10 else 0) | \
                   (2 if c11 else 0) | (1 if c01 else 0)
            if case == 0 or case == 15:
                continue
            # edge crossing points in local (s,t):
            #   bottom edge t=0: P00->P10  (corners 00,10)
            #   right  edge s=1: P10->P11  (corners 10,11)
            #   top    edge t=1: P01->P11  (corners 01,11)
            #   left   edge s=0: P00->P01  (corners 00,01)
            seglist = _marching_segments(case, P00, P10, P01, P11, p_target)
            for (s0, t0, s1, t1) in seglist:
                A0d, A1d = _integrate_segment(
                    s0, t0, s1, t1, ua_i, ub_j, du, invdu,
                    P00, P10, P01, P11, tau_a, tau_b)
                A0 = A0 + A0d
                A1 = A1 + A1d
    return A0, A1


def _marching_segments(case, P00, P10, P01, P11, p):
    """Return list of (s0,t0,s1,t1) segment endpoint local coords for a cell.

    Edge param helpers (each returns a point on the named edge):
      bottom: (e_b, 0)    e_b = frac(P00->P10)
      right : (1, e_r)    e_r = frac(P10->P11)
      top   : (e_t, 1)    e_t = frac(P01->P11)
      left  : (0, e_l)    e_l = frac(P00->P01)
    """
    def eb():
        return _interp_edge(P00, P10, p)

    def er():
        return _interp_edge(P10, P11, p)

    def et():
        return _interp_edge(P01, P11, p)

    def el():
        return _interp_edge(P00, P01, p)

    # standard 16 marching-squares cases (case bits: 8=00,4=10,2=11,1=01)
    if case == 1 or case == 14:      # corner 01 isolated -> left/top
        return [(0, el(), et(), 1)]
    if case == 2 or case == 13:      # corner 11 -> top/right
        return [(et(), 1, 1, er())]
    if case == 4 or case == 11:      # corner 10 -> bottom/right
        return [(eb(), 0, 1, er())]
    if case == 8 or case == 7:       # corner 00 -> left/bottom
        return [(0, el(), eb(), 0)]
    if case == 3 or case == 12:      # 01,11 vs 00,10 -> horizontal: left/right
        return [(0, el(), 1, er())]
    if case == 6 or case == 9:       # 10,11 vs 00,01 -> vertical: bottom/top
        return [(eb(), 0, et(), 1)]
    if case == 5 or case == 10:      # saddle (00,11 vs 10,01)
        # resolve via bilinear center value
        center = (P00 + P10 + P01 + P11) * arb('0.25')
        if case == 5:   # corners 00,11 high (case bits 8|... no): 5=4|1 ->10,01 high
            # case 5: c10 and c01 true (value>=p), c00,c11 false
            if center >= p:
                # connect such that high regions joined: bottom-right & top-left
                return [(eb(), 0, 1, er()), (0, el(), et(), 1)]
            else:
                return [(eb(), 0, 0, el()), (et(), 1, 1, er())]
        else:  # case 10: c00,c11 true
            if center >= p:
                return [(0, el(), eb(), 0), (et(), 1, 1, er())]
            else:
                return [(0, el(), et(), 1), (eb(), 0, 1, er())]
    return []


def _integrate_segment(s0, t0, s1, t1, ua_i, ub_j, du, invdu,
                       P00, P10, P01, P11, tau_a, tau_b):
    """3-pt Gauss-Legendre integral of f_a f_b / |gradP| along the segment
    from local (s0,t0) to (s1,t1). dsigma is arc length in (ua,ub) units."""
    ds = s1 - s0
    dt = t1 - t0
    # physical segment length (ua=ua_i+s*du, ub=ub_j+t*du)
    seg_len = du * (ds * ds + dt * dt).sqrt()
    if seg_len <= 0:
        return arb(0), arb(0)
    A0 = arb(0)
    A1 = arb(0)
    for k in range(3):
        # node in [-1,1] -> param r in [0,1]
        r = (GL_NODES[k] + ONE) * HALF
        s = s0 + r * ds
        t = t0 + r * dt
        ua = ua_i + s * du
        ub = ub_j + t * du
        # bilinear gradient in local coords
        # P(s,t)=P00(1-s)(1-t)+P10 s(1-t)+P01(1-s)t+P11 s t
        dPds = (P10 - P00) * (ONE - t) + (P11 - P01) * t
        dPdt = (P01 - P00) * (ONE - s) + (P11 - P10) * s
        gx = dPds * invdu
        gy = dPdt * invdu
        gnorm = (gx * gx + gy * gy).sqrt()
        f0 = f_signal(ua, VM0, tau_a) * f_signal(ub, VM0, tau_b)
        f1 = f_signal(ua, VM1, tau_a) * f_signal(ub, VM1, tau_b)
        # weight: GL weight * (jacobian of [-1,1]->[0,1] is 1/2) * seg_len
        w = GL_W[k] * HALF * seg_len
        A0 = A0 + w * f0 / gnorm
        A1 = A1 + w * f1 / gnorm
    return A0, A1


# ---------------------------------------------------------------------------
# Full Phi operator on the 3D grid (flint arb everywhere)
# ---------------------------------------------------------------------------

def lam(z):
    """logistic 1/(1+e^-z) in arb."""
    if z >= 0:
        e = (-z).exp()
        return ONE / (ONE + e)
    e = z.exp()
    return e / (ONE + e)


def init_no_learning(u_full, tau_vec, gamma_vec, W_vec):
    """No-learning surface (own signal only). Returns G x G x G list of arb."""
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


def _col(P, axis, fix_a, fix_b):
    """Extract the 2D slice of P over the two axes != `axis` at fixed indices.

    axis=0: slice over (j,l) at i fixed=fix_a  (but fix_a is i) -> handled below
    We return slice[p][q] as arb. Convention matches reference:
      agent0 sees (u2,u3) slice at fixed u1=i -> P[i,:,:]
      agent1 sees (u1,u3) slice at fixed u2=j -> P[:,j,:]
      agent2 sees (u1,u2) slice at fixed u3=l -> P[:,:,l]
    """
    pass


def phi_strict(P, u_full, lo, hi, tau_vec, gamma_vec, W_vec, du):
    """One application of the strict h=0 operator. P is G^3 list-of-lists of arb.
    Returns new P (full grid; boundary copied)."""
    G = len(u_full)
    Pnew = [[[P[i][j][l] for l in range(G)] for j in range(G)] for i in range(G)]
    n = G  # contour integration uses full grid extent
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P[i][j][l]
                # agent0: slice P[i,:,:] over (u2,u3); a-axis=u2(tau1), b-axis=u3(tau2)
                sl0 = [[P[i][a][b] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_strict(sl0, p, u_full,
                                               tau_vec[1], tau_vec[2], du, n)
                mu0 = bayes(u_full[i], tau_vec[0], A0, A1)
                # agent1: slice P[:,j,:] over (u1,u3); a-axis=u1(tau0), b-axis=u3(tau2)
                sl1 = [[P[a][j][b] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_strict(sl1, p, u_full,
                                               tau_vec[0], tau_vec[2], du, n)
                mu1 = bayes(u_full[j], tau_vec[1], A0, A1)
                # agent2: slice P[:,:,l] over (u1,u2); a-axis=u1(tau0), b-axis=u2(tau1)
                sl2 = [[P[a][b][l] for b in range(G)] for a in range(G)]
                A0, A1 = agent_evidence_strict(sl2, p, u_full,
                                               tau_vec[0], tau_vec[1], du, n)
                mu2 = bayes(u_full[l], tau_vec[2], A0, A1)
                Pnew[i][j][l] = clear_crra([mu0, mu1, mu2], gamma_vec, W_vec)
    return Pnew


def build_grid(G_inner, pad, UMAX):
    du = TWO * arb(UMAX) / arb(G_inner - 1)
    Gf = G_inner + 2 * pad
    u_full = [arb(-UMAX) + (arb(q) - arb(pad)) * du for q in range(Gf)]
    lo, hi = pad, pad + G_inner
    return du, u_full, lo, hi
