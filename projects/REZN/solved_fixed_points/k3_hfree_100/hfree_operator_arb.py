"""h-FREE SMOOTH co-area operator for the K=3 CRRA REE -- 100-DECIMAL (arb).

EXACT port of /tmp/hfree_operator.py (Numba float64) to python-flint (arb).
Only the arithmetic is changed to arb (Arb ball arithmetic at ~110-decimal
working precision, headroom for a 1e-100 fixed-point nail). The algorithm is
mirrored line-for-line:

  lam, logit, f_signal, x_crra, clear_crra (bisection), bayes,
  natural_spline_M (tridiagonal Thomas solve), spline_eval (C2 cubic),
  spline_roots (sub-bracket sign change + Newton/bisection for ALL roots),
  slice_evidence (co-area partition-of-unity over fixed Gauss-Legendre
  transverse nodes, w2=d2P^2/(d2P^2+d3P^2), w3=1-w2), phi_hfree.

NO kernel, NO bandwidth, NO smoothing parameter. The only discretizations are
the working grid G and the number of Gauss-Legendre nodes Nq.
"""
from __future__ import annotations
import flint
from flint import arb

# ---- precision: ~110 decimals, headroom over the 1e-100 target ----
DPS = 110
flint.ctx.prec = int(DPS * 3.3219) + 30  # 395 bits at DPS=110

ZERO = arb(0)
ONE = arb(1)
HALF = arb("0.5")
EPS_PRICE = arb(10) ** (-12)        # mirrors 1e-12 float64
NEWTON_TOL = arb(10) ** (-30)       # tighter than float64 1e-14
ROOT_FTOL = arb(10) ** (-40)
BISECT_TOL = arb(10) ** (-40)       # clear_crra bracket tol


# ----------------------------------------------------------------------
def lam(z):
    if z >= 0:
        e = (-z).exp()
        return ONE / (ONE + e)
    e = z.exp()
    return e / (ONE + e)


def logit(p):
    return p.log() - (ONE - p).log()


def _twopi():
    return arb(2) * arb.pi()


def f_signal(u, v, tau, twopi):
    mean = HALF if v == 1 else -HALF
    d = u - mean
    return (tau / twopi).sqrt() * (arb("-0.5") * tau * d * d).exp()


def x_crra(mu, p, gamma, W):
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0:
        e = (-z).exp()
        return W * (ONE - e) / ((ONE - p) * e + p)
    e = z.exp()
    return W * (e - ONE) / ((ONE - p) + p * e)


def clear_crra(mu0, mu1, mu2, g0, g1, g2, W0, W1, W2):
    a = EPS_PRICE
    b = ONE - EPS_PRICE

    def ex(p):
        return (x_crra(mu0, p, g0, W0) + x_crra(mu1, p, g1, W1)
                + x_crra(mu2, p, g2, W2))
    fa = ex(a)
    fb = ex(b)
    if fa <= 0:
        return a
    if fb >= 0:
        return b
    for _ in range(400):
        c = HALF * (a + b)
        fc = ex(c)
        if fc >= 0:
            a = c
        else:
            b = c
        if (b - a) < BISECT_TOL:
            break
    return HALF * (a + b)


def bayes(u_own, tau_own, A0, A1, twopi):
    f0 = f_signal(u_own, 0, tau_own, twopi)
    f1 = f_signal(u_own, 1, tau_own, twopi)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0:
        return HALF
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > ONE - EPS_PRICE:
        return ONE - EPS_PRICE
    return mu


# ----------------------------------------------------------------------
# Natural cubic spline 2nd derivatives (Thomas) -- y is a python list of arb.
# ----------------------------------------------------------------------
def natural_spline_M(y, h):
    n = len(y)
    M = [ZERO] * n
    if n < 3:
        return M
    h2 = h * h
    six_h2 = arb(6) / h2
    rhs = [ZERO] * n
    for i in range(1, n - 1):
        rhs[i] = six_h2 * (y[i - 1] - arb(2) * y[i] + y[i + 1])
    c = [ZERO] * n
    d = [ZERO] * n
    b0 = arb(4)
    c[1] = ONE / b0
    d[1] = rhs[1] / b0
    for i in range(2, n - 1):
        m = arb(4) - c[i - 1]
        c[i] = ONE / m
        d[i] = (rhs[i] - d[i - 1]) / m
    for i in range(n - 2, 0, -1):
        M[i] = d[i] - c[i] * M[i + 1]
    return M


def spline_eval(y, M, h, u0, t):
    """Return (value, derivative) of natural cubic spline at coordinate t."""
    n = len(y)
    x = (t - u0) / h
    # i = floor(x), clamped to [0, n-2]. Index selection only -- the float
    # cast is for picking the spline interval; all arithmetic stays in arb.
    # Robust to non-finite / wide balls produced by transient Newton steps
    # (mirrors the float64 source's clamp of i to [0, n-2]).
    import math as _math
    xf = float(x)
    if not _math.isfinite(xf):
        # undecidable / blown-up coordinate: clamp by arb sign comparisons
        if (x > 0) is False:        # x <= 0 or undecidable-near-0 -> low end
            i = 0
        else:
            i = n - 2
    else:
        i = int(_math.floor(xf))
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    xi = u0 + arb(i) * h
    a = (xi + h - t) / h
    b = (t - xi) / h
    yi = y[i]
    yi1 = y[i + 1]
    Mi = M[i]
    Mi1 = M[i + 1]
    h2_6 = (h * h) / arb(6)
    val = (a * yi + b * yi1
           + ((a * a * a - a) * Mi + (b * b * b - b) * Mi1) * h2_6)
    der = ((yi1 - yi) / h
           - (arb(3) * a * a - ONE) / arb(6) * h * Mi
           + (arb(3) * b * b - ONE) / arb(6) * h * Mi1)
    return val, der


# ----------------------------------------------------------------------
# All roots of spline(line)=p_target on [u0, u0+(n-1)h] + |spline'| there.
# Returns list of (root_u, abs_der).
# ----------------------------------------------------------------------
def spline_roots(y, M, h, u0, p_target, sub):
    n = len(y)
    nseg = (n - 1) * sub
    out = []
    t_prev = u0
    v_prev, _ = spline_eval(y, M, h, u0, t_prev)
    step = (h * arb(n - 1)) / arb(nseg)
    for s in range(1, nseg + 1):
        t_cur = u0 + arb(s) * step
        v_cur, _ = spline_eval(y, M, h, u0, t_cur)
        dp = v_prev - p_target
        dc = v_cur - p_target
        zp = (dp == 0)
        zc = (dc == 0)
        if zp and zc:
            t_prev = t_cur
            v_prev = v_cur
            continue
        if (dp * dc) <= 0:
            # Robust bracketed Newton (mirrors float64 source, but uses a
            # provably-inside test so wide/non-finite arb balls never escape;
            # falls back to bisection on the maintained sign bracket).
            lo = t_prev
            hi = t_cur
            flo = dp     # = spline(lo) - p_target
            t = HALF * (lo + hi)
            for _ in range(160):
                val, der = spline_eval(y, M, h, u0, t)
                fval = val - p_target
                # try a Newton step, accept only if PROVABLY inside (lo, hi)
                accepted = False
                if not (der == 0):
                    tn = t - fval / der
                    if (tn > lo) and (tn < hi):
                        accepted = True
                if not accepted:
                    tn = HALF * (lo + hi)   # bisection
                # update bracket using sign of fval at t (the current point)
                if (flo * fval) <= 0:
                    hi = t
                else:
                    lo = t
                    flo = fval
                if abs(tn - t) < NEWTON_TOL:
                    t = tn
                    break
                t = tn
                if abs(fval) < ROOT_FTOL:
                    break
            val, der = spline_eval(y, M, h, u0, t)
            out.append((t, abs(der)))
        t_prev = t_cur
        v_prev = v_cur
    return out


# ----------------------------------------------------------------------
# Co-area evidence for a single 2-D slice S (n x n list-of-lists of arb).
# Partition-of-unity over the two parameterization directions, fixed GL nodes.
# Returns (A0, A1).
# ----------------------------------------------------------------------
def slice_evidence(S, h, u0, p_target, gnodes, gweights, tauA, tauB, sub,
                   twopi):
    n = len(S)
    Nq = len(gnodes)
    A0 = ZERO
    A1 = ZERO

    # column splines over rows: colM[ib] = 2nd derivs of column ib
    cols = [[S[ia][ib] for ia in range(n)] for ib in range(n)]
    colM = [natural_spline_M(cols[ib], h) for ib in range(n)]

    # ---- term A^(B): fix uA at GL nodes, root-find along axis B (cols) ----
    for q in range(Nq):
        uA = gnodes[q]
        wA = gweights[q]
        fA0 = f_signal(uA, 0, tauA, twopi)
        fA1 = f_signal(uA, 1, tauA, twopi)
        rowvals = [ZERO] * n
        rowdA = [ZERO] * n
        for ib in range(n):
            v, d = spline_eval(cols[ib], colM[ib], h, u0, uA)
            rowvals[ib] = v
            rowdA[ib] = d
        Mrow = natural_spline_M(rowvals, h)
        MrowdA = natural_spline_M(rowdA, h)
        roots = spline_roots(rowvals, Mrow, h, u0, p_target, sub)
        for (uB, dB) in roots:
            dAval, _ = spline_eval(rowdA, MrowdA, h, u0, uB)
            dA2 = dAval * dAval
            dB2 = dB * dB
            denom = dA2 + dB2
            if denom <= 0:
                continue
            wB = dB2 / denom
            if dB <= 0:
                continue
            fB0 = f_signal(uB, 0, tauB, twopi)
            fB1 = f_signal(uB, 1, tauB, twopi)
            A0 += wA * wB * fA0 * fB0 / dB
            A1 += wA * wB * fA1 * fB1 / dB

    # ---- term A^(A): fix uB at GL nodes, root-find along axis A (rows) ----
    rows = [[S[ia][ib] for ib in range(n)] for ia in range(n)]
    rowM = [natural_spline_M(rows[ia], h) for ia in range(n)]
    for q in range(Nq):
        uB = gnodes[q]
        wB_node = gweights[q]
        fB0 = f_signal(uB, 0, tauB, twopi)
        fB1 = f_signal(uB, 1, tauB, twopi)
        colvals = [ZERO] * n
        coldB = [ZERO] * n
        for ia in range(n):
            v, d = spline_eval(rows[ia], rowM[ia], h, u0, uB)
            colvals[ia] = v
            coldB[ia] = d
        Mcol = natural_spline_M(colvals, h)
        McoldB = natural_spline_M(coldB, h)
        roots = spline_roots(colvals, Mcol, h, u0, p_target, sub)
        for (uA, dA) in roots:
            dBval, _ = spline_eval(coldB, McoldB, h, u0, uA)
            dA2 = dA * dA
            dB2 = dBval * dBval
            denom = dA2 + dB2
            if denom <= 0:
                continue
            wA = dA2 / denom
            if dA <= 0:
                continue
            fA0 = f_signal(uA, 0, tauA, twopi)
            fA1 = f_signal(uA, 1, tauA, twopi)
            A0 += wB_node * wA * fA0 * fB0 / dA
            A1 += wB_node * wA * fA1 * fB1 / dA

    return A0, A1


# ----------------------------------------------------------------------
# Full Phi operator. P is a (G,G,G) nested list of arb. ui is a list of arb.
# Returns nested list (G,G,G) of arb.
# ----------------------------------------------------------------------
def phi_hfree(P, ui, gnodes, gweights, tau_vec, gamma_vec, W_vec, sub):
    G = len(ui)
    h = ui[1] - ui[0]
    u0 = ui[0]
    twopi = _twopi()
    out = [[[ZERO] * G for _ in range(G)] for _ in range(G)]
    for i in range(G):
        # S0 = P[i,:,:]
        S0 = [[P[i][j][l] for l in range(G)] for j in range(G)]
        for j in range(G):
            # S1 = P[:,j,:]
            S1 = [[P[a][j][l] for l in range(G)] for a in range(G)]
            for l in range(G):
                p = P[i][j][l]
                A0a, A1a = slice_evidence(S0, h, u0, p, gnodes, gweights,
                                          tau_vec[1], tau_vec[2], sub, twopi)
                mu0 = bayes(ui[i], tau_vec[0], A0a, A1a, twopi)
                A0b, A1b = slice_evidence(S1, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[2], sub, twopi)
                mu1 = bayes(ui[j], tau_vec[1], A0b, A1b, twopi)
                # S2 = P[:,:,l]
                S2 = [[P[a][b][l] for b in range(G)] for a in range(G)]
                A0c, A1c = slice_evidence(S2, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[1], sub, twopi)
                mu2 = bayes(ui[l], tau_vec[2], A0c, A1c, twopi)
                out[i][j][l] = clear_crra(mu0, mu1, mu2,
                                          gamma_vec[0], gamma_vec[1],
                                          gamma_vec[2],
                                          W_vec[0], W_vec[1], W_vec[2])
    return out


# ----------------------------------------------------------------------
# Gauss-Legendre nodes/weights at full arb precision via Newton refinement of
# float64 leggauss seeds on the Legendre polynomial (computed in arb).
# ----------------------------------------------------------------------
def _legendre_P_and_deriv(n, x):
    """P_n(x) and P_n'(x) by recurrence, in arb."""
    p0 = ONE
    p1 = x
    if n == 0:
        return ONE, ZERO
    if n == 1:
        return x, ONE
    for k in range(2, n + 1):
        pk = ((arb(2 * k - 1) * x * p1) - arb(k - 1) * p0) / arb(k)
        p0 = p1
        p1 = pk
    pn = p1
    dpn = arb(n) * (x * pn - p0) / (x * x - ONE)
    return pn, dpn


def gauss_legendre(Nq, a, b):
    """arb GL nodes/weights on [a,b], Newton-refined from float64 seeds."""
    import numpy as np
    x0, _ = np.polynomial.legendre.leggauss(Nq)
    a = arb(a) if not isinstance(a, arb) else a
    b = arb(b) if not isinstance(b, arb) else b
    nodes = []
    weights = []
    tiny = arb(10) ** (-105)
    for xi in x0:
        x = arb(repr(float(xi)))
        # Fixed, modest number of Newton steps: each step roughly doubles the
        # number of correct digits (12 -> 24 -> 48 -> 96 -> 192...), so ~6
        # steps from a float64 seed reaches full ~110-dec precision. We do NOT
        # use the arb ball comparison to break, because once pn straddles zero
        # the comparison is undecidable and further pn/dpn can blow up.
        for _ in range(8):
            pn, dpn = _legendre_P_and_deriv(Nq, x)
            dx = pn / dpn
            xn = x - dx
            if not (abs(dx) > tiny):  # converged at full precision
                x = xn
                break
            x = xn
        pn, dpn = _legendre_P_and_deriv(Nq, x)
        w = arb(2) / ((ONE - x * x) * dpn * dpn)
        nodes.append(x)
        weights.append(w)
    # map [-1,1] -> [a,b]
    half = HALF * (b - a)
    mid = HALF * (a + b)
    xm = [half * x + mid for x in nodes]
    wm = [half * w for w in weights]
    return xm, wm
