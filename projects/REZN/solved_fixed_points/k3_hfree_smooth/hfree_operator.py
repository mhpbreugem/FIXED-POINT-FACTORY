"""h-FREE SMOOTH co-area operator for the K=3 CRRA REE.

NO kernel, NO bandwidth, NO smoothing parameter. The only discretizations
are (i) the working grid resolution G and (ii) the number of Gauss-Legendre
quadrature nodes Nq -- both standard discretizations that converge as ->inf.

The evidence integral for agent 0 (slice over (u2,u3)) is the co-area form

    A_v(p) = int_{P=p}  f_v(u2) f_v(u3) / |grad_{2,3} P|  dsigma

computed by a partition-of-unity over the two parameterization directions:

    A_v(p) = A_v^(2)(p) + A_v^(3)(p)
    A_v^(2)(p) = sum over FIXED u3-nodes  sum_{roots u2* : P(u2*,u3)=p}
                    w2(u2*,u3) f_v(u2*) f_v(u3) / |d2 P(u2*,u3)|   * gl_weight(u3)
    A_v^(3)(p) = sum over FIXED u2-nodes  sum_{roots u3* : P(u2,u3*)=p}
                    w3(u2,u3*) f_v(u2) f_v(u3*) / |d3 P(u2,u3*)|   * gl_weight(u2)

with w2 = d2P^2/(d2P^2+d3P^2), w3 = d3P^2/(d2P^2+d3P^2),  w2+w3=1.
Near a u2-turning point (d2P->0) the weight shifts to the well-behaved
u3-direction term, cancelling the 1/|d2P| blowup -> SMOOTH everywhere except
true Morse-critical points (both gradients vanish, measure zero).

P is represented by a SMOOTH C2 natural-cubic tensor spline on the working
grid so that d2P,d3P and the implicit contour roots are smooth in p.

Quadrature nodes are DECOUPLED from the grid -> A_v(p) is smooth in p
(no kink at former grid-node values). THIS is the cure for the grid floor.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import math
import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------
# Model primitives (CRRA, signals) -- copied so this file is self-contained
# ----------------------------------------------------------------------
EPS_PRICE = 1.0e-12


@njit(cache=True, fastmath=False, inline="always")
def lam(z):
    if z >= 0.0:
        e = math.exp(-z); return 1.0 / (1.0 + e)
    e = math.exp(z); return e / (1.0 + e)


@njit(cache=True, fastmath=False, inline="always")
def logit(p):
    return math.log(p) - math.log(1.0 - p)


@njit(cache=True, fastmath=False, inline="always")
def f_signal(u, v, tau):
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * d * d)


@njit(cache=True, fastmath=False, inline="always")
def x_crra(mu, p, gamma, W):
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0.0:
        e = math.exp(-z); return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = math.exp(z); return W * (e - 1.0) / ((1.0 - p) + p * e)


@njit(cache=True, fastmath=False)
def clear_crra(mu0, mu1, mu2, g0, g1, g2, W0, W1, W2):
    a = EPS_PRICE; b = 1.0 - EPS_PRICE

    def ex(p):
        return (x_crra(mu0, p, g0, W0) + x_crra(mu1, p, g1, W1)
                + x_crra(mu2, p, g2, W2))
    fa = ex(a); fb = ex(b)
    if fa <= 0.0:
        return a
    if fb >= 0.0:
        return b
    for _ in range(80):
        c = 0.5 * (a + b); fc = ex(c)
        if fc >= 0.0:
            a = c
        else:
            b = c
        if (b - a) < 1.0e-15:
            break
    return 0.5 * (a + b)


@njit(cache=True, fastmath=False, inline="always")
def bayes(u_own, tau_own, A0, A1):
    f0 = f_signal(u_own, 0, tau_own)
    f1 = f_signal(u_own, 1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > 1.0 - EPS_PRICE:
        return 1.0 - EPS_PRICE
    return mu


# ----------------------------------------------------------------------
# 1-D natural cubic spline: coefficients, then C2 evaluation + derivative.
# We build per-line second-derivatives (the classic tridiagonal solve) and
# evaluate the cubic piece. This gives a globally C2 interpolant of P along
# any grid line, smooth in the evaluation coordinate -> smooth roots/derivs.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def natural_spline_M(y, h):
    """Second derivatives M at the n knots of a natural cubic spline.
    Uniform spacing h. Returns array length n."""
    n = y.size
    M = np.zeros(n)
    if n < 3:
        return M
    # tridiagonal: for interior i: M[i-1] + 4 M[i] + M[i+1] = 6/h^2 (y[i-1]-2y[i]+y[i+1])
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        rhs[i] = 6.0 / (h * h) * (y[i - 1] - 2.0 * y[i] + y[i + 1])
    # Thomas algorithm on interior nodes 1..n-2, with natural BC M[0]=M[n-1]=0
    c = np.zeros(n)  # super-diagonal after elimination
    d = np.zeros(n)  # rhs after elimination
    # diagonal = 4, off = 1
    b0 = 4.0
    c[1] = 1.0 / b0
    d[1] = rhs[1] / b0
    for i in range(2, n - 1):
        m = 4.0 - 1.0 * c[i - 1]
        c[i] = 1.0 / m
        d[i] = (rhs[i] - 1.0 * d[i - 1]) / m
    for i in range(n - 2, 0, -1):
        M[i] = d[i] - c[i] * M[i + 1]
    return M


@njit(cache=True, fastmath=False, inline="always")
def spline_eval(y, M, h, u0, t):
    """Evaluate natural cubic spline at coordinate t (absolute), knots at
    u0 + k*h. Returns (value, derivative)."""
    n = y.size
    # locate interval
    x = (t - u0) / h
    i = int(math.floor(x))
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    xi = u0 + i * h
    a = (xi + h - t) / h   # (x_{i+1}-t)/h
    b = (t - xi) / h       # (t-x_i)/h
    yi = y[i]; yi1 = y[i + 1]
    Mi = M[i]; Mi1 = M[i + 1]
    val = (a * yi + b * yi1
           + ((a * a * a - a) * Mi + (b * b * b - b) * Mi1) * (h * h) / 6.0)
    # derivative wrt t
    der = ((yi1 - yi) / h
           - (3.0 * a * a - 1.0) / 6.0 * h * Mi
           + (3.0 * b * b - 1.0) / 6.0 * h * Mi1)
    return val, der


# ----------------------------------------------------------------------
# Find ALL roots of spline(line)=p_target in [u0, u0+(n-1)h] and the
# transverse derivative there. We do a fine sub-scan for sign changes then
# Newton-polish on the C2 spline (smooth).
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def spline_roots(y, M, h, u0, p_target, roots_u, roots_d, sub):
    """Fill roots_u[k]=root coordinate, roots_d[k]=|spline'| there.
    `sub` sub-intervals per cell for bracketing. Returns number of roots."""
    n = y.size
    nseg = (n - 1) * sub
    cnt = 0
    t_prev = u0
    v_prev, _ = spline_eval(y, M, h, u0, t_prev)
    step = (h * (n - 1)) / nseg
    for s in range(1, nseg + 1):
        t_cur = u0 + s * step
        v_cur, _ = spline_eval(y, M, h, u0, t_cur)
        dp = v_prev - p_target
        dc = v_cur - p_target
        if dp == 0.0 and dc == 0.0:
            t_prev = t_cur; v_prev = v_cur; continue
        if dp * dc <= 0.0:
            # bracket [t_prev, t_cur]; Newton-polish from midpoint
            t = 0.5 * (t_prev + t_cur)
            ok = False
            for _ in range(40):
                val, der = spline_eval(y, M, h, u0, t)
                fval = val - p_target
                if der != 0.0:
                    tn = t - fval / der
                else:
                    tn = t
                # keep inside bracket
                if tn < t_prev or tn > t_cur or der == 0.0:
                    # bisection step
                    va, _ = spline_eval(y, M, h, u0, t_prev)
                    if (va - p_target) * fval <= 0.0:
                        tn = 0.5 * (t_prev + t)
                    else:
                        tn = 0.5 * (t + t_cur)
                if abs(tn - t) < 1.0e-14:
                    t = tn; ok = True; break
                t = tn
                if abs(fval) < 1.0e-15:
                    ok = True; break
            val, der = spline_eval(y, M, h, u0, t)
            if cnt < roots_u.size:
                roots_u[cnt] = t
                roots_d[cnt] = abs(der)
                cnt += 1
        t_prev = t_cur; v_prev = v_cur
    return cnt


# ----------------------------------------------------------------------
# Co-area evidence for a single 2-D slice S[ia,ib] = P over (axisA, axisB),
# at level p_target. Partition-of-unity over the two parameterization
# directions, FIXED Gauss-Legendre nodes (gnodes,gweights) on [u0, u0+(n-1)h].
# tauA = precision of axis A (rows), tauB = precision of axis B (cols).
# Returns A0, A1 (evidence under v=0, v=1).
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def slice_evidence(S, h, u0, p_target, gnodes, gweights, tauA, tauB,
                   sub, Mcols_buf, Mrows_buf):
    n = S.shape[0]
    Nq = gnodes.size
    A0 = 0.0
    A1 = 0.0
    maxroots = 2 * n + 4
    roots_u = np.empty(maxroots)
    roots_d = np.empty(maxroots)

    # ---- term A^(B): fix uA at GL nodes, root-find along axis B (cols) ----
    # For a fixed uA node, we need the spline of S(uA, .) over cols. That
    # requires S at a non-grid row uA -> interpolate columns first via the
    # column-splines, OR: build a row by spline-evaluating each column at uA.
    # We do: for each column ib, value at uA = spline over rows. Then build a
    # row-spline over ib of those values. d_B P = row-spline'. d_A P needs the
    # row-derivatives -> we also get them from the column splines.
    colM = Mcols_buf  # (n, n): colM[ib] = 2nd-derivs of column ib over rows
    for ib in range(n):
        # column ib as function of row index: S[:, ib]
        col = S[:, ib]
        colM[ib, :] = natural_spline_M(col, h)

    rowvals = np.empty(n)
    rowdA = np.empty(n)   # d/dA value at uA for each col
    for q in range(Nq):
        uA = gnodes[q]
        wA = gweights[q]
        fA0 = f_signal(uA, 0, tauA)
        fA1 = f_signal(uA, 1, tauA)
        for ib in range(n):
            v, d = spline_eval(S[:, ib], colM[ib, :], h, u0, uA)
            rowvals[ib] = v
            rowdA[ib] = d  # d P / d uA at (uA, grid col ib)
        # spline of rowvals over cols, and of rowdA over cols (for dA at root)
        Mrow = natural_spline_M(rowvals, h)
        MrowdA = natural_spline_M(rowdA, h)
        ncnt = spline_roots(rowvals, Mrow, h, u0, p_target,
                            roots_u, roots_d, sub)
        for r in range(ncnt):
            uB = roots_u[r]
            dB = roots_d[r]            # |d P / d uB|
            dAval, _ = spline_eval(rowdA, MrowdA, h, u0, uB)  # d P / d uA
            dA2 = dAval * dAval
            dB2 = dB * dB
            denom = dA2 + dB2
            if denom <= 0.0:
                continue
            wB = dB2 / denom           # partition-of-unity weight for B-term
            if dB <= 0.0:
                continue
            fB0 = f_signal(uB, 0, tauB)
            fB1 = f_signal(uB, 1, tauB)
            A0 += wA * wB * fA0 * fB0 / dB
            A1 += wA * wB * fA1 * fB1 / dB

    # ---- term A^(A): fix uB at GL nodes, root-find along axis A (rows) ----
    rowM = Mrows_buf  # (n, n): rowM[ia] = 2nd-derivs of row ia over cols
    for ia in range(n):
        row = S[ia, :]
        rowM[ia, :] = natural_spline_M(row, h)
    colvals = np.empty(n)
    coldB = np.empty(n)
    for q in range(Nq):
        uB = gnodes[q]
        wB_node = gweights[q]
        fB0 = f_signal(uB, 0, tauB)
        fB1 = f_signal(uB, 1, tauB)
        for ia in range(n):
            v, d = spline_eval(S[ia, :], rowM[ia, :], h, u0, uB)
            colvals[ia] = v
            coldB[ia] = d  # d P / d uB at (grid row ia, uB)
        Mcol = natural_spline_M(colvals, h)
        McoldB = natural_spline_M(coldB, h)
        ncnt = spline_roots(colvals, Mcol, h, u0, p_target,
                            roots_u, roots_d, sub)
        for r in range(ncnt):
            uA = roots_u[r]
            dA = roots_d[r]            # |d P / d uA|
            dBval, _ = spline_eval(coldB, McoldB, h, u0, uA)
            dA2 = dA * dA
            dB2 = dBval * dBval
            denom = dA2 + dB2
            if denom <= 0.0:
                continue
            wA = dA2 / denom           # partition-of-unity weight for A-term
            if dA <= 0.0:
                continue
            fA0 = f_signal(uA, 0, tauA)
            fA1 = f_signal(uA, 1, tauA)
            A0 += wB_node * wA * fA0 * fB0 / dA
            A1 += wB_node * wA * fA1 * fB1 / dA

    return A0, A1


# ----------------------------------------------------------------------
# Full Phi operator on the inner grid (G,G,G). P_inner only (no halo needed:
# the spline is C2 over the inner grid). Quadrature nodes fixed on
# [-UMAX, UMAX]. NO kernel, NO bandwidth.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False, parallel=True)
def phi_hfree(P, ui, gnodes, gweights, tau_vec, gamma_vec, W_vec, sub):
    G = ui.size
    h = ui[1] - ui[0]
    u0 = ui[0]
    out = np.empty_like(P)
    for i in prange(G):
        Mc = np.empty((G, G))
        Mr = np.empty((G, G))
        for j in range(G):
            for l in range(G):
                p = P[i, j, l]
                # Agent 0: slice over (u2,u3) = P[i, :, :], axisA=1(tau1) axisB=2(tau2)
                S0 = P[i, :, :]
                A0a, A1a = slice_evidence(S0, h, u0, p, gnodes, gweights,
                                          tau_vec[1], tau_vec[2], sub, Mc, Mr)
                mu0 = bayes(ui[i], tau_vec[0], A0a, A1a)
                # Agent 1: slice over (u1,u3) = P[:, j, :], axisA=0(tau0) axisB=2(tau2)
                S1 = P[:, j, :]
                A0b, A1b = slice_evidence(S1, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[2], sub, Mc, Mr)
                mu1 = bayes(ui[j], tau_vec[1], A0b, A1b)
                # Agent 2: slice over (u1,u2) = P[:, :, l], axisA=0(tau0) axisB=1(tau1)
                S2 = P[:, :, l]
                A0c, A1c = slice_evidence(S2, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[1], sub, Mc, Mr)
                mu2 = bayes(ui[l], tau_vec[2], A0c, A1c)
                out[i, j, l] = clear_crra(mu0, mu1, mu2,
                                          gamma_vec[0], gamma_vec[1], gamma_vec[2],
                                          W_vec[0], W_vec[1], W_vec[2])
    return out


# ----------------------------------------------------------------------
# Single-slice A_v(p) sweep for the SMOOTHNESS test (decoupled p sweep).
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def slice_Av_sweep(S, h, u0, ps, gnodes, gweights, tauA, tauB, sub):
    np_ = ps.size
    G = S.shape[0]
    A0o = np.empty(np_)
    A1o = np.empty(np_)
    Mc = np.empty((G, G))
    Mr = np.empty((G, G))
    for k in range(np_):
        a0, a1 = slice_evidence(S, h, u0, ps[k], gnodes, gweights,
                                tauA, tauB, sub, Mc, Mr)
        A0o[k] = a0; A1o[k] = a1
    return A0o, A1o


def gauss_legendre(Nq, a, b):
    x, w = np.polynomial.legendre.leggauss(Nq)
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    wm = 0.5 * (b - a) * w
    return xm, wm
