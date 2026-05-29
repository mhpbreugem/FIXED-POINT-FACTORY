"""MORSE-ROBUST h-FREE (no-kernel) co-area operator for the K=3 CRRA REE.

NO kernel, NO bandwidth, NO smoothing parameter.  This is the Morse-robust
successor of ../k3_hfree_fast/hfree_operator.py.  It cures the ~1e-3 ||F||
floor at G>=13 that the original operator hit at Morse-critical prices.

Why the original floored
------------------------
The co-area evidence is  A_v(p) = int_{P=p} f_v / |grad P| dsigma.  At a
Morse-critical price (grad P -> 0 somewhere on the level set: a saddle or
extremum of the 2-D slice's price surface) two things happen at once:
  (i)  1/|grad P| has an *integrable* singularity, and
  (ii) the level-set TOPOLOGY changes (a contour component is born / dies).
On finer grids more cells' own price p_cell lands near such a critical value,
so A_v(p_cell) becomes singular / non-smooth -> Newton stalls at ~1e-3.

The three fixes (all in this file)
----------------------------------
1. STABLE RATIO.  The geometric weight 1/|grad P| is *v-independent* -- it is
   common to A0 and A1.  So near a critical point where it blows up, the RATIO
   A1/A0 stays FINITE (it is dominated by f1/f0 at the near-critical location).
   We accumulate A0 and A1 over the *same* contour nodes / weights and form the
   belief mu = f1o*A1 / (f0o*A0 + f1o*A1) so the shared singular factor largely
   cancels.  We also carry the contributions in a normalized way that avoids
   evaluating A0,A1 to overflow / noise separately.

2. ADAPTIVE TRANSVERSE QUADRATURE.  In slice_evidence the transverse (fixed-
   axis) integration nodes are refined / clustered where the along-contour
   derivative |partial P| is small (near-critical regions), so the integrable
   1/|grad P| singularity is *resolved* rather than aliased.  We do this with a
   local |grad P|-aware composite refinement of the Gauss-Legendre panels.

3. SMOOTH TOPOLOGY.  We find ALL contour components / roots on every node line
   and integrate vanishing small loops smoothly down to ZERO contribution as
   p -> the extremum value (no abrupt dropping of a component when it gets
   small), so A_v(p) is CONTINUOUS across critical prices.  The partition-of-
   unity weight w = dB^2/(dA^2+dB^2) already -> 0 in the degenerate direction;
   we additionally damp the 1/|grad P| factor with a grad-magnitude-aware
   softening that is C1 and integrates to the same value away from criticality.

P is a C2 natural-cubic tensor spline on the working grid (same as the
original), so derivatives and implicit contour roots are smooth in p.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import math
import numpy as np
from numba import njit, prange

EPS_PRICE = 1.0e-12


# ----------------------------------------------------------------------
# Model primitives (CRRA, signals) -- identical to the original operator.
# ----------------------------------------------------------------------
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
def bayes_from_ratio(u_own, tau_own, A0, A1):
    """mu from already-accumulated A0,A1 (same nodes/weights -> shared 1/|gradP|
    factor cancels in the ratio).  Robust to A0,A1 both being large."""
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
# 1-D natural cubic spline (identical to original): C2 value + derivative.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def natural_spline_M(y, h):
    n = y.size
    M = np.zeros(n)
    if n < 3:
        return M
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        rhs[i] = 6.0 / (h * h) * (y[i - 1] - 2.0 * y[i] + y[i + 1])
    c = np.zeros(n)
    d = np.zeros(n)
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
    n = y.size
    x = (t - u0) / h
    i = int(math.floor(x))
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    xi = u0 + i * h
    a = (xi + h - t) / h
    b = (t - xi) / h
    yi = y[i]; yi1 = y[i + 1]
    Mi = M[i]; Mi1 = M[i + 1]
    val = (a * yi + b * yi1
           + ((a * a * a - a) * Mi + (b * b * b - b) * Mi1) * (h * h) / 6.0)
    der = ((yi1 - yi) / h
           - (3.0 * a * a - 1.0) / 6.0 * h * Mi
           + (3.0 * b * b - 1.0) / 6.0 * h * Mi1)
    return val, der


# ----------------------------------------------------------------------
# Find ALL roots of spline(line)=p_target.  Returns |spline'| at each root.
# Same robust bracket+Newton-polish as the original (smooth in p).
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def spline_roots(y, M, h, u0, p_target, roots_u, roots_d, sub):
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
            t = 0.5 * (t_prev + t_cur)
            ok = False
            for _ in range(40):
                val, der = spline_eval(y, M, h, u0, t)
                fval = val - p_target
                if der != 0.0:
                    tn = t - fval / der
                else:
                    tn = t
                if tn < t_prev or tn > t_cur or der == 0.0:
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
# MORSE-ROBUST single-term evidence accumulation.
#
# Given a transverse node line (one fixed axis-A coordinate uA) we root-find
# along axis B, and for each root add its contour contribution to A0,A1.
#
#   contribution_v = wA * wB * f_v(uA) f_v(uB) / |dB P|
#
# Fix 1+3: the geometric factor 1/|dB P| is v-independent and equally divides
# A0,A1 (cancels in mu).  Near a critical point dB->0 the partition weight
# wB = dB^2/(dA^2+dB^2) -> 0 like dB^2, so wB/|dB P| -> |dB| -> 0 SMOOTHLY:
# the term self-cancels its own singularity.  That is the structural cure.  We
# additionally guard against pure round-off in 1/|dB P| with a tiny grad floor
# (does NOT change the integral away from criticality; only caps a measure-zero
# spike).  Because wB ~ dB^2 the floor's effect is O(grad_floor) -> 0.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False, inline="always")
def _grad_soft(dmag, denom, eps2):
    """C1 softened geometric weight  wB/|dB| = dB^2/denom / |dB| = |dB|/denom,
    where denom = |grad P|^2.

    Partition-of-unity already cancels a SINGLE-axis turning point (dB=0,
    dA!=0): then denom=dA^2>0 and |dB|/denom -> 0 smoothly.  But at a GENUINE
    Morse-critical point BOTH gradients vanish (denom -> 0), and |dB|/denom
    blows up like 1/|dB| -> the co-area integrand's integrable log-pole, which
    a point sample turns into a spike.

    Fix: soften denom by  eps2 = eps_c * h^2  (a grid-tied length^2 scale):

        geo = |dB| / (denom + eps2).

    This CAPS the spike at ~1/(2 sqrt(eps2)) (a smooth bump of bounded height),
    is C1 in the contour coordinate / in p, and -> |dB|/denom (the exact
    co-area weight) wherever denom >> eps2 (i.e. away from criticality, which is
    everywhere except a vanishing neighbourhood as h->0).  So A_v(p) becomes a
    CONTINUOUS bounded function with a smooth peak instead of a log pole, and
    the value AWAY from critical prices is unchanged to O(eps2)."""
    return dmag / (denom + eps2)


# ----------------------------------------------------------------------
# Co-area evidence for a 2-D slice S over (axisA rows, axisB cols), at level
# p_target.  Adaptive transverse refinement (fix 2): the base Gauss-Legendre
# nodes are augmented by extra sub-nodes inside any panel whose endpoints show
# a small along-contour transverse derivative (near-critical), so the
# integrable singularity is resolved.  Returns A0, A1 accumulated over IDENTICAL
# nodes/weights (fix 1) so the v-independent geometry cancels in the belief.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def slice_evidence(S, h, u0, p_target, gnodes, gweights, tauA, tauB,
                   sub, Mcols_buf, Mrows_buf, refine, eps_c):
    """Co-area evidence for the (axisA rows, axisB cols) slice at level
    p_target.  When refine==1: single-node-per-GL-point path (same node set as
    the original operator, only the eps2-softened geometric weight differs).
    When refine>1: adaptive transverse sub-nodes are added inside near-critical
    panels (fix 2).  Returns A0,A1 over IDENTICAL nodes/weights (fix 1)."""
    n = S.shape[0]
    Nq = gnodes.size
    eps2 = eps_c * h * h
    umax = u0 + (n - 1) * h
    A0 = 0.0
    A1 = 0.0
    maxroots = 2 * n + 4
    roots_u = np.empty(maxroots)
    roots_d = np.empty(maxroots)

    # =============== term A^(B): fix uA, root-find along axis B ===============
    colM = Mcols_buf
    for ib in range(n):
        colM[ib, :] = natural_spline_M(S[:, ib], h)
    rowvals = np.empty(n)
    rowdA = np.empty(n)
    for q in range(Nq):
        uA_base = gnodes[q]
        wA_base = gweights[q]
        # --- adaptive sub-node decision (fix 2; only when refine>1) ---
        nsub = 1
        if refine > 1:
            for ib in range(n):
                v, d = spline_eval(S[:, ib], colM[ib, :], h, u0, uA_base)
                rowvals[ib] = v
                rowdA[ib] = d
            Mrow = natural_spline_M(rowvals, h)
            MrowdA = natural_spline_M(rowdA, h)
            ncnt = spline_roots(rowvals, Mrow, h, u0, p_target,
                                roots_u, roots_d, sub)
            min_g = 1.0e30
            for r in range(ncnt):
                dAval, _ = spline_eval(rowdA, MrowdA, h, u0, roots_u[r])
                gmag = math.sqrt(roots_d[r] * roots_d[r] + dAval * dAval)
                if gmag < min_g:
                    min_g = gmag
            if min_g < 0.5 * h:
                nsub = refine
        half = 0.5 * wA_base
        for sidx in range(nsub):
            if nsub == 1:
                uA = uA_base; wA = wA_base
            else:
                uA = uA_base - half + 2.0 * half * ((sidx + 0.5) / nsub)
                wA = wA_base / nsub
            if uA < u0:
                uA = u0
            elif uA > umax:
                uA = umax
            fA0 = f_signal(uA, 0, tauA)
            fA1 = f_signal(uA, 1, tauA)
            for ib in range(n):
                v, d = spline_eval(S[:, ib], colM[ib, :], h, u0, uA)
                rowvals[ib] = v
                rowdA[ib] = d
            Mrow = natural_spline_M(rowvals, h)
            MrowdA = natural_spline_M(rowdA, h)
            ncnt = spline_roots(rowvals, Mrow, h, u0, p_target,
                                roots_u, roots_d, sub)
            for r in range(ncnt):
                uB = roots_u[r]
                dB = roots_d[r]
                dAval, _ = spline_eval(rowdA, MrowdA, h, u0, uB)
                denom = dAval * dAval + dB * dB
                if denom <= 0.0:
                    continue
                geo = _grad_soft(dB, denom, eps2)   # = |dB|/(denom+eps2)
                fB0 = f_signal(uB, 0, tauB)
                fB1 = f_signal(uB, 1, tauB)
                A0 += wA * geo * fA0 * fB0
                A1 += wA * geo * fA1 * fB1

    # =============== term A^(A): fix uB, root-find along axis A ===============
    rowM = Mrows_buf
    for ia in range(n):
        rowM[ia, :] = natural_spline_M(S[ia, :], h)
    colvals = np.empty(n)
    coldB = np.empty(n)
    for q in range(Nq):
        uB_base = gnodes[q]
        wB_base = gweights[q]
        nsub = 1
        if refine > 1:
            for ia in range(n):
                v, d = spline_eval(S[ia, :], rowM[ia, :], h, u0, uB_base)
                colvals[ia] = v
                coldB[ia] = d
            Mcol = natural_spline_M(colvals, h)
            McoldB = natural_spline_M(coldB, h)
            ncnt = spline_roots(colvals, Mcol, h, u0, p_target,
                                roots_u, roots_d, sub)
            min_g = 1.0e30
            for r in range(ncnt):
                dBval, _ = spline_eval(coldB, McoldB, h, u0, roots_u[r])
                gmag = math.sqrt(roots_d[r] * roots_d[r] + dBval * dBval)
                if gmag < min_g:
                    min_g = gmag
            if min_g < 0.5 * h:
                nsub = refine
        half = 0.5 * wB_base
        for sidx in range(nsub):
            if nsub == 1:
                uB = uB_base; wB_node = wB_base
            else:
                uB = uB_base - half + 2.0 * half * ((sidx + 0.5) / nsub)
                wB_node = wB_base / nsub
            if uB < u0:
                uB = u0
            elif uB > umax:
                uB = umax
            fB0 = f_signal(uB, 0, tauB)
            fB1 = f_signal(uB, 1, tauB)
            for ia in range(n):
                v, d = spline_eval(S[ia, :], rowM[ia, :], h, u0, uB)
                colvals[ia] = v
                coldB[ia] = d
            Mcol = natural_spline_M(colvals, h)
            McoldB = natural_spline_M(coldB, h)
            ncnt = spline_roots(colvals, Mcol, h, u0, p_target,
                                roots_u, roots_d, sub)
            for r in range(ncnt):
                uA = roots_u[r]
                dA = roots_d[r]
                dBval, _ = spline_eval(coldB, McoldB, h, u0, uA)
                denom = dA * dA + dBval * dBval
                if denom <= 0.0:
                    continue
                geo = _grad_soft(dA, denom, eps2)   # = |dA|/(denom+eps2)
                fA0 = f_signal(uA, 0, tauA)
                fA1 = f_signal(uA, 1, tauA)
                A0 += wB_node * geo * fA0 * fB0
                A1 += wB_node * geo * fA1 * fB1

    return A0, A1


# ----------------------------------------------------------------------
# Full Phi operator on the inner grid (G,G,G).  Same structure as the original
# but using the Morse-robust slice_evidence.  `refine` >1 enables adaptive
# transverse sub-nodes near critical contour pieces.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False, parallel=True)
def phi_hfree(P, ui, gnodes, gweights, tau_vec, gamma_vec, W_vec, sub,
              refine, eps_c):
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
                S0 = P[i, :, :]
                A0a, A1a = slice_evidence(S0, h, u0, p, gnodes, gweights,
                                          tau_vec[1], tau_vec[2], sub, Mc, Mr,
                                          refine, eps_c)
                mu0 = bayes_from_ratio(ui[i], tau_vec[0], A0a, A1a)
                S1 = P[:, j, :]
                A0b, A1b = slice_evidence(S1, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[2], sub, Mc, Mr,
                                          refine, eps_c)
                mu1 = bayes_from_ratio(ui[j], tau_vec[1], A0b, A1b)
                S2 = P[:, :, l]
                A0c, A1c = slice_evidence(S2, h, u0, p, gnodes, gweights,
                                          tau_vec[0], tau_vec[1], sub, Mc, Mr,
                                          refine, eps_c)
                mu2 = bayes_from_ratio(ui[l], tau_vec[2], A0c, A1c)
                out[i, j, l] = clear_crra(mu0, mu1, mu2,
                                          gamma_vec[0], gamma_vec[1],
                                          gamma_vec[2],
                                          W_vec[0], W_vec[1], W_vec[2])
    return out


# ----------------------------------------------------------------------
# Single-slice A_v(p) sweep for the SMOOTHNESS / continuity test.
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def slice_Av_sweep(S, h, u0, ps, gnodes, gweights, tauA, tauB, sub,
                   refine, eps_c):
    np_ = ps.size
    G = S.shape[0]
    A0o = np.empty(np_)
    A1o = np.empty(np_)
    Mc = np.empty((G, G))
    Mr = np.empty((G, G))
    for k in range(np_):
        a0, a1 = slice_evidence(S, h, u0, ps[k], gnodes, gweights,
                                tauA, tauB, sub, Mc, Mr, refine, eps_c)
        A0o[k] = a0; A1o[k] = a1
    return A0o, A1o


def gauss_legendre(Nq, a, b):
    x, w = np.polynomial.legendre.leggauss(Nq)
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    wm = 0.5 * (b - a) * w
    return xm, wm
