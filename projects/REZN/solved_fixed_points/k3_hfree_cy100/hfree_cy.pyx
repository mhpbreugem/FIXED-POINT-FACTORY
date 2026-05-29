# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""TRUE Cython + libflint (arb_t) port of the h-FREE SMOOTH co-area K=3 CRRA REE
operator. 1:1 translation of hfree_operator_arb.py into C-level arb_t calls.

NO kernel, NO bandwidth, NO smoothing parameter. Pure co-area change-of-variables,
identical math to hfree_operator_arb.py:
  f_signal, x_crra, clear_crra (bisection), bayes,
  natural_spline_M (Thomas), spline_eval (C2 cubic val+der),
  spline_roots (rtsafe Newton/bisection for ALL contour roots),
  slice_evidence (partition-of-unity, w2=d2P^2/(d2P^2+d3P^2)),
  phi_hfree.

GL nodes/weights are passed in as python-flint arb (computed once at 110-dec).
"""
from libc.stdlib cimport malloc, free
from libc.math cimport floor as c_floor, isfinite as c_isfinite

from flint.types.arb cimport arb as py_arb
from flint.flintlib.types.arb cimport arb_t, arb_midref
from flint.flintlib.functions.arb cimport (
    arb_init, arb_clear, arb_set, arb_set_d, arb_set_si, arb_set_ui,
    arb_add, arb_sub, arb_mul, arb_div, arb_neg, arb_abs,
    arb_sqrt, arb_exp, arb_log,
    arb_const_pi, arb_zero, arb_one,
    arb_is_zero, arb_is_positive, arb_is_negative, arb_is_finite,
    arb_mul_2exp_si,
)
from flint.flintlib.functions.arf cimport arf_get_d

cdef extern from "flint/arf.h":
    ctypedef int arf_rnd_t
    cdef arf_rnd_t ARF_RND_NEAR "ARF_RND_NEAR"

cdef extern from "flint/arb.h":
    int arb_lt(const arb_t x, const arb_t y)
    int arb_gt(const arb_t x, const arb_t y)
    int arb_le(const arb_t x, const arb_t y)
    int arb_ge(const arb_t x, const arb_t y)
    int arb_eq(const arb_t x, const arb_t y)


# ---------------- precision (global, set per call) ----------------
cdef long PREC = 0

# ---------------- shared constants (init once per call) ----------------
cdef arb_t C_ZERO, C_ONE, C_HALF, C_EPS_PRICE, C_TOL, C_TWO, C_THREE, C_FOUR, C_SIX
cdef arb_t C_TWOPI, C_ONE_M_EPS


# ---------------- py <-> arb helpers ----------------
cdef inline void c_from_pyarb(arb_t y, object obj):
    if isinstance(obj, py_arb):
        arb_set(y, (<py_arb>obj).val)
    else:
        tmp = py_arb(obj)
        arb_set(y, (<py_arb>tmp).val)

cdef inline object pyarb_from_c(const arb_t x):
    cdef py_arb out = py_arb.__new__(py_arb)
    arb_init(out.val)
    arb_set(out.val, x)
    return out

cdef inline bint c_gt(const arb_t x, const arb_t y):
    return arb_gt(x, y) != 0
cdef inline bint c_ge(const arb_t x, const arb_t y):
    return arb_ge(x, y) != 0
cdef inline bint c_lt(const arb_t x, const arb_t y):
    return arb_lt(x, y) != 0
cdef inline bint c_le(const arb_t x, const arb_t y):
    return arb_le(x, y) != 0


# ---------------- model primitives ----------------
cdef inline void c_logit(arb_t out, const arb_t p, arb_t t1, long prec):
    # out = log(p) - log(1-p)
    arb_log(t1, p, prec)
    arb_sub(out, C_ONE, p, prec)
    arb_log(out, out, prec)
    arb_sub(out, t1, out, prec)


cdef void c_fsignal(arb_t out, const arb_t u, int v, const arb_t tau,
                    arb_t t1, arb_t t2, long prec):
    # mean = 0.5 if v==1 else -0.5 ; coef = sqrt(tau/twopi)
    # out = coef * exp(-0.5*tau*(u-mean)^2)
    if v == 1:
        arb_sub(t1, u, C_HALF, prec)
    else:
        arb_add(t1, u, C_HALF, prec)
    arb_mul(t1, t1, t1, prec)        # (u-mean)^2
    arb_mul(t1, t1, tau, prec)       # tau*d^2
    arb_mul_2exp_si(t1, t1, -1)      # 0.5*tau*d^2
    arb_neg(t1, t1)
    arb_exp(t1, t1, prec)            # exp(...)
    arb_div(t2, tau, C_TWOPI, prec)
    arb_sqrt(t2, t2, prec)           # coef
    arb_mul(out, t1, t2, prec)


cdef void c_xcrra(arb_t out, const arb_t mu, const arb_t p, const arb_t gamma,
                  const arb_t W, arb_t z, arb_t e, arb_t t1, arb_t t2, long prec):
    # z = (logit(mu)-logit(p))/gamma
    c_logit(t1, mu, t2, prec)
    c_logit(t2, p, out, prec)
    arb_sub(z, t1, t2, prec)
    arb_div(z, z, gamma, prec)
    if c_ge(z, C_ZERO):
        # e = exp(-z); out = W*(1-e)/((1-p)*e + p)
        arb_neg(t1, z)
        arb_exp(e, t1, prec)
        arb_sub(t1, C_ONE, e, prec)        # 1-e
        arb_mul(t1, t1, W, prec)           # W*(1-e)
        arb_sub(t2, C_ONE, p, prec)        # 1-p
        arb_mul(t2, t2, e, prec)           # (1-p)*e
        arb_add(t2, t2, p, prec)           # (1-p)*e + p
        arb_div(out, t1, t2, prec)
    else:
        # e = exp(z); out = W*(e-1)/((1-p) + p*e)
        arb_exp(e, z, prec)
        arb_sub(t1, e, C_ONE, prec)        # e-1
        arb_mul(t1, t1, W, prec)           # W*(e-1)
        arb_sub(t2, C_ONE, p, prec)        # 1-p
        arb_mul(out, p, e, prec)           # p*e
        arb_add(t2, t2, out, prec)         # (1-p) + p*e
        arb_div(out, t1, t2, prec)


cdef void c_clear_crra(arb_t out, const arb_t mu0, const arb_t mu1, const arb_t mu2,
                       const arb_t g0, const arb_t g1, const arb_t g2,
                       const arb_t W0, const arb_t W1, const arb_t W2, long prec):
    cdef arb_t a, b, c, fa, fb, fc, d0, d1, d2, z, e, t1, t2, width
    cdef int step
    arb_init(a); arb_init(b); arb_init(c); arb_init(fa); arb_init(fb); arb_init(fc)
    arb_init(d0); arb_init(d1); arb_init(d2); arb_init(z); arb_init(e)
    arb_init(t1); arb_init(t2); arb_init(width)
    arb_set(a, C_EPS_PRICE)
    arb_sub(b, C_ONE, C_EPS_PRICE, prec)
    # fa = ex(a)
    c_xcrra(d0, mu0, a, g0, W0, z, e, t1, t2, prec)
    c_xcrra(d1, mu1, a, g1, W1, z, e, t1, t2, prec)
    c_xcrra(d2, mu2, a, g2, W2, z, e, t1, t2, prec)
    arb_add(fa, d0, d1, prec); arb_add(fa, fa, d2, prec)
    c_xcrra(d0, mu0, b, g0, W0, z, e, t1, t2, prec)
    c_xcrra(d1, mu1, b, g1, W1, z, e, t1, t2, prec)
    c_xcrra(d2, mu2, b, g2, W2, z, e, t1, t2, prec)
    arb_add(fb, d0, d1, prec); arb_add(fb, fb, d2, prec)
    if c_le(fa, C_ZERO):
        arb_set(out, a)
    elif c_ge(fb, C_ZERO):
        arb_set(out, b)
    else:
        for step in range(600):
            arb_add(c, a, b, prec)
            arb_mul_2exp_si(c, c, -1)        # c = 0.5*(a+b)
            c_xcrra(d0, mu0, c, g0, W0, z, e, t1, t2, prec)
            c_xcrra(d1, mu1, c, g1, W1, z, e, t1, t2, prec)
            c_xcrra(d2, mu2, c, g2, W2, z, e, t1, t2, prec)
            arb_add(fc, d0, d1, prec); arb_add(fc, fc, d2, prec)
            if c_ge(fc, C_ZERO):
                arb_set(a, c)
            else:
                arb_set(b, c)
            arb_sub(width, b, a, prec)
            if c_lt(width, C_TOL):
                break
        arb_add(out, a, b, prec)
        arb_mul_2exp_si(out, out, -1)
    arb_clear(a); arb_clear(b); arb_clear(c); arb_clear(fa); arb_clear(fb); arb_clear(fc)
    arb_clear(d0); arb_clear(d1); arb_clear(d2); arb_clear(z); arb_clear(e)
    arb_clear(t1); arb_clear(t2); arb_clear(width)


cdef void c_bayes(arb_t mu, const arb_t u_own, const arb_t tau_own,
                  const arb_t A0, const arb_t A1, long prec):
    cdef arb_t f0, f1, num, den, t1, t2, omeps
    arb_init(f0); arb_init(f1); arb_init(num); arb_init(den)
    arb_init(t1); arb_init(t2); arb_init(omeps)
    c_fsignal(f0, u_own, 0, tau_own, t1, t2, prec)
    c_fsignal(f1, u_own, 1, tau_own, t1, t2, prec)
    arb_mul(num, f1, A1, prec)
    arb_mul(t1, f0, A0, prec)
    arb_add(den, t1, num, prec)
    if c_le(den, C_ZERO):
        arb_set(mu, C_HALF)
    else:
        arb_div(mu, num, den, prec)
        arb_sub(omeps, C_ONE, C_EPS_PRICE, prec)
        if c_lt(mu, C_EPS_PRICE):
            arb_set(mu, C_EPS_PRICE)
        elif c_gt(mu, omeps):
            arb_set(mu, omeps)
    arb_clear(f0); arb_clear(f1); arb_clear(num); arb_clear(den)
    arb_clear(t1); arb_clear(t2); arb_clear(omeps)


# ---------------- natural cubic spline (Thomas) ----------------
# y, M are arb_t* of length n. h is arb_t.
cdef void c_spline_M(arb_t *M, arb_t *y, int n, const arb_t h, long prec):
    cdef int i
    cdef arb_t six_h2, t1, mm
    cdef arb_t *rhs
    cdef arb_t *cc
    cdef arb_t *dd
    for i in range(n):
        arb_set_ui(M[i], 0)
    if n < 3:
        return
    arb_init(six_h2); arb_init(t1); arb_init(mm)
    arb_mul(six_h2, h, h, prec)            # h^2
    arb_div(six_h2, C_SIX, six_h2, prec)   # 6/h^2
    rhs = <arb_t *> malloc(n * sizeof(arb_t))
    cc = <arb_t *> malloc(n * sizeof(arb_t))
    dd = <arb_t *> malloc(n * sizeof(arb_t))
    for i in range(n):
        arb_init(rhs[i]); arb_init(cc[i]); arb_init(dd[i])
    for i in range(1, n - 1):
        arb_mul_2exp_si(t1, y[i], 1)        # 2*y[i]
        arb_sub(rhs[i], y[i-1], t1, prec)   # y[i-1]-2y[i]
        arb_add(rhs[i], rhs[i], y[i+1], prec)
        arb_mul(rhs[i], rhs[i], six_h2, prec)
    # b0 = 4
    arb_div(cc[1], C_ONE, C_FOUR, prec)
    arb_div(dd[1], rhs[1], C_FOUR, prec)
    for i in range(2, n - 1):
        arb_sub(mm, C_FOUR, cc[i-1], prec)
        arb_div(cc[i], C_ONE, mm, prec)
        arb_sub(t1, rhs[i], dd[i-1], prec)
        arb_div(dd[i], t1, mm, prec)
    for i in range(n - 2, 0, -1):
        arb_mul(t1, cc[i], M[i+1], prec)
        arb_sub(M[i], dd[i], t1, prec)
    for i in range(n):
        arb_clear(rhs[i]); arb_clear(cc[i]); arb_clear(dd[i])
    free(rhs); free(cc); free(dd)
    arb_clear(six_h2); arb_clear(t1); arb_clear(mm)


# Evaluate spline value + derivative at coordinate t. Writes val,der.
cdef void c_spline_eval(arb_t val, arb_t der, arb_t *y, arb_t *M, int n,
                        const arb_t h, const arb_t u0, const arb_t t, long prec):
    cdef arb_t x, xi, a, b, t1, t2, h2_6, tmp
    cdef double xf
    cdef int i
    arb_init(x); arb_init(xi); arb_init(a); arb_init(b)
    arb_init(t1); arb_init(t2); arb_init(h2_6); arb_init(tmp)
    arb_sub(x, t, u0, prec)
    arb_div(x, x, h, prec)               # x = (t-u0)/h
    xf = arf_get_d(arb_midref(x), ARF_RND_NEAR)
    if not c_isfinite(xf):
        if not c_gt(x, C_ZERO):
            i = 0
        else:
            i = n - 2
    else:
        i = <int> c_floor(xf)
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    # xi = u0 + i*h
    arb_set_si(t1, i)
    arb_mul(t1, t1, h, prec)
    arb_add(xi, u0, t1, prec)
    # a = (xi+h-t)/h ; b = (t-xi)/h
    arb_add(a, xi, h, prec)
    arb_sub(a, a, t, prec)
    arb_div(a, a, h, prec)
    arb_sub(b, t, xi, prec)
    arb_div(b, b, h, prec)
    # h2_6 = h*h/6
    arb_mul(h2_6, h, h, prec)
    arb_div(h2_6, h2_6, C_SIX, prec)
    # val = a*y[i] + b*y[i+1] + ((a^3-a)*M[i] + (b^3-b)*M[i+1])*h2_6
    arb_mul(val, a, y[i], prec)
    arb_mul(tmp, b, y[i+1], prec)
    arb_add(val, val, tmp, prec)
    arb_mul(t1, a, a, prec); arb_mul(t1, t1, a, prec)   # a^3
    arb_sub(t1, t1, a, prec)                            # a^3-a
    arb_mul(t1, t1, M[i], prec)
    arb_mul(t2, b, b, prec); arb_mul(t2, t2, b, prec)   # b^3
    arb_sub(t2, t2, b, prec)
    arb_mul(t2, t2, M[i+1], prec)
    arb_add(t1, t1, t2, prec)
    arb_mul(t1, t1, h2_6, prec)
    arb_add(val, val, t1, prec)
    # der = (y[i+1]-y[i])/h - (3a^2-1)/6 * h * M[i] + (3b^2-1)/6 * h * M[i+1]
    arb_sub(der, y[i+1], y[i], prec)
    arb_div(der, der, h, prec)
    arb_mul(t1, a, a, prec)
    arb_mul_2exp_si(t2, t1, 1)                # 2a^2
    arb_add(t1, t1, t2, prec)                 # 3a^2
    arb_sub(t1, t1, C_ONE, prec)              # 3a^2-1
    arb_div(t1, t1, C_SIX, prec)
    arb_mul(t1, t1, h, prec)
    arb_mul(t1, t1, M[i], prec)
    arb_sub(der, der, t1, prec)
    arb_mul(t1, b, b, prec)
    arb_mul_2exp_si(t2, t1, 1)
    arb_add(t1, t1, t2, prec)                 # 3b^2
    arb_sub(t1, t1, C_ONE, prec)
    arb_div(t1, t1, C_SIX, prec)
    arb_mul(t1, t1, h, prec)
    arb_mul(t1, t1, M[i+1], prec)
    arb_add(der, der, t1, prec)
    arb_clear(x); arb_clear(xi); arb_clear(a); arb_clear(b)
    arb_clear(t1); arb_clear(t2); arb_clear(h2_6); arb_clear(tmp)


# Find ALL roots of spline(y,M)=p_target on [u0,u0+(n-1)h]. Fill roots_u, roots_d
# (|der|). Returns count. sub = sub-intervals per cell.
cdef int c_spline_roots(arb_t *roots_u, arb_t *roots_d, arb_t *y, arb_t *M,
                        int n, const arb_t h, const arb_t u0,
                        const arb_t p_target, int sub, long prec):
    cdef int nseg = (n - 1) * sub
    cdef int cnt = 0
    cdef int s, it
    cdef arb_t t_prev, t_cur, v_prev, v_cur, der_dummy, step
    cdef arb_t dp, dc, lo, hi, flo, fhi, t, val, der, fval, tn, cross, adiff, afval
    cdef bint use_newton
    arb_init(t_prev); arb_init(t_cur); arb_init(v_prev); arb_init(v_cur)
    arb_init(der_dummy); arb_init(step)
    arb_init(dp); arb_init(dc); arb_init(lo); arb_init(hi); arb_init(flo); arb_init(fhi)
    arb_init(t); arb_init(val); arb_init(der); arb_init(fval); arb_init(tn)
    arb_init(cross); arb_init(adiff); arb_init(afval)
    arb_set(t_prev, u0)
    c_spline_eval(v_prev, der_dummy, y, M, n, h, u0, t_prev, prec)
    # step = h*(n-1)/nseg
    arb_set_si(step, n - 1)
    arb_mul(step, step, h, prec)
    arb_set_si(t, nseg)
    arb_div(step, step, t, prec)
    for s in range(1, nseg + 1):
        # t_cur = u0 + s*step
        arb_set_si(t, s)
        arb_mul(t, t, step, prec)
        arb_add(t_cur, u0, t, prec)
        c_spline_eval(v_cur, der_dummy, y, M, n, h, u0, t_cur, prec)
        arb_sub(dp, v_prev, p_target, prec)
        arb_sub(dc, v_cur, p_target, prec)
        if arb_is_zero(dp) and arb_is_zero(dc):
            arb_set(t_prev, t_cur)
            arb_set(v_prev, v_cur)
            continue
        arb_mul(cross, dp, dc, prec)
        if c_le(cross, C_ZERO):
            # rtsafe -- mirrors hfree_operator_arb.spline_roots EXACTLY,
            # including the INVERTED tolerance tests. Once a ball straddles 0
            # the direct "x < tol" is undecidable (False forever); "not (x>tol)"
            # flips True at the tolerance, which is what makes Newton terminate
            # at the tight root instead of drifting to a bracket endpoint.
            arb_set(lo, t_prev); arb_set(hi, t_cur)
            arb_set(flo, dp)                    # sign at lo
            arb_add(t, lo, hi, prec); arb_mul_2exp_si(t, t, -1)
            for it in range(400):
                c_spline_eval(val, der, y, M, n, h, u0, t, prec)
                arb_sub(fval, val, p_target, prec)
                # tighten the sign-bracket with sign of fval at t
                arb_mul(cross, flo, fval, prec)
                if c_le(cross, C_ZERO):
                    arb_set(hi, t)
                else:
                    arb_set(lo, t)
                    arb_set(flo, fval)
                # converged: residual no longer provably above tol, OR bracket
                # collapsed below tol (inverted comparisons).
                arb_abs(afval, fval)
                arb_sub(adiff, hi, lo, prec)
                if (not c_gt(afval, C_TOL)) or (not c_gt(adiff, C_TOL)):
                    break
                # next iterate: Newton if provably strictly inside (lo,hi),
                # else bisection of the maintained sign-bracket.
                use_newton = False
                if not arb_is_zero(der):
                    arb_div(tn, fval, der, prec)
                    arb_sub(tn, t, tn, prec)      # tn = t - fval/der
                    if c_gt(tn, lo) and c_lt(tn, hi):
                        arb_set(t, tn)
                        use_newton = True
                if not use_newton:
                    arb_add(t, lo, hi, prec); arb_mul_2exp_si(t, t, -1)
            c_spline_eval(val, der, y, M, n, h, u0, t, prec)
            arb_set(roots_u[cnt], t)
            arb_abs(roots_d[cnt], der)
            cnt += 1
        arb_set(t_prev, t_cur)
        arb_set(v_prev, v_cur)
    arb_clear(t_prev); arb_clear(t_cur); arb_clear(v_prev); arb_clear(v_cur)
    arb_clear(der_dummy); arb_clear(step)
    arb_clear(dp); arb_clear(dc); arb_clear(lo); arb_clear(hi); arb_clear(flo); arb_clear(fhi)
    arb_clear(t); arb_clear(val); arb_clear(der); arb_clear(fval); arb_clear(tn)
    arb_clear(cross); arb_clear(adiff); arb_clear(afval)
    return cnt


# ---------------- slice_evidence ----------------
# S is n x n stored row-major: S[ia*n+ib]. gnodes/gweights arb_t* length Nq.
cdef void c_slice_evidence(arb_t A0, arb_t A1, arb_t *S, int n,
                           const arb_t h, const arb_t u0, const arb_t p_target,
                           arb_t *gnodes, arb_t *gweights, int Nq,
                           const arb_t tauA, const arb_t tauB, int sub, long prec):
    cdef int ia, ib, q, r, ncnt, maxroots
    cdef arb_t fA0, fA1, fB0, fB1, v, d, dAval, dBval, dA2, dB2, denom, wA, wB, dB, dA
    cdef arb_t t1, t2, prod, der_dummy
    cdef arb_t *cols      # n*n col-major: cols[ib*n+ia] = S[ia,ib]
    cdef arb_t *colM
    cdef arb_t *rows      # n*n row-major: rows[ia*n+ib] = S[ia,ib]
    cdef arb_t *rowM
    cdef arb_t *rowvals
    cdef arb_t *rowdA
    cdef arb_t *Mrow
    cdef arb_t *MrowdA
    cdef arb_t *colvals
    cdef arb_t *coldB
    cdef arb_t *Mcol
    cdef arb_t *McoldB
    cdef arb_t *roots_u
    cdef arb_t *roots_d

    arb_set_ui(A0, 0)
    arb_set_ui(A1, 0)
    maxroots = 2 * n + 8

    arb_init(fA0); arb_init(fA1); arb_init(fB0); arb_init(fB1)
    arb_init(v); arb_init(d); arb_init(dAval); arb_init(dBval)
    arb_init(dA2); arb_init(dB2); arb_init(denom); arb_init(wA); arb_init(wB)
    arb_init(dB); arb_init(dA); arb_init(t1); arb_init(t2); arb_init(prod); arb_init(der_dummy)

    cols = <arb_t *> malloc(n * n * sizeof(arb_t))
    colM = <arb_t *> malloc(n * n * sizeof(arb_t))
    rows = <arb_t *> malloc(n * n * sizeof(arb_t))
    rowM = <arb_t *> malloc(n * n * sizeof(arb_t))
    rowvals = <arb_t *> malloc(n * sizeof(arb_t))
    rowdA = <arb_t *> malloc(n * sizeof(arb_t))
    Mrow = <arb_t *> malloc(n * sizeof(arb_t))
    MrowdA = <arb_t *> malloc(n * sizeof(arb_t))
    colvals = <arb_t *> malloc(n * sizeof(arb_t))
    coldB = <arb_t *> malloc(n * sizeof(arb_t))
    Mcol = <arb_t *> malloc(n * sizeof(arb_t))
    McoldB = <arb_t *> malloc(n * sizeof(arb_t))
    roots_u = <arb_t *> malloc(maxroots * sizeof(arb_t))
    roots_d = <arb_t *> malloc(maxroots * sizeof(arb_t))
    for ia in range(n * n):
        arb_init(cols[ia]); arb_init(colM[ia]); arb_init(rows[ia]); arb_init(rowM[ia])
    for ia in range(n):
        arb_init(rowvals[ia]); arb_init(rowdA[ia]); arb_init(Mrow[ia]); arb_init(MrowdA[ia])
        arb_init(colvals[ia]); arb_init(coldB[ia]); arb_init(Mcol[ia]); arb_init(McoldB[ia])
    for ia in range(maxroots):
        arb_init(roots_u[ia]); arb_init(roots_d[ia])

    # cols[ib][ia] = S[ia][ib]; colM[ib] = natural_spline_M(cols[ib])
    for ib in range(n):
        for ia in range(n):
            arb_set(cols[ib*n+ia], S[ia*n+ib])
    for ib in range(n):
        c_spline_M(&colM[ib*n], &cols[ib*n], n, h, prec)

    # ---- term A^(B): fix uA at GL nodes, root-find along axis B (cols) ----
    for q in range(Nq):
        c_fsignal(fA0, gnodes[q], 0, tauA, t1, t2, prec)
        c_fsignal(fA1, gnodes[q], 1, tauA, t1, t2, prec)
        for ib in range(n):
            c_spline_eval(v, d, &cols[ib*n], &colM[ib*n], n, h, u0, gnodes[q], prec)
            arb_set(rowvals[ib], v)
            arb_set(rowdA[ib], d)
        c_spline_M(Mrow, rowvals, n, h, prec)
        c_spline_M(MrowdA, rowdA, n, h, prec)
        ncnt = c_spline_roots(roots_u, roots_d, rowvals, Mrow, n, h, u0,
                              p_target, sub, prec)
        for r in range(ncnt):
            # uB = roots_u[r], dB = roots_d[r]
            arb_set(dB, roots_d[r])
            c_spline_eval(dAval, der_dummy, rowdA, MrowdA, n, h, u0, roots_u[r], prec)
            arb_mul(dA2, dAval, dAval, prec)
            arb_mul(dB2, dB, dB, prec)
            arb_add(denom, dA2, dB2, prec)
            if c_le(denom, C_ZERO):
                continue
            arb_div(wB, dB2, denom, prec)
            if c_le(dB, C_ZERO):
                continue
            c_fsignal(fB0, roots_u[r], 0, tauB, t1, t2, prec)
            c_fsignal(fB1, roots_u[r], 1, tauB, t1, t2, prec)
            # A0 += wA*wB*fA0*fB0/dB ; wA = gweights[q]
            arb_mul(prod, gweights[q], wB, prec)
            arb_mul(prod, prod, fA0, prec)
            arb_mul(prod, prod, fB0, prec)
            arb_div(prod, prod, dB, prec)
            arb_add(A0, A0, prod, prec)
            arb_mul(prod, gweights[q], wB, prec)
            arb_mul(prod, prod, fA1, prec)
            arb_mul(prod, prod, fB1, prec)
            arb_div(prod, prod, dB, prec)
            arb_add(A1, A1, prod, prec)

    # rows[ia][ib] = S[ia][ib]; rowM[ia] = natural_spline_M(rows[ia])
    for ia in range(n):
        for ib in range(n):
            arb_set(rows[ia*n+ib], S[ia*n+ib])
    for ia in range(n):
        c_spline_M(&rowM[ia*n], &rows[ia*n], n, h, prec)

    # ---- term A^(A): fix uB at GL nodes, root-find along axis A (rows) ----
    for q in range(Nq):
        c_fsignal(fB0, gnodes[q], 0, tauB, t1, t2, prec)
        c_fsignal(fB1, gnodes[q], 1, tauB, t1, t2, prec)
        for ia in range(n):
            c_spline_eval(v, d, &rows[ia*n], &rowM[ia*n], n, h, u0, gnodes[q], prec)
            arb_set(colvals[ia], v)
            arb_set(coldB[ia], d)
        c_spline_M(Mcol, colvals, n, h, prec)
        c_spline_M(McoldB, coldB, n, h, prec)
        ncnt = c_spline_roots(roots_u, roots_d, colvals, Mcol, n, h, u0,
                              p_target, sub, prec)
        for r in range(ncnt):
            arb_set(dA, roots_d[r])
            c_spline_eval(dBval, der_dummy, coldB, McoldB, n, h, u0, roots_u[r], prec)
            arb_mul(dA2, dA, dA, prec)
            arb_mul(dB2, dBval, dBval, prec)
            arb_add(denom, dA2, dB2, prec)
            if c_le(denom, C_ZERO):
                continue
            arb_div(wA, dA2, denom, prec)
            if c_le(dA, C_ZERO):
                continue
            c_fsignal(fA0, roots_u[r], 0, tauA, t1, t2, prec)
            c_fsignal(fA1, roots_u[r], 1, tauA, t1, t2, prec)
            arb_mul(prod, gweights[q], wA, prec)
            arb_mul(prod, prod, fA0, prec)
            arb_mul(prod, prod, fB0, prec)
            arb_div(prod, prod, dA, prec)
            arb_add(A0, A0, prod, prec)
            arb_mul(prod, gweights[q], wA, prec)
            arb_mul(prod, prod, fA1, prec)
            arb_mul(prod, prod, fB1, prec)
            arb_div(prod, prod, dA, prec)
            arb_add(A1, A1, prod, prec)

    for ia in range(n * n):
        arb_clear(cols[ia]); arb_clear(colM[ia]); arb_clear(rows[ia]); arb_clear(rowM[ia])
    for ia in range(n):
        arb_clear(rowvals[ia]); arb_clear(rowdA[ia]); arb_clear(Mrow[ia]); arb_clear(MrowdA[ia])
        arb_clear(colvals[ia]); arb_clear(coldB[ia]); arb_clear(Mcol[ia]); arb_clear(McoldB[ia])
    for ia in range(maxroots):
        arb_clear(roots_u[ia]); arb_clear(roots_d[ia])
    free(cols); free(colM); free(rows); free(rowM)
    free(rowvals); free(rowdA); free(Mrow); free(MrowdA)
    free(colvals); free(coldB); free(Mcol); free(McoldB)
    free(roots_u); free(roots_d)
    arb_clear(fA0); arb_clear(fA1); arb_clear(fB0); arb_clear(fB1)
    arb_clear(v); arb_clear(d); arb_clear(dAval); arb_clear(dBval)
    arb_clear(dA2); arb_clear(dB2); arb_clear(denom); arb_clear(wA); arb_clear(wB)
    arb_clear(dB); arb_clear(dA); arb_clear(t1); arb_clear(t2); arb_clear(prod); arb_clear(der_dummy)


# ---------------- constants init/clear ----------------
cdef void _init_consts(long prec):
    arb_init(C_ZERO); arb_init(C_ONE); arb_init(C_HALF); arb_init(C_EPS_PRICE)
    arb_init(C_TOL); arb_init(C_TWO); arb_init(C_THREE); arb_init(C_FOUR); arb_init(C_SIX)
    arb_init(C_TWOPI); arb_init(C_ONE_M_EPS)
    arb_set_ui(C_ZERO, 0)
    arb_one(C_ONE)
    arb_one(C_HALF); arb_mul_2exp_si(C_HALF, C_HALF, -1)
    arb_set_ui(C_TWO, 2); arb_set_ui(C_THREE, 3); arb_set_ui(C_FOUR, 4); arb_set_ui(C_SIX, 6)
    cdef py_arb eps_py = py_arb('1e-12')
    arb_set(C_EPS_PRICE, eps_py.val)
    cdef py_arb tol_py = py_arb('1e-130')
    arb_set(C_TOL, tol_py.val)
    arb_const_pi(C_TWOPI, prec)
    arb_mul_2exp_si(C_TWOPI, C_TWOPI, 1)   # 2*pi
    arb_sub(C_ONE_M_EPS, C_ONE, C_EPS_PRICE, prec)

cdef void _clear_consts():
    arb_clear(C_ZERO); arb_clear(C_ONE); arb_clear(C_HALF); arb_clear(C_EPS_PRICE)
    arb_clear(C_TOL); arb_clear(C_TWO); arb_clear(C_THREE); arb_clear(C_FOUR); arb_clear(C_SIX)
    arb_clear(C_TWOPI); arb_clear(C_ONE_M_EPS)


# ---------------- main entry ----------------
def phi_hfree(P, ui, gnodes, gweights, tau_vec, gamma_vec, W_vec, int sub):
    """Cython arb port of hfree_operator_arb.phi_hfree.
    P: (G,G,G) nested list of arb. ui: list of arb. gnodes/gweights: list arb len Nq.
    tau_vec/gamma_vec/W_vec: len-3 lists of arb. Returns (G,G,G) nested list of arb.
    """
    cdef int G = len(ui)
    cdef int Nq = len(gnodes)
    cdef long prec
    import flint
    prec = <long> flint.ctx.prec
    global PREC
    PREC = prec
    _init_consts(prec)

    cdef int i, j, l, a, b, idx
    cdef arb_t h, u0
    arb_init(h); arb_init(u0)

    # ---- copy in ----
    cdef arb_t *P_c = <arb_t *> malloc(G * G * G * sizeof(arb_t))
    cdef arb_t *ui_c = <arb_t *> malloc(G * sizeof(arb_t))
    cdef arb_t *gn_c = <arb_t *> malloc(Nq * sizeof(arb_t))
    cdef arb_t *gw_c = <arb_t *> malloc(Nq * sizeof(arb_t))
    cdef arb_t tau0, tau1, tau2, gam0, gam1, gam2, W0, W1, W2
    arb_init(tau0); arb_init(tau1); arb_init(tau2)
    arb_init(gam0); arb_init(gam1); arb_init(gam2)
    arb_init(W0); arb_init(W1); arb_init(W2)
    for i in range(G*G*G):
        arb_init(P_c[i])
    for i in range(G):
        arb_init(ui_c[i])
    for i in range(Nq):
        arb_init(gn_c[i]); arb_init(gw_c[i])
    for i in range(G):
        for j in range(G):
            for l in range(G):
                c_from_pyarb(P_c[(i*G+j)*G+l], P[i][j][l])
    for i in range(G):
        c_from_pyarb(ui_c[i], ui[i])
    for i in range(Nq):
        c_from_pyarb(gn_c[i], gnodes[i])
        c_from_pyarb(gw_c[i], gweights[i])
    c_from_pyarb(tau0, tau_vec[0]); c_from_pyarb(tau1, tau_vec[1]); c_from_pyarb(tau2, tau_vec[2])
    c_from_pyarb(gam0, gamma_vec[0]); c_from_pyarb(gam1, gamma_vec[1]); c_from_pyarb(gam2, gamma_vec[2])
    c_from_pyarb(W0, W_vec[0]); c_from_pyarb(W1, W_vec[1]); c_from_pyarb(W2, W_vec[2])
    # h = ui[1]-ui[0]; u0 = ui[0]
    arb_sub(h, ui_c[1], ui_c[0], prec)
    arb_set(u0, ui_c[0])

    cdef arb_t *out_c = <arb_t *> malloc(G * G * G * sizeof(arb_t))
    for i in range(G*G*G):
        arb_init(out_c[i])

    # slice buffers (n=G)
    cdef arb_t *S = <arb_t *> malloc(G * G * sizeof(arb_t))
    for i in range(G*G):
        arb_init(S[i])
    cdef arb_t A0a, A1a, A0b, A1b, A0c, A1c, mu0, mu1, mu2, p
    arb_init(A0a); arb_init(A1a); arb_init(A0b); arb_init(A1b); arb_init(A0c); arb_init(A1c)
    arb_init(mu0); arb_init(mu1); arb_init(mu2); arb_init(p)

    for i in range(G):
        for j in range(G):
            for l in range(G):
                arb_set(p, P_c[(i*G+j)*G+l])
                # Agent 0: S0 = P[i,:,:] -> S[jj*G+ll] = P[i][jj][ll]
                for a in range(G):
                    for b in range(G):
                        arb_set(S[a*G+b], P_c[(i*G+a)*G+b])
                c_slice_evidence(A0a, A1a, S, G, h, u0, p, gn_c, gw_c, Nq,
                                 tau1, tau2, sub, prec)
                c_bayes(mu0, ui_c[i], tau0, A0a, A1a, prec)
                # Agent 1: S1 = P[:,j,:] -> S[aa*G+ll] = P[aa][j][ll]
                for a in range(G):
                    for b in range(G):
                        arb_set(S[a*G+b], P_c[(a*G+j)*G+b])
                c_slice_evidence(A0b, A1b, S, G, h, u0, p, gn_c, gw_c, Nq,
                                 tau0, tau2, sub, prec)
                c_bayes(mu1, ui_c[j], tau1, A0b, A1b, prec)
                # Agent 2: S2 = P[:,:,l] -> S[aa*G+bb] = P[aa][bb][l]
                for a in range(G):
                    for b in range(G):
                        arb_set(S[a*G+b], P_c[(a*G+b)*G+l])
                c_slice_evidence(A0c, A1c, S, G, h, u0, p, gn_c, gw_c, Nq,
                                 tau0, tau1, sub, prec)
                c_bayes(mu2, ui_c[l], tau2, A0c, A1c, prec)
                c_clear_crra(out_c[(i*G+j)*G+l], mu0, mu1, mu2,
                             gam0, gam1, gam2, W0, W1, W2, prec)

    # ---- copy out ----
    out = [[[None] * G for _ in range(G)] for _ in range(G)]
    for i in range(G):
        for j in range(G):
            for l in range(G):
                out[i][j][l] = pyarb_from_c(out_c[(i*G+j)*G+l])

    # ---- cleanup ----
    for i in range(G*G*G):
        arb_clear(P_c[i]); arb_clear(out_c[i])
    for i in range(G):
        arb_clear(ui_c[i])
    for i in range(Nq):
        arb_clear(gn_c[i]); arb_clear(gw_c[i])
    for i in range(G*G):
        arb_clear(S[i])
    free(P_c); free(ui_c); free(gn_c); free(gw_c); free(out_c); free(S)
    arb_clear(h); arb_clear(u0)
    arb_clear(tau0); arb_clear(tau1); arb_clear(tau2)
    arb_clear(gam0); arb_clear(gam1); arb_clear(gam2)
    arb_clear(W0); arb_clear(W1); arb_clear(W2)
    arb_clear(A0a); arb_clear(A1a); arb_clear(A0b); arb_clear(A1b); arb_clear(A0c); arb_clear(A1c)
    arb_clear(mu0); arb_clear(mu1); arb_clear(mu2); arb_clear(p)
    _clear_consts()
    return out

