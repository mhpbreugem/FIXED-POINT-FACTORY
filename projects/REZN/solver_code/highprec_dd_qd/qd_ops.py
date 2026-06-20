"""Quad-double (4x float64, ~64 digits, eps~1e-64) arithmetic for numba.
Bailey/Hida/Li algorithms. QD scalars are 4-tuples (a0,a1,a2,a3), |a_{i+1}|<=ulp(a_i)/2.
All njit -> SIMD/parallel friendly, no mpmath."""
import numpy as np
from numba import njit

# constants (ln2, pi) as quad-double
LN2 = (0.6931471805599453, 2.3190468138462996e-17, 5.707708438416212e-34, -3.5824322106018114e-50)
PI  = (3.141592653589793, 1.2246467991473532e-16, -2.9947698097183397e-33, 1.1124542208633655e-49)

@njit
def two_sum(a, b):
    s = a + b; bb = s - a
    return s, (a - (s - bb)) + (b - bb)

@njit
def quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)

@njit
def _split(a):
    c = 134217729.0 * a; hi = c - (c - a)
    return hi, a - hi

@njit
def two_prod(a, b):
    p = a * b
    ah, al = _split(a); bh, bl = _split(b)
    return p, ((ah*bh - p) + ah*bl + al*bh) + al*bl

@njit
def three_sum(a, b, c):
    t1, t2 = two_sum(a, b)
    a1, t3 = two_sum(t1, c)
    b1, c1 = two_sum(t2, t3)
    return a1, b1, c1

@njit
def three_sum2(a, b, c):
    t1, t2 = two_sum(a, b)
    a1, t3 = two_sum(t1, c)
    return a1, t2 + t3

@njit
def renorm(c0, c1, c2, c3, c4):
    s, c4 = quick_two_sum(c3, c4)
    s, c3 = quick_two_sum(c2, s)
    s, c2 = quick_two_sum(c1, s)
    c0, c1 = quick_two_sum(c0, s)
    s0 = c0; s1 = c1; s2 = 0.0; s3 = 0.0
    s0, s1 = quick_two_sum(c0, c1)
    if s1 != 0.0:
        s1, s2 = quick_two_sum(s1, c2)
        if s2 != 0.0:
            s2, s3 = quick_two_sum(s2, c3)
            if s3 != 0.0: s3 += c4
            else: s2, s3 = quick_two_sum(s2, c4)
        else:
            s1, s2 = quick_two_sum(s1, c3)
            if s2 != 0.0: s2, s3 = quick_two_sum(s2, c4)
            else: s1, s2 = quick_two_sum(s1, c4)
    else:
        s0, s1 = quick_two_sum(s0, c2)
        if s1 != 0.0:
            s1, s2 = quick_two_sum(s1, c3)
            if s2 != 0.0: s2, s3 = quick_two_sum(s2, c4)
            else: s1, s2 = quick_two_sum(s1, c4)
        else:
            s0, s1 = quick_two_sum(s0, c3)
            if s1 != 0.0: s1, s2 = quick_two_sum(s1, c4)
            else: s0, s1 = quick_two_sum(s0, c4)
    return s0, s1, s2, s3

@njit
def qd_add(a0, a1, a2, a3, b0, b1, b2, b3):
    s0, t0 = two_sum(a0, b0)
    s1, t1 = two_sum(a1, b1)
    s2, t2 = two_sum(a2, b2)
    s3, t3 = two_sum(a3, b3)
    s1, t0 = two_sum(s1, t0)
    s2, t0, t1 = three_sum(s2, t0, t1)
    s3, t0 = three_sum2(s3, t0, t2)
    t0 = t0 + t1 + t3
    return renorm(s0, s1, s2, s3, t0)

@njit
def qd_neg(a0, a1, a2, a3):
    return -a0, -a1, -a2, -a3

@njit
def qd_sub(a0, a1, a2, a3, b0, b1, b2, b3):
    return qd_add(a0, a1, a2, a3, -b0, -b1, -b2, -b3)

@njit
def qd_mul(a0, a1, a2, a3, b0, b1, b2, b3):
    p0, q0 = two_prod(a0, b0)
    p1, q1 = two_prod(a0, b1)
    p2, q2 = two_prod(a1, b0)
    p3, q3 = two_prod(a0, b2)
    p4, q4 = two_prod(a1, b1)
    p5, q5 = two_prod(a2, b0)
    p1, p2, q0 = three_sum(p1, p2, q0)
    p2, q1, q2 = three_sum(p2, q1, q2)
    p3, p4, p5 = three_sum(p3, p4, p5)
    s0, t0 = two_sum(p2, p3)
    s1, t1 = two_sum(q1, p4)
    s2 = q2 + p5
    s1, t0 = two_sum(s1, t0)
    s2 += (t0 + t1)
    s1 += a0*b3 + a1*b2 + a2*b1 + a3*b0 + q0 + q3 + q4 + q5
    return renorm(p0, p1, s0, s1, s2)

@njit
def qd_mul_d(a0, a1, a2, a3, b):
    p0, q0 = two_prod(a0, b)
    p1, q1 = two_prod(a1, b)
    p2, q2 = two_prod(a2, b)
    p3 = a3 * b
    s0 = p0
    s1, s2 = two_sum(q0, p1)
    s2, q1, p2 = three_sum(s2, q1, p2)
    q1, q2 = three_sum2(q1, q2, p3)
    s3 = q1
    s4 = q2 + p2
    return renorm(s0, s1, s2, s3, s4)

@njit
def qd_sqr(a0, a1, a2, a3):
    return qd_mul(a0, a1, a2, a3, a0, a1, a2, a3)

@njit
def qd_div(a0, a1, a2, a3, b0, b1, b2, b3):
    q0 = a0 / b0
    r0, r1, r2, r3 = qd_sub(a0, a1, a2, a3, *qd_mul_d(b0, b1, b2, b3, q0))
    q1 = r0 / b0
    r0, r1, r2, r3 = qd_sub(r0, r1, r2, r3, *qd_mul_d(b0, b1, b2, b3, q1))
    q2 = r0 / b0
    r0, r1, r2, r3 = qd_sub(r0, r1, r2, r3, *qd_mul_d(b0, b1, b2, b3, q2))
    q3 = r0 / b0
    return renorm(q0, q1, q2, q3, 0.0)

@njit
def qd_ldexp(a0, a1, a2, a3, k):
    s = 2.0 ** k
    return a0*s, a1*s, a2*s, a3*s

@njit
def qd_sqrt(a0, a1, a2, a3):
    if a0 <= 0.0: return 0.0, 0.0, 0.0, 0.0
    y0 = np.sqrt(a0)
    yh, yl = quick_two_sum(y0, 0.0)
    Yh, Yl, Y2, Y3 = yh, 0.0, 0.0, 0.0
    # Newton: y = 0.5*(y + a/y), twice (16->32->64)
    for _ in range(2):
        qh, ql, q2, q3 = qd_div(a0, a1, a2, a3, Yh, Yl, Y2, Y3)
        sh, sl, s2, s3 = qd_add(Yh, Yl, Y2, Y3, qh, ql, q2, q3)
        Yh, Yl, Y2, Y3 = qd_mul_d(sh, sl, s2, s3, 0.5)
    return Yh, Yl, Y2, Y3

@njit
def qd_exp(a0, a1, a2, a3):
    if a0 < -700.0: return 0.0, 0.0, 0.0, 0.0
    if a0 > 700.0: return 1e308, 0.0, 0.0, 0.0
    # k = round(a0/ln2)
    k = np.floor(a0 / LN2[0] + 0.5)
    # r = (a - k*ln2) / 2^m  (m=8 -> reduce magnitude, fewer Taylor terms)
    mh, ml, m2, m3 = qd_mul_d(LN2[0], LN2[1], LN2[2], LN2[3], k)
    rh, rl, r2, r3 = qd_sub(a0, a1, a2, a3, mh, ml, m2, m3)
    m = 8
    rh, rl, r2, r3 = qd_ldexp(rh, rl, r2, r3, -m)
    # Taylor of exp(r)-1
    sh, sl, s2, s3 = rh, rl, r2, r3
    th, tl, t2, t3 = rh, rl, r2, r3
    fac = 1.0
    for n in range(2, 22):
        fac *= n
        th, tl, t2, t3 = qd_mul(th, tl, t2, t3, rh, rl, r2, r3)
        # term = t / fac
        ah, al, a2_, a3_ = qd_div(th, tl, t2, t3, fac, 0.0, 0.0, 0.0)
        sh, sl, s2, s3 = qd_add(sh, sl, s2, s3, ah, al, a2_, a3_)
    # square m times: (1+s)^(2^m) - 1
    for _ in range(m):
        # s = 2*s + s*s
        twos = qd_mul_d(sh, sl, s2, s3, 2.0)
        sq = qd_sqr(sh, sl, s2, s3)
        sh, sl, s2, s3 = qd_add(twos[0], twos[1], twos[2], twos[3], sq[0], sq[1], sq[2], sq[3])
    # exp = (1+s)*2^k
    sh, sl, s2, s3 = qd_add(sh, sl, s2, s3, 1.0, 0.0, 0.0, 0.0)
    ki = int(k)
    return np.ldexp(sh, ki), np.ldexp(sl, ki), np.ldexp(s2, ki), np.ldexp(s3, ki)

@njit
def qd_log(a0, a1, a2, a3):
    y0 = np.log(a0)
    yh, yl, y2, y3 = y0, 0.0, 0.0, 0.0
    # Newton: y += a*exp(-y) - 1, three times (16->32->64; extra step for safety)
    for _ in range(3):
        eh, el, e2, e3 = qd_exp(-yh, -yl, -y2, -y3)
        ph, pl, p2, p3 = qd_mul(a0, a1, a2, a3, eh, el, e2, e3)
        ch, cl, c2, c3 = qd_sub(ph, pl, p2, p3, 1.0, 0.0, 0.0, 0.0)
        yh, yl, y2, y3 = qd_add(yh, yl, y2, y3, ch, cl, c2, c3)
    return yh, yl, y2, y3

@njit
def qd_atanh(a0, a1, a2, a3):
    nh, nl, n2, n3 = qd_add(1.0, 0.0, 0.0, 0.0, a0, a1, a2, a3)
    dh, dl, d2, d3 = qd_sub(1.0, 0.0, 0.0, 0.0, a0, a1, a2, a3)
    qh, ql, q2, q3 = qd_div(nh, nl, n2, n3, dh, dl, d2, d3)
    lh, ll, l2, l3 = qd_log(qh, ql, q2, q3)
    return qd_mul_d(lh, ll, l2, l3, 0.5)

@njit
def qd_sigmoid(a0, a1, a2, a3):
    eh, el, e2, e3 = qd_exp(-a0, -a1, -a2, -a3)
    dh, dl, d2, d3 = qd_add(1.0, 0.0, 0.0, 0.0, eh, el, e2, e3)
    return qd_div(1.0, 0.0, 0.0, 0.0, dh, dl, d2, d3)

@njit
def qd_logit(a0, a1, a2, a3):
    dh, dl, d2, d3 = qd_sub(1.0, 0.0, 0.0, 0.0, a0, a1, a2, a3)
    qh, ql, q2, q3 = qd_div(a0, a1, a2, a3, dh, dl, d2, d3)
    return qd_log(qh, ql, q2, q3)
