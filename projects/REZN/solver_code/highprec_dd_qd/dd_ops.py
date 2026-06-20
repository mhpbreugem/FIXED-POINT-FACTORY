"""Double-double (hi+lo float64) arithmetic for numba — ~32 digits, eps~1e-32.
All njit, FMA-free (Dekker split), so SIMD-vectorizable / parallelizable."""
import numpy as np
from numba import njit

# ln2, pi in double-double
LN2H = 0.6931471805599453; LN2L = 2.3190468138462996e-17
PIH = 3.141592653589793; PIL = 1.2246467991473532e-16

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
def dd_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    e += al + bl
    return quick_two_sum(s, e)

@njit
def dd_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e += ah*bl + al*bh
    return quick_two_sum(p, e)

@njit
def dd_div(ah, al, bh, bl):
    q1 = ah / bh
    # r = a - q1*b
    ph, pl = dd_mul(q1, 0.0, bh, bl)
    rh, rl = dd_add(ah, al, -ph, -pl)
    q2 = rh / bh
    ph2, pl2 = dd_mul(q2, 0.0, bh, bl)
    rh2, rl2 = dd_add(rh, rl, -ph2, -pl2)
    q3 = rh2 / bh
    s, e = quick_two_sum(q1, q2)
    return dd_add(s, e, q3, 0.0)

@njit
def dd_exp(xh, xl):
    if xh < -700.0: return 0.0, 0.0
    k = np.floor(xh / LN2H + 0.5)
    mh, ml = dd_mul(k, 0.0, LN2H, LN2L)
    rh, rl = dd_add(xh, xl, -mh, -ml)         # r = x - k*ln2,  |r|<=ln2/2
    # Taylor sum_{n>=0} r^n/n!
    sh, sl = 1.0, 0.0
    th, tl = 1.0, 0.0
    for n in range(1, 22):
        th, tl = dd_mul(th, tl, rh, rl)
        th, tl = dd_div(th, tl, float(n), 0.0)
        sh, sl = dd_add(sh, sl, th, tl)
    # * 2^k  (ldexp is exact for power-of-two scaling; 2.0**k via pow() is not)
    ki = int(k)
    return np.ldexp(sh, ki), np.ldexp(sl, ki)

@njit
def dd_log(xh, xl):
    # Newton: y = log(xh); y += x*exp(-y) - 1   (doubles precision)
    y = np.log(xh)
    eh, el = dd_exp(-y, 0.0)                   # exp(-y)
    ph, pl = dd_mul(xh, xl, eh, el)            # x*exp(-y)
    ch, cl = dd_add(ph, pl, -1.0, 0.0)        # x*exp(-y) - 1
    yh, yl = dd_add(y, 0.0, ch, cl)
    # one more Newton step for full precision
    eh, el = dd_exp(-yh, -yl)
    ph, pl = dd_mul(xh, xl, eh, el)
    ch, cl = dd_add(ph, pl, -1.0, 0.0)
    return dd_add(yh, yl, ch, cl)

@njit
def dd_sqrt(xh, xl):
    if xh <= 0.0: return 0.0, 0.0
    y = np.sqrt(xh)
    # y = (y + x/y)/2  in DD
    qh, ql = dd_div(xh, xl, y, 0.0)
    sh, sl = dd_add(y, 0.0, qh, ql)
    return dd_mul(sh, sl, 0.5, 0.0)

@njit
def dd_atanh(xh, xl):
    # 0.5*log((1+x)/(1-x))
    nh, nl = dd_add(1.0, 0.0, xh, xl)
    dh, dl = dd_add(1.0, 0.0, -xh, -xl)
    qh, ql = dd_div(nh, nl, dh, dl)
    lh, ll = dd_log(qh, ql)
    return dd_mul(lh, ll, 0.5, 0.0)

@njit
def dd_sigmoid(zh, zl):
    eh, el = dd_exp(-zh, -zl)
    dh, dl = dd_add(1.0, 0.0, eh, el)
    return dd_div(1.0, 0.0, dh, dl)

@njit
def dd_logit(ph, pl):
    dh, dl = dd_add(1.0, 0.0, -ph, -pl)
    qh, ql = dd_div(ph, pl, dh, dl)
    return dd_log(qh, ql)
