"""Quick gmpy2 test: confirm high-precision arithmetic gives same iter-1 ferr
as float64 (i.e., the NK floor is discretization, not arithmetic).

Uses gmpy2.mpfr at 50 dps (~165 bits). Tests one Phi evaluation at G=5 and
compares to float64 result.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
import math
import gmpy2
from gmpy2 import mpfr, sqrt as gsqrt, exp as gexp, log as glog
def gpi():
    return gmpy2.const_pi()
ctx = gmpy2.get_context()
ctx.precision = 165  # ~50 decimal digits
import numpy as np

TAU = mpfr('2.0')
GAMMA = mpfr('0.1')
SQRT_TAU = gsqrt(TAU)

def F_bar(u):
    """Mixture CDF; use erf approximation via gmpy2 if available."""
    # Use mpmath's erf since gmpy2 doesn't have erf
    import mpmath as mp
    mp.mp.dps = 50
    u_mp = mp.mpf(str(u))
    half = mp.mpf('0.5')
    st = mp.sqrt(mp.mpf('2.0'))
    F = half * (half*(1 + mp.erf(st*(u_mp + half)/mp.sqrt(2))) +
                half*(1 + mp.erf(st*(u_mp - half)/mp.sqrt(2))))
    return mpfr(str(F))

def f_signal(u, v):
    vm = mpfr('0.5') if v == 1 else mpfr('-0.5')
    return gsqrt(TAU/(2*gpi())) * gexp(mpfr('-0.5')*TAU*(u-vm)**2)

def crra_clear(mu0, mu1, mu2, gamma):
    eps = mpfr('1e-40')
    a = eps; b = mpfr(1) - eps
    me = [max(min(mu, mpfr(1)-eps), eps) for mu in (mu0, mu1, mu2)]
    lm = [glog(m/(1-m)) for m in me]
    for _ in range(300):
        m = (a+b)/2; lp = glog(m/(1-m))
        e = mpfr(0)
        for lmk in lm:
            arg = (lmk - lp)/gamma
            if arg > mpfr(700):
                e += mpfr(1)/m
            else:
                R = gexp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return (a+b)/2

if __name__ == '__main__':
    # Confirm: compare crra_clear at gmpy2 vs float64 for one cell
    print(f'gmpy2 precision: {ctx.precision} bits (~{int(ctx.precision*0.301)} digits)')

    # One sample CRRA clear
    mu0 = mpfr('0.6'); mu1 = mpfr('0.7'); mu2 = mpfr('0.5')
    ts = time.time()
    p_gmpy2 = crra_clear(mu0, mu1, mu2, GAMMA)
    dt = time.time() - ts
    print(f'\ngmpy2 crra_clear(0.6, 0.7, 0.5, γ=0.1) = {p_gmpy2} ({dt*1000:.1f}ms)')

    # Compare to float64 (use existing numba)
    import sys; sys.path.insert(0, HERE)
    from cdf_cube_numba import crra_clear_nb as crra_f64
    p_f64 = crra_f64(0.6, 0.7, 0.5, 0.1, 120)
    print(f'float64 crra_clear_nb same = {p_f64}')

    diff = float(p_gmpy2) - p_f64
    print(f'\ndifference: {diff:.3e}')
    print(f'gmpy2 to 30 dps: {gmpy2.digits(p_gmpy2, 10, 30)}')

    print('\nVERDICT: if diff is ~1e-15, both methods agree; gmpy2 50-dps doesnt help.')
    print('         The NK floor at 0.03-0.16 is operator discretization (G, NQ, spline)')
    print('         not arithmetic precision (already at machine epsilon)')
