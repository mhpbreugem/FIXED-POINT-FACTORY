"""PCHIP variant of the CDF-cube CRRA operator.

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is monotone and
shape-preserving — it cannot overshoot, so probabilities stay in [0, 1]
intrinsically (no need to clip). Per-interval slopes are set by Fritsch-Carlson
to ensure monotonicity. Hermite cubic root-find on each interval is a quadratic
in t = (x-xL)/h.

This replaces natural_spline_M + spline_eval in cdf_cube_numba.py while keeping
the rest of the architecture (ζ-grid + GL quadrature + Bayes + CRRA clear).
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from numba import njit, prange
from scipy.stats import norm
from scipy.optimize import brentq

TAU = 2.0
GAMMA = 0.1
EPS_PRICE = 1e-9

SQRT_TAU = math.sqrt(TAU)
def F_bar(u):
    return 0.5*norm.cdf(SQRT_TAU*(u + 0.5)) + 0.5*norm.cdf(SQRT_TAU*(u - 0.5))
def f_bar(u):
    return 0.5*SQRT_TAU/math.sqrt(2*math.pi)*(math.exp(-0.5*TAU*(u+0.5)**2)+math.exp(-0.5*TAU*(u-0.5)**2))
def F_bar_inv(zeta):
    return brentq(lambda u: F_bar(u) - zeta, -20.0, 20.0, xtol=1e-12)

TAB_N = 2001
TAB_Z = np.linspace(1e-6, 1.0-1e-6, TAB_N)
TAB_U = np.array([F_bar_inv(z) for z in TAB_Z])
TAB_DUDZ = 1.0 / np.array([f_bar(u) for u in TAB_U])

@njit(cache=True, inline='always')
def lookup_uz(z, TAB_Z, TAB_U, TAB_DUDZ):
    n = TAB_Z.size
    if z <= TAB_Z[0]:
        return TAB_U[0], TAB_DUDZ[0]
    if z >= TAB_Z[-1]:
        return TAB_U[-1], TAB_DUDZ[-1]
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if TAB_Z[mid] <= z: lo = mid
        else: hi = mid
    t = (z - TAB_Z[lo]) / (TAB_Z[hi] - TAB_Z[lo])
    u = (1.0-t)*TAB_U[lo] + t*TAB_U[hi]
    duz = (1.0-t)*TAB_DUDZ[lo] + t*TAB_DUDZ[hi]
    return u, duz

@njit(cache=True, inline='always')
def f_signal_nb(u, vm, tau):
    return math.sqrt(tau/(2*math.pi)) * math.exp(-0.5*tau*(u-vm)**2)

# ---------- PCHIP slopes (Fritsch-Carlson) ----------
@njit(cache=True)
def pchip_slopes(y, h):
    """Per-knot slopes m[i] for monotone Hermite cubic on uniform-h knots.
    Fritsch-Carlson rule: m[i] is harmonic mean of adjacent secants when same
    sign, else 0 (preserves monotonicity).
    """
    n = y.size
    m = np.zeros(n)
    s = np.empty(n - 1)  # secants
    for i in range(n - 1):
        s[i] = (y[i+1] - y[i]) / h
    # Endpoints: 3-pt formula clipped to monotone
    m[0] = ((2*h + h)*s[0] - h*s[1]) / (2*h)
    if m[0]*s[0] <= 0.0:
        m[0] = 0.0
    elif abs(m[0]) > 3*abs(s[0]):
        m[0] = 3*s[0]
    for i in range(1, n - 1):
        if s[i-1]*s[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2*h + h; w2 = h + 2*h  # = 3h, 3h for uniform; weighted harmonic mean
            m[i] = (w1 + w2) / (w1/s[i-1] + w2/s[i])
    m[n-1] = ((2*h + h)*s[n-2] - h*s[n-3]) / (2*h)
    if m[n-1]*s[n-2] <= 0.0:
        m[n-1] = 0.0
    elif abs(m[n-1]) > 3*abs(s[n-2]):
        m[n-1] = 3*s[n-2]
    return m

# ---------- Hermite cubic eval (uniform h) ----------
@njit(cache=True, inline='always')
def hermite_eval(y, m, h, z0, z):
    """Evaluate the Hermite cubic at z; returns (value, derivative wrt z)."""
    i = int((z - z0) / h)
    n = y.size
    if i < 0: i = 0
    if i > n - 2: i = n - 2
    zL = z0 + i*h; zR = zL + h
    t = (z - zL) / h
    t2 = t*t; t3 = t2*t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    val = h00*y[i] + h10*h*m[i] + h01*y[i+1] + h11*h*m[i+1]
    # dval/dz = (dval/dt) / h
    dh00 = 6*t2 - 6*t
    dh10 = 3*t2 - 4*t + 1
    dh01 = -6*t2 + 6*t
    dh11 = 3*t2 - 2*t
    der = (dh00*y[i] + dh10*h*m[i] + dh01*y[i+1] + dh11*h*m[i+1]) / h
    return val, der

# ---------- Hermite cubic roots (per interval, Newton from t=0.5) ----------
@njit(cache=True)
def hermite_roots(y, m, h, z0, p_target, out_roots, out_ders, sub):
    n = y.size; cnt = 0
    for i in range(n - 1):
        zL = z0 + i*h
        for s in range(sub):
            za = zL + s*h/sub; zb = zL + (s+1)*h/sub
            va, _ = hermite_eval(y, m, h, z0, za)
            vb, _ = hermite_eval(y, m, h, z0, zb)
            if (va - p_target)*(vb - p_target) <= 0.0:
                # bisect a few, then Newton refine
                a = za; b = zb
                fa = va - p_target
                for _ in range(20):
                    mp = 0.5*(a+b)
                    vm, _ = hermite_eval(y, m, h, z0, mp)
                    fm = vm - p_target
                    if fa*fm <= 0.0: b = mp
                    else: a = mp; fa = fm
                z = 0.5*(a+b)
                for _ in range(8):
                    v, dv = hermite_eval(y, m, h, z0, z)
                    if abs(dv) < 1e-30: break
                    dz = (v - p_target)/dv
                    z -= dz
                    if z < zL: z = zL
                    if z > zL + h: z = zL + h
                    if abs(dz) < 1e-13: break
                vf, df = hermite_eval(y, m, h, z0, z)
                if cnt < out_roots.size:
                    out_roots[cnt] = z
                    out_ders[cnt] = df
                    cnt += 1
    return cnt

# ---------- Slice evidence using PCHIP ----------
@njit(cache=True)
def slice_evidence_pchip(P_slice, zeta_grid, h, z0,
                          gl_z_nodes, gl_z_weights,
                          TAB_Z, TAB_U, TAB_DUDZ, tau, p_target):
    G = zeta_grid.size; nq = gl_z_nodes.size
    out_roots = np.empty(20); out_ders = np.empty(20)
    A0 = 0.0; A1 = 0.0
    # PASS 0: ζ_a at GL, root-find ζ_b
    mb_cache = np.empty((G, G))
    for kb in range(G):
        mb_cache[kb, :] = pchip_slopes(P_slice[:, kb], h)
    for ia in range(nq):
        za_gl = gl_z_nodes[ia]; wa = gl_z_weights[ia]
        ua, dudza = lookup_uz(za_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
        P_line = np.empty(G)
        for kb in range(G):
            v, _ = hermite_eval(P_slice[:, kb], mb_cache[kb, :], h, z0, za_gl)
            P_line[kb] = v
        ma = pchip_slopes(P_line, h)
        nr = hermite_roots(P_line, ma, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            zb = out_roots[r]; dPdz_b = out_ders[r]
            if abs(dPdz_b) < 1e-30: continue
            ub, dudzb = lookup_uz(zb, TAB_Z, TAB_U, TAB_DUDZ)
            f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
            dPdu_b = dPdz_b / dudzb
            A0 += wa * dudza * f0a * f0b / abs(dPdu_b)
            A1 += wa * dudza * f1a * f1b / abs(dPdu_b)
    # PASS 1
    ma_cache = np.empty((G, G))
    for ka in range(G):
        ma_cache[ka, :] = pchip_slopes(P_slice[ka, :], h)
    for ib in range(nq):
        zb_gl = gl_z_nodes[ib]; wb = gl_z_weights[ib]
        ub, dudzb = lookup_uz(zb_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
        P_line = np.empty(G)
        for ka in range(G):
            v, _ = hermite_eval(P_slice[ka, :], ma_cache[ka, :], h, z0, zb_gl)
            P_line[ka] = v
        mb = pchip_slopes(P_line, h)
        nr = hermite_roots(P_line, mb, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            za = out_roots[r]; dPdz_a = out_ders[r]
            if abs(dPdz_a) < 1e-30: continue
            ua, dudza = lookup_uz(za, TAB_Z, TAB_U, TAB_DUDZ)
            f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
            dPdu_a = dPdz_a / dudza
            A0 += wb * dudzb * f0a * f0b / abs(dPdu_a)
            A1 += wb * dudzb * f1a * f1b / abs(dPdu_a)
    return 0.5*A0, 0.5*A1

@njit(cache=True)
def crra_clear_nb(mu0, mu1, mu2, gamma, steps):
    eps = 1e-30
    a = eps; b = 1.0 - eps
    me0 = mu0 if mu0 > EPS_PRICE else EPS_PRICE
    if me0 > 1-EPS_PRICE: me0 = 1-EPS_PRICE
    me1 = mu1 if mu1 > EPS_PRICE else EPS_PRICE
    if me1 > 1-EPS_PRICE: me1 = 1-EPS_PRICE
    me2 = mu2 if mu2 > EPS_PRICE else EPS_PRICE
    if me2 > 1-EPS_PRICE: me2 = 1-EPS_PRICE
    lm0 = math.log(me0/(1-me0)); lm1 = math.log(me1/(1-me1)); lm2 = math.log(me2/(1-me2))
    for _ in range(steps):
        m = 0.5*(a+b); lp = math.log(m/(1-m))
        e = 0.0
        for k in range(3):
            lm = lm0 if k==0 else (lm1 if k==1 else lm2)
            arg = (lm - lp)/gamma
            if arg > 700: R = 1e300
            elif arg < -700: R = 0.0
            else: R = math.exp(arg)
            if R > 1e290: e += 1.0/m
            else: e += (R-1.0)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)

@njit(cache=True, parallel=True)
def phi_cdf_pchip(P, zeta_grid, h, z0,
                   gl_z_nodes, gl_z_weights,
                   TAB_Z, TAB_U, TAB_DUDZ,
                   gamma, tau):
    G = zeta_grid.size
    P_new = P.copy()
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                A0a, A1a = slice_evidence_pchip(P[i, :, :], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_i, _ = lookup_uz(zeta_grid[i], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_i, -0.5, tau); f1o = f_signal_nb(u_i, 0.5, tau)
                den = f0o*A0a + f1o*A1a
                mu0 = (f1o*A1a)/den if den>1e-30 else 0.5
                A0b, A1b = slice_evidence_pchip(P[:, j, :], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_j, _ = lookup_uz(zeta_grid[j], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_j, -0.5, tau); f1o = f_signal_nb(u_j, 0.5, tau)
                den = f0o*A0b + f1o*A1b
                mu1 = (f1o*A1b)/den if den>1e-30 else 0.5
                A0c, A1c = slice_evidence_pchip(P[:, :, k], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_k, _ = lookup_uz(zeta_grid[k], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_k, -0.5, tau); f1o = f_signal_nb(u_k, 0.5, tau)
                den = f0o*A0c + f1o*A1c
                mu2 = (f1o*A1c)/den if den>1e-30 else 0.5
                P_new[i, j, k] = crra_clear_nb(mu0, mu1, mu2, gamma, 120)
    return P_new

def gauss_legendre(n, a, b):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5*(b-a)*nodes + 0.5*(a+b), 0.5*(b-a)*weights

def build_zeta_grid(G, zeta_lo=1e-3, zeta_hi=1.0-1e-3):
    zeta = np.linspace(zeta_lo, zeta_hi, G)
    u = np.array([F_bar_inv(z) for z in zeta])
    h = float(zeta[1] - zeta[0])
    z0 = float(zeta[0])
    return zeta, u, h, z0

def metrics(P, u_arr):
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    T = TAU*(U1+U2+U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
    P_FR = 1/(1+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=defi, d_FR=d_FR, slope_T=float(a[0]))

if __name__ == '__main__':
    G = 13
    NQ = 64
    zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
    print(f'PCHIP CDF-cube G={G}, NQ={NQ}, ζ∈[{zeta_arr[0]:.4f},{zeta_arr[-1]:.4f}]')
    gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])

    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P_NL = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
                P_NL[i,j,k] = crra_clear_nb(mu1, mu2, mu3, GAMMA, 120)
    m_ic = metrics(P_NL, u_arr)
    print(f'IC: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}', flush=True)

    print('\nJIT warmup phi_cdf_pchip...', flush=True); t = time.time()
    _ = phi_cdf_pchip(P_NL.copy(), zeta_arr, h, z0,
                       gl_z_nodes, gl_z_weights,
                       TAB_Z, TAB_U, TAB_DUDZ,
                       GAMMA, TAU)
    print(f'  warmup {time.time()-t:.0f}s', flush=True)

    print('\nPicard PCHIP from NL IC (test 3)...', flush=True)
    Pf = P_NL.copy()
    for it in range(15):
        ts = time.time()
        Pn = phi_cdf_pchip(Pf, zeta_arr, h, z0,
                            gl_z_nodes, gl_z_weights,
                            TAB_Z, TAB_U, TAB_DUDZ,
                            GAMMA, TAU)
        ferr = float(np.max(np.abs(Pn - Pf)))
        m = metrics(Pn, u_arr)
        print(f'  it {it+1:2d} ferr={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
        Pf = Pn
        if ferr < 1e-7: break
    np.save(os.path.join(HERE, 'cdf_cube_PCHIP_g0.1_G13.npy'), Pf)
    print('saved')
