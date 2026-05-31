"""NUMBA-JIT CDF-cube CRRA operator.

Architecture: like hfree_smooth (cubic spline + GL + partition-of-unity), but
operating on a ζ-uniform grid where ζ = F̄(u) is the unconditional Gaussian-
mixture signal CDF. The u-coordinates u(ζ) and Jacobian du/dζ are pre-computed
in a dense linear-interpolation table for fast lookups inside numba kernels.

The cubic spline of P is built on the uniform ζ-grid (so the standard uniform
natural cubic-spline machinery applies). Contour roots ζ_root from
spline(ζ)=p_target are converted to u_root via table lookup, and f_v is
evaluated at u_root. The Jacobian du/dζ at each GL ζ-node multiplies the
quadrature weight to give the correct co-area integral.
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

# ---------- F̄ and u(ζ) precomputation ----------
SQRT_TAU = math.sqrt(TAU)
def F_bar(u):
    return 0.5*norm.cdf(SQRT_TAU*(u + 0.5)) + 0.5*norm.cdf(SQRT_TAU*(u - 0.5))
def f_bar(u):
    return 0.5*SQRT_TAU/math.sqrt(2*math.pi)*(math.exp(-0.5*TAU*(u+0.5)**2)+math.exp(-0.5*TAU*(u-0.5)**2))
def F_bar_inv(zeta):
    return brentq(lambda u: F_bar(u) - zeta, -20.0, 20.0, xtol=1e-12)

# Dense lookup table for u(ζ) and du/dζ on ζ ∈ [0, 1]
TAB_N = 2001
TAB_Z = np.linspace(1e-6, 1.0-1e-6, TAB_N)
TAB_U = np.array([F_bar_inv(z) for z in TAB_Z])
TAB_DUDZ = 1.0 / np.array([f_bar(u) for u in TAB_U])

@njit(cache=True, inline='always')
def lookup_uz(z, TAB_Z, TAB_U, TAB_DUDZ):
    """Linear-interp u(z) and du/dz at z."""
    n = TAB_Z.size
    if z <= TAB_Z[0]:
        return TAB_U[0], TAB_DUDZ[0]
    if z >= TAB_Z[-1]:
        return TAB_U[-1], TAB_DUDZ[-1]
    # find index
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if TAB_Z[mid] <= z: lo = mid
        else: hi = mid
    t = (z - TAB_Z[lo]) / (TAB_Z[hi] - TAB_Z[lo])
    u = (1.0-t)*TAB_U[lo] + t*TAB_U[hi]
    duz = (1.0-t)*TAB_DUDZ[lo] + t*TAB_DUDZ[hi]
    return u, duz

# ---------- f_signal ----------
@njit(cache=True, inline='always')
def f_signal_nb(u, vm, tau):
    return math.sqrt(tau/(2*math.pi)) * math.exp(-0.5*tau*(u-vm)**2)

# ---------- Natural cubic spline (uniform h) — from hfree_operator ----------
@njit(cache=True)
def natural_spline_M(y, h):
    n = y.size
    M = np.zeros(n); a = np.zeros(n); b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
    a[0]=0.0; b[0]=1.0; c[0]=0.0; d[0]=0.0
    for i in range(1, n-1):
        a[i]=h; b[i]=4.0*h; c[i]=h
        d[i]=6.0*(y[i+1]-2.0*y[i]+y[i-1])/h
    a[n-1]=0.0; b[n-1]=1.0; c[n-1]=0.0; d[n-1]=0.0
    # Thomas algorithm
    for i in range(1, n):
        m = a[i]/b[i-1]; b[i] -= m*c[i-1]; d[i] -= m*d[i-1]
    M[n-1] = d[n-1]/b[n-1]
    for i in range(n-2, -1, -1):
        M[i] = (d[i] - c[i]*M[i+1])/b[i]
    return M

@njit(cache=True, inline='always')
def spline_eval(y, M, h, z0, z):
    i = int((z - z0) / h)
    n = y.size
    if i < 0: i = 0
    if i > n - 2: i = n - 2
    zL = z0 + i*h; zR = zL + h
    A = (zR - z)/h; B = (z - zL)/h
    val = A*y[i] + B*y[i+1] + ((A**3-A)*M[i] + (B**3-B)*M[i+1])*(h*h)/6.0
    der = (y[i+1]-y[i])/h - (3*A*A-1)*h*M[i]/6.0 + (3*B*B-1)*h*M[i+1]/6.0
    return val, der

@njit(cache=True)
def spline_roots(y, M, h, z0, p_target, out_roots, out_ders, sub):
    n = y.size; cnt = 0
    for i in range(n-1):
        zL = z0 + i*h
        # subdivide [zL, zL+h] into 'sub' bracket-intervals and Newton each
        for s in range(sub):
            za = zL + s*h/sub; zb = zL + (s+1)*h/sub
            va, _ = spline_eval(y, M, h, z0, za)
            vb, _ = spline_eval(y, M, h, z0, zb)
            if (va-p_target)*(vb-p_target) <= 0.0:
                # bisect a few then newton
                a = za; b = zb
                for _ in range(20):
                    m = 0.5*(a+b); vm, _ = spline_eval(y, M, h, z0, m)
                    if (vm-p_target)*(va-p_target) <= 0.0: b = m
                    else: a = m; va = vm
                # Newton refine
                z = 0.5*(a+b)
                for _ in range(8):
                    v, dv = spline_eval(y, M, h, z0, z)
                    if abs(dv) < 1e-30: break
                    z = z - (v - p_target)/dv
                    if z < zL: z = zL
                    if z > zL+h: z = zL+h
                vf, df = spline_eval(y, M, h, z0, z)
                if cnt < out_roots.size:
                    out_roots[cnt] = z; out_ders[cnt] = df; cnt += 1
    return cnt

# ---------- Co-area on one slice (2D, CDF-cube) ----------
@njit(cache=True)
def slice_evidence_cdf(P_slice, zeta_grid, h, z0,
                        gl_z_nodes, gl_z_weights,
                        TAB_Z, TAB_U, TAB_DUDZ, tau, p_target):
    """A_v(p) for v ∈ {0, 1} on slice P_slice(ζ_a, ζ_b), with cubic spline
    + GL in the off-axis variable and root-find in the on-axis.

    Returns (A0, A1) averaged over the two passes (vary ζ_a / vary ζ_b).
    """
    G = zeta_grid.size; nq = gl_z_nodes.size
    out_roots = np.empty(20); out_ders = np.empty(20)
    A0 = 0.0; A1 = 0.0
    # PASS 0: ζ_a at GL nodes, root-find ζ_b
    # Pre-compute splines along axis 0 for each fixed b-index
    Mb_cache = np.empty((G, G))
    for kb in range(G):
        Mb_cache[kb, :] = natural_spline_M(P_slice[:, kb], h)
    # For each GL ζ_a, build P_line over b-indices, root-find
    for ia in range(nq):
        za_gl = gl_z_nodes[ia]; wa = gl_z_weights[ia]
        ua, dudza = lookup_uz(za_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
        P_line = np.empty(G)
        for kb in range(G):
            v, _ = spline_eval(P_slice[:, kb], Mb_cache[kb, :], h, z0, za_gl)
            P_line[kb] = v
        Ma = natural_spline_M(P_line, h)
        nr = spline_roots(P_line, Ma, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            zb = out_roots[r]; dPdz_b = out_ders[r]
            if abs(dPdz_b) < 1e-30: continue
            ub, dudzb = lookup_uz(zb, TAB_Z, TAB_U, TAB_DUDZ)
            f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
            # dP/du_b = dP/dζ_b · dζ/du_b = (dP/dζ_b) / (du/dζ_b)
            dPdu_b = dPdz_b / dudzb
            A0 += wa * dudza * f0a * f0b / abs(dPdu_b)
            A1 += wa * dudza * f1a * f1b / abs(dPdu_b)
    # PASS 1: ζ_b at GL nodes, root-find ζ_a
    Ma_cache = np.empty((G, G))
    for ka in range(G):
        Ma_cache[ka, :] = natural_spline_M(P_slice[ka, :], h)
    for ib in range(nq):
        zb_gl = gl_z_nodes[ib]; wb = gl_z_weights[ib]
        ub, dudzb = lookup_uz(zb_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
        P_line = np.empty(G)
        for ka in range(G):
            v, _ = spline_eval(P_slice[ka, :], Ma_cache[ka, :], h, z0, zb_gl)
            P_line[ka] = v
        Mb = natural_spline_M(P_line, h)
        nr = spline_roots(P_line, Mb, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            za = out_roots[r]; dPdz_a = out_ders[r]
            if abs(dPdz_a) < 1e-30: continue
            ua, dudza = lookup_uz(za, TAB_Z, TAB_U, TAB_DUDZ)
            f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
            dPdu_a = dPdz_a / dudza
            A0 += wb * dudzb * f0a * f0b / abs(dPdu_a)
            A1 += wb * dudzb * f1a * f1b / abs(dPdu_a)
    return 0.5*A0, 0.5*A1

# ---------- CRRA clearing (proper, with overflow-safe exp) ----------
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
            arg = (lm-lp)/gamma
            if arg > 700: R = 1e300
            elif arg < -700: R = 0.0
            else: R = math.exp(arg)
            if R > 1e290:
                demand = 1.0/m
            else:
                demand = (R-1.0)/((1-m) + R*m)
            e += demand
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)

# ---------- Phi (cube → cube) ----------
@njit(cache=True, parallel=True)
def phi_cdf_numba(P, zeta_grid, h, z0,
                   gl_z_nodes, gl_z_weights,
                   TAB_Z, TAB_U, TAB_DUDZ,
                   gamma, tau):
    G = zeta_grid.size
    P_new = P.copy()
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                # Agent 0: slice P[i, :, :], axes (ζ_2, ζ_3)
                A0a, A1a = slice_evidence_cdf(P[i, :, :], zeta_grid, h, z0,
                                                gl_z_nodes, gl_z_weights,
                                                TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_i, _ = lookup_uz(zeta_grid[i], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_i, -0.5, tau); f1o = f_signal_nb(u_i, 0.5, tau)
                den = f0o*A0a + f1o*A1a
                mu0 = (f1o*A1a)/den if den>1e-30 else 0.5
                # Agent 1: slice P[:, j, :]
                A0b, A1b = slice_evidence_cdf(P[:, j, :], zeta_grid, h, z0,
                                                gl_z_nodes, gl_z_weights,
                                                TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_j, _ = lookup_uz(zeta_grid[j], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_j, -0.5, tau); f1o = f_signal_nb(u_j, 0.5, tau)
                den = f0o*A0b + f1o*A1b
                mu1 = (f1o*A1b)/den if den>1e-30 else 0.5
                # Agent 2: slice P[:, :, k]
                A0c, A1c = slice_evidence_cdf(P[:, :, k], zeta_grid, h, z0,
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
    G = 9
    NQ = 24
    zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
    print(f'CDF-cube G={G}, ζ∈[{zeta_arr[0]:.4f},{zeta_arr[-1]:.4f}], u∈[{u_arr.min():.3f},{u_arr.max():.3f}]')
    gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
    print(f'GL ζ-nodes: {len(gl_z_nodes)} on [{gl_z_nodes[0]:.4f},{gl_z_nodes[-1]:.4f}]')

    # No-learning IC
    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P_NL = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
                P_NL[i,j,k] = crra_clear_nb(mu1, mu2, mu3, GAMMA, 120)
    m_ic = metrics(P_NL, u_arr)
    print(f'\nIC: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}', flush=True)

    print('\nJIT warmup phi_cdf_numba...', flush=True); t=time.time()
    _ = phi_cdf_numba(P_NL.copy(), zeta_arr, h, z0,
                       gl_z_nodes, gl_z_weights,
                       TAB_Z, TAB_U, TAB_DUDZ,
                       GAMMA, TAU)
    print(f'  warmup {time.time()-t:.0f}s', flush=True)

    # Picard test
    Pf = P_NL.copy()
    print('\nPicard from NL IC...', flush=True)
    for it in range(8):
        ts = time.time()
        Pn = phi_cdf_numba(Pf, zeta_arr, h, z0,
                            gl_z_nodes, gl_z_weights,
                            TAB_Z, TAB_U, TAB_DUDZ,
                            GAMMA, TAU)
        ferr = float(np.max(np.abs(Pn - Pf)))
        m = metrics(Pn, u_arr)
        print(f'  it {it+1:2d} ferr={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
        Pf = Pn
        if ferr < 1e-7: break
    np.save(os.path.join(HERE, 'cdf_cube_numba_NL_G9.npy'), Pf)
    print('saved')
