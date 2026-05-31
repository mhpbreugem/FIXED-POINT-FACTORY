"""CDF-cube CRRA operator in mpmath at dps=50 (~165 bits).

Tests whether the float64 NK floor at ||F||≈0.03-0.16 is due to arithmetic
precision (mpmath would crack it) or due to discretization (mpmath wouldn't
help; need finer G, more GL nodes, spline clipping).

Same architecture as cdf_cube_operator.py (scipy version) but all arithmetic
through mpmath.mpf at dps=50. SLOW (no JIT, no vectorization) — only viable
at small G (G=5 or 7) for diagnostic comparison.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
import mpmath as mp
mp.mp.dps = 50

import numpy as np

TAU = mp.mpf('2.0')
GAMMA_DEFAULT = mp.mpf('0.1')
EPS_PRICE = mp.mpf('1e-9')
EPSB = mp.mpf('1e-3')

def F_bar(u):
    """F̄(u) = ½Φ(√τ(u+½)) + ½Φ(√τ(u-½)) via mpmath erf."""
    st = mp.sqrt(TAU)
    return mp.mpf('0.5')*(mp.mpf('0.5')*(1 + mp.erf(st*(u+mp.mpf('0.5'))/mp.sqrt(2))) +
                            mp.mpf('0.5')*(1 + mp.erf(st*(u-mp.mpf('0.5'))/mp.sqrt(2))))

def f_bar(u):
    """f̄(u) = ½f_v0(u) + ½f_v1(u)."""
    s = mp.sqrt(TAU/(2*mp.pi))
    return mp.mpf('0.5')*s*(mp.exp(-mp.mpf('0.5')*TAU*(u+mp.mpf('0.5'))**2) +
                              mp.exp(-mp.mpf('0.5')*TAU*(u-mp.mpf('0.5'))**2))

def f_signal(u, v):
    vm = mp.mpf('0.5') if v == 1 else mp.mpf('-0.5')
    return mp.sqrt(TAU/(2*mp.pi)) * mp.exp(-mp.mpf('0.5')*TAU*(u-vm)**2)

def F_bar_inv(zeta, lo=-mp.mpf(20), hi=mp.mpf(20)):
    return mp.findroot(lambda u: F_bar(u) - zeta, mp.mpf(0))

def build_zeta_grid_mp(G):
    """G uniform ζ-nodes in [EPSB, 1-EPSB] (mpmath); u_k = F̄⁻¹(ζ_k)."""
    zeta = [EPSB + (mp.mpf(1) - 2*EPSB)*mp.mpf(i)/(G-1) for i in range(G)]
    u = [F_bar_inv(z) for z in zeta]
    return zeta, u

def cubic_spline_natural(x_arr, y_arr):
    """Natural cubic spline coefficients (a, b, c, d) for given non-uniform knots.
    Returns piece-wise polynomials in form (a_i, b_i, c_i, d_i) for
    s_i(x) = a_i + b_i (x-x_i) + c_i (x-x_i)² + d_i (x-x_i)³.
    """
    n = len(x_arr) - 1
    h = [x_arr[i+1] - x_arr[i] for i in range(n)]
    # Tridiagonal solve for second derivatives M_i (natural: M_0 = M_n = 0)
    A = mp.zeros(n+1, n+1)
    b = mp.zeros(n+1, 1)
    A[0,0] = 1; A[n,n] = 1
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i, 0] = 6 * ((y_arr[i+1]-y_arr[i])/h[i] - (y_arr[i]-y_arr[i-1])/h[i-1])
    M = mp.lu_solve(A, b)
    M = [M[i, 0] for i in range(n+1)]
    return M  # second derivatives at each knot

def spline_eval_natural(x_arr, y_arr, M, x):
    n = len(x_arr) - 1
    # Find interval
    if x <= x_arr[0]: i = 0
    elif x >= x_arr[-1]: i = n - 1
    else:
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x_arr[mid] <= x: lo = mid
            else: hi = mid
        i = lo
    h = x_arr[i+1] - x_arr[i]
    A = (x_arr[i+1] - x) / h
    B = (x - x_arr[i]) / h
    val = A*y_arr[i] + B*y_arr[i+1] + ((A**3-A)*M[i] + (B**3-B)*M[i+1])*(h*h)/6
    der = (y_arr[i+1]-y_arr[i])/h + (-3*A**2 + 1)/6 * h * M[i] + (3*B**2 - 1)/6 * h * M[i+1]
    return val, der

def find_all_roots(x_arr, y_arr, p_target, sub=4):
    """Return list of (x_root, dy/dx_root) for natural spline of y_arr=P at p=p_target."""
    n = len(x_arr) - 1
    M = cubic_spline_natural(x_arr, y_arr)
    roots = []
    for i in range(n):
        for s in range(sub):
            xa = x_arr[i] + s*(x_arr[i+1]-x_arr[i])/sub
            xb = x_arr[i] + (s+1)*(x_arr[i+1]-x_arr[i])/sub
            va, _ = spline_eval_natural(x_arr, y_arr, M, xa)
            vb, _ = spline_eval_natural(x_arr, y_arr, M, xb)
            if (va-p_target)*(vb-p_target) <= 0:
                # bisect 30 times to high precision
                a, b = xa, xb
                fa = va - p_target
                for _ in range(30):
                    m = (a+b)/2
                    vm, _ = spline_eval_natural(x_arr, y_arr, M, m)
                    fm = vm - p_target
                    if fa*fm <= 0: b = m
                    else: a = m; fa = fm
                xr = (a+b)/2
                vr, dr = spline_eval_natural(x_arr, y_arr, M, xr)
                roots.append((xr, dr))
    return roots

def slice_evidence_mp(P_slice, u_grid, p_target, v, gl_u_nodes, gl_u_weights):
    """A_v(p) on 2D slice via PCHIP-like contour. P_slice has shape (G, G).
    Average two passes (vary ua / vary ub).
    """
    G = len(u_grid)
    A = mp.mpf(0)
    # PASS 0: fix ua at GL nodes, root-find ub on each
    # For each GL ua, evaluate P(ua, ub) at each grid ub via column splines, then root-find ub
    col_M = [cubic_spline_natural(u_grid, [P_slice[a][b] for a in range(G)]) for b in range(G)]
    for ua_gl, wa in zip(gl_u_nodes, gl_u_weights):
        f_a = f_signal(ua_gl, v)
        P_line_b = []
        for b in range(G):
            v_val, _ = spline_eval_natural(u_grid, [P_slice[a][b] for a in range(G)], col_M[b], ua_gl)
            P_line_b.append(v_val)
        roots = find_all_roots(u_grid, P_line_b, p_target)
        for ub_r, dPdub in roots:
            if abs(dPdub) < mp.mpf('1e-30'): continue
            f_b = f_signal(ub_r, v)
            A += wa * f_a * f_b / abs(dPdub)
    # PASS 1: fix ub at GL nodes, root-find ua
    row_M = [cubic_spline_natural(u_grid, [P_slice[a][b] for b in range(G)]) for a in range(G)]
    for ub_gl, wb in zip(gl_u_nodes, gl_u_weights):
        f_b = f_signal(ub_gl, v)
        P_line_a = []
        for a in range(G):
            v_val, _ = spline_eval_natural(u_grid, [P_slice[a][b] for b in range(G)], row_M[a], ub_gl)
            P_line_a.append(v_val)
        roots = find_all_roots(u_grid, P_line_a, p_target)
        for ua_r, dPdua in roots:
            if abs(dPdua) < mp.mpf('1e-30'): continue
            f_a = f_signal(ua_r, v)
            A += wb * f_a * f_b / abs(dPdua)
    return A/2

def crra_clear_mp(mu0, mu1, mu2, gamma, steps=300):
    """CRRA bisection at high precision."""
    eps = mp.mpf('1e-50')
    a = eps; b = mp.mpf(1) - eps
    for mu in (mu0, mu1, mu2):
        if mu < EPS_PRICE or mu > 1-EPS_PRICE:
            pass
    me = [max(min(m, mp.mpf(1)-EPS_PRICE), EPS_PRICE) for m in (mu0, mu1, mu2)]
    lm = [mp.log(m/(1-m)) for m in me]
    for _ in range(steps):
        m = (a+b)/2
        lp = mp.log(m/(1-m))
        e = mp.mpf(0)
        for lm_k in lm:
            arg = (lm_k - lp)/gamma
            if arg > mp.mpf(700):
                e += mp.mpf(1)/m
            else:
                R = mp.exp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return (a+b)/2

def phi_cdf_mp(P_lst, u_arr, gamma, gl_u_nodes, gl_u_weights):
    """Phi: cube list-of-lists-of-lists → cube. Slow but high-precision.
    P_lst is a G×G×G list of mpf values.
    """
    G = len(u_arr)
    P_new = [[[mp.mpf(0) for _ in range(G)] for _ in range(G)] for _ in range(G)]
    for i in range(G):
        u_i = u_arr[i]
        for j in range(G):
            u_j = u_arr[j]
            for k in range(G):
                u_k = u_arr[k]
                p = P_lst[i][j][k]
                # Agent 0
                A0 = slice_evidence_mp([[P_lst[i][b][c] for c in range(G)] for b in range(G)], u_arr, p, 0, gl_u_nodes, gl_u_weights)
                A1 = slice_evidence_mp([[P_lst[i][b][c] for c in range(G)] for b in range(G)], u_arr, p, 1, gl_u_nodes, gl_u_weights)
                f0 = f_signal(u_i, 0); f1 = f_signal(u_i, 1)
                den = f0*A0 + f1*A1
                mu0 = (f1*A1)/den if den > mp.mpf('1e-30') else mp.mpf('0.5')
                # Agent 1
                A0 = slice_evidence_mp([[P_lst[a][j][c] for c in range(G)] for a in range(G)], u_arr, p, 0, gl_u_nodes, gl_u_weights)
                A1 = slice_evidence_mp([[P_lst[a][j][c] for c in range(G)] for a in range(G)], u_arr, p, 1, gl_u_nodes, gl_u_weights)
                f0 = f_signal(u_j, 0); f1 = f_signal(u_j, 1)
                den = f0*A0 + f1*A1
                mu1 = (f1*A1)/den if den > mp.mpf('1e-30') else mp.mpf('0.5')
                # Agent 2
                A0 = slice_evidence_mp([[P_lst[a][b][k] for b in range(G)] for a in range(G)], u_arr, p, 0, gl_u_nodes, gl_u_weights)
                A1 = slice_evidence_mp([[P_lst[a][b][k] for b in range(G)] for a in range(G)], u_arr, p, 1, gl_u_nodes, gl_u_weights)
                f0 = f_signal(u_k, 0); f1 = f_signal(u_k, 1)
                den = f0*A0 + f1*A1
                mu2 = (f1*A1)/den if den > mp.mpf('1e-30') else mp.mpf('0.5')
                P_new[i][j][k] = crra_clear_mp(mu0, mu1, mu2, gamma)
    return P_new

def gauss_legendre_mp(n, a, b):
    """GL nodes+weights at high precision via mpmath."""
    nodes, weights = mp.mp.gauss_legendre(n)
    # mp.mp.gauss_legendre returns on [-1, 1]; rescale
    A = (b - a)/2; B = (a + b)/2
    return [A*n + B for n in nodes], [A*w for w in weights]

def metrics_mp(P_lst, u_arr):
    G = len(u_arr)
    X = []; Y = []
    Pfr = [[[mp.mpf(0) for _ in range(G)] for _ in range(G)] for _ in range(G)]
    for i in range(G):
        for j in range(G):
            for k in range(G):
                T = TAU * (u_arr[i] + u_arr[j] + u_arr[k])
                X.append(float(T))
                p = float(P_lst[i][j][k])
                p = max(1e-12, min(1-1e-12, p))
                Y.append(math.log(p/(1-p)))
                Pfr[i][j][k] = 1/(1 + mp.exp(-T))
    X = np.array(X); Y = np.array(Y)
    a = np.polyfit(X, Y, 1); pred = a[0]*X + a[1]
    defi = float(np.sum((Y-pred)**2) / max(np.sum((Y-Y.mean())**2), 1e-30))
    diff_sq = mp.mpf(0)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                diff_sq += (P_lst[i][j][k] - Pfr[i][j][k])**2
    d_FR = float(mp.sqrt(diff_sq / G**3))
    return dict(deficit=defi, d_FR=d_FR, slope_T=float(a[0]))

if __name__ == '__main__':
    G = 5
    NQ = 12
    print(f'mpmath dps={mp.mp.dps}, G={G}, NQ={NQ}', flush=True)
    zeta_arr, u_arr = build_zeta_grid_mp(G)
    print(f'  u range: [{float(u_arr[0]):.3f}, {float(u_arr[-1]):.3f}]', flush=True)
    print(f'  ζ_arr: {[float(z) for z in zeta_arr]}')
    print(f'  u_arr: {[float(u) for u in u_arr]}', flush=True)
    gl_u_nodes, gl_u_weights = gauss_legendre_mp(NQ, u_arr[0], u_arr[-1])
    print(f'  GL: {NQ} nodes on [{float(gl_u_nodes[0]):.3f}, {float(gl_u_nodes[-1]):.3f}]', flush=True)

    # NL IC
    def sg_mp(x): return 1/(1+mp.exp(-x))
    P_IC = []
    for i in range(G):
        layer = []
        for j in range(G):
            row = []
            for k in range(G):
                mu1 = sg_mp(TAU*u_arr[i]); mu2 = sg_mp(TAU*u_arr[j]); mu3 = sg_mp(TAU*u_arr[k])
                row.append(crra_clear_mp(mu1, mu2, mu3, GAMMA_DEFAULT))
            layer.append(row)
        P_IC.append(layer)
    m_ic = metrics_mp(P_IC, u_arr)
    print(f'\nIC: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}', flush=True)

    print(f'\nPicard mpmath dps=50 from NL IC...', flush=True)
    P = P_IC
    for it in range(5):
        ts = time.time()
        Pn = phi_cdf_mp(P, u_arr, GAMMA_DEFAULT, gl_u_nodes, gl_u_weights)
        ferr = mp.mpf(0)
        for i in range(G):
            for j in range(G):
                for k in range(G):
                    diff = abs(Pn[i][j][k] - P[i][j][k])
                    if diff > ferr: ferr = diff
        m = metrics_mp(Pn, u_arr)
        print(f'  it {it+1:2d} ferr={float(ferr):.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.0f}s)', flush=True)
        P = Pn
    # Convert to numpy and save
    P_np = np.array([[[float(P[i][j][k]) for k in range(G)] for j in range(G)] for i in range(G)])
    np.save(os.path.join(HERE, f'cdf_cube_mp_G{G}.npy'), P_np)
    print('saved')
