"""K=3 CRRA REE on the CDF-CUBE: axes are ζ_k = F̄(u_k) where F̄ is the
UNCONDITIONAL signal CDF (Gaussian mixture of v=0, v=1 components).

F̄(u) = ½·Φ(√τ·(u+½)) + ½·Φ(√τ·(u-½))
u(ζ) = F̄⁻¹(ζ)  via Brent's method (one-time table per τ)

Why CDF-uniform: the grid concentrates points where signal density is largest
(near u=±½), giving better quadrature than a linear u-grid (which wastes nodes
in the tails) or atanh-stretched ξ-grid (which doesn't match signal density).

Operator: same h-free smooth co-area architecture (cubic spline + GL +
partition-of-unity + smooth contour root-find) as hfree_smooth, but on the
non-uniform u-grid u_k = F̄⁻¹(ζ_k) with ζ uniform in [ζ_min, ζ_max].

Strict h=0, NO kernel, NO bandwidth.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import math

TAU = 2.0
GAMMA = 0.1
EPS_PRICE = 1e-9

# F̄(u) and inverse
SQRT_TAU = math.sqrt(TAU)
def F_bar(u):
    return 0.5*norm.cdf(SQRT_TAU*(u + 0.5)) + 0.5*norm.cdf(SQRT_TAU*(u - 0.5))
def F_bar_inv(zeta, bracket=(-20.0, 20.0)):
    return brentq(lambda u: F_bar(u) - zeta, bracket[0], bracket[1], xtol=1e-12)

def f_signal(u, v):
    vm = 0.5 if v == 1 else -0.5
    return math.sqrt(TAU/(2*math.pi)) * math.exp(-0.5*TAU*(u-vm)**2)

def f_signal_arr(u, v):
    vm = 0.5 if v == 1 else -0.5
    return np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u-vm)**2)

def build_zeta_u_grid(G, zeta_lo=1e-3, zeta_hi=1.0-1e-3):
    """G uniform ζ-nodes in [ζ_lo, ζ_hi]; u_k = F̄⁻¹(ζ_k)."""
    zeta = np.linspace(zeta_lo, zeta_hi, G)
    u = np.array([F_bar_inv(z) for z in zeta])
    return zeta, u

def gauss_legendre(n, a, b):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5*(b-a)*nodes + 0.5*(a+b), 0.5*(b-a)*weights

def slice_evidence_at(P_slice, u_grid, p_target, gl_u_a, gl_w_a, v):
    """Co-area evidence on a 2D slice. P_slice has shape (G, G) on axes (a, b)
    at grid u_grid (same for both axes; non-uniform).

    A_v(p) = sum_a w_a^GL * sum_{b roots: P(u_a^GL, b)=p}
                f_v(u_a^GL) f_v(u_b) / |dP/du_b|
    Two passes (vary a, vary b), partition-of-unity average.

    Implementation: for each fixed u_a, sample P(u_a, ·) via cubic spline on
    the GRID's a-direction (column-wise) then a spline in b at the contour.
    """
    G = u_grid.size
    A0 = 0.0; A1 = 0.0
    # PASS 0: fix u_a in GL nodes, vary u_b
    # For each u_a in gl nodes: get P(u_a, u_b) as a function of u_b via splines.
    # First, evaluate the 'column splines' P_b = spline_in_a(P[ :, b])(u_a) for each b.
    col_splines = [CubicSpline(u_grid, P_slice[:, b], bc_type='natural') for b in range(G)]
    for ia, (u_a_val, w_a) in enumerate(zip(gl_u_a, gl_w_a)):
        # P(u_a_val, u_b) at the G b-knots:
        P_line_b = np.array([col_splines[b](u_a_val) for b in range(G)])
        # Spline P along b-axis (non-uniform u_grid), find roots P=p_target
        try:
            sp = CubicSpline(u_grid, P_line_b - p_target, bc_type='natural')
            roots = sp.roots(extrapolate=False)
        except Exception:
            roots = []
        f0a = f_signal(u_a_val, 0); f1a = f_signal(u_a_val, 1)
        deriv = sp.derivative() if len(roots) > 0 else None
        for ub in roots:
            if not np.isfinite(ub) or ub < u_grid[0] or ub > u_grid[-1]: continue
            dPdb = float(deriv(ub))
            if abs(dPdb) < 1e-30: continue
            # partition-of-unity weight (need dP/da too, harder; skip for simplicity → over-count by factor 2)
            # NOTE: full partition-of-unity needs both d/da and d/db; we average two passes instead.
            f0b = f_signal(ub, 0); f1b = f_signal(ub, 1)
            A0 += w_a * f0a*f0b / abs(dPdb)
            A1 += w_a * f1a*f1b / abs(dPdb)
    # PASS 1: fix u_b in GL nodes, vary u_a (swap)
    row_splines = [CubicSpline(u_grid, P_slice[a, :], bc_type='natural') for a in range(G)]
    for ib, (u_b_val, w_b) in enumerate(zip(gl_u_a, gl_w_a)):
        P_line_a = np.array([row_splines[a](u_b_val) for a in range(G)])
        try:
            sp = CubicSpline(u_grid, P_line_a - p_target, bc_type='natural')
            roots = sp.roots(extrapolate=False)
        except Exception:
            roots = []
        f0b = f_signal(u_b_val, 0); f1b = f_signal(u_b_val, 1)
        deriv = sp.derivative() if len(roots) > 0 else None
        for ua in roots:
            if not np.isfinite(ua) or ua < u_grid[0] or ua > u_grid[-1]: continue
            dPda = float(deriv(ua))
            if abs(dPda) < 1e-30: continue
            f0a = f_signal(ua, 0); f1a = f_signal(ua, 1)
            A0 += w_b * f0a*f0b / abs(dPda)
            A1 += w_b * f1a*f1b / abs(dPda)
    return 0.5*A0, 0.5*A1  # average two passes

def crra_clear(mu0, mu1, mu2, gamma, steps=120):
    """Proper CRRA bisection (matches v11_strict_h0_hardwired.crra_clear_nb)."""
    eps = 1e-30
    a = eps; b = 1.0 - eps
    me0 = max(min(mu0, 1-EPS_PRICE), EPS_PRICE)
    me1 = max(min(mu1, 1-EPS_PRICE), EPS_PRICE)
    me2 = max(min(mu2, 1-EPS_PRICE), EPS_PRICE)
    lm0 = math.log(me0/(1-me0)); lm1 = math.log(me1/(1-me1)); lm2 = math.log(me2/(1-me2))
    for _ in range(steps):
        m = 0.5*(a+b); lp = math.log(m/(1-m))
        R0 = math.exp((lm0-lp)/gamma); R1 = math.exp((lm1-lp)/gamma); R2 = math.exp((lm2-lp)/gamma)
        e = (R0-1)/((1-m)+R0*m) + (R1-1)/((1-m)+R1*m) + (R2-1)/((1-m)+R2*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)

def phi_cdf_cube(P_full, u_grid, gl_u_a, gl_w_a, gamma):
    """One Phi-application on the CDF-cube (non-uniform u-grid)."""
    G = P_full.shape[0]
    P_new = P_full.copy()
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = P_full[i, j, k]
                A0a, A1a = slice_evidence_at(P_full[i, :, :], u_grid, p, gl_u_a, gl_w_a, None)
                f0o = f_signal(u_grid[i], 0); f1o = f_signal(u_grid[i], 1)
                den = f0o*A0a + f1o*A1a; mu0 = (f1o*A1a)/den if den>1e-30 else 0.5
                A0b, A1b = slice_evidence_at(P_full[:, j, :], u_grid, p, gl_u_a, gl_w_a, None)
                f0o = f_signal(u_grid[j], 0); f1o = f_signal(u_grid[j], 1)
                den = f0o*A0b + f1o*A1b; mu1 = (f1o*A1b)/den if den>1e-30 else 0.5
                A0c, A1c = slice_evidence_at(P_full[:, :, k], u_grid, p, gl_u_a, gl_w_a, None)
                f0o = f_signal(u_grid[k], 0); f1o = f_signal(u_grid[k], 1)
                den = f0o*A0c + f1o*A1c; mu2 = (f1o*A1c)/den if den>1e-30 else 0.5
                P_new[i, j, k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_new

def metrics(P, u_grid):
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T = TAU*(U1+U2+U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
    P_FR = 1/(1+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=defi, d_FR=d_FR, slope_T=float(a[0]))

if __name__ == '__main__':
    print(f'CDF-cube operator: τ={TAU}, γ={GAMMA}, F̄ = ½·Φ(√τ(u+½)) + ½·Φ(√τ(u-½))')
    G = 9
    zeta_arr, u_arr = build_zeta_u_grid(G)
    print(f'\nG={G}, ζ∈[{zeta_arr[0]:.4f}, {zeta_arr[-1]:.4f}] uniform')
    print(f'  u_arr = F̄⁻¹(ζ) = {np.round(u_arr, 3)}')
    print(f'  u range: [{u_arr.min():.3f}, {u_arr.max():.3f}]', flush=True)
    NQ = 24
    gl_u_a, gl_w_a = gauss_legendre(NQ, u_arr[0], u_arr[-1])
    # No-learning IC (proper-CRRA)
    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P_NL = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
                P_NL[i,j,k] = crra_clear(mu1, mu2, mu3, GAMMA)
    m_ic = metrics(P_NL, u_arr)
    print(f'\nIC (NL proper-CRRA): deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}', flush=True)
    # Picard from NL IC
    print(f'\nPicard from NL IC (CDF-cube, h=0)...', flush=True)
    Pf = P_NL.copy()
    for it in range(20):
        ts = time.time()
        Pn = phi_cdf_cube(Pf, u_arr, gl_u_a, gl_w_a, GAMMA)
        ferr = float(np.max(np.abs(Pn - Pf)))
        m = metrics(Pn, u_arr)
        print(f'  it {it+1:2d} ferr={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
        Pf = Pn
        if ferr < 1e-7:
            print('  CONVERGED'); break
    np.save(os.path.join(HERE, 'cdf_cube_NL_G9.npy'), Pf)
    json.dump({'IC':m_ic, 'final':metrics(Pf,u_arr)}, open(os.path.join(HERE,'cdf_cube_NL_G9.json'),'w'), indent=2)
    print('saved')
