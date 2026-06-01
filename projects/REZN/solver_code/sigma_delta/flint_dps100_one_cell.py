"""Diagnostic: compute Phi(FR_ansatz) at G=9, γ=10 in flint arb at dps=100,
compare ||F|| to float64 result. Same operator structure as hfree_operator
but all arithmetic through flint.arb. Tests whether the NK plateau at
||F||~5e-3 from FR ansatz is arithmetic precision (would shrink at dps=100)
or discretization (would stay).
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import flint
flint.ctx.prec = 333   # ~100 dec digits (log2(10^100) ≈ 332.2)
print(f'flint prec: {flint.ctx.prec} bits (~{int(flint.ctx.prec*0.301)} dec digits)')
arb = flint.arb

UMAX = arb(4)
TAU = arb(2)
GAMMA = arb('10.0')
G = 9
NQ = 40
SUB = 4

ui = [arb(-4) + arb(i)*arb(8)/arb(G-1) for i in range(G)]
h = ui[1] - ui[0]

def f_signal(u, vm, tau):
    """f_v Gaussian density at u with v-mean vm and precision tau."""
    return (tau/(2*arb.pi())).sqrt() * (-(u-vm)**2 * tau / 2).exp()

# FR ansatz
P_FR = []
for i in range(G):
    layer_i = []
    for j in range(G):
        row_j = []
        for k in range(G):
            T = TAU*(ui[i] + ui[j] + ui[k])
            p = arb(1) / (arb(1) + (-T).exp())
            row_j.append(p)
        layer_i.append(row_j)
    P_FR.append(layer_i)
print(f'FR ansatz built ({G**3} cells)', flush=True)

# Natural cubic spline (uniform h, arb arithmetic)
def natural_spline_M(y):
    n = len(y)
    M = [arb(0)]*n
    # Tridiagonal: A_i M_{i-1} + B_i M_i + C_i M_{i+1} = D_i
    # Natural: M[0] = M[n-1] = 0
    A = [arb(0)]*n; B = [arb(0)]*n; C = [arb(0)]*n; D = [arb(0)]*n
    B[0] = arb(1); B[n-1] = arb(1)
    for i in range(1, n-1):
        A[i] = h; B[i] = arb(4)*h; C[i] = h
        D[i] = arb(6) * (y[i+1] - 2*y[i] + y[i-1]) / h
    # Thomas
    for i in range(1, n):
        m = A[i] / B[i-1]
        B[i] -= m * C[i-1]
        D[i] -= m * D[i-1]
    M[n-1] = D[n-1] / B[n-1]
    for i in range(n-2, -1, -1):
        M[i] = (D[i] - C[i]*M[i+1]) / B[i]
    return M

def spline_eval(y, M, t):
    """Evaluate cubic spline at t. Returns (val, deriv)."""
    n = len(y)
    i_int = int(float((t - ui[0]) / h))
    if i_int < 0: i_int = 0
    if i_int > n-2: i_int = n-2
    zL = ui[i_int]; zR = ui[i_int+1]
    A_ = (zR - t)/h; B_ = (t - zL)/h
    val = A_*y[i_int] + B_*y[i_int+1] + ((A_**3 - A_)*M[i_int] + (B_**3 - B_)*M[i_int+1])*h*h/6
    der = (y[i_int+1] - y[i_int])/h - (3*A_**2 - 1)*h*M[i_int]/6 + (3*B_**2 - 1)*h*M[i_int+1]/6
    return val, der

# Spline roots via bisection+newton
def spline_roots(y, M, p_target, sub=4):
    n = len(y); roots = []
    for i in range(n-1):
        for s_idx in range(sub):
            za = ui[i] + arb(s_idx)*h/sub
            zb = ui[i] + arb(s_idx+1)*h/sub
            va, _ = spline_eval(y, M, za)
            vb, _ = spline_eval(y, M, zb)
            if float((va - p_target)*(vb - p_target)) <= 0:
                a, b = za, zb
                fa = va - p_target
                for _ in range(40):
                    mp_ = (a+b)/2
                    vm, _ = spline_eval(y, M, mp_)
                    fm = vm - p_target
                    if float(fa*fm) <= 0: b = mp_
                    else: a = mp_; fa = fm
                xr = (a+b)/2
                # Newton refine
                for _ in range(10):
                    v, d = spline_eval(y, M, xr)
                    if abs(float(d)) < 1e-50: break
                    xr = xr - (v - p_target)/d
                vf, df = spline_eval(y, M, xr)
                roots.append((xr, df))
    return roots

# Gauss-Legendre nodes for arb (compute in float, lift to arb)
gnodes_f, gweights_f = np.polynomial.legendre.leggauss(NQ)
gnodes = [arb(-4) + (arb(1)+arb(float(g_)))*arb(4) for g_ in gnodes_f]
gweights = [arb(float(w_))*arb(4) for w_ in gweights_f]

def slice_evidence(S, p_target, tauA, tauB):
    """Slice evidence (A0, A1) on 2D slice S[a, b] of arbs.
    Uses cubic spline + GL + partition-of-unity (simplified: 2-pass average)."""
    A0 = arb(0); A1 = arb(0)
    # Pre-cache splines along axis a (per b column)
    col_M = []
    for kb in range(G):
        col = [S[ka][kb] for ka in range(G)]
        col_M.append(natural_spline_M(col))
    row_M = []
    for ka in range(G):
        row_M.append(natural_spline_M(S[ka]))
    # PASS 0: ua at GL, find ub roots
    for ia in range(NQ):
        u_a = gnodes[ia]; w_a = gweights[ia]
        f0a = f_signal(u_a, arb('-0.5'), tauA)
        f1a = f_signal(u_a, arb('0.5'), tauA)
        P_line = []
        for kb in range(G):
            col = [S[ka][kb] for ka in range(G)]
            v, _ = spline_eval(col, col_M[kb], u_a)
            P_line.append(v)
        Ma = natural_spline_M(P_line)
        roots = spline_roots(P_line, Ma, p_target)
        for ub, dPdu in roots:
            if abs(float(dPdu)) < 1e-50: continue
            f0b = f_signal(ub, arb('-0.5'), tauB)
            f1b = f_signal(ub, arb('0.5'), tauB)
            A0 = A0 + w_a * f0a * f0b / abs(dPdu)
            A1 = A1 + w_a * f1a * f1b / abs(dPdu)
    # PASS 1: ub at GL, find ua roots
    for ib in range(NQ):
        u_b = gnodes[ib]; w_b = gweights[ib]
        f0b = f_signal(u_b, arb('-0.5'), tauB)
        f1b = f_signal(u_b, arb('0.5'), tauB)
        P_line = []
        for ka in range(G):
            v, _ = spline_eval(S[ka], row_M[ka], u_b)
            P_line.append(v)
        Mb = natural_spline_M(P_line)
        roots = spline_roots(P_line, Mb, p_target)
        for ua, dPdu in roots:
            if abs(float(dPdu)) < 1e-50: continue
            f0a = f_signal(ua, arb('-0.5'), tauA)
            f1a = f_signal(ua, arb('0.5'), tauA)
            A0 = A0 + w_b * f0a * f0b / abs(dPdu)
            A1 = A1 + w_b * f1a * f1b / abs(dPdu)
    return A0/2, A1/2

def bayes(u_own, tau_own, A0, A1):
    f0 = f_signal(u_own, arb('-0.5'), tau_own)
    f1 = f_signal(u_own, arb('0.5'), tau_own)
    den = f0*A0 + f1*A1
    if float(den) <= 1e-50: return arb('0.5')
    return f1*A1 / den

def crra_clear_3(mu0, mu1, mu2, gamma):
    eps = arb(10)**(-100)
    a = eps; b = arb(1) - eps
    mes = [m if float(m) > 1e-50 else eps for m in (mu0, mu1, mu2)]
    mes = [m if float(m) < 1.0 - 1e-50 else (arb(1) - eps) for m in mes]
    lms = [(m/(arb(1)-m)).log() for m in mes]
    for _ in range(200):
        m = (a+b)/2; lp = (m/(arb(1)-m)).log()
        e = arb(0)
        for lmk in lms:
            arg = (lmk - lp)/gamma
            if float(arg) > 700: e += arb(1)/m
            else:
                R = arg.exp()
                e += (R - arb(1))/((arb(1) - m) + R*m)
        if float(e) > 0: a = m
        else: b = m
    return (a+b)/2

# Test: compute Phi(P_FR) at ONE cell to get a sample
print('\nComputing Phi at one cell (i,j,k)=(4,4,4) (center)...', flush=True)
ts = time.time()
i, j, k = 4, 4, 4
p_cell = P_FR[i][j][k]
print(f'  Sample: u=({float(ui[i])},{float(ui[j])},{float(ui[k])}), p_cell={float(p_cell):.6f}', flush=True)
S_a = [[P_FR[i][a][b] for b in range(G)] for a in range(G)]
A0a, A1a = slice_evidence(S_a, p_cell, TAU, TAU)
mu0 = bayes(ui[i], TAU, A0a, A1a)
print(f'  μ_0 = {float(mu0):.10f}  (time {time.time()-ts:.0f}s)', flush=True)

S_b = [[P_FR[a][j][b] for b in range(G)] for a in range(G)]
A0b, A1b = slice_evidence(S_b, p_cell, TAU, TAU)
mu1 = bayes(ui[j], TAU, A0b, A1b)
print(f'  μ_1 = {float(mu1):.10f}', flush=True)

S_c = [[P_FR[a][b][k] for b in range(G)] for a in range(G)]
A0c, A1c = slice_evidence(S_c, p_cell, TAU, TAU)
mu2 = bayes(ui[k], TAU, A0c, A1c)
print(f'  μ_2 = {float(mu2):.10f}', flush=True)

p_new = crra_clear_3(mu0, mu1, mu2, GAMMA)
diff = p_new - p_cell
print(f'\n  Phi(P_FR)[4,4,4] - P_FR[4,4,4] = {float(diff):.6e}', flush=True)
print(f'  P_FR[4,4,4] = {float(p_cell):.12f}', flush=True)
print(f'  Phi[4,4,4]  = {float(p_new):.12f}', flush=True)
print(f'  arb interval width: {float(p_new.rad()):.3e}', flush=True)

# Compare to float64
import sys as sys2; sys2.path.insert(0, '/tmp')
import hfree_operator as H
ui_f = np.linspace(-4, 4, G)
U1, U2, U3 = np.meshgrid(ui_f, ui_f, ui_f, indexing='ij')
T_f = 2.0*(U1+U2+U3)
P_FR_f = 1.0/(1.0+np.exp(-T_f))
gnodes_f, gweights_f = H.gauss_legendre(NQ, -4.0, 4.0)
PhP = H.phi_hfree(P_FR_f, ui_f, gnodes_f, gweights_f, np.full(3,2.0), np.full(3,10.0), np.full(3,1.0), SUB)
diff_f64 = float(PhP[4,4,4] - P_FR_f[4,4,4])
print(f'\n  float64 Phi[4,4,4] - FR = {diff_f64:.6e}', flush=True)
print(f'  flint  Phi[4,4,4] - FR = {float(diff):.6e}', flush=True)
print(f'  flint - float64 difference = {(float(diff) - diff_f64):.3e}', flush=True)

json.dump({
    'arb_precision_bits': flint.ctx.prec,
    'arb_dps': int(flint.ctx.prec*0.301),
    'gamma': float(GAMMA), 'G': G, 'NQ': NQ,
    'cell_(i,j,k)': [i,j,k],
    'P_FR_cell': float(p_cell),
    'mu_0': float(mu0), 'mu_1': float(mu1), 'mu_2': float(mu2),
    'Phi_arb_cell': float(p_new),
    'Phi_arb_minus_FR': float(diff),
    'arb_interval_rad': float(p_new.rad()),
    'Phi_float64_cell': float(PhP[4,4,4]),
    'Phi_float64_minus_FR': diff_f64,
    'flint_vs_float64_diff': float(diff) - diff_f64,
}, open(os.path.join(HERE,'flint_dps100_one_cell.json'),'w'), indent=2, default=str)
print('\nsaved')
