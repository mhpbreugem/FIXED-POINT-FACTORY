"""Adaptive-damped Picard with gmpy2 at ~200-decimal precision.

Mirrors phi_K3_smooth_mp (kernel-smoothed Φ on (u_1, u_2, u_3) cube) but
uses gmpy2.mpfr for arithmetic. gmpy2 wraps GMP/MPFR — typically 2-5×
faster than mpmath at the same precision.

K=3 sym, h=0, γ=0.1, FR-ansatz IC, kernel_h=0.05.
"""
import os, sys, time, math
import numpy as np
import gmpy2
from gmpy2 import mpfr, exp, log, sqrt

# Set precision: 200 decimal digits ≈ 665 binary bits + headroom
DPS = 200
ctx = gmpy2.get_context()
ctx.precision = int(DPS * 3.33) + 20  # ~685 bits

# Config
G_FULL = 11
INNER_LO, INNER_HI = 2, 9
UMAX_INNER = 3.0
TAU = 2.0; GAMMA = 0.1; W_S = 1.0
KERNEL_H = 0.05
MAX_ITER = 50
REPORT_EVERY = 1

LOG = '/tmp/gmpy2_damp.log'
open(LOG, 'w').close()
def lg(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

# ===== gmpy2 helpers =====
def to_mpfr(x):
    return mpfr(str(x))

def fsig_lp(u, vm, tau, coef):
    """f_v(u) = coef * exp(-0.5 τ (u - vm)²)."""
    d = u - vm
    return coef * exp(-mpfr('0.5') * tau * d * d)

def crra_demand(mu, p, gamma, W):
    lm = log(mu / (mpfr('1') - mu))
    lp = log(p / (mpfr('1') - p))
    R = exp((lm - lp) / gamma)
    return W * (R - mpfr('1')) / ((mpfr('1') - p) + R * p)

def crra_clear_sym(mus, gamma, W, steps=300):
    eps = mpfr('1e-150')
    a, b = eps, mpfr('1') - eps
    for _ in range(steps):
        m = (a + b) / mpfr('2')
        ex = mpfr('0')
        for mu in mus:
            ex += crra_demand(mu, m, gamma, W)
        if ex > 0: a = m
        else: b = m
    return (a + b) / mpfr('2')

def phi_kernel(P, u_full, tau, gamma, W, kernel_h, coef):
    """One kernel-smoothed Φ step. P is a 3D list-of-lists-of-lists of mpfr."""
    G = len(P)
    vm0 = mpfr('-0.5'); vm1 = mpfr('0.5')
    inv_2h2 = mpfr('1') / (mpfr('2') * kernel_h * kernel_h)
    # precompute f0, f1 per axis
    f0 = [[fsig_lp(u_full[i], vm0, tau, coef) for i in range(G)] for _ in range(3)]
    f1 = [[fsig_lp(u_full[i], vm1, tau, coef) for i in range(G)] for _ in range(3)]

    P_new = [[[P[i][j][k] for k in range(G)] for j in range(G)] for i in range(G)]

    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                p = P[i][j][k]
                mus = []
                # Agent 0: slice fixing i
                A0_acc = mpfr('0'); A1_acc = mpfr('0')
                for a in range(G):
                    for b in range(G):
                        diff = P[i][a][b] - p
                        w = exp(-diff * diff * inv_2h2)
                        A0_acc += w * f0[1][a] * f0[2][b]
                        A1_acc += w * f1[1][a] * f1[2][b]
                den = f0[0][i] * A0_acc + f1[0][i] * A1_acc
                mu0 = f1[0][i] * A1_acc / den if den > 0 else mpfr('0.5')
                mus.append(mu0)
                # Agent 1: slice fixing j
                A0_acc = mpfr('0'); A1_acc = mpfr('0')
                for a in range(G):
                    for b in range(G):
                        diff = P[a][j][b] - p
                        w = exp(-diff * diff * inv_2h2)
                        A0_acc += w * f0[0][a] * f0[2][b]
                        A1_acc += w * f1[0][a] * f1[2][b]
                den = f0[1][j] * A0_acc + f1[1][j] * A1_acc
                mu1 = f1[1][j] * A1_acc / den if den > 0 else mpfr('0.5')
                mus.append(mu1)
                # Agent 2: slice fixing k
                A0_acc = mpfr('0'); A1_acc = mpfr('0')
                for a in range(G):
                    for b in range(G):
                        diff = P[a][b][k] - p
                        w = exp(-diff * diff * inv_2h2)
                        A0_acc += w * f0[0][a] * f0[1][b]
                        A1_acc += w * f1[0][a] * f1[1][b]
                den = f0[2][k] * A0_acc + f1[2][k] * A1_acc
                mu2 = f1[2][k] * A1_acc / den if den > 0 else mpfr('0.5')
                mus.append(mu2)
                # clear
                eps_p = mpfr('1e-100')
                mus = [max(eps_p, min(mpfr('1') - eps_p, mu)) for mu in mus]
                P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
    return P_new

def f_inf(A, B):
    G = len(A)
    m = mpfr('0')
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if d > m: m = d
    return m

# ===== setup =====
u_full_np = np.linspace(-UMAX_INNER, UMAX_INNER, G_FULL)
u_full = [to_mpfr(u) for u in u_full_np]
tau = to_mpfr(TAU); gamma = to_mpfr(GAMMA); W = to_mpfr(W_S); kh = to_mpfr(KERNEL_H)
coef = sqrt(tau / (mpfr('2') * gmpy2.const_pi()))

# FR ansatz
U1, U2, U3 = np.meshgrid(u_full_np, u_full_np, u_full_np, indexing='ij')
S_np = U1 + U2 + U3
P_FR_np = 1.0 / (1.0 + np.exp(-TAU * S_np))
P = [[[to_mpfr(P_FR_np[i,j,k]) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]

lg(f'gmpy2 Picard, precision={ctx.precision} bits (≈{DPS} decimal)')
lg(f'γ={GAMMA}, kernel_h={KERNEL_H}, G={G_FULL}, MAX_ITER={MAX_ITER}')

omega = mpfr('1.0')
res_prev = mpfr('1e100')
res_history = []; omega_history = []
t0 = time.time()

for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = phi_kernel(P, u_full, tau, gamma, W, kh, coef)
    # damped: P_new = (1-ω) P + ω Φ(P)
    P_damped = [[[((mpfr('1') - omega) * P[i][j][k] + omega * P_phi[i][j][k])
                   for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    res_mp = f_inf(P_damped, P)
    res = float(res_mp)
    if it > 3:
        if res_mp > res_prev * mpfr('0.99'):
            omega = max(omega * mpfr('0.7'), mpfr('0.001'))
        elif res_mp < res_prev * mpfr('0.6'):
            omega = min(omega * mpfr('1.05'), mpfr('1.0'))
    P = P_damped
    res_prev = res_mp
    res_history.append(res); omega_history.append(float(omega))
    if it % REPORT_EVERY == 0 or it == 1:
        lg(f'  iter {it:3d}  ω={float(omega):.4f}  res={res:.3e}  ({time.time()-t_step:.1f}s)')
    if res < 1e-180:
        lg(f'  CONVERGED iter {it}')
        break

lg(f'Total: {(time.time()-t0)/60:.1f} min')
lg(f'FINAL: ω={float(omega):.4f}, res={res:.3e}')

# save trace
import json
with open('/tmp/gmpy2_history.json', 'w') as f:
    json.dump({'res': res_history, 'omega': omega_history,
                'precision_bits': ctx.precision, 'dps': DPS,
                'MAX_ITER': MAX_ITER, 'GAMMA': GAMMA, 'KERNEL_H': KERNEL_H}, f, indent=2)
