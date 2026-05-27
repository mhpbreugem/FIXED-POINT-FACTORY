"""Strict-contour Φ in python-flint arb at 200-decimal precision.

Single-iter benchmark vs gmpy2.
"""
import time, math
import numpy as np
import flint
from flint import arb

DPS = 200
flint.ctx.prec = int(DPS * 3.33) + 20

G_FULL = 11
INNER_LO, INNER_HI = 2, 9
UMAX = 3.0
TAU = 2.0; GAMMA = 0.1; W_S = 1.0


def mp(x):
    if isinstance(x, str):
        return arb(x)
    return arb(float(x))

def fsig(u, vm, tau, coef):
    d = u - vm
    return coef * (mp('-0.5') * tau * d * d).exp()

def crra_demand(mu, p, gamma, W):
    one = mp(1)
    lm = (mu / (one - mu)).log()
    lp = (p / (one - p)).log()
    R = ((lm - lp) / gamma).exp()
    return W * (R - one) / ((one - p) + R * p)

def crra_clear_sym(mus, gamma, W, steps=300):
    eps = mp('1e-150')
    one = mp(1); two = mp(2); zero = mp(0)
    a, b = eps, one - eps
    for _ in range(steps):
        m = (a + b) / two
        ex = zero
        for mu in mus: ex = ex + crra_demand(mu, m, gamma, W)
        if ex > 0: a = m
        else: b = m
    return (a + b) / two


def contour_evidence_2D(P_slice, p, u_full, tau, coef):
    G = len(P_slice)
    vm0 = mp('-0.5'); vm1 = mp('0.5')
    zero = mp(0); one = mp(1); two = mp(2)
    A0 = zero; A1 = zero
    for b in range(G):
        f0b = fsig(u_full[b], vm0, tau, coef)
        f1b = fsig(u_full[b], vm1, tau, coef)
        prev = P_slice[0][b]
        for a in range(G-1):
            nxt = P_slice[a+1][b]
            dp = prev - p; dn = nxt - p
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    u_off = (one - frac) * u_full[a] + frac * u_full[a+1]
                    A0 = A0 + fsig(u_off, vm0, tau, coef) * f0b
                    A1 = A1 + fsig(u_off, vm1, tau, coef) * f1b
            prev = nxt
    for a in range(G):
        f0a = fsig(u_full[a], vm0, tau, coef)
        f1a = fsig(u_full[a], vm1, tau, coef)
        prev = P_slice[a][0]
        for b in range(G-1):
            nxt = P_slice[a][b+1]
            dp = prev - p; dn = nxt - p
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    u_off = (one - frac) * u_full[b] + frac * u_full[b+1]
                    A0 = A0 + f0a * fsig(u_off, vm0, tau, coef)
                    A1 = A1 + f1a * fsig(u_off, vm1, tau, coef)
            prev = nxt
    return A0 / two, A1 / two


def phi_strict(P, u_full, tau, gamma, W, coef):
    G = len(P)
    P_new = [[[P[i][j][k] for k in range(G)] for j in range(G)] for i in range(G)]
    one = mp(1); vm0 = mp('-0.5'); vm1 = mp('0.5')
    eps_p = mp('1e-100')
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                p = P[i][j][k]
                mus = []
                # agent 0
                slice0 = [[P[i][a][b] for b in range(G)] for a in range(G)]
                A0, A1 = contour_evidence_2D(slice0, p, u_full, tau, coef)
                f0_i = fsig(u_full[i], vm0, tau, coef)
                f1_i = fsig(u_full[i], vm1, tau, coef)
                den = f0_i * A0 + f1_i * A1
                mu0 = f1_i * A1 / den if den > 0 else mp('0.5')
                mus.append(mu0)
                # agent 1
                slice1 = [[P[a][j][b] for b in range(G)] for a in range(G)]
                A0, A1 = contour_evidence_2D(slice1, p, u_full, tau, coef)
                f0_j = fsig(u_full[j], vm0, tau, coef)
                f1_j = fsig(u_full[j], vm1, tau, coef)
                den = f0_j * A0 + f1_j * A1
                mu1 = f1_j * A1 / den if den > 0 else mp('0.5')
                mus.append(mu1)
                # agent 2
                slice2 = [[P[a][b][k] for b in range(G)] for a in range(G)]
                A0, A1 = contour_evidence_2D(slice2, p, u_full, tau, coef)
                f0_k = fsig(u_full[k], vm0, tau, coef)
                f1_k = fsig(u_full[k], vm1, tau, coef)
                den = f0_k * A0 + f1_k * A1
                mu2 = f1_k * A1 / den if den > 0 else mp('0.5')
                mus.append(mu2)
                mus = [max(eps_p, min(one - eps_p, mu)) for mu in mus]
                P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
    return P_new


# setup
u_full_np = np.linspace(-UMAX, UMAX, G_FULL)
u_full = [mp(str(u)) for u in u_full_np]
tau = mp(TAU); gamma = mp(GAMMA); W = mp(W_S)
coef = (tau / (mp(2) * arb.pi())).sqrt()

U1, U2, U3 = np.meshgrid(u_full_np, u_full_np, u_full_np, indexing='ij')
P_FR_np = 1.0 / (1.0 + np.exp(-TAU * (U1+U2+U3)))
P = [[[arb(float(P_FR_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]

print(f'flint strict-contour benchmark, prec={flint.ctx.prec} bits (≈{DPS} dec)')
t0 = time.time()
P_new = phi_strict(P, u_full, tau, gamma, W, coef)
t_flint = time.time() - t0
print(f'  one Φ step: {t_flint:.2f}s')

# residual
def f_inf(A, B):
    G = len(A); m = mp(0)
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if d > m: m = d
    return m

res = f_inf(P_new, P)
print(f'  ferr from FR ansatz: {float(res):.3e}')
