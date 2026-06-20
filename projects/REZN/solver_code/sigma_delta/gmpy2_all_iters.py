"""Rerun gmpy2 Picard saving snapshots every iter, then plot all 50."""
import os, sys, time, math
import numpy as np
import gmpy2
from gmpy2 import mpfr, exp, log, sqrt

DPS = 200
ctx = gmpy2.get_context()
ctx.precision = int(DPS * 3.33) + 20

G_FULL = 11
INNER_LO, INNER_HI = 2, 9
UMAX_INNER = 3.0
TAU = 2.0; GAMMA = 0.1; W_S = 1.0
KERNEL_H = 0.05
MAX_ITER = 50

LOG = '/tmp/gmpy2_all.log'
open(LOG, 'w').close()
def lg(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def to_mpfr(x): return mpfr(str(x))
def fsig_lp(u, vm, tau, coef):
    d = u - vm
    return coef * exp(-mpfr('0.5') * tau * d * d)
def crra_demand(mu, p, gamma, W):
    lm = log(mu / (mpfr('1') - mu)); lp = log(p / (mpfr('1') - p))
    R = exp((lm - lp) / gamma)
    return W * (R - mpfr('1')) / ((mpfr('1') - p) + R * p)
def crra_clear_sym(mus, gamma, W, steps=300):
    eps = mpfr('1e-150')
    a, b = eps, mpfr('1') - eps
    for _ in range(steps):
        m = (a + b) / mpfr('2')
        ex = mpfr('0')
        for mu in mus: ex += crra_demand(mu, m, gamma, W)
        if ex > 0: a = m
        else: b = m
    return (a + b) / mpfr('2')

def phi_kernel(P, u_full, tau, gamma, W, kernel_h, coef):
    G = len(P)
    vm0 = mpfr('-0.5'); vm1 = mpfr('0.5')
    inv_2h2 = mpfr('1') / (mpfr('2') * kernel_h * kernel_h)
    f0 = [[fsig_lp(u_full[i], vm0, tau, coef) for i in range(G)] for _ in range(3)]
    f1 = [[fsig_lp(u_full[i], vm1, tau, coef) for i in range(G)] for _ in range(3)]
    P_new = [[[P[i][j][k] for k in range(G)] for j in range(G)] for i in range(G)]
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                p = P[i][j][k]
                mus = []
                for agent in range(3):
                    A0_acc = mpfr('0'); A1_acc = mpfr('0')
                    for a in range(G):
                        for b in range(G):
                            if agent == 0:
                                diff = P[i][a][b] - p; fa0 = f0[1][a]; fa1 = f1[1][a]; fb0 = f0[2][b]; fb1 = f1[2][b]
                            elif agent == 1:
                                diff = P[a][j][b] - p; fa0 = f0[0][a]; fa1 = f1[0][a]; fb0 = f0[2][b]; fb1 = f1[2][b]
                            else:
                                diff = P[a][b][k] - p; fa0 = f0[0][a]; fa1 = f1[0][a]; fb0 = f0[1][b]; fb1 = f1[1][b]
                            w = exp(-diff * diff * inv_2h2)
                            A0_acc += w * fa0 * fb0
                            A1_acc += w * fa1 * fb1
                    if agent == 0:   fk0 = f0[0][i]; fk1 = f1[0][i]
                    elif agent == 1: fk0 = f0[1][j]; fk1 = f1[1][j]
                    else:            fk0 = f0[2][k]; fk1 = f1[2][k]
                    den = fk0 * A0_acc + fk1 * A1_acc
                    mu = fk1 * A1_acc / den if den > 0 else mpfr('0.5')
                    eps_p = mpfr('1e-100')
                    mu = max(eps_p, min(mpfr('1') - eps_p, mu))
                    mus.append(mu)
                P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
    return P_new

def f_inf(A, B):
    m = mpfr('0')
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if d > m: m = d
    return m

def to_np(P):
    G = len(P)
    return np.array([[[float(P[i][j][k]) for k in range(G)] for j in range(G)] for i in range(G)])

# setup
u_full_np = np.linspace(-UMAX_INNER, UMAX_INNER, G_FULL)
u_full = [to_mpfr(u) for u in u_full_np]
tau = to_mpfr(TAU); gamma = to_mpfr(GAMMA); W = to_mpfr(W_S); kh = to_mpfr(KERNEL_H)
coef = sqrt(tau / (mpfr('2') * gmpy2.const_pi()))
U1, U2, U3 = np.meshgrid(u_full_np, u_full_np, u_full_np, indexing='ij')
P_FR_np = 1.0 / (1.0 + np.exp(-TAU * (U1+U2+U3)))
P = [[[to_mpfr(P_FR_np[i,j,k]) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]

snapshots = [to_np(P)]   # iter 0 = IC
ferr_t = [0.0]
lg(f'gmpy2 Picard, dps={DPS}, γ={GAMMA}, MAX_ITER={MAX_ITER}; saving every iter')
t0 = time.time()
for it in range(1, MAX_ITER+1):
    P_phi = phi_kernel(P, u_full, tau, gamma, W, kh, coef)
    res_mp = f_inf(P_phi, P)
    P = P_phi
    snapshots.append(to_np(P))
    ferr_t.append(float(res_mp))
    lg(f'  iter {it:2d}: ferr={ferr_t[-1]:.3e}')

lg(f'Total {(time.time()-t0)/60:.1f} min')

# Save snapshots
np.save('/tmp/gmpy2_snapshots.npy', np.array(snapshots))
import json
with open('/tmp/gmpy2_ferr.json', 'w') as f:
    json.dump({'ferr': ferr_t, 'MAX_ITER': MAX_ITER, 'GAMMA': GAMMA, 'KERNEL_H': KERNEL_H,
                'G_FULL': G_FULL}, f, indent=2)
lg('saved snapshots and ferr')
