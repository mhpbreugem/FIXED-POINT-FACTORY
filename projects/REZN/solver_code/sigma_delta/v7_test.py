"""V7 strict h=0 marching-squares co-area on sigma-delta -- test at G=10.

CARA from FR-IC: expect slope -> 1.000 (Hellwig exactly).
CRRA gamma=0.1 from NL-IC: expect proper Jensen gap (slope < 1).
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from v7_strict_h0 import (phi_strict_sigdelta, set_boundary, make_grids, crra_clear,
                            TAU, TOT_u, TOT_S, TOT_d)

G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
W = 1.0

xi_arr, u_arr, S_arr, d_arr, Ju, JS, Jd = make_grids(G_FULL)

# Inner physical coords for metrics
u_inner = u_arr[INNER_LO:INNER_HI]
S_inner = S_arr[INNER_LO:INNER_HI]
d_inner = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_inner, S_inner, d_inner, indexing='ij')
S_full_inner = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
Tstar = TAU * S_full_inner
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_inner = sg(Tstar)
def fa(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5) +
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def d_FR_w(P): return float(np.sqrt(np.sum((P - P_FR_inner)**2 * Wd)))
def w_R2(P):
    eps = 1e-30; Pc = np.clip(P,eps,1-eps); lp = np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x+intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intc)

# --- CARA test from FR-IC ---
print('='*70)
print('V7 STRICT h=0 marching-squares sigma-delta TEST')
print(f'  G_FULL={G_FULL}, G_inner={G_INNER}, dps=float64')
print('='*70)
print('\n(1) CARA from FR-ansatz IC')
P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_inner
P = set_boundary(P)
omr20, slope0, intc0 = w_R2(P_FR_inner)
print(f'  IC: slope={slope0:.5f}  1-R²={omr20:.3e}  d_FR_w={d_FR_w(P_FR_inner):.3e}')
omega = 1.0; res_prev = 1e100
for it in range(1, 11):
    t = time.time()
    P_phi = phi_strict_sigdelta(P, INNER_LO, INNER_HI, xi_arr, u_arr, S_arr, d_arr, JS, Jd,
                                  gamma=0.1, clearing='cara')
    P_damp = (1-omega)*P + omega*P_phi
    P_damp = set_boundary(P_damp)
    res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    omr2, slope, intc = w_R2(inner); d_w = d_FR_w(inner)
    sec = time.time() - t
    print(f'  it {it:2d}  ω={omega:.3f}  ferr={res:.3e}  slope={slope:.5f}  1-R²={omr2:.3e}  d_FR_w={d_w:.3e}  ({sec:.0f}s)')

print('\n(2) CRRA γ=0.1 from no-learning IC')
mu1 = sg(TAU*U1m); mu2 = sg(TAU*0.5*(SIm+DEm)); mu3 = sg(TAU*0.5*(SIm-DEm))
P_NL = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL[i,j,k] = crra_clear(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], 0.1)
P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL
P = set_boundary(P)
omr20, slope0, intc0 = w_R2(P_NL); d0 = d_FR_w(P_NL)
print(f'  IC: slope={slope0:.5f}  1-R²={omr20:.3e}  d_FR_w={d0:.3e}')
omega = 1.0; res_prev = 1e100
for it in range(1, 11):
    t = time.time()
    P_phi = phi_strict_sigdelta(P, INNER_LO, INNER_HI, xi_arr, u_arr, S_arr, d_arr, JS, Jd,
                                  gamma=0.1, clearing='crra')
    P_damp = (1-omega)*P + omega*P_phi
    P_damp = set_boundary(P_damp)
    res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    omr2, slope, intc = w_R2(inner); d_w = d_FR_w(inner)
    sec = time.time() - t
    print(f'  it {it:2d}  ω={omega:.3f}  ferr={res:.3e}  slope={slope:.5f}  1-R²={omr2:.3e}  d_FR_w={d_w:.3e}  ({sec:.0f}s)')

print('\nDONE.')
