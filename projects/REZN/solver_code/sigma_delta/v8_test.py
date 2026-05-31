"""V8 = h-free SMOOTH co-area sigma-delta (cubic spline + partition-of-unity).
Test at G=10. NO kernel, NO bandwidth.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from v8_hfree_smooth import (phi_hfree_sigdelta, set_boundary, crra_clear,
                              TAU, TOT_u, TOT_S, TOT_d)

G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO

# grids
xi_arr = np.linspace(-1.0, 1.0, G_FULL)
dxi = xi_arr[1] - xi_arr[0]
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe)
S_arr = TOT_S * np.arctanh(safe)
d_arr = TOT_d * np.arctanh(safe)
Ju = TOT_u / (1-safe**2); Js = TOT_S / (1-safe**2); Jd = TOT_d / (1-safe**2)

u_in = u_arr[INNER_LO:INNER_HI]
S_in = S_arr[INNER_LO:INNER_HI]
d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5) +
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x+intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    d_FR_w = float(np.sqrt(np.sum((P - P_FR_in)**2 * Wd)))
    return slope, intc, vr/vt if vt>0 else float('nan'), d_FR_w

print('='*70)
print(f'V8 h-FREE SMOOTH sigma-delta test G_inner={G_INNER} float64')
print('  NO kernel, NO bandwidth, only cubic spline + GL quadrature')
print('='*70)

# (1) CARA from FR-ansatz
print('\n(1) CARA from FR-ansatz IC')
P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P = set_boundary(P)
s0, i0, o0, d0 = metrics(P_FR_in)
print(f'  IC: slope={s0:.5f}  1-R²={o0:.3e}  d_FR_w={d0:.3e}')
omega = 1.0; res_prev = 1e100
for it in range(1, 8):
    t = time.time()
    P_phi = phi_hfree_sigdelta(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                                  Ju, Js, Jd, gamma=0.1, clearing='cara')
    P_damp = (1-omega)*P + omega*P_phi
    P_damp = set_boundary(P_damp)
    res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    s, i, o, d = metrics(inner)
    print(f'  it {it}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({time.time()-t:.0f}s)')

# (2) CRRA γ=0.1 from NL-IC
print('\n(2) CRRA γ=0.1 from no-learning IC')
mu1 = sg(TAU*U1m); mu2 = sg(TAU*0.5*(SIm+DEm)); mu3 = sg(TAU*0.5*(SIm-DEm))
P_NL = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL[i,j,k] = crra_clear(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], 0.1)
P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL
P = set_boundary(P)
s0, i0, o0, d0 = metrics(P_NL)
print(f'  IC: slope={s0:.5f}  1-R²={o0:.3e}  d_FR_w={d0:.3e}')
omega = 1.0; res_prev = 1e100
for it in range(1, 8):
    t = time.time()
    P_phi = phi_hfree_sigdelta(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                                  Ju, Js, Jd, gamma=0.1, clearing='crra')
    P_damp = (1-omega)*P + omega*P_phi
    P_damp = set_boundary(P_damp)
    res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    s, i, o, d = metrics(inner)
    print(f'  it {it}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({time.time()-t:.0f}s)')

print('\nDONE')
