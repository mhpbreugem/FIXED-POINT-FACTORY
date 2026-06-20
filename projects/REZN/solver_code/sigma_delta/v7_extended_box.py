"""V7 sigma-delta strict h=0 with EXTENDED box (TOT_Σ=10 instead of 3).
Test: does extending the box allow CRRA γ=0.1 to find the real PR
equilibrium (slope < 1) instead of being pinned to FR by the box-edge BC?

If extending the box recovers the u-grid kernel slope ≈ 0.18, the
discrepancy is fully explained: σ-δ at TOT_Σ=3 was BC-constrained.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

# Override TOT values BEFORE importing V7
import v7_strict_h0
v7_strict_h0.TOT_u = 4.0   # was 2.0
v7_strict_h0.TOT_S = 10.0  # was 3.0
v7_strict_h0.TOT_d = 10.0  # was 3.0
from v7_strict_h0 import (phi_strict_sigdelta, set_boundary, crra_clear,
                            TAU, TOT_u, TOT_S, TOT_d)

G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO

# rebuild grids with the new TOT
xi_arr = np.linspace(-1.0, 1.0, G_FULL)
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe)
S_arr = TOT_S * np.arctanh(safe)
d_arr = TOT_d * np.arctanh(safe)
interior = np.abs(xi_arr) < 1 - 1e-12
Ju = np.where(interior, TOT_u/(1-xi_arr**2), 0.0)
JS = np.where(interior, TOT_S/(1-xi_arr**2), 0.0)
Jd = np.where(interior, TOT_d/(1-xi_arr**2), 0.0)

print(f'EXTENDED BOX: TOT_u={TOT_u}, TOT_Σ={TOT_S}, TOT_δ={TOT_d}')
print(f'  inner u  range: {u_arr[INNER_LO]:.2f} .. {u_arr[INNER_HI-1]:.2f}')
print(f'  inner Σ  range: {S_arr[INNER_LO]:.2f} .. {S_arr[INNER_HI-1]:.2f}')
print(f'  inner δ  range: {d_arr[INNER_LO]:.2f} .. {d_arr[INNER_HI-1]:.2f}')

# physical grids for metrics
u_in = u_arr[INNER_LO:INNER_HI]
S_in = S_arr[INNER_LO:INNER_HI]
d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
S_full_in = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
Tstar = TAU * S_full_in
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

# Build NL IC
mu1=sg(TAU*U1m); mu2=sg(TAU*0.5*(SIm+DEm)); mu3=sg(TAU*0.5*(SIm-DEm))
P_NL = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL[i,j,k] = crra_clear(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], 0.1)

P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL
P = set_boundary(P)
slope0, intc0, omr20, dfr0 = metrics(P_NL)
print(f'IC: slope={slope0:.5f}  1-R²={omr20:.3e}  d_FR_w={dfr0:.3e}')

omega = 1.0; res_prev = 1e100
for it in range(1, 12):
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
    slope, intc, omr2, dfr = metrics(inner)
    print(f'  it {it:2d}  ω={omega:.3f}  ferr={res:.3e}  slope={slope:.5f}  1-R²={omr2:.3e}  d_FR_w={dfr:.3e}  ({time.time()-t:.0f}s)')
print('\nDONE -- TOT_Σ=10 box vs default TOT_Σ=3')
