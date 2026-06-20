"""V4 sanity: Phi_CARA_v4(P_FR) at G=10. Expect: even SMALLER max|delta| than V3,
because Jacobian fixes the agent 2,3 oblique-contour bias.
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint; flint.ctx.prec = 150
from flint import arb

from phi_sigma_delta import set_boundary as set_bd_f64
from phi_sigma_delta_cara import phi_sigmadelta_cara
from flint_sd_v3 import phi_sigdelta_v3
from flint_sd_v4 import (mp, phi_sigdelta_v4, set_boundary as set_bd_v4, to_np)

G = 10
G_FULL = G + 2
INNER_LO, INNER_HI = 1, G + 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0

dxi = 2.0 / (G + 1)
xi_inner = np.linspace(-1+dxi, 1-dxi, G)
xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])

# Build P_FR
u_full = np.zeros(G_FULL); S_full = np.zeros(G_FULL)
for i, x in enumerate(xi_full):
    if abs(x) < 1 - 1e-15:
        u_full[i] = TOT_u * math.atanh(x); S_full[i] = TOT_S * math.atanh(x)
    else:
        u_full[i] = math.copysign(1e10, x); S_full[i] = math.copysign(1e10, x)
P_FR = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    for j in range(G_FULL):
        for k in range(G_FULL):
            S = u_full[i] + S_full[j]
            if S > 50: P_FR[i,j,k] = 1
            elif S < -50: P_FR[i,j,k] = 0
            else: P_FR[i,j,k] = 1/(1+math.exp(-TAU*S))
P_FR = set_bd_f64(P_FR, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)

# float64 ref
print("Computing float64 ref + V3 + V4 at G=10...")
t = time.time()
Pf64 = phi_sigmadelta_cara(P_FR, xi_full, xi_full, xi_full, TOT_u, TOT_S, TOT_d, TAU, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
Pf64 = set_bd_f64(Pf64, TOT_u, TOT_S, TOT_d, xi_full, xi_full, xi_full)
inner_f64 = Pf64[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
inner_FR = P_FR[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f"  float64: max|Phi(P_FR)-P_FR| = {np.max(np.abs(inner_f64-inner_FR)):.4e}  ({time.time()-t:.0f}s)")

# flint v3
xi_mp = [mp(float(x)) for x in xi_full]
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU); gamma = mp(0.1); W_mp = mp(W)
P_mp = [[[mp(float(P_FR[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P_mp = set_bd_v4(P_mp)
t = time.time()
P_v3 = phi_sigdelta_v3(P_mp, xi_mp, xi_mp, xi_mp, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                        tau, gamma, W_mp, INNER_LO, INNER_HI, clearing='cara')
P_v3 = set_bd_v4(P_v3)
P_v3_np = to_np(P_v3)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f"  V3: max|Phi(P_FR)-P_FR| = {np.max(np.abs(P_v3_np-inner_FR)):.4e}  ({time.time()-t:.0f}s)")

t = time.time()
P_v4 = phi_sigdelta_v4(P_mp, xi_mp, xi_mp, xi_mp, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                        tau, gamma, W_mp, INNER_LO, INNER_HI, clearing='cara')
P_v4 = set_bd_v4(P_v4)
P_v4_np = to_np(P_v4)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f"  V4: max|Phi(P_FR)-P_FR| = {np.max(np.abs(P_v4_np-inner_FR)):.4e}  ({time.time()-t:.0f}s)")

# slope and 1-R^2 for each
TAU = 2.0
U1m, SIm, DEm = np.meshgrid(TOT_u*np.arctanh(xi_inner), TOT_S*np.arctanh(xi_inner), TOT_d*np.arctanh(xi_inner), indexing='ij')
U2m=0.5*(SIm+DEm); U3m=0.5*(SIm-DEm)
Tstar = TAU*(U1m+U2m+U3m)
def fa(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(U2m,-0.5)*fa(U3m,-0.5) + fa(U1m,0.5)*fa(U2m,0.5)*fa(U3m,0.5))
Wd /= Wd.sum()
def metrics(P):
    eps = 1e-30; Pc = np.clip(P,eps,1-eps); lp = np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x + intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    d_FR_w = float(np.sqrt(np.sum((P - 1/(1+np.exp(-Tstar)))**2 * Wd)))
    return slope, intc, vr/vt, d_FR_w

print()
for label, P_inner in [('float64', inner_f64), ('V3', P_v3_np), ('V4', P_v4_np)]:
    s, ic, omr2, d = metrics(P_inner)
    print(f"  {label:>7s}: slope={s:.5f}  intc={ic:+.5f}  1-R²={omr2:.3e}  d_FR_w={d:.3e}")
