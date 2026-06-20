"""V5 G=10 CARA FR-IC test with VERY SMALL h. If the kernel-bandwidth-bias
hypothesis is correct, shrinking h should restore slope toward 1 (kernel
approaches delta).

Tests h_factor in {0.45 (default), 0.10, 0.05, 0.02}. Each is 5 iters."""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint; flint.ctx.prec = 180
from flint import arb
from flint_sd_v5_kernel import (mp, phi_sigdelta_v5, set_boundary, f_inf, to_np)

G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0; TAU_F = 2.0; W_F = 1.0

dxi = 2.0 / (G_FULL - 1)
xi_full_np = np.linspace(-1.0, 1.0, G_FULL)
xi_inner_np = xi_full_np[INNER_LO:INNER_HI]
xi_u1 = [mp(float(x)) for x in xi_full_np]; xi_S = list(xi_u1); xi_d = list(xi_u1)
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU_F); gamma = mp(0); W = mp(W_F)

u_phys = TOT_u * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
S_full = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm); Tstar = TAU_F*S_full
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u,vm): return np.sqrt(TAU_F/(2*np.pi))*np.exp(-0.5*TAU_F*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5)+
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def w_R2(P):
    eps = 1e-30; Pc = np.clip(P,eps,1-eps); lp = np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x+intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intc)
def d_FR_w(P): return float(np.sqrt(np.sum((P - P_FR_in)**2 * Wd)))

for C_H in [0.45, 0.10, 0.05, 0.02]:
    h_kernel = mp(C_H * (dxi ** 0.5))
    P_full_np = np.zeros((G_FULL,)*3); P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
    P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P = set_boundary(P)
    print(f'C_H={C_H:.2f}  h={float(h_kernel):.4f}')
    print(f'  IC: slope={w_R2(P_FR_in)[1]:.4f}  d_FR_w={d_FR_w(P_FR_in):.3e}')
    omega = mp(1); res_prev = mp('1e100')
    for it in range(1, 8):
        P_phi = phi_sigdelta_v5(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                                 INNER_LO, INNER_HI, h_kernel, clearing='cara')
        one = mp(1)
        P_damp = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                     for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
        P_damp = set_boundary(P_damp)
        res_mp = f_inf(P_damp, P, INNER_LO, INNER_HI); res = float(res_mp)
        if it > 3:
            if float(res_mp) > float(res_prev)*0.99: omega = max(omega*mp('0.7'), mp('0.05'))
            elif float(res_mp) < float(res_prev)*0.6: omega = min(omega*mp('1.05'), mp(1))
        P = P_damp; res_prev = res_mp
        Pnp = to_np(P); inner = Pnp[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        omr2, slope, intc = w_R2(inner)
        print(f'  it {it} ferr={res:.3e} slope={slope:.5f} 1-R²={omr2:.3e} d_FR_w={d_FR_w(inner):.3e}')
    print()
