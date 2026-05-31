"""V11 (STRICT h=0 HARDWIRED) quick test at G_inner=11. CARA from FR-IC + CRRA γ=0.1 from NL-IC."""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from v11_strict_h0_hardwired import (phi_v10, set_boundary, crra_clear_nb,
                                       XI_GL, W_GL, NQ, EPSB,
                                       TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)
print(f'V11 STRICT h=0 HARDWIRED: EPSB={EPSB}  NQ={NQ}')

G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)
u_in = u_arr[INNER_LO:INNER_HI]; S_in = S_arr[INNER_LO:INNER_HI]; d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm); Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u,vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5)+
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred=slope*fl_x+intc; m=float(np.average(fl_lp,weights=fl_w))
    vt=float(np.average((fl_lp-m)**2,weights=fl_w)); vr=float(np.average((fl_lp-pred)**2,weights=fl_w))
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))

def picard(P_ic, clear_cara, gv, tag, max_iter=6):
    P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_ic
    P = set_boundary(P)
    s0,_,o0,d0 = metrics(P_ic)
    print(f'\n--- {tag}  IC: slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}', flush=True)
    omega=1.0; res_prev=1e100
    for it in range(1, max_iter+1):
        t=time.time()
        P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                         XI_GL, W_GL, NQ, gv, clear_cara,
                         TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
        P_damp = (1-omega)*P + omega*P_phi
        P_damp = set_boundary(P_damp)
        res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
        if it>3:
            if res>res_prev*0.99: omega=max(omega*0.7,0.05)
            elif res<res_prev*0.6: omega=min(omega*1.05,1.0)
        P=P_damp; res_prev=res
        inner=P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        s,_,o,d=metrics(inner)
        print(f'  it {it}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({time.time()-t:.1f}s)',flush=True)
    return inner

# JIT warmup
print('JIT warmup...', flush=True); t=time.time()
P0 = np.zeros((G_FULL,)*3); P0[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P0 = set_boundary(P0)
_ = phi_v10(P0, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, True,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

picard(P_FR_in.copy(), True, 0.1, 'CARA from FR-IC', max_iter=8)
# NL IC for CRRA
mu1=sg(TAU*U1m); mu2=sg(TAU*0.5*(SIm+DEm)); mu3=sg(TAU*0.5*(SIm-DEm))
P_NL = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL[i,j,k] = crra_clear_nb(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], 0.1, 120)
picard(P_NL, False, 0.1, 'CRRA γ=0.1 from NL-IC', max_iter=10)
print('DONE')
