"""V10 pure-Picard strict-tol ladder at G_inner=11. Each Picard step is one
phi_v10 call (numba JIT'd, parallel). NO scipy in the loop.

Strict tol = 1e-7. Damped Picard with adaptive omega. Up to 600 iters per gamma.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from v10_hfree_gl_numba import (phi_v10, set_boundary, crra_clear_nb,
                                  XI_GL, W_GL, NQ,
                                  TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
TOL = 1e-7
MAX_ITER = 600
GAMMAS = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]

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
    pred=slope*fl_x+intc
    m=float(np.average(fl_lp,weights=fl_w)); vt=float(np.average((fl_lp-m)**2,weights=fl_w))
    vr=float(np.average((fl_lp-pred)**2,weights=fl_w))
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))

LOG = os.path.join(HERE, 'v10_picard_strict_G11.log')
open(LOG,'w').close()
def lg(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG,'a').write(line+'\n')

lg(f'V10 PURE-Picard strict-tol gamma-ladder  G_inner={G_INNER}  TOL={TOL}  MAX_ITER={MAX_ITER}')
lg('JIT warmup...')
t0=time.time()
P_warm = np.zeros((G_FULL,)*3); P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_warm = set_boundary(P_warm)
_ = phi_v10(P_warm, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
lg(f'  warmup {time.time()-t0:.0f}s')

results=[]
t_g=time.time()
for gv in GAMMAS:
    lg(f'\n=== γ={gv} ===')
    mu1=sg(TAU*U1m); mu2=sg(TAU*0.5*(SIm+DEm)); mu3=sg(TAU*0.5*(SIm-DEm))
    P_NL = np.empty_like(U1m)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for k in range(G_INNER):
                P_NL[i,j,k] = crra_clear_nb(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], gv, 120)
    P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL
    P = set_boundary(P)
    s0,i0,o0,d0=metrics(P_NL)
    lg(f'  IC (NL): slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}')
    omega=1.0; res_prev=1e100; conv=False
    history = {'ferr':[], 'slope':[], '1mR2':[], 'omega':[]}
    for it in range(1, MAX_ITER+1):
        ts=time.time()
        P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                         XI_GL, W_GL, NQ, gv, False,
                         TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
        P_damp = (1-omega)*P + omega*P_phi
        P_damp = set_boundary(P_damp)
        res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
        if it > 3:
            if res > res_prev*0.99: omega = max(omega*0.7, 0.02)
            elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
        P = P_damp; res_prev=res
        inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        s,_,o,d = metrics(inner)
        history['ferr'].append(res); history['slope'].append(s); history['1mR2'].append(o); history['omega'].append(omega)
        if it % 20 == 0 or it == 1 or res < TOL:
            lg(f'  it {it:4d}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({time.time()-ts:.1f}s)')
        if res < TOL:
            conv=True; lg(f'  CONVERGED at iter {it}, ferr={res:.3e} < TOL={TOL}'); break
    rec=dict(gamma=gv, converged=conv, iters=it, final_ferr=res,
             slope=s, one_minus_R2=o, d_FR_w=d, history=history)
    results.append(rec)
    np.save(os.path.join(HERE, f'v10_strict_G11_gamma{gv:g}_P.npy'), inner)
    json.dump({'tau':TAU,'G_inner':G_INNER,'TOL':TOL,'results':results,
               'note':'V10 strict pure-Picard gamma ladder G=11'},
              open(os.path.join(HERE, 'v10_strict_G11.json'), 'w'), indent=2)
    lg(f'  {"ACCEPTED" if conv else "REJECTED"} γ={gv}: slope={s:.5f} 1-R²={o:.3e} (cum {(time.time()-t_g)/60:.1f}m)')

lg(f'\nDONE total {(time.time()-t_g)/60:.1f}m  accepted={sum(1 for r in results if r["converged"])}/{len(results)}')
