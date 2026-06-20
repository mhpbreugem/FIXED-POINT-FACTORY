"""V10 (numba h-free smooth GL co-area) gamma-ladder at G_inner=17, STRICT TOL.

CRRA only. Accept only fixed points with ferr < TOL (1e-5). Skip if not
converged in MAX_ITER.

Gammas: {0.05, 0.1, 0.3, 1.0, 3.0, 10.0}. NL-IC. Damped Picard with
Anderson-like acceleration scheme. Per-γ wall clock estimate ~ minutes
to ~hour depending on convergence.

Saves per-γ:
  - final P inner array
  - iter trajectory (ferr, slope, 1-R²)
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from v10_hfree_gl_numba import (phi_v10, set_boundary, crra_clear_nb,
                                  XI_GL, W_GL, NQ,
                                  TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

G_FULL = 19           # G_inner = 25
INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = INNER_HI - INNER_LO
TOL = 1e-5
MAX_ITER = 120
GAMMAS = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]

xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe)
S_arr = TOT_S * np.arctanh(safe)
d_arr = TOT_d * np.arctanh(safe)
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
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    d_FR_w = float(np.sqrt(np.sum((P - P_FR_in)**2 * Wd)))
    return slope, intc, vr/vt if vt>0 else float('nan'), d_FR_w

LOG = os.path.join(HERE, 'v10_G17_ladder.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    open(LOG, 'a').write(line + '\n')

# JIT warmup at this G
lg(f'V10 G_inner={G_INNER} STRICT TOL={TOL}  gamma ladder {GAMMAS}')
lg('JIT warmup at G=17...')
t0 = time.time()
P_warm = np.zeros((G_FULL,)*3); P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_warm = set_boundary(P_warm)
_ = phi_v10(P_warm, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
warm_sec = time.time() - t0
lg(f'  warmup: 1 iter = {warm_sec:.0f}s')

results = []
t_global = time.time()
for gv in GAMMAS:
    lg(f'\n=== GAMMA = {gv}  (TOL={TOL}, MAX_ITER={MAX_ITER}) ===')
    # NL IC
    mu1 = sg(TAU*U1m); mu2 = sg(TAU*0.5*(SIm+DEm)); mu3 = sg(TAU*0.5*(SIm-DEm))
    P_NL = np.empty_like(U1m)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for k in range(G_INNER):
                P_NL[i,j,k] = crra_clear_nb(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], gv, 120)
    P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL
    P = set_boundary(P)
    s0, i0, o0, d0 = metrics(P_NL)
    lg(f'  IC (NL): slope={s0:.5f}  1-R²={o0:.3e}  d_FR_w={d0:.3e}')
    omega = 1.0; res_prev = 1e100
    iter_data = {'ferr':[], 'omega':[], 'd_FR_w':[], 'slope':[], '1mR2':[], 'sec':[]}
    converged = False
    for it in range(1, MAX_ITER+1):
        t = time.time()
        P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                         XI_GL, W_GL, NQ, gv, False,
                         TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
        P_damp = (1-omega)*P + omega*P_phi
        P_damp = set_boundary(P_damp)
        res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
        if it > 3:
            if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
            elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
        P = P_damp; res_prev = res
        inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        s, i, o, d = metrics(inner)
        sec = time.time() - t
        iter_data['ferr'].append(res); iter_data['omega'].append(omega)
        iter_data['d_FR_w'].append(d); iter_data['slope'].append(s); iter_data['1mR2'].append(o); iter_data['sec'].append(sec)
        if it % 5 == 0 or it == 1 or res < TOL:
            lg(f'  it {it:3d}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({sec:.0f}s)')
        if res < TOL:
            converged = True
            lg(f'  CONVERGED at iter {it}, ferr={res:.3e} < TOL={TOL}')
            break
    rec = dict(gamma=gv, converged=converged, iters=len(iter_data['ferr']),
               final_ferr=iter_data['ferr'][-1] if iter_data['ferr'] else None,
               slope_final=s, one_minus_R2_final=o, d_FR_w_final=d, iter_data=iter_data)
    results.append(rec)
    np.save(os.path.join(HERE, f'v10_G17_gamma{gv:g}_P.npy'), inner)
    json.dump({'tau':TAU, 'G_inner':G_INNER, 'TOL':TOL, 'results':results,
               'note':'V10 h-free smooth GL strict-tol gamma ladder'},
              open(os.path.join(HERE, 'v10_G17_ladder.json'), 'w'), indent=2)
    if converged:
        lg(f'  ACCEPTED γ={gv}: slope={s:.5f} 1-R²={o:.3e} (cum {(time.time()-t_global)/60:.1f}m)')
    else:
        lg(f'  REJECTED γ={gv}: ferr={res:.3e} > TOL after {it} iters; slope={s:.5f} 1-R²={o:.3e} (cum {(time.time()-t_global)/60:.1f}m)')

lg(f'\nDONE total {(time.time()-t_global)/60:.1f}m')
n_acc = sum(1 for r in results if r['converged'])
lg(f'Accepted: {n_acc}/{len(results)}')
