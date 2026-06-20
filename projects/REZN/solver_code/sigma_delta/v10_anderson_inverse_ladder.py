"""V10 + Anderson INVERSE gamma ladder with CARA WARM-START.

Start from FR-ansatz (CARA limit), nail at γ=100 (effectively CARA), then
DESCEND γ: 100 → 30 → 10 → 3 → 1 → 0.3 → 0.1 → 0.05. Each γ's FP is used
as warm-start for the next-lower γ. This continuation should converge
much faster than NL-IC because each FP is close to the previous.

tol=1e-7, Anderson(m=8), MAX_ITER=100/γ.
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
MAX_ITER = 100
M_AND = 8
GAMMAS = [100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.05, 0.02]  # DESCENDING

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

halo = np.zeros((G_FULL,)*3); halo = set_boundary(halo)
def phi_x(x, gv):
    P = halo.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x.reshape((G_INNER,)*3)
    P = set_boundary(P)
    Pn = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                   XI_GL, W_GL, NQ, gv, False,
                   TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return Pn[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

def anderson(x0, gv, m_max, max_it, tol, log):
    x = x0.copy()
    G_h=[]; F_h=[]; hist={'ferr':[]}
    for it in range(1, max_it+1):
        ts = time.time()
        g = phi_x(x, gv); f = g - x
        ferr = float(np.max(np.abs(f)))
        hist['ferr'].append(ferr)
        if ferr < tol:
            log(f'  it {it:3d}  ferr={ferr:.3e}  CONVERGED ({time.time()-ts:.1f}s)')
            return x, True, hist
        G_h.append(g.copy()); F_h.append(f.copy())
        if len(F_h) > m_max+1: G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0:
            x_new = g
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x_new = g - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g
        x_new = np.clip(x_new, 1e-12, 1-1e-12)
        x = x_new
        if it % 10 == 0 or it == 1:
            log(f'  it {it:3d}  ferr={ferr:.3e}  ({time.time()-ts:.1f}s)')
    return x, False, hist

LOG = os.path.join(HERE, 'v10_anderson_inverse_G11.log')
open(LOG, 'w').close()
def lg(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG,'a').write(line+'\n')

lg(f'V10 Anderson(m={M_AND}) INVERSE gamma ladder (CARA warm-start) G={G_INNER} tol={TOL}')
lg(f'Gammas (descending): {GAMMAS}')
lg('JIT warmup...')
t0=time.time(); _ = phi_x(P_FR_in.ravel(), 100.0); lg(f'  warmup {time.time()-t0:.1f}s')

# Initial: CARA warm-start = FR ansatz
x = P_FR_in.ravel().copy()

results=[]; t_g=time.time()
for gv in GAMMAS:
    lg(f'\n=== γ={gv} (warm-start from prev FP) ===')
    s0,_,o0,d0 = metrics(x.reshape((G_INNER,)*3))
    lg(f'  IC: slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}')
    x_new, conv, hist = anderson(x, gv, M_AND, MAX_ITER, TOL, lg)
    inner = x_new.reshape((G_INNER,)*3)
    s,_,o,d = metrics(inner)
    rec = dict(gamma=gv, converged=conv, iters=len(hist['ferr']),
               final_ferr=hist['ferr'][-1], slope=s, one_minus_R2=o, d_FR_w=d,
               ferr_history=hist['ferr'])
    results.append(rec)
    np.save(os.path.join(HERE, f'v10_inverse_G11_gamma{gv:g}_P.npy'), inner)
    json.dump({'tau':TAU,'G_inner':G_INNER,'TOL':TOL,'m':M_AND,'results':results,
               'note':'V10 Anderson INVERSE gamma ladder, CARA warm-start'},
              open(os.path.join(HERE, 'v10_anderson_inverse_G11.json'),'w'), indent=2)
    lg(f'  {"ACCEPTED" if conv else "REJECTED"} γ={gv}: slope={s:.5f} 1-R²={o:.3e} d_FR_w={d:.3e}'
       f' (cum {(time.time()-t_g)/60:.1f}m)')
    x = x_new  # warm-start next γ
lg(f'\nDONE total {(time.time()-t_g)/60:.1f}m  accepted={sum(1 for r in results if r["converged"])}/{len(results)}')
