"""V10 + Anderson acceleration. Much faster convergence than damped Picard
on slow-converging FPs. Each iter = one phi_v10 call (numba) + tiny LS solve.

Anderson(m): keep history of last m residuals, find LS combination minimizing
||F||, take Anderson step. Convergence often 5-20x faster than Picard.
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
MAX_ITER = 80
M_AND = 8     # Anderson memory depth
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

halo = np.zeros((G_FULL,)*3); halo = set_boundary(halo)

def phi_x(x, gamma_val):
    """Phi applied to inner-block flat vector x. Returns flat phi(x)."""
    P = halo.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x.reshape((G_INNER,)*3)
    P = set_boundary(P)
    P_new = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                     XI_GL, W_GL, NQ, gamma_val, False,
                     TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

def anderson_solve(x0, gamma_val, m_max, max_iter, tol, log):
    """Anderson(m_max) accelerated FP solve. F(x) = phi(x) - x."""
    N = x0.size
    x = x0.copy()
    G_hist = []  # G(x) := phi(x), stored as columns
    F_hist = []  # F(x) := G(x) - x
    history = {'ferr':[]}
    for it in range(1, max_iter+1):
        ts = time.time()
        g = phi_x(x, gamma_val)
        f = g - x
        ferr = float(np.max(np.abs(f)))
        history['ferr'].append(ferr)
        if ferr < tol:
            log(f'  it {it:3d}  ferr={ferr:.3e}  CONVERGED  ({time.time()-ts:.1f}s)')
            return x, True, history
        # Anderson update
        G_hist.append(g.copy()); F_hist.append(f.copy())
        if len(F_hist) > m_max + 1:
            G_hist.pop(0); F_hist.pop(0)
        mk = len(F_hist) - 1
        if mk == 0:
            # First iter: plain fixed-point update x <- g
            x_new = g
        else:
            # Compute ΔF matrix (N x mk) and ΔG matrix
            dF = np.column_stack([F_hist[k+1] - F_hist[k] for k in range(mk)])
            dG = np.column_stack([G_hist[k+1] - G_hist[k] for k in range(mk)])
            # Solve dF * γ = f_current  (least squares)
            try:
                gamma_coef, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x_new = g - dG @ gamma_coef
            except np.linalg.LinAlgError:
                x_new = g
        # clip to [eps, 1-eps]
        x_new = np.clip(x_new, 1e-12, 1-1e-12)
        x = x_new
        if it % 10 == 0 or it == 1:
            log(f'  it {it:3d}  ferr={ferr:.3e}  ({time.time()-ts:.1f}s)')
    return x, False, history

LOG = os.path.join(HERE, 'v10_anderson_G11.log')
open(LOG,'w').close()
def lg(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(LOG,'a').write(line+'\n')

lg(f'V10 Anderson(m={M_AND}) strict-tol gamma ladder G_inner={G_INNER}  TOL={TOL}  MAX_ITER={MAX_ITER}')
lg('JIT warmup...')
t0=time.time()
_ = phi_x(P_FR_in.ravel(), 0.1)
lg(f'  warmup {time.time()-t0:.1f}s')

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
    s0,_,o0,d0 = metrics(P_NL)
    lg(f'  IC (NL): slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}')
    x_final, conv, hist = anderson_solve(P_NL.ravel(), gv, M_AND, MAX_ITER, TOL, lg)
    inner = x_final.reshape((G_INNER,)*3)
    s,_,o,d = metrics(inner)
    rec=dict(gamma=gv, converged=conv, iters=len(hist['ferr']),
             final_ferr=hist['ferr'][-1], slope=s, one_minus_R2=o, d_FR_w=d,
             ferr_history=hist['ferr'])
    results.append(rec)
    np.save(os.path.join(HERE, f'v10_AND_G11_gamma{gv:g}_P.npy'), inner)
    json.dump({'tau':TAU,'G_inner':G_INNER,'TOL':TOL,'m':M_AND,'results':results},
              open(os.path.join(HERE, 'v10_anderson_G11.json'), 'w'), indent=2)
    lg(f'  {"ACCEPTED" if conv else "REJECTED"} γ={gv}: slope={s:.5f} 1-R²={o:.3e} d_FR_w={d:.3e}'
       f' (cum {(time.time()-t_g)/60:.1f}m)')
lg(f'\nDONE total {(time.time()-t_g)/60:.1f}m  accepted={sum(1 for r in results if r["converged"])}/{len(results)}')
