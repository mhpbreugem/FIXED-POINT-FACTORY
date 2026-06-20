"""V10 Newton-Krylov nail at G_inner=11. Strict tol via NK (not Picard).

NK handles the saddle/slow-Picard convergence issue. If a PR FP exists in
V10's basin, NK will nail it; if not, NK won't converge.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from v10_hfree_gl_numba import (phi_v10, set_boundary, crra_clear_nb,
                                  XI_GL, W_GL, NQ,
                                  TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

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
    pred=slope*fl_x+intc
    m=float(np.average(fl_lp,weights=fl_w)); vt=float(np.average((fl_lp-m)**2,weights=fl_w))
    vr=float(np.average((fl_lp-pred)**2,weights=fl_w))
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))

halo = np.zeros((G_FULL,)*3); halo = set_boundary(halo)

def F(x, gamma_val, clearing_cara):
    P = halo.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x.reshape((G_INNER,)*3)
    P = set_boundary(P)
    P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                     XI_GL, W_GL, NQ, gamma_val, clearing_cara,
                     TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return (P_phi - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

# JIT warmup
print('JIT warmup...', flush=True)
t = time.time()
_ = F(P_FR_in.ravel(), 0.1, True)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

results = []
GAMMAS = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
TOL = 1e-7
MAXIT = 80
for gv in GAMMAS:
    print(f'\n=== γ={gv} (NK, tol={TOL}, maxiter={MAXIT}) ===', flush=True)
    # NL IC
    mu1=sg(TAU*U1m); mu2=sg(TAU*0.5*(SIm+DEm)); mu3=sg(TAU*0.5*(SIm-DEm))
    P_NL = np.empty_like(U1m)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for k in range(G_INNER):
                P_NL[i,j,k] = crra_clear_nb(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], gv, 120)
    s0,i0,o0,d0 = metrics(P_NL)
    print(f'  IC: slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}', flush=True)
    x0 = P_NL.ravel().copy()
    t0 = time.time()
    try:
        sol = newton_krylov(lambda x: F(x, gv, False), x0, f_tol=TOL, maxiter=MAXIT,
                              method='lgmres', verbose=True)
        converged = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); converged = False
    nk_sec = time.time() - t0
    ferr = float(np.max(np.abs(F(sol, gv, False))))
    inner = sol.reshape((G_INNER,)*3)
    s, i, o, d = metrics(inner)
    print(f'  NK done {nk_sec:.0f}s  conv={converged}  ferr={ferr:.3e}', flush=True)
    print(f'  FINAL: slope={s:.5f} 1-R²={o:.3e} d_FR_w={d:.3e}', flush=True)
    rec = dict(gamma=gv, converged=converged, ferr_final=ferr, sec=nk_sec,
               slope=s, intc=i, one_minus_R2=o, d_FR_w=d)
    results.append(rec)
    np.save(os.path.join(HERE, f'v10_NK_G11_gamma{gv:g}_P.npy'), inner)
    json.dump({'tau':TAU,'G_inner':G_INNER,'TOL':TOL,'results':results,
               'note':'V10 Newton-Krylov gamma ladder G_inner=11'},
              open(os.path.join(HERE, 'v10_NK_G11.json'), 'w'), indent=2)

print('\nDONE.', flush=True)
