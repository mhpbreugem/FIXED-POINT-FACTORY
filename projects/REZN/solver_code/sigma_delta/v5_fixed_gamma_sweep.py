"""V5 KERNEL CO-AREA at G=10 with FIXED h_factor=0.02 -- much smaller kernel
to escape the bandwidth bias.

Gamma in {0.05, 0.1, 1, 10, 100}. NL IC, 12 iters per gamma. Should show:
  - high gamma -> CARA limit (slope -> 1, 1-R² -> 0)
  - low gamma -> Jensen gap visible (slope < 1)
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint; flint.ctx.prec = 180
from flint import arb
from flint_sd_v5_kernel import (mp, phi_sigdelta_v5, set_boundary, f_inf, to_np)

G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0; TAU_F = 2.0; W_F = 1.0
C_H = 0.02   # MUCH smaller h
MAX_ITER = 12
GAMMAS = [0.05, 0.1, 1.0, 10.0, 100.0]

dxi = 2.0 / (G_FULL - 1)
h_kernel = mp(C_H * (dxi ** 0.5))

xi_full_np = np.linspace(-1.0, 1.0, G_FULL)
xi_inner_np = xi_full_np[INNER_LO:INNER_HI]
xi_u1 = [mp(float(x)) for x in xi_full_np]; xi_S = list(xi_u1); xi_d = list(xi_u1)
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU_F); W = mp(W_F)

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

def build_NL_IC(g):
    mu1=sg(TAU_F*U1m); mu2=sg(TAU_F*0.5*(SIm+DEm)); mu3=sg(TAU_F*0.5*(SIm-DEm))
    def cf(m0,m1,m2,gf,steps=300):
        eps=1e-30
        def dd(mu,p): lm=math.log(mu/(1-mu)); lp=math.log(p/(1-p)); R=math.exp((lm-lp)/gf); return (R-1)/((1-p)+R*p)
        a,b=eps,1-eps
        for _ in range(steps):
            m=(a+b)/2; e=dd(m0,m)+dd(m1,m)+dd(m2,m)
            if e>0: a=m
            else: b=m
        return (a+b)/2
    P_inner = np.empty_like(U1m)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for k in range(G_INNER):
                P_inner[i,j,k] = cf(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], g)
    return P_inner

results = []
print(f"V5 FIXED (C_H={C_H}, h={float(h_kernel):.4f}) G={G_INNER} dps=50 gamma-sweep")
t0 = time.time()
for gv in GAMMAS:
    print(f'=== γ={gv} ===')
    gamma_mp = mp(gv)
    P_ic = build_NL_IC(gv)
    P_full_np = np.zeros((G_FULL,)*3); P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_ic
    P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P = set_boundary(P)
    omr20, slope0, intc0 = w_R2(P_ic); d0 = d_FR_w(P_ic)
    print(f'  IC: slope={slope0:.4f} 1-R²={omr20:.3e} d_FR_w={d0:.3e}')
    omega = mp(1); res_prev = mp('1e100')
    iter_data = {'res':[], 'omega':[], 'd_FR_w':[], 'slope':[], '1mR2':[]}
    for it in range(1, MAX_ITER+1):
        t_step = time.time()
        P_phi = phi_sigdelta_v5(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma_mp, W,
                                 INNER_LO, INNER_HI, h_kernel, clearing='crra')
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
        omr2, slope, intc = w_R2(inner); d_w = d_FR_w(inner)
        iter_data['res'].append(res); iter_data['omega'].append(float(omega))
        iter_data['d_FR_w'].append(d_w); iter_data['slope'].append(slope); iter_data['1mR2'].append(omr2)
    rec = dict(gamma=gv, slope_final=slope, one_minus_R2_final=omr2, d_FR_w_final=d_w, iter_data=iter_data)
    results.append(rec)
    np.save(os.path.join(HERE, f'flint_v5fixed_gamma{gv:g}_P.npy'), inner)
    print(f'  FINAL γ={gv}: slope={slope:.5f}  1-R²={omr2:.3e}  d_FR_w={d_w:.3e}  ({(time.time()-t0)/60:.1f}m cum)')
    json.dump({'tau':TAU_F, 'G_FULL':G_FULL, 'C_H':C_H, 'h':float(h_kernel), 'results':results,
               'note':'V5 KERNEL with FIXED small h_factor=0.02'},
              open(os.path.join(HERE, 'flint_v5fixed_gamma_sweep.json'), 'w'), indent=2)
print(f'DONE total {(time.time()-t0)/60:.1f}m')
