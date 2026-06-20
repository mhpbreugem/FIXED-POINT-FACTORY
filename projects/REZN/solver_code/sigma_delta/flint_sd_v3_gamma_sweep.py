"""V3 CRRA gamma-sweep at G=15 dps=50 -- verify CRRA -> CARA (FR) as gamma -> infty.

For each gamma in {0.1, 0.3, 1, 3, 10, 30, 100}: run CRRA Picard with V3 operator
(duplicate-crossing fix; proven match to float64 at G=10 to 8e-3) starting from
no-learning IC. Track final slope, d_FR_w (signal-weighted), 1-R^2(T*) and
compare to CARA reference (slope=1.013, d_FR_w=2.2e-3, 1-R^2=7e-6).

Hellwig prediction: CRRA gap proportional to 1/gamma. At gamma=100 we should
be within ~1% of CARA FR.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20

from flint_sd_v3 import (mp, phi_sigdelta_v3, set_boundary, f_inf, to_np, crra_clear_sym)

G_FULL = 17
INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = INNER_HI - INNER_LO
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; W_F = 1.0
MAX_ITER = 20
GAMMAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

LOG = os.path.join(HERE, 'flint_v3_gamma_sweep.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

xi_full_np = np.linspace(-1.0, 1.0, G_FULL)
xi_inner_np = xi_full_np[INNER_LO:INNER_HI]
xi_u1 = [mp(float(x)) for x in xi_full_np]
xi_S = [mp(float(x)) for x in xi_full_np]
xi_d = [mp(float(x)) for x in xi_full_np]
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU_F); W = mp(W_F)

u_phys = TOT_u * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
S_full = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
Tstar = TAU_F * S_full
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def f_arr(u, vm): return np.sqrt(TAU_F/(2*np.pi))*np.exp(-0.5*TAU_F*(u-vm)**2)
F0 = f_arr(U1m,-0.5)*f_arr(0.5*(SIm+DEm),-0.5)*f_arr(0.5*(SIm-DEm),-0.5)
F1 = f_arr(U1m,+0.5)*f_arr(0.5*(SIm+DEm),+0.5)*f_arr(0.5*(SIm-DEm),+0.5)
Wd = 0.5*F0 + 0.5*F1; Wd /= max(Wd.sum(), 1e-30)

def d_FR_w(P): return float(np.sqrt(np.sum((P - P_FR_in)**2 * Wd)))
def weighted_R2(P):
    eps = 1e-30; Pc = np.clip(P,eps,1-eps); lp = np.log(Pc/(1-Pc))
    fl_x = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x + intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intc)

# no-learning IC builder (depends on gamma)
def build_NL_IC(g):
    mu1 = sg(TAU_F*U1m); mu2 = sg(TAU_F*0.5*(SIm+DEm)); mu3 = sg(TAU_F*0.5*(SIm-DEm))
    def cf(mu0, m1, m2, gf, steps=120):
        eps=1e-30
        def dd(mu,p):
            lm=math.log(mu/(1-mu)); lp=math.log(p/(1-p))
            R=math.exp((lm-lp)/gf); return (R-1)/((1-p)+R*p)
        a,b=eps,1-eps
        for _ in range(steps):
            m=(a+b)/2; e=dd(mu0,m)+dd(m1,m)+dd(m2,m)
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
overall_t0 = time.time()
for gamma_val in GAMMAS:
    lg(f'=== GAMMA = {gamma_val} ===')
    gamma_mp = mp(gamma_val)
    # IC: NL at this gamma
    P_inner_ic = build_NL_IC(gamma_val)
    P_full_np = np.zeros((G_FULL,)*3)
    P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_inner_ic
    P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P = set_boundary(P)
    # initial metrics
    d0 = d_FR_w(P_inner_ic); omr20, slope0, intc0 = weighted_R2(P_inner_ic)
    lg(f'  IC: d_FR_w={d0:.3e}  slope={slope0:.5f}  1-R²={omr20:.3e}')

    omega = mp(1); res_prev = mp('1e100')
    iter_data = {'res': [], 'omega': [], 'd_FR_w': [], 'slope': [], 'one_minus_R2': []}
    for it in range(1, MAX_ITER+1):
        t_step = time.time()
        P_phi = phi_sigdelta_v3(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma_mp, W,
                                 INNER_LO, INNER_HI, clearing='crra')
        one = mp(1)
        P_damp = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                     for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
        P_damp = set_boundary(P_damp)
        res_mp = f_inf(P_damp, P, INNER_LO, INNER_HI); res = float(res_mp)
        if it > 3:
            if float(res_mp) > float(res_prev) * 0.99:   omega = max(omega * mp('0.7'), mp('0.001'))
            elif float(res_mp) < float(res_prev) * 0.6:  omega = min(omega * mp('1.05'), mp(1))
        P = P_damp; res_prev = res_mp
        Pnp = to_np(P); inner = Pnp[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        d_w = d_FR_w(inner); omr2, slope, intc = weighted_R2(inner)
        sec = time.time() - t_step
        iter_data['res'].append(res); iter_data['omega'].append(float(omega))
        iter_data['d_FR_w'].append(d_w); iter_data['slope'].append(slope)
        iter_data['one_minus_R2'].append(omr2)
        lg(f'  it {it:3d}  ω={float(omega):.3f}  ferr={res:.3e}  d_FR_w={d_w:.3e}  '
           f'slope={slope:.5f}  1-R²={omr2:.3e}  ({sec:.0f}s)')

    # final
    Pnp = to_np(P); inner_final = Pnp[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    d_final = d_FR_w(inner_final); omr2_f, slope_f, intc_f = weighted_R2(inner_final)
    rec = dict(gamma=gamma_val, max_iter=MAX_ITER,
               d_FR_w=d_final, slope=slope_f, intercept=intc_f, one_minus_R2=omr2_f,
               final_res=float(res_prev), iter_data=iter_data)
    results.append(rec)
    np.save(os.path.join(HERE, f'flint_v3_gamma{gamma_val:g}_G15_P.npy'), inner_final)
    json.dump({'tau': TAU_F, 'G_FULL': G_FULL, 'dps': DPS, 'results': results,
               'note': 'V3 (duplicate-crossing fix); test CRRA -> CARA(FR) as gamma -> infty'},
              open(os.path.join(HERE, 'flint_v3_gamma_sweep.json'), 'w'), indent=2)
    lg(f'  FINAL γ={gamma_val:g}: d_FR_w={d_final:.3e}  slope={slope_f:.5f}  intc={intc_f:+.5f}  '
       f'1-R²={omr2_f:.3e}  (cumulative {(time.time()-overall_t0)/60:.1f}m)')

lg(f'DONE total {(time.time()-overall_t0)/60:.1f}m')
