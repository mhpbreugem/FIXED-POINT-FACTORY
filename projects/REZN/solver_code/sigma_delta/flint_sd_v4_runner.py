"""V4 Picard runner -- proper Jacobian + co-area weights.

Usage: python flint_sd_v4_runner.py <crra|cara> <G_FULL> <dps> <max_iter> [IC]
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

if len(sys.argv) < 5:
    print(__doc__); sys.exit(1)
CLEARING = sys.argv[1]; assert CLEARING in ('crra', 'cara')
G_FULL = int(sys.argv[2])
DPS = int(sys.argv[3])
MAX_ITER = int(sys.argv[4])
IC = sys.argv[5] if len(sys.argv) > 5 else ('FR' if CLEARING == 'cara' else 'NL')
flint.ctx.prec = int(DPS * 3.33) + 20

from flint_sd_v4 import (mp, phi_sigdelta_v4, set_boundary, f_inf, to_np)

INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = INNER_HI - INNER_LO
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; GAMMA_F = 0.1; W_F = 1.0

tag = f"v4_{CLEARING}_G{G_INNER}_dps{DPS}_{IC}"
LOG = os.path.join(HERE, f'flint_{tag}.log')
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
tau = mp(TAU_F); gamma = mp(GAMMA_F); W = mp(W_F)

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

if IC == 'FR':
    P_inner_ic = P_FR_in.copy()
else:
    mu1 = sg(TAU_F*U1m); mu2 = sg(TAU_F*0.5*(SIm+DEm)); mu3 = sg(TAU_F*0.5*(SIm-DEm))
    def cf(mu0,m1,m2,g,steps=80):
        eps=1e-30
        def dd(mu,p):
            lm=math.log(mu/(1-mu)); lp=math.log(p/(1-p))
            R=math.exp((lm-lp)/g); return (R-1)/((1-p)+R*p)
        a,b=eps,1-eps
        for _ in range(steps):
            m=(a+b)/2; e=dd(mu0,m)+dd(m1,m)+dd(m2,m)
            if e>0: a=m
            else: b=m
        return (a+b)/2
    P_inner_ic = np.empty_like(U1m)
    for i in range(G_INNER):
        for j in range(G_INNER):
            for k in range(G_INNER):
                P_inner_ic[i,j,k] = cf(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA_F)

P_full_np = np.zeros((G_FULL,)*3)
P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_inner_ic
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P = set_boundary(P)

lg(f"V4 sigma-delta -- {CLEARING.upper()} G_FULL={G_FULL} (G_inner={G_INNER}), dps={DPS}, IC={IC}, damped Picard.")
lg("V4: V3 dedupe + proper Jacobian/co-area weight 1/((1-xi_off^2)(1-xi_fix^2)|nxt-prev|).")

omega = mp(1); res_prev = mp('1e100')
hist = {'res': [], 'omega': [], 'd_FR_w': [], 'slope': [], 'one_minus_R2': [], 'sec': []}
t0 = time.time()
for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = phi_sigdelta_v4(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                             INNER_LO, INNER_HI, clearing=CLEARING)
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
    hist['res'].append(res); hist['omega'].append(float(omega))
    hist['d_FR_w'].append(d_w); hist['slope'].append(slope)
    hist['one_minus_R2'].append(omr2); hist['sec'].append(sec)
    lg(f"  it {it:3d}  ω={float(omega):.3f}  ferr={res:.3e}  d_FR_w={d_w:.3e}  "
       f"slope={slope:.5f}  1-R²={omr2:.3e}  ({sec:.0f}s)")
    json.dump({**hist, 'G_FULL': G_FULL, 'dps': DPS, 'tau': TAU_F, 'gamma': GAMMA_F,
               'IC': IC, 'clearing': CLEARING, 'version': 'V4'},
              open(os.path.join(HERE, f'flint_{tag}.json'), 'w'), indent=2)
    np.save(os.path.join(HERE, f'flint_{tag}_P.npy'), inner)
lg(f"DONE total {(time.time()-t0)/60:.1f}m")
