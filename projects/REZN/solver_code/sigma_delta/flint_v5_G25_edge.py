"""V5 KERNEL CO-AREA at G_inner=25, "go to the edge": inner cells extend
to within one dxi of xi=+-1 (smaller halo buffer). Boundary cells xi=+-1
carry the exact FR limit P=0/1.

Runs CARA from FR-ansatz and CRRA at gamma=0.1 from no-learning, each 12
iters. dps=50. Single script -- launches both sequentially.

At G_inner=25 the per-step cost is roughly (25/15)^5 ~ 13x slower than G=15,
so ~13 min per Picard iter. Total: 12 iters * 13 min * 2 = ~5 hours.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20

from flint_sd_v5_kernel import (mp, phi_sigdelta_v5, set_boundary, f_inf, to_np)

G_FULL = 27
INNER_LO, INNER_HI = 1, G_FULL - 1   # 25 inner cells; xi[1]=-0.923 .. xi[25]=+0.923 -- closer to edge than G=15
G_INNER = INNER_HI - INNER_LO        # = 25
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; W_F = 1.0
C_H = 0.45
MAX_ITER = 12

dxi = 2.0 / (G_FULL - 1)
h_kernel = mp(C_H * (dxi ** 0.5))

LOG = os.path.join(HERE, 'flint_v5_G25_edge.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

xi_full_np = np.linspace(-1.0, 1.0, G_FULL)
xi_inner_np = xi_full_np[INNER_LO:INNER_HI]
xi_u1 = [mp(float(x)) for x in xi_full_np]
xi_S = list(xi_u1); xi_d = list(xi_u1)
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
def w_R2(P):
    eps = 1e-30; Pc = np.clip(P,eps,1-eps); lp = np.log(Pc/(1-Pc))
    fl_x = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x + intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intc)

def picard_run(P_inner_init, clearing, gamma_val, tag):
    gamma_mp = mp(gamma_val)
    P_full_np = np.zeros((G_FULL,)*3); P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_inner_init
    P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P = set_boundary(P)
    d0 = d_FR_w(P_inner_init); omr20, slope0, intc0 = w_R2(P_inner_init)
    lg(f'  IC ({tag}): d_FR_w={d0:.3e}  slope={slope0:.4f}  1-R²={omr20:.3e}')
    omega = mp(1); res_prev = mp('1e100')
    iter_data = {'res':[], 'omega':[], 'd_FR_w':[], 'slope':[], '1mR2':[], 'sec':[]}
    t0 = time.time()
    for it in range(1, MAX_ITER+1):
        t_step = time.time()
        P_phi = phi_sigdelta_v5(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma_mp, W,
                                 INNER_LO, INNER_HI, h_kernel, clearing=clearing)
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
        d_w = d_FR_w(inner); omr2, slope, intc = w_R2(inner)
        sec = time.time() - t_step
        iter_data['res'].append(res); iter_data['omega'].append(float(omega))
        iter_data['d_FR_w'].append(d_w); iter_data['slope'].append(slope); iter_data['1mR2'].append(omr2); iter_data['sec'].append(sec)
        lg(f'  {tag} it {it:3d}  ω={float(omega):.3f}  ferr={res:.3e}  d_FR_w={d_w:.3e}  slope={slope:.5f}  1-R²={omr2:.3e}  ({sec:.0f}s)')
        np.save(os.path.join(HERE, f'flint_v5_G25_edge_{tag}_P.npy'), inner)
        json.dump({'tau':TAU_F, 'gamma':gamma_val, 'G_FULL':G_FULL, 'G_inner':G_INNER, 'dps':DPS,
                   'h':float(h_kernel), 'clearing':clearing, 'tag':tag, 'iter_data':iter_data},
                  open(os.path.join(HERE, f'flint_v5_G25_edge_{tag}.json'), 'w'), indent=2)
    lg(f'  {tag} DONE total {(time.time()-t0)/60:.1f}m')
    return iter_data

# Run CARA first (faster -- no bisection), then CRRA at γ=0.1
lg(f"V5 G_inner={G_INNER} (G_FULL={G_FULL}) edge sweep, dps={DPS}, h={float(h_kernel):.4f}.")
lg("(1) CARA from FR-ansatz IC.")
picard_run(P_FR_in.copy(), 'cara', 0.0, 'cara_FR_IC')

lg("(2) CRRA gamma=0.1 from no-learning IC.")
mu1_NL = sg(TAU_F*U1m); mu2_NL = sg(TAU_F*0.5*(SIm+DEm)); mu3_NL = sg(TAU_F*0.5*(SIm-DEm))
def crra_clear_f64(m0, m1, m2, gf, steps=200):
    eps=1e-30
    def dd(mu, p):
        lm=math.log(mu/(1-mu)); lp=math.log(p/(1-p))
        R=math.exp((lm-lp)/gf); return (R-1)/((1-p)+R*p)
    a,b=eps,1-eps
    for _ in range(steps):
        m=(a+b)/2; e=dd(m0,m)+dd(m1,m)+dd(m2,m)
        if e>0: a=m
        else: b=m
    return (a+b)/2
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_f64(mu1_NL[i,j,k], mu2_NL[i,j,k], mu3_NL[i,j,k], 0.1)
picard_run(P_NL_in, 'crra', 0.1, 'crra_g0p1_NL_IC')

lg('ALL DONE')
