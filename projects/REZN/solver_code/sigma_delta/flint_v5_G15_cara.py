"""V5 KERNEL CO-AREA CARA test at G_inner=15 -- starts from FR-ansatz IC
and runs Picard to test Hellwig (CARA = FR with deficit -> 0)."""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20

from flint_sd_v5_kernel import (mp, phi_sigdelta_v5, set_boundary, f_inf, to_np)

G_FULL = 17
INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = INNER_HI - INNER_LO
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; W_F = 1.0
C_H = 0.45
MAX_ITER = 30
GAMMA_unused = 0.0  # unused in CARA path

dxi = 2.0 / (G_FULL - 1)
h_kernel = mp(C_H * (dxi ** 0.5))

LOG = os.path.join(HERE, 'flint_v5_G15_cara.log')
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
tau = mp(TAU_F); gamma = mp(GAMMA_unused); W = mp(W_F)

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

P_full_np = np.zeros((G_FULL,)*3); P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P = set_boundary(P)

lg(f"V5 CARA flint sigma-delta G_FULL={G_FULL} (G_inner={G_INNER}), dps={DPS}, FR-ANSATZ IC, h={float(h_kernel):.3f}, damped Picard.")
lg("Hellwig: expect d_FR_w -> small (~1e-2), slope -> 1, 1-R^2 -> small as the FP nails near FR.")

omega = mp(1); res_prev = mp('1e100')
hist = {'res':[], 'omega':[], 'd_FR_w':[], 'slope':[], '1mR2':[], 'sec':[]}
t0 = time.time()
for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = phi_sigdelta_v5(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                             INNER_LO, INNER_HI, h_kernel, clearing='cara')
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
    hist['res'].append(res); hist['omega'].append(float(omega))
    hist['d_FR_w'].append(d_w); hist['slope'].append(slope); hist['1mR2'].append(omr2)
    hist['sec'].append(sec)
    lg(f"  it {it:3d}  ω={float(omega):.3f}  ferr={res:.3e}  d_FR_w={d_w:.3e}  "
       f"slope={slope:.5f}  1-R²={omr2:.3e}  ({sec:.0f}s)")
    json.dump({**hist, 'G_FULL':G_FULL, 'dps':DPS, 'tau':TAU_F, 'h':float(h_kernel), 'IC':'FR_ansatz'},
              open(os.path.join(HERE, 'flint_v5_G15_cara.json'), 'w'), indent=2)
    np.save(os.path.join(HERE, 'flint_v5_G15_cara_P.npy'), inner)
lg(f"DONE total {(time.time()-t0)/60:.1f}m")
