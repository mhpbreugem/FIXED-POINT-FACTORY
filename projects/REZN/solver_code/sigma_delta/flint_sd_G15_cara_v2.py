"""flint sigma-delta CARA V2: float()-comparison fix, FR-ansatz IC."""
import time, math, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb
from flint_sd_G15_crra_v2 import (
    mp, fsig, phi_sigdelta, set_boundary, f_inf, to_np,
    G_FULL, INNER_LO, INNER_HI, TOT_u, TOT_S, TOT_d, TAU_F, W_F,
    xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau,
    P_FR_in, xi_full_np, xi_inner_np,
)

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20
MAX_ITER = 40

LOG = os.path.join(HERE, 'flint_sd_G15_cara_v2.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

# FR-ansatz IC
P_full_np = np.zeros((G_FULL,)*3)
P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P = set_boundary(P)
gamma = mp(0.0); W = mp(W_F)

lg(f"CARA flint sigma-delta V2 G_FULL={G_FULL} (G_inner={INNER_HI-INNER_LO}), dps={DPS}, FR-ANSATZ IC, damped Picard.")
lg(f"V2: float() branch-compare fix. Hellwig predicts d_FR -> ~7e-3 (matching float64).")

omega = mp(1); res_prev = mp('1e100')
hist = {'res': [], 'omega': [], 'd_FR': [], 'sec_per_iter': []}
t0 = time.time()
for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = phi_sigdelta(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W, clearing='cara')
    one = mp(1)
    P_damp = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                 for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P_damp = set_boundary(P_damp)
    res_mp = f_inf(P_damp, P); res = float(res_mp)
    if it > 3:
        if float(res_mp) > float(res_prev) * 0.99:   omega = max(omega * mp('0.7'), mp('0.001'))
        elif float(res_mp) < float(res_prev) * 0.6:  omega = min(omega * mp('1.05'), mp(1))
    P = P_damp; res_prev = res_mp
    Pnp = to_np(P)
    d_FR = float(np.sqrt(np.mean((Pnp[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI] - P_FR_in)**2)))
    sec = time.time() - t_step
    hist['res'].append(res); hist['omega'].append(float(omega)); hist['d_FR'].append(d_FR); hist['sec_per_iter'].append(sec)
    lg(f"  iter {it:3d}  ω={float(omega):.4f}  ferr={res:.3e}  d_FR={d_FR:.3e}  ({sec:.0f}s)")
    json.dump({**hist, 'G_FULL': G_FULL, 'dps': DPS, 'tau': TAU_F, 'IC': 'FR_ansatz', 'fix': 'V2 float comparisons'},
              open(os.path.join(HERE, 'flint_sd_G15_cara_v2.json'), 'w'), indent=2)
lg(f"DONE total {(time.time()-t0)/60:.1f}m")
