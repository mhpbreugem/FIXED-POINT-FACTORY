"""V2 SANITY CHECK at G=11 (matching the float64 G=11 from sd_cara_G_sweep).
Should give d_FR ~ 6e-3 if the V2 float() comparison fix works.
"""
import os, sys, time, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20

# Override constants from the v2 module by importing the FUNCTIONS only
# and rebuilding setup for G_FULL=12 (G_inner=10)
G_FULL_NEW = 12
INNER_LO_NEW, INNER_HI_NEW = 1, G_FULL_NEW - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; W_F = 1.0
MAX_ITER = 40

# Monkey-patch INNER_LO / INNER_HI / G_FULL by mutating module attrs
import flint_sd_G15_crra_v2 as M
M.G_FULL = G_FULL_NEW
M.INNER_LO = INNER_LO_NEW
M.INNER_HI = INNER_HI_NEW

LOG = os.path.join(HERE, 'flint_sd_G11_cara_v2.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

mp = M.mp
xi_full_np = np.linspace(-1.0, 1.0, G_FULL_NEW)
xi_inner_np = xi_full_np[INNER_LO_NEW:INNER_HI_NEW]
xi_u1 = [mp(float(x)) for x in xi_full_np]
xi_S = [mp(float(x)) for x in xi_full_np]
xi_d = [mp(float(x)) for x in xi_full_np]
TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU_F); gamma = mp(0); W = mp(W_F)

u_phys = TOT_u * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
S_full = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(TAU_F * S_full)

P_full_np = np.zeros((G_FULL_NEW,)*3)
P_full_np[INNER_LO_NEW:INNER_HI_NEW, INNER_LO_NEW:INNER_HI_NEW, INNER_LO_NEW:INNER_HI_NEW] = P_FR_in
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL_NEW)] for j in range(G_FULL_NEW)] for i in range(G_FULL_NEW)]
P = M.set_boundary(P)

lg(f"V2 SANITY CHECK: CARA at G_FULL={G_FULL_NEW} (G_inner=10), dps={DPS}, FR-IC.")
lg(f"Expect: d_FR -> ~6e-3 (matching float64 G=10 result).")

omega = mp(1); res_prev = mp('1e100')
hist = {'res': [], 'd_FR': []}
t0 = time.time()
for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = M.phi_sigdelta(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W, clearing='cara')
    one = mp(1)
    P_damp = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                 for k in range(G_FULL_NEW)] for j in range(G_FULL_NEW)] for i in range(G_FULL_NEW)]
    P_damp = M.set_boundary(P_damp)
    res_mp = M.f_inf(P_damp, P); res = float(res_mp)
    if it > 3:
        if float(res_mp) > float(res_prev) * 0.99: omega = max(omega * mp('0.7'), mp('0.001'))
        elif float(res_mp) < float(res_prev) * 0.6: omega = min(omega * mp('1.05'), mp(1))
    P = P_damp; res_prev = res_mp
    Pnp = M.to_np(P)
    d_FR = float(np.sqrt(np.mean((Pnp[INNER_LO_NEW:INNER_HI_NEW,INNER_LO_NEW:INNER_HI_NEW,INNER_LO_NEW:INNER_HI_NEW] - P_FR_in)**2)))
    hist['res'].append(res); hist['d_FR'].append(d_FR)
    lg(f"  iter {it:3d}  ω={float(omega):.4f}  ferr={res:.3e}  d_FR={d_FR:.3e}  ({time.time()-t_step:.0f}s)")
    json.dump({**hist, 'G_FULL': G_FULL_NEW, 'dps': DPS, 'tau': TAU_F},
              open(os.path.join(HERE, 'flint_sd_G11_cara_v2.json'), 'w'), indent=2)
lg(f"DONE total {(time.time()-t0)/60:.1f}m")
