"""Picard the CARA Phi on the (u_1, Σ, δ) ξ-cube with exact FR boundary
conditions. The Hellwig prediction: with FR-consistent BCs, the analytic
FR P* = sigmoid(tau*(u_1+Σ)) IS a fixed point to machine precision.
"""
import os, sys, time, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phi_sigma_delta import set_boundary, finf_interior, fsig
from phi_sigma_delta_cara import phi_sigmadelta_cara

# config
G = 10
G_FULL = G + 2
INNER_LO, INNER_HI = 1, G + 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0
MAX_ITER = 200
REPORT_EVERY = 5

LOG = os.path.join(HERE, 'sigmadelta_cara_picard.log')
open(LOG, 'w').close()
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# grid (uniform xi on inner, +/-1 boundary cells)
dxi = 2.0 / (G + 1)
xi_inner = np.linspace(-1+dxi, 1-dxi, G)
xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

u1_inner = TOT_u * np.arctanh(xi_inner)
S_inner_vals = TOT_S * np.arctanh(xi_inner)
d_inner = TOT_d * np.arctanh(xi_inner)
U1_m, SI_m, DE_m = np.meshgrid(u1_inner, S_inner_vals, d_inner, indexing='ij')
U2_m = 0.5 * (SI_m + DE_m); U3_m = 0.5 * (SI_m - DE_m)
S_full = U1_m + U2_m + U3_m  # = u1 + Σ (δ-flat for FR)
Tstar = TAU * S_full
P_FR_inner = 1.0 / (1.0 + np.exp(-Tstar))

# weight = density-mixture, used for d_FR and R²
def f_arr(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
F0 = f_arr(U1_m, -0.5)*f_arr(U2_m, -0.5)*f_arr(U3_m, -0.5)
F1 = f_arr(U1_m, +0.5)*f_arr(U2_m, +0.5)*f_arr(U3_m, +0.5)
Wd = 0.5*F0 + 0.5*F1
Wd = Wd / max(Wd.sum(), 1e-30)

def dist_to_FR(P_inner):
    return float(np.sqrt(np.sum((P_inner - P_FR_inner)**2 * Wd)))

def weighted_R2(P_inner, x_field):
    eps = 1e-30
    Pc = np.clip(P_inner, eps, 1 - eps)
    lp = np.log(Pc / (1 - Pc))
    fl_x = x_field.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intercept = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope * fl_x + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp - m)**2, weights=fl_w))
    vr = float(np.average((fl_lp - pred)**2, weights=fl_w))
    return (vr / vt if vt > 0 else float('nan')), float(slope), float(intercept)

def IC_FR(): return P_FR_inner.copy()
def IC_no_learn():
    # each agent uses only own signal: λ(τ u_k); naive K=3 average in log-odds
    m1 = 1.0/(1.0+np.exp(-TAU*U1_m)); m2 = 1.0/(1.0+np.exp(-TAU*U2_m)); m3 = 1.0/(1.0+np.exp(-TAU*U3_m))
    eps = 1e-12
    l1, l2, l3 = (np.log(np.clip(m1,eps,1-eps)/(1-np.clip(m1,eps,1-eps))),
                  np.log(np.clip(m2,eps,1-eps)/(1-np.clip(m2,eps,1-eps))),
                  np.log(np.clip(m3,eps,1-eps)/(1-np.clip(m3,eps,1-eps))))
    return 1.0/(1.0+np.exp(-(l1+l2+l3)/3.0))

ICs = [('FR_ansatz', IC_FR()), ('no_learning', IC_no_learn())]

log("=" * 78)
log(f"CARA Φ on (u_1, Σ, δ) cube, G={G}/axis, TOT_u={TOT_u}, TOT_Σ={TOT_S}, TOT_δ={TOT_d}")
log(f"BC: Σ=-∞,+∞ → P=0,1 (FR exact); δ=±∞ → zero-order extrap; u_1=±∞ → P=0,1")
log(f"τ={TAU}  CARA log-odds clearing.  Hellwig predicts FR is the FP with deficit=0.")
log("=" * 78)

# JIT warmup
log("JIT warmup...")
P_warm = np.zeros((G_FULL,)*3)
P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_inner
P_warm = set_boundary(P_warm, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
t0 = time.time()
_ = phi_sigmadelta_cara(P_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, W,
                         INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
log(f"  JIT compile + 1 step: {time.time()-t0:.1f}s")

t_global = time.time()
results = {}
for ic_name, P_ic in ICs:
    log("")
    log(f">>> {ic_name}  (elapsed {(time.time()-t_global)/60:.1f}m)")
    log(f"  IC dist_to_FR = {dist_to_FR(P_ic):.4e}")

    P = np.zeros((G_FULL,)*3)
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_ic
    P = set_boundary(P, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)

    t_ic = time.time(); res = float('inf'); d = float('inf')
    for it in range(1, MAX_ITER+1):
        P_new = phi_sigmadelta_cara(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, W,
                                     INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_new = set_boundary(P_new, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
        res = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_inner_now = P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        d = dist_to_FR(P_inner_now)
        P = P_new
        if it % REPORT_EVERY == 0 or it == 1:
            omr2_T, sl_T, int_T = weighted_R2(P_inner_now, Tstar)
            log(f"  iter {it:4d}  ferr={res:.3e}  1-R²(T*)={omr2_T:.3e}  "
                f"d_FR={d:.4e}  slope_T*={sl_T:.5f}  ({(time.time()-t_ic)/60:.1f}m)")
        if res < 1e-15:
            log(f"  CONVERGED iter {it}, ferr={res:.3e}")
            break

    P_final = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()
    omr2_T, sl_T, int_T = weighted_R2(P_final, Tstar)
    log(f"  FINAL {ic_name}: ferr={res:.3e}  d_FR={d:.4e}")
    log(f"    1-R²(T*)={omr2_T:.4e}  slope_T*={sl_T:.6f}  intc_T*={int_T:+.6f}")
    np.save(os.path.join(HERE, f'sigmadelta_cara_{ic_name}.npy'), P_final)
    results[ic_name] = {
        'ferr': res, 'd_FR': d, '1mR2_Tstar': omr2_T,
        'slope_Tstar': sl_T, 'intercept_Tstar': int_T, 'iters': it,
    }
    with open(os.path.join(HERE, 'sigmadelta_cara_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
log("")
log(f"DONE (total {(time.time()-t_global)/60:.1f}m).")
