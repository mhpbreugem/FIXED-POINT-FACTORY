"""Picard on the (u_1, Σ, δ) cube — G=10 per axis, float64.
K=3 sym, γ=0.1, τ=2, h=0. Reports every 5 iter.
"""
import os, sys, time, math, json
sys.path.insert(0, '/tmp')
import numpy as np
from dd_phi_sigma_delta import (phi_sigmadelta, set_boundary, finf_interior, fsig)

# config
G = 10
G_FULL = G + 2
INNER_LO, INNER_HI = 1, G + 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
MAX_ITER = 200
REPORT_EVERY = 5

LOG = '/tmp/sigmadelta_picard.log'
open(LOG, 'w').close()
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

# grid
dxi = 2.0 / (G + 1)
xi_inner = np.linspace(-1+dxi, 1-dxi, G)
xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

# precompute physical values for the inner grid
u1_inner = TOT_u * np.arctanh(xi_inner)
S_inner_vals = TOT_S * np.arctanh(xi_inner)
d_inner = TOT_d * np.arctanh(xi_inner)
U1_m, SI_m, DE_m = np.meshgrid(u1_inner, S_inner_vals, d_inner, indexing='ij')
U2_m = 0.5 * (SI_m + DE_m); U3_m = 0.5 * (SI_m - DE_m)
S_full = U1_m + U2_m + U3_m   # = u_1 + u_2 + u_3 = u_1 + Σ (regardless of δ)

P_FR_inner = 1.0 / (1.0 + np.exp(-TAU * S_full))

# joint density of (u_1, u_2, u_3) at each (i, j, k) of (u_1, Σ, δ) cube
# density transforms: f(u_1) f(u_2) f(u_3) but we need to weight by Jacobian.
# Actually since we sum density across both v's, joint density is mixture.
# weight ~ ½ Π_k f_1(u_k) + ½ Π_k f_0(u_k)
f0_1 = fsig_arr = np.vectorize(lambda u, vm: np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2))
F0_full = f0_1(U1_m, -0.5) * f0_1(U2_m, -0.5) * f0_1(U3_m, -0.5)
F1_full = f0_1(U1_m, +0.5) * f0_1(U2_m, +0.5) * f0_1(U3_m, +0.5)
Wd = 0.5 * F0_full + 0.5 * F1_full
Wd_sum = Wd.sum(); Wd = Wd / max(Wd_sum, 1e-30)

Tstar = TAU * S_full

def dist_to_FR(P_inner):
    return float(np.sqrt(np.sum((P_inner - P_FR_inner) ** 2 * Wd)))

def weighted_R2(P_inner, x_field, x_name=''):
    eps = 1e-30
    Pc = np.clip(P_inner, eps, 1 - eps)
    lp = np.log(Pc / (1 - Pc))
    fl_x = x_field.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intercept = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope * fl_x + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp - m) ** 2, weights=fl_w))
    vr = float(np.average((fl_lp - pred) ** 2, weights=fl_w))
    return (vr / vt if vt > 0 else float('nan')), float(slope), float(intercept)

# ICs (on inner grid)
def IC_FR(): return P_FR_inner.copy()
def IC_two_step():
    return np.where(S_full > 0, 0.85, 0.15).astype(np.float64)

initials = [
    ('FR_ansatz', IC_FR()),
    ('two_step',  IC_two_step()),
]

log("=" * 78)
log(f"PHI ON (u_1, Σ, δ) CUBE, G={G} per axis, K=3 sym")
log(f"  TOT_u={TOT_u}, TOT_Σ={TOT_S}, TOT_δ={TOT_d}")
log(f"  Σ=-∞,+∞ → P=0,1 (FR exact); δ=±∞ → zero-order extrap")
log(f"  τ={TAU}, γ={GAMMA}, max_iter={MAX_ITER}, report every {REPORT_EVERY}")
log("=" * 78)

# warmup
log("JIT warmup...")
P_warm = np.zeros((G_FULL,)*3)
P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_inner
P_warm = set_boundary(P_warm, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
t0 = time.time()
_ = phi_sigmadelta(P_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
log(f"  JIT compile + 1 step: {time.time()-t0:.1f}s")

t_global = time.time()
results = {}

for ic_name, P_ic in initials:
    log("")
    log(f">>> {ic_name}  (elapsed {(time.time()-t_global)/60:.1f}m)")
    log(f"  IC dist_to_FR = {dist_to_FR(P_ic):.4e}")

    P = np.zeros((G_FULL,)*3)
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_ic
    P = set_boundary(P, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)

    t_ic = time.time()
    res = float('inf'); d = float('inf')
    for it in range(1, MAX_ITER+1):
        P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                                INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_new = set_boundary(P_new, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
        res = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_inner_now = P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        d = dist_to_FR(P_inner_now)
        P = P_new
        if it % REPORT_EVERY == 0 or it == 1:
            omr2_T, sl_T, int_T = weighted_R2(P_inner_now, Tstar, 'T*')
            log(f"  iter {it:4d}  ferr={res:.3e}  1-R²(T*)={omr2_T:.3e}  "
                f"d_FR={d:.4e}  ({(time.time()-t_ic)/60:.1f}m)")
        if res < 1e-29:
            log(f"  CONVERGED iter {it}")
            break

    P_final = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()
    omr2_T, sl_T, int_T = weighted_R2(P_final, Tstar, 'T*')
    log(f"  FINAL {ic_name}: ferr={res:.3e}  d_FR={d:.4e}")
    log(f"    1-R²(T*)={omr2_T:.4e}  slope_T*={sl_T:.6f}  intc_T*={int_T:+.6f}")
    np.save(f'/tmp/sigmadelta_{ic_name}.npy', P_final)
    results[ic_name] = {
        'ferr': res, 'd_FR': d,
        '1mR2_Tstar': omr2_T, 'slope_Tstar': sl_T, 'intercept_Tstar': int_T,
        'iters': it,
    }
    with open('/tmp/sigmadelta_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

log("")
log(f"Done. Total elapsed: {(time.time()-t_global)/60:.1f} min")
