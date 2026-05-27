"""G=21 with max_iter=5000 — let trajectories run to find fixed points or limit cycles.

10 scenarios: γ ∈ {0.05, 0.1, 0.3, 0.5, 1.0} × IC ∈ {FR, no_learn}.
Reports every 100 iter. Auto-pushes per scenario.
"""
import os, sys, time, math, json
sys.path.insert(0, '/tmp')
import numpy as np
from phi_sigma_delta_folded import (phi_folded, set_boundary_folded, unfold_full,
                                      finf_interior_folded, crra_clear_sym)

OUT_DIR = '/tmp/g21_max5000_out'
os.makedirs(OUT_DIR, exist_ok=True)
LOG = '/tmp/g21_max5000.log'
open(LOG, 'w').close()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [5K] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

G = 21
mid = G // 2
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0
MAX_ITER = 5000
REPORT_EVERY = 100

xi_full = np.linspace(-1.0, 1.0, G)
xi_inner = xi_full[1:-1]
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
u_p = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
S_p = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
d_p = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_p, S_p, d_p, indexing='ij')
U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
S_phys = U1m + SIm
P_FR_inner = sigmoid(TAU * S_phys)
Tstar = TAU * S_phys
f_v_at = lambda U, v: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(U-v)**2)
F0_full = f_v_at(U1m,-0.5)*f_v_at(U2m,-0.5)*f_v_at(U3m,-0.5)
F1_full = f_v_at(U1m,+0.5)*f_v_at(U2m,+0.5)*f_v_at(U3m,+0.5)
Wd_inner = 0.5*F0_full + 0.5*F1_full; Wd_inner /= max(Wd_inner.sum(), 1e-30)

def compute_ic_no_learn(GAMMA, W):
    mu1 = sigmoid(TAU * U1m); mu2 = sigmoid(TAU * U2m); mu3 = sigmoid(TAU * U3m)
    P = np.empty_like(U1m)
    for i in range(U1m.shape[0]):
        for j in range(U1m.shape[1]):
            for k in range(U1m.shape[2]):
                P[i,j,k] = crra_clear_sym(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA, W)
    return P

def fold_from_inner(P_inner, G):
    P_stored = np.zeros((G, mid+1, mid+1))
    for i_full in range(1, G-1):
        for j_full in range(mid, G-1):
            for k_full in range(mid, G-1):
                P_stored[i_full, j_full-mid, k_full-mid] = P_inner[i_full-1, j_full-1, k_full-1]
    return P_stored

def weighted_R2_T(P_in):
    eps = 1e-30
    Pc = np.clip(P_in, eps, 1-eps)
    lp = np.log(Pc/(1-Pc))
    fl_t = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd_inner.flatten()
    slope, intercept = np.polyfit(fl_t, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_t + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intercept)

scenarios = []
for GAMMA in [0.05, 0.1, 0.3, 0.5, 1.0]:
    for ic in ['FR_ansatz', 'no_learn']:
        scenarios.append({'gamma': GAMMA, 'ic': ic})

log(f'G=21 max_iter=5000 battery: {len(scenarios)} scenarios')

# JIT warmup
log('JIT warmup...')
P_stored_warm = fold_from_inner(P_FR_inner, G)
P_stored_warm = set_boundary_folded(P_stored_warm, G)
t0 = time.time()
_ = phi_folded(P_stored_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, 0.1, W, G)
log(f'  JIT: {time.time()-t0:.1f}s')

t_global = time.time()
results = {}

for sc_i, sc in enumerate(scenarios):
    GAMMA = sc['gamma']; ic = sc['ic']
    label = f"5K_G21_g{GAMMA}_{ic}"
    log("")
    log(f'>>> {sc_i+1}/{len(scenarios)}: {label} (elapsed {(time.time()-t_global)/60:.1f}m)')
    if ic == 'FR_ansatz':
        P_in = P_FR_inner.copy()
    else:
        P_in = compute_ic_no_learn(GAMMA, W)
    P_stored = fold_from_inner(P_in, G)
    P_stored = set_boundary_folded(P_stored, G)

    t_ic = time.time()
    converged = False
    for it in range(1, MAX_ITER+1):
        P_new = phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
        P_new = set_boundary_folded(P_new, G)
        res = finf_interior_folded(P_new, P_stored, G)
        P_stored = P_new
        if it % REPORT_EVERY == 0 or it == 1:
            P_full = unfold_full(P_stored, G); P_inner = P_full[1:-1, 1:-1, 1:-1]
            d = float(np.sqrt(np.sum((P_inner - P_FR_inner)**2 * Wd_inner)))
            omr2, sl, intc = weighted_R2_T(P_inner)
            log(f'  iter {it:5d} res={res:.3e} d_FR={d:.4e} 1-R²={omr2:.3e}')
        if res < 1e-25:
            log(f'  CONVERGED iter {it}')
            converged = True
            break

    P_full = unfold_full(P_stored, G); P_inner = P_full[1:-1, 1:-1, 1:-1]
    final_d = float(np.sqrt(np.sum((P_inner - P_FR_inner)**2 * Wd_inner)))
    final_omr2, final_sl, final_intc = weighted_R2_T(P_inner)
    log(f'  FINAL {label}: res={res:.3e} d_FR={final_d:.4e} 1-R²={final_omr2:.3e} '
        f'({(time.time()-t_ic)/60:.1f}m)')
    np.save(f'{OUT_DIR}/P_{label}.npy', P_full)
    results[label] = {
        'gamma': GAMMA, 'ic': ic,
        'final_ferr': res, 'final_d_FR': final_d,
        'final_1mR2': final_omr2, 'slope': final_sl, 'intercept': final_intc,
        'iters': it, 'converged': converged,
    }
    with open(f'{OUT_DIR}/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    # auto-commit
    repo_out = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/folded/g21_5000_results'
    os.makedirs(repo_out, exist_ok=True)
    os.system(f'cp {OUT_DIR}/P_{label}.npy {repo_out}/ 2>&1')
    os.system(f'cp {OUT_DIR}/summary.json {repo_out}/ 2>&1')
    os.system(f'cp {LOG} {repo_out}/log.txt 2>&1')
    os.system(f'cd /home/user/FIXED-POINT-FACTORY && '
              f'git add projects/REZN/solver_code/sigma_delta/folded/g21_5000_results/ 2>&1 | tail -1 && '
              f'git commit -m "5K-iter {label} (ferr={res:.2e} d_FR={final_d:.3e})" 2>&1 | tail -1 && '
              f'git push origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -1')

log(f'Total elapsed: {(time.time()-t_global)/60:.1f} min')
