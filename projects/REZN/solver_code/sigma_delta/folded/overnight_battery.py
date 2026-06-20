"""Overnight battery: folded (Σ̂, δ̂) Picard across G, γ, ICs.

Tests:
  G ∈ {21, 41, 61}
  γ ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
  IC ∈ {FR_ansatz, no_learn}
  max_iter = 200

Logs every 10 iter per scenario. Saves final P (folded) and summary JSON.
"""
import os, sys, time, math, json
sys.path.insert(0, '/tmp')
import numpy as np
from phi_sigma_delta_folded import (phi_folded, set_boundary_folded, unfold_full,
                                      finf_interior_folded, crra_clear_sym)

OUT_DIR = '/tmp/overnight_out'
os.makedirs(OUT_DIR, exist_ok=True)
LOG = '/tmp/overnight_battery.log'
open(LOG, 'w').close()

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def build_grid(G):
    mid = G // 2
    xi_full = np.linspace(-1.0, 1.0, G)
    xi_inner = xi_full[1:-1]
    u_phys = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    S_phys = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    d_phys = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    return xi_full, mid, u_phys, S_phys, d_phys

def build_storage_from_full_inner(P_full_inner, G):
    """Build folded storage from a full-cube inner P (shape G-2 ×3).

    Convention: P_stored[i, j_s, k_s] = P_full[i, mid+j_s, mid+k_s] for
    i ∈ stored grid coords (0..G-1 with halo set later).
    """
    mid = G // 2
    P_stored = np.zeros((G, mid+1, mid+1))
    # Map inner cells of P_full (size G-2 per axis) to full indices [1, G-1)
    for i_full in range(1, G-1):
        for j_full in range(mid, G-1):
            for k_full in range(mid, G-1):
                P_stored[i_full, j_full-mid, k_full-mid] = P_full_inner[i_full-1, j_full-1, k_full-1]
    return P_stored

def compute_ic_fr(G, TAU):
    mid = G // 2
    xi_full = np.linspace(-1.0, 1.0, G)
    xi_inner = xi_full[1:-1]
    u_p = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    S_p = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    d_p = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    U1, SI, DE = np.meshgrid(u_p, S_p, d_p, indexing='ij')
    S_phys = U1 + SI
    return sigmoid(TAU * S_phys)

def compute_ic_no_learn(G, TAU, GAMMA, W):
    mid = G // 2
    xi_full = np.linspace(-1.0, 1.0, G)
    xi_inner = xi_full[1:-1]
    u_p = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    S_p = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    d_p = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
    U1, SI, DE = np.meshgrid(u_p, S_p, d_p, indexing='ij')
    U2 = 0.5*(SI+DE); U3 = 0.5*(SI-DE)
    mu1 = sigmoid(TAU * U1); mu2 = sigmoid(TAU * U2); mu3 = sigmoid(TAU * U3)
    P = np.empty_like(U1)
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            for k in range(U1.shape[2]):
                P[i,j,k] = crra_clear_sym(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], GAMMA, W)
    return P

def weighted_R2_T(P_in, P_FR_inner, Tstar, Wd):
    eps = 1e-30
    Pc = np.clip(P_in, eps, 1-eps)
    lp = np.log(Pc/(1-Pc))
    fl_t = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intercept = np.polyfit(fl_t, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_t + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    return (vr/vt if vt>0 else float('nan')), float(slope), float(intercept)


TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0
MAX_ITER = 200
REPORT_EVERY = 10

scenarios = []
for G in [21, 41, 61]:
    for GAMMA in [0.05, 0.1, 0.3, 0.5, 1.0]:
        for ic in ['FR_ansatz', 'no_learn']:
            scenarios.append({'G': G, 'gamma': GAMMA, 'ic': ic})

log(f'Overnight battery: {len(scenarios)} scenarios')
log(f'Tests: G ∈ {{21, 41, 61}}, γ ∈ {{0.05, 0.1, 0.3, 0.5, 1.0}}, IC ∈ {{FR_ansatz, no_learn}}')

# JIT warmup at smallest G
log('JIT warmup at G=21...')
xi_full, mid, u_p, S_p, d_p = build_grid(21)
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
P_FR_in_21 = compute_ic_fr(21, TAU)
P_stored_warm = build_storage_from_full_inner(P_FR_in_21, 21)
P_stored_warm = set_boundary_folded(P_stored_warm, 21)
t0 = time.time()
_ = phi_folded(P_stored_warm, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, 0.1, W, 21)
log(f'  G=21 JIT compile: {time.time()-t0:.1f}s')

t_global = time.time()
results = {}

for sc_i, sc in enumerate(scenarios):
    G = sc['G']; GAMMA = sc['gamma']; ic = sc['ic']
    label = f"G{G}_g{GAMMA}_{ic}"
    log("")
    log(f'>>> {sc_i+1}/{len(scenarios)}: {label} (elapsed {(time.time()-t_global)/60:.1f}m)')

    xi_full, mid, u_p, S_p, d_p = build_grid(G)
    xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
    P_FR_in = compute_ic_fr(G, TAU)
    if ic == 'FR_ansatz':
        P_in = P_FR_in.copy()
    else:
        P_in = compute_ic_no_learn(G, TAU, GAMMA, W)

    P_stored = build_storage_from_full_inner(P_in, G)
    P_stored = set_boundary_folded(P_stored, G)

    # density weights on inner
    f0n = np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u_p+0.5)**2)
    f1n = np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u_p-0.5)**2)
    F0_u = f0n[:,None,None]*f0n[None,:,None]*f0n[None,None,:]
    F1_u = f1n[:,None,None]*f1n[None,:,None]*f1n[None,None,:]
    # Note: weights here are wrt (u_1, Σ, δ) physical coords; we approximate
    # by using f_v(u_2(Σ,δ)) f_v(u_3(Σ,δ)) replacing the f_v in second and third
    # axes — keeps the right shape for the diagnostic regression
    U1m, SIm, DEm = np.meshgrid(u_p, S_p, d_p, indexing='ij')
    U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
    f_v_at = lambda U, v: np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(U-v)**2)
    F0_full = f_v_at(U1m,-0.5)*f_v_at(U2m,-0.5)*f_v_at(U3m,-0.5)
    F1_full = f_v_at(U1m,+0.5)*f_v_at(U2m,+0.5)*f_v_at(U3m,+0.5)
    Wd_inner = 0.5*F0_full + 0.5*F1_full; Wd_inner /= max(Wd_inner.sum(), 1e-30)
    Tstar = TAU * (U1m + SIm)  # T* = τS = τ(u_1+Σ)

    # iter
    t_ic = time.time()
    trace = []
    converged = False
    for it in range(1, MAX_ITER+1):
        P_new = phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
        P_new = set_boundary_folded(P_new, G)
        res = finf_interior_folded(P_new, P_stored, G)
        # extract inner full version for diagnostics
        P_full = unfold_full(P_new, G)
        P_inner = P_full[1:-1, 1:-1, 1:-1]
        d = float(np.sqrt(np.sum((P_inner - P_FR_in)**2 * Wd_inner)))
        omr2, sl, intc = weighted_R2_T(P_inner, P_FR_in, Tstar, Wd_inner)
        trace.append({'iter': it, 'res': res, 'd_FR': d, '1mR2': omr2})
        if it % REPORT_EVERY == 0 or it == 1:
            log(f'  iter {it:4d} res={res:.3e} d_FR={d:.4e} 1-R²(T*)={omr2:.3e}')
        P_stored = P_new
        if res < 1e-25:
            log(f'  CONVERGED iter {it}')
            converged = True
            break

    final_d_FR = trace[-1]['d_FR']
    final_ferr = trace[-1]['res']
    final_R2 = trace[-1]['1mR2']
    log(f'  FINAL: res={final_ferr:.3e} d_FR={final_d_FR:.4e} 1-R²(T*)={final_R2:.3e} '
        f'({len(trace)} iter, {(time.time()-t_ic)/60:.1f}m)')
    # save
    np.save(f'{OUT_DIR}/P_{label}.npy', unfold_full(P_stored, G))
    results[label] = {
        'G': G, 'gamma': GAMMA, 'ic': ic,
        'final_ferr': final_ferr, 'final_d_FR': final_d_FR,
        'final_1mR2': final_R2, 'slope': sl, 'intercept': intc,
        'iters': len(trace), 'converged': converged,
    }
    with open(f'{OUT_DIR}/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    # copy results to repo & push
    repo_out = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/folded/overnight_results'
    os.makedirs(repo_out, exist_ok=True)
    os.system(f'cp {OUT_DIR}/P_{label}.npy {repo_out}/ 2>&1')
    os.system(f'cp {OUT_DIR}/summary.json {repo_out}/ 2>&1')
    os.system(f'cp {LOG} {repo_out}/battery_log.txt 2>&1')
    os.system(f'cd /home/user/FIXED-POINT-FACTORY && '
              f'git add projects/REZN/solver_code/sigma_delta/folded/overnight_results/ 2>&1 | tail -1 && '
              f'git commit -m "overnight: {label} (ferr={final_ferr:.2e}, d_FR={final_d_FR:.3e}, iters={len(trace)})" 2>&1 | tail -2 && '
              f'git push origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -1')

log('')
log(f'Total elapsed: {(time.time()-t_global)/60:.1f} min')
log('SUMMARY:')
for k, r in results.items():
    log(f'  {k:35s} ferr={r["final_ferr"]:.3e} d_FR={r["final_d_FR"]:.4e} '
        f'1-R²={r["final_1mR2"]:.3e} iters={r["iters"]}')
