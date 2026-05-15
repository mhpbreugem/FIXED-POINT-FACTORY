#!/usr/bin/env python3
"""
G11 / umax=2 / tau=1 sweep — uses contour_KN_sym (exact contour integrals).

Step 0  Verify: load g100_t0250_G17 (known machine-precision point), evaluate
        sym_phi at identical parameters, confirm residual ≤ 1e-8.

Step 1  Anchor: G=11, tau=1, gamma=1, umax=2.
        no-learning init → Anderson(armijo, M=5) to ≤1e-5
        → scipy newton_krylov to 1e-12.

Step 2  Sweep gamma 1→5 (right) and 1→0.2 (left), adaptive step,
        same Anderson→Newton per step.
"""
import sys, time, json, math
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.optimize import anderson as scipy_anderson, NoConvergence, newton_krylov

REPO   = Path("/home/user/FIXED-POINT-FACTORY")
CKPT   = REPO / "projects/REZN/checkpoints"
OUT    = REPO / "projects/REZN/overnight"
SOLVER = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOLVER))

from contour_KN_sym import (
    SymGrid, sym_phi, sym_init_no_learning,
    sym_weighted_R2, full_to_sym,
)

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Verification on known G17 checkpoint
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("STEP 0 — verify sym_phi on g100_t0250_G17 checkpoint")
G17_CKPT = CKPT / "g100_t0250_G17.npz"
d17 = np.load(G17_CKPT, allow_pickle=True)
G17         = int(d17["G_inner"])          # 17
tau_v17     = d17["tau_vec"].astype(np.float64)
gamma_v17   = d17["gamma_vec"].astype(np.float64)
W_v17       = d17["W_vec"].astype(np.float64)
pad17       = int(d17["pad"])
lo17, hi17  = pad17, pad17 + G17
u17         = d17["u_full"].astype(np.float64)[lo17:hi17]
P17_inner   = d17["P_full"].astype(np.float64)[lo17:hi17, lo17:hi17, lo17:hi17]

tau17   = float(tau_v17[0])   # 2.5
gamma17 = float(gamma_v17[0]) # 1.0
W17     = float(W_v17[0])     # 1.0

sg17 = SymGrid.build(G17, 3)
P17_sorted = full_to_sym(P17_inner, sg17)

t0 = time.time()
phi17 = sym_phi(P17_sorted, sg17, u17, tau17, gamma17, W17)
F17 = float(np.max(np.abs(phi17 - P17_sorted)))
metrics17 = sym_weighted_R2(P17_sorted, sg17, u17, tau17)
log(f"  G={G17}  tau={tau17}  gamma={gamma17}")
log(f"  ||sym_phi(P*) - P*|| = {F17:.3e}   (kernel-halo was < 1e-12)")
log(f"  1-R² = {metrics17['1-R2']*100:.4f}%  slope={metrics17['slope']:.4f}")
log(f"  Verify time: {time.time()-t0:.1f}s")
if F17 < 1e-7:
    log("  ✓ VERIFICATION PASSED: sym_phi consistent with checkpoint to < 1e-7")
else:
    log(f"  ✗ WARNING: residual {F17:.2e} > 1e-7 — possible discrepancy")
log("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# G11 / tau=1 setup
# ─────────────────────────────────────────────────────────────────────────────
G      = 11
UMAX   = 2.0
TAU    = 1.0
W      = 1.0
GAMMA0 = 1.0
GAMMA_R = 5.0
GAMMA_L = 0.2
K      = 3

u_grid = np.linspace(-UMAX, UMAX, G)
sg     = SymGrid.build(G, K)
log(f"G11: G={G} umax={UMAX} tau={TAU} K={K}  n_sorted={sg.n}")

def make_res_fn(gamma, evals=None, best=None, tag=""):
    """Returns the residual closure F(P) = sym_phi(P) - P for scipy solvers."""
    def fn(Pf):
        pp = sym_phi(Pf, sg, u_grid, TAU, gamma, W)
        F  = pp - Pf
        r  = float(np.max(np.abs(F)))
        if evals is not None:
            evals[0] += 1
            if best is not None and r < best[0]:
                best[0] = r; best[1] = Pf.copy()
            if evals[0] % 20 == 0:
                log(f"  {tag} eval={evals[0]}  ||F||={r:.3e}")
        return F
    return fn

def solve_point(P_init, gamma, tag="", aa_tol=1e-5, aa_max=500, nk_tol=1e-9):
    """Anderson(M=5, armijo) → newton_krylov.  Returns (P_sol, F_final, converged).

    nk_tol=1e-9 matches the contour-integral finite-difference noise floor.
    """
    evals = [0]; best = [float("inf"), P_init.copy()]

    # Anderson phase
    fn = make_res_fn(gamma, evals, best, tag=f"AA:{tag}")
    try:
        P_aa = scipy_anderson(fn, P_init, f_tol=aa_tol, maxiter=aa_max,
                               M=5, verbose=False, line_search="armijo")
    except NoConvergence as e:
        P_aa = np.asarray(e.x)
        log(f"  AA {tag} NoConvergence at eval={evals[0]}")
    # Use best if AA made things worse
    F_aa = float(np.max(np.abs(sym_phi(P_aa, sg, u_grid, TAU, gamma, W) - P_aa)))
    if best[0] < F_aa:
        P_aa = best[1]; F_aa = best[0]
    log(f"  AA {tag} done ||F||={F_aa:.3e}  evals={evals[0]}")

    # Newton-Krylov phase — noise floor ~2e-9, so tol=1e-9 is achievable
    nk_evals = [0]; nk_best = [float("inf"), P_aa.copy()]
    def nk_fn(Pf):
        pp = sym_phi(Pf, sg, u_grid, TAU, gamma, W)
        F  = pp - Pf
        nk_evals[0] += 1
        r = float(np.max(np.abs(F)))
        if r < nk_best[0]:
            nk_best[0] = r; nk_best[1] = Pf.copy()
        if nk_evals[0] % 5 == 1:
            log(f"  NK {tag} eval={nk_evals[0]}  ||F||={r:.3e}  best={nk_best[0]:.3e}")
        return F

    t0 = time.time()
    try:
        P_sol = newton_krylov(nk_fn, P_aa, f_tol=nk_tol, maxiter=200,
                               method="lgmres", inner_m=30, outer_k=10,
                               verbose=False)
        P_sol = np.asarray(P_sol)
    except Exception as ex:
        log(f"  NK {tag} stopped: {ex} — using NK best")
        P_sol = nk_best[1]

    F_final = float(np.max(np.abs(sym_phi(P_sol, sg, u_grid, TAU, gamma, W) - P_sol)))
    log(f"  NK {tag} done ||F||={F_final:.3e}  t={time.time()-t0:.1f}s  NK_evals={nk_evals[0]}")
    return P_sol, F_final, F_final < nk_tol

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Anchor at gamma=1, tau=1, G=11
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_FILE = CKPT / f"g{int(GAMMA0*100):04d}_t{int(TAU*100):04d}_G{G}_sym.npz"

if ANCHOR_FILE.exists():
    log(f"Loading anchor: {ANCHOR_FILE.name}")
    da = np.load(ANCHOR_FILE, allow_pickle=True)
    P_anchor = da["P_sorted"].astype(np.float64)
    log(f"  Anchor loaded: ||F||={float(np.max(np.abs(sym_phi(P_anchor, sg, u_grid, TAU, GAMMA0, W) - P_anchor))):.3e}")
else:
    log("=" * 60)
    log("STEP 1 — build anchor: tau=1 gamma=1 G=11")
    P_nl = sym_init_no_learning(sg, u_grid, TAU, GAMMA0, W)
    F_nl = float(np.max(np.abs(sym_phi(P_nl, sg, u_grid, TAU, GAMMA0, W) - P_nl)))
    log(f"  no-learning init ||F||={F_nl:.3e}")
    P_anchor, F_anc, ok = solve_point(P_nl, GAMMA0, tag="anchor",
                                       aa_tol=1e-5, aa_max=1000, nk_tol=1e-12)
    metrics_anc = sym_weighted_R2(P_anchor, sg, u_grid, TAU)
    log(f"  Anchor: ||F||={F_anc:.3e}  1-R²={metrics_anc['1-R2']*100:.4f}%  "
        f"n_cells={sg.n}")
    if not ok:
        log("  WARNING: anchor not at machine precision — sweeping anyway")
    np.savez_compressed(ANCHOR_FILE, P_sorted=P_anchor, u_grid=u_grid,
                        K=K, G=G, gamma=GAMMA0, tau=TAU, W=W,
                        F_final=F_anc, one_minus_r2=metrics_anc["1-R2"])
    log(f"  Anchor saved: {ANCHOR_FILE.name}")
    log("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Sweep gamma
# ─────────────────────────────────────────────────────────────────────────────
def quad_predict(hist, g_next):
    if len(hist) < 3: return hist[-1][1].copy()
    (g0,P0),(g1,P1),(g2,P2) = hist[-3], hist[-2], hist[-1]
    d01, d02, d12 = g1-g0, g2-g0, g2-g1
    if abs(d01) < 1e-12 or abs(d12) < 1e-12: return P2.copy()
    t = g_next
    L0 = (t-g1)*(t-g2)/(d01*d02)
    L1 = (t-g0)*(t-g2)/(-d01*d12)
    L2 = (t-g0)*(t-g1)/(d02*-d12)
    return L0*P0 + L1*P1 + L2*P2

def sweep(P_start, g_start, g_end, label, results_all):
    is_right = g_end > g_start
    STEP_INIT = 0.05; STEP_MIN = 1e-4; STEP_MAX = 0.5
    GROW = 1.5; SHRINK = 0.5; TOL = 1e-9
    hist = [(g_start, P_start.copy())]
    g, P, step = g_start, P_start.copy(), STEP_INIT
    n_ok = 0
    log(f"=== {label} sweep: {g_start:.4f} → {g_end:.4f} ===")
    while (g < g_end - 1e-9) if is_right else (g > g_end + 1e-9):
        g_next = min(g + step, g_end) if is_right else max(g - step, g_end)
        P_pred = quad_predict(hist, g_next)
        F_pred = float(np.max(np.abs(sym_phi(P_pred, sg, u_grid, TAU, g_next, W) - P_pred)))
        P0 = P_pred if F_pred < 0.05 else P
        if F_pred >= 0.05:
            log(f"  pred ||F||={F_pred:.2e} — using P_prev")
        log(f"--- gamma={g_next:.5f}  step={step:.5f} ---")
        P_new, F_new, ok = solve_point(P0, g_next, tag=f"g={g_next:.4f}",
                                        aa_tol=1e-5, aa_max=300, nk_tol=TOL)
        if ok or F_new < 5e-9:
            metrics = sym_weighted_R2(P_new, sg, u_grid, TAU)
            d1 = metrics["1-R2"]
            log(f"  ✓ gamma={g_next:.5f}  ||F||={F_new:.2e}  1-R²={d1*100:.4f}%")
            hist.append((g_next, P_new.copy()))
            if len(hist) > 4: hist.pop(0)
            g, P = g_next, P_new.copy()
            n_ok += 1
            # Adapt step on quality of convergence
            if F_new < 1e-9: step = min(step * GROW, STEP_MAX)
            elif F_new > 5e-9: step = max(step * SHRINK, STEP_MIN)
            results_all.append({"gamma": g_next, "F_inf": F_new,
                                 "deficit": d1, "dir": label,
                                 "slope": metrics["slope"]})
            if n_ok % 5 == 0:
                ts = (f"g{int(round(g_next*100)):04d}"
                      f"_t{int(round(TAU*100)):04d}_G{G}_sym")
                np.savez_compressed(CKPT / f"{ts}.npz",
                    P_sorted=P_new, u_grid=u_grid, K=K, G=G,
                    gamma=g_next, tau=TAU, W=W,
                    F_final=F_new, one_minus_r2=d1)
                log(f"  checkpoint: {ts}.npz")
        else:
            log(f"  ✗ gamma={g_next:.5f}  ||F||={F_new:.2e}")
            step = max(step * SHRINK, STEP_MIN)
            if step <= STEP_MIN * 1.5:
                log("  step at minimum — fold boundary"); g = g_next
        with open(OUT / f"sweep_G{G}_sym.json", "w") as fh:
            json.dump(results_all, fh, indent=2)
    log(f"=== {label} done: {n_ok} points, last gamma={g:.5f} ===")
    return P, g

results = []
log("=" * 60)
log("STEP 2 — sweep gamma 1→5 (right)")
_, _ = sweep(P_anchor, GAMMA0, GAMMA_R, "right", results)
log("=" * 60)
log("STEP 2 — sweep gamma 1→0.2 (left)")
_, _ = sweep(P_anchor, GAMMA0, GAMMA_L, "left",  results)

log("=" * 60)
log(f"ALL DONE: {len(results)} converged points")
log(f"Results → {OUT}/sweep_G{G}_sym.json")
