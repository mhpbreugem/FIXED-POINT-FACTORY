#!/usr/bin/env python3
"""
Adaptive machine-precision gamma sweep for G17/umax4 (tau=2.5).

Algorithm at each gamma step
─────────────────────────────
1. Quadratic Lagrange predictor from the last three converged points.
2. Anderson mixing (AA-I, depth m=10) to reduce ||F||_inf below AA_TOL.
3. Jacobian-free Newton-Krylov (LGMRES) to reach machine precision (1e-12).

Step adaptation (based on Newton iterations only, not AA count)
────────────────────────────────────────────────────────────────
  NK ≤ 4 iters  →  grow   step × 1.5  (easy region)
  NK ≥ 8 iters  →  shrink step × 0.5  (stiff region)
  failure        →  shrink step × 0.5; advance past stiff point at STEP_MIN

Usage
─────
  python sweep_G17.py left     # gamma 1.0 → 0.01
  python sweep_G17.py right    # gamma 1.0 → 5.0
"""
import sys, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.sparse.linalg import lgmres, LinearOperator

# ── Paths ────────────────────────────────────────────────────────────────────
REPO   = Path("/home/user/FIXED-POINT-FACTORY")
CKPT   = REPO / "projects/REZN/checkpoints"
OUT    = REPO / "projects/REZN/overnight"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "projects/REZN/solver_code"))
sys.path.insert(0, "/home/user/rezn-src")

from code.contour_K3_halo import phi_K3_halo_smooth
from code.metrics import revelation_deficit

# ── Direction from command line ───────────────────────────────────────────────
DIRECTION = sys.argv[1].lower() if len(sys.argv) > 1 else "left"
assert DIRECTION in ("left", "right"), "Usage: sweep_G17.py left|right"

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Load G17 anchor (gamma=1.0, tau=2.5) ─────────────────────────────────────
d        = np.load(CKPT / "g100_t0250_G17.npz", allow_pickle=True)
G_inner  = int(d["G_inner"])     # 17
pad      = int(d["pad"])          # 4
lo, hi   = pad, pad + G_inner    # inner-grid slice bounds
u_full   = d["u_full"].astype(np.float64)
tau      = d["tau_vec"].astype(np.float64)
W        = d["W_vec"].astype(np.float64)
P_anchor = d["P_full"].astype(np.float64)
kernel_h = max(0.005, 0.05 * float(u_full[1] - u_full[0]))  # Gaussian bandwidth
n_inner  = G_inner ** 3
sl       = slice(lo, hi)

log(f"G17  tau={tau[0]}  kernel_h={kernel_h:.4f}  n_inner={n_inner}  dir={DIRECTION}")

# ── Phi factory ──────────────────────────────────────────────────────────────
def make_phi(g):
    """Return phi(P) = phi_K3_halo_smooth(P, gamma=g)."""
    gv = np.full(3, g)
    return lambda P: phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)

# ── Anderson mixing (AA-I) ────────────────────────────────────────────────────
def anderson_mix(phi_fn, P0, m=10, n_iter=30, tol=1e-7, tag=""):
    """
    AA-I fixed-point iteration (Walker & Ni 2011).
    Update rule: x_{k+1} = (x_k + f_k) − (dX + dF) @ gamma_k
    where gamma_k = argmin ||f_k + dF @ gamma||^2.
    Falls back to Picard when history is empty.
    """
    P = P0.copy()
    X_hist, F_hist = [], []

    for it in range(n_iter):
        phi_P = phi_fn(P)
        f_k   = (phi_P - P)[sl, sl, sl].ravel()
        res   = float(np.max(np.abs(f_k)))
        if it % 10 == 0 or res < tol:
            log(f"    AA {tag} it={it}  ||F||={res:.3e}")
        if res < tol:
            return P, res, it

        x_k = P[sl, sl, sl].ravel().copy()
        X_hist.append(x_k)
        F_hist.append(f_k.copy())

        n_diff = min(m, len(X_hist) - 1)   # number of history columns
        if n_diff == 0:
            x_new = x_k + f_k               # Picard step (first iteration)
        else:
            i0 = len(X_hist) - n_diff - 1
            dX = np.column_stack([X_hist[i0+j+1] - X_hist[i0+j] for j in range(n_diff)])
            dF = np.column_stack([F_hist[i0+j+1] - F_hist[i0+j] for j in range(n_diff)])
            try:
                gamma, _, _, _ = np.linalg.lstsq(dF, -f_k, rcond=None)
                x_new = (x_k + f_k) - (dX + dF) @ gamma
            except Exception:
                x_new = x_k + f_k           # lstsq failed → Picard fallback

        P_new = P.copy()
        P_new[sl, sl, sl] = x_new.reshape(G_inner, G_inner, G_inner)
        P = np.clip(P_new, 1e-12, 1.0 - 1e-12)

    # Return whatever we have after n_iter steps
    phi_P = phi_fn(P)
    res   = float(np.max(np.abs((phi_P - P)[sl, sl, sl])))
    return P, res, n_iter

# ── Jacobian-free Newton-Krylov ───────────────────────────────────────────────
def newton_krylov(phi_fn, P0, tol=1e-12, max_iter=10, tag=""):
    """
    JFNK: solve (I − J_phi) delta = F via LGMRES with finite-difference matvec.
    Finite-difference step: delta = 1.5e-8 * (1 + ||P||) / ||w||  (Knoll & Keyes).
    """
    P     = P0.copy()
    phi_P = phi_fn(P)
    F_in  = (phi_P - P)[sl, sl, sl].ravel()
    res   = float(np.max(np.abs(F_in)))

    for it in range(max_iter):
        if res < tol:
            log(f"    NK {tag} it={it}  ||F||={res:.3e}  [converged]")
            return P, res, it

        normP = np.linalg.norm(P[sl, sl, sl])
        fp_in = phi_P[sl, sl, sl].ravel()

        def mv(w, _n=normP, _fp=fp_in, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0.0: return w
            eps = 1.5e-8 * (1.0 + _n) / wn
            Pp  = _P.copy()
            Pp[sl, sl, sl] += eps * w.reshape(G_inner, G_inner, G_inner)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _fp) / eps

        A = LinearOperator((n_inner, n_inner), matvec=mv, dtype=np.float64)
        t0 = time.time()
        try:
            dv, _ = lgmres(A, F_in, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError:
            dv, _ = lgmres(A, F_in, tol=1e-4, maxiter=40, inner_m=25)

        P[sl, sl, sl] += dv.reshape(G_inner, G_inner, G_inner)
        P     = np.clip(P, 1e-12, 1.0 - 1e-12)
        phi_P = phi_fn(P)
        F_in  = (phi_P - P)[sl, sl, sl].ravel()
        res   = float(np.max(np.abs(F_in)))
        log(f"    NK {tag} it={it}  ||F||={res:.3e}  t={time.time()-t0:.0f}s")

    return P, res, max_iter

# ── Quadratic Lagrange predictor ──────────────────────────────────────────────
def predict(history, g_next):
    """O(Δg³) predictor using the last three converged (gamma, P) pairs."""
    if len(history) < 3:
        return history[-1][1].copy()
    (g0, P0), (g1, P1), (g2, P2) = history[-3], history[-2], history[-1]
    d01 = g1 - g0;  d02 = g2 - g0;  d12 = g2 - g1
    if abs(d01) < 1e-12 or abs(d12) < 1e-12:
        return P2.copy()
    t  = g_next
    L0 = (t-g1)*(t-g2) / ( d01 * d02)
    L1 = (t-g0)*(t-g2) / (-d01 * d12)
    L2 = (t-g0)*(t-g1) / ( d02 *-d12)
    return np.clip(L0*P0 + L1*P1 + L2*P2, 1e-12, 1.0 - 1e-12)

# ── Sweep parameters ─────────────────────────────────────────────────────────
GAMMA_START = 1.0
GAMMA_MIN   = 0.01
GAMMA_MAX   = 5.0
TOL         = 1e-12
STEP_INIT   = 0.005
STEP_MIN    = 1e-5
STEP_MAX    = 0.05
GROW        = 1.5    # step multiplier when NK ≤ 4 iters
SHRINK      = 0.5    # step multiplier when NK ≥ 8 iters or failure

# ── JIT warm-up ──────────────────────────────────────────────────────────────
log("Warming up Numba JIT...")
_ = make_phi(1.0)(P_anchor)
log("JIT ready.")
log("=" * 60)

# ── Main continuation loop ────────────────────────────────────────────────────
history = [(GAMMA_START, P_anchor.copy())]
g_cur   = GAMMA_START
P_cur   = P_anchor.copy()
step    = STEP_INIT
results = []
n_steps = 0

is_left = (DIRECTION == "left")
gamma_limit = GAMMA_MIN if is_left else GAMMA_MAX

while (g_cur > gamma_limit + 1e-9) if is_left else (g_cur < gamma_limit - 1e-9):
    g_next = max(g_cur - step, gamma_limit) if is_left else min(g_cur + step, gamma_limit)
    phi    = make_phi(g_next)
    P_pred = predict(history, g_next)

    log(f"--- gamma={g_next:.6f}  step={step:.5f} ---")

    # Step 1: Anderson mixing to get close to the fixed point
    P_aa, _, aa_it = anderson_mix(phi, P_pred, m=10, n_iter=30, tol=1e-7,
                                   tag=f"g={g_next:.5f}")

    # Step 2: Newton-Krylov to reach machine precision
    P_new, res_nk, nk_it = newton_krylov(phi, P_aa, tol=TOL, max_iter=10,
                                           tag=f"g={g_next:.5f}")
    converged = res_nk < TOL

    if converged:
        deficit = float(revelation_deficit(P_new[lo:hi,lo:hi,lo:hi],
                                           u_full[lo:hi], tau, 3))
        log(f"  ✓ gamma={g_next:.6f}  ||F||={res_nk:.2e}  "
            f"1-R²={deficit*100:.4f}%  AA:{aa_it}+NK:{nk_it}")

        history.append((g_next, P_new.copy()))
        if len(history) > 4: history.pop(0)
        g_cur = g_next
        P_cur = P_new.copy()
        results.append({"gamma": g_next, "F_inf": res_nk,
                        "deficit": deficit, "aa_iters": aa_it, "nk_iters": nk_it})
        n_steps += 1

        # Adapt step size based on Newton difficulty
        if nk_it <= 4:
            step = min(step * GROW, STEP_MAX)
        elif nk_it >= 8:
            step = max(step * SHRINK, STEP_MIN)

        # Save checkpoint every 5 converged points
        if n_steps % 5 == 0:
            tag_s = (f"g{int(round(g_next*1000)):04d}"
                     f"_t{int(round(tau[0]*100)):04d}_G{G_inner}_{DIRECTION}")
            np.savez_compressed(CKPT / f"{tag_s}.npz",
                                P_inner=P_new[lo:hi,lo:hi,lo:hi], P_full=P_new,
                                u_full=u_full, u_grid_inner=u_full[lo:hi],
                                gamma_vec=np.full(3, g_next),
                                tau_vec=tau, W_vec=W, G_inner=G_inner, pad=pad, K=3,
                                stage_F_inf=res_nk, stage_deficit=deficit)
            log(f"  checkpoint: {tag_s}.npz")
    else:
        log(f"  ✗ gamma={g_next:.6f}  ||F||={res_nk:.2e}  AA:{aa_it}+NK:{nk_it}")
        step = max(step * SHRINK, STEP_MIN)
        if step <= STEP_MIN * 1.5:
            # Advance past the stiff point rather than halting
            log(f"  step at minimum — advancing past stiff point")
            g_cur = g_next

    # Flush results after every step
    out_file = OUT / f"sweep_G17_{DIRECTION}.json"
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2)

log("=" * 60)
log(f"Sweep done ({DIRECTION}): {n_steps} converged points, "
    f"last gamma={g_cur:.6f}")
