#!/usr/bin/env python3
"""
Left-sweep for G17/umax4 (tau=2.5, gamma=1.0→0.01).
Each gamma step:
  1. Quadratic Lagrange predictor (once ≥3 converged points).
  2. Anderson mixing (m=10) to reduce residual to ~1e-8.
  3. JFNK (Newton-Krylov) to converge to 1e-12.
Adaptive step: GROW=1.4 (≤4 iters), SHRINK=0.5 (≥8 or fail), step ∈ [1e-5, 0.04].
"""
import sys, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.sparse.linalg import lgmres, LinearOperator

REPO   = Path("/home/user/FIXED-POINT-FACTORY")
CKPT   = REPO / "projects/REZN/checkpoints"
OUT    = REPO / "projects/REZN/overnight"
SOLVER = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOLVER))
sys.path.insert(0, "/home/user/rezn-src")

from code.contour_K3_halo import phi_K3_halo_smooth
from code.metrics import revelation_deficit

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Load anchor ──────────────────────────────────────────────────────────────
ANCHOR = CKPT / "g100_t0250_G17.npz"
d = np.load(ANCHOR, allow_pickle=True)
G_inner = int(d["G_inner"]); pad = int(d["pad"])
lo, hi  = pad, pad + G_inner
u_full  = d["u_full"].astype(np.float64)
tau     = d["tau_vec"].astype(np.float64)
W       = d["W_vec"].astype(np.float64)
P_anchor = d["P_full"].astype(np.float64)
du      = float(u_full[1] - u_full[0])
kernel_h = max(0.005, 0.05 * du)
n_inner  = G_inner ** 3
sl       = slice(lo, hi)

log(f"G17 anchor: G_inner={G_inner} pad={pad} tau={tau} W={W} kernel_h={kernel_h:.4f}")
log(f"  u_full: [{u_full[0]:.2f}, {u_full[-1]:.2f}]  n_inner={n_inner}")

# ── Operators ────────────────────────────────────────────────────────────────
def make_phi(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)
    return fn

# ── Anderson mixing AA-I (Walker & Ni 2011) ───────────────────────────────────
def anderson_mix(phi_fn, P0, m=10, n_iter=30, tol=1e-8, tag=""):
    """AA-I: x_{k+1} = (x_k + f_k) - (dX + dF) @ gamma
    where gamma = argmin ||f_k + dF @ gamma||^2.
    Falls back to Picard if AA step is worse than Picard."""
    P = P0.copy()
    X_hist = []   # inner-cell iterates (raveled)
    F_hist = []   # residuals f = phi(P) - P (inner cells raveled)

    for it in range(n_iter):
        phi_P = phi_fn(P)
        f_k = (phi_P - P)[sl, sl, sl].ravel()
        res = float(np.max(np.abs(f_k)))
        if it % 5 == 0 or res < tol:
            log(f"    AA {tag} it={it}  ||F||={res:.3e}")
        if res < tol:
            return P, res, it

        x_k = P[sl, sl, sl].ravel().copy()
        X_hist.append(x_k)
        F_hist.append(f_k.copy())

        # Number of finite-difference columns we can form
        n_diff = min(m, len(X_hist) - 1)

        if n_diff == 0:
            # Pure Picard
            x_new = x_k + f_k
        else:
            i0 = len(X_hist) - n_diff - 1
            dX = np.column_stack([X_hist[i0+j+1] - X_hist[i0+j] for j in range(n_diff)])
            dF = np.column_stack([F_hist[i0+j+1] - F_hist[i0+j] for j in range(n_diff)])
            try:
                gamma, _, _, _ = np.linalg.lstsq(dF, -f_k, rcond=None)
                x_aa = (x_k + f_k) - (dX + dF) @ gamma
                # Only accept if AA doesn't overshoot badly; else Picard
                x_new = x_aa
            except Exception:
                x_new = x_k + f_k  # fallback to Picard

        P_new = P.copy()
        P_new[sl, sl, sl] = x_new.reshape(G_inner, G_inner, G_inner)
        P_new = np.clip(P_new, 1e-12, 1.0 - 1e-12)
        P = P_new

    phi_P = phi_fn(P)
    f_fin = (phi_P - P)[sl, sl, sl].ravel()
    res = float(np.max(np.abs(f_fin)))
    return P, res, n_iter

# ── Newton-Krylov (JFNK) ─────────────────────────────────────────────────────
def newton_jfnk(phi_fn, P0, tol=1e-12, max_iter=10, tag=""):
    P = P0.copy()
    phi_P = phi_fn(P)
    F_in = (phi_P - P)[sl, sl, sl].ravel()
    res = float(np.max(np.abs(F_in)))
    for it in range(max_iter):
        if res < tol:
            log(f"    NK {tag} it={it}  ||F||={res:.3e}  [converged]")
            return P, res, it
        normP = np.linalg.norm(P[sl, sl, sl])
        fp_in = phi_P[sl, sl, sl].ravel()
        def mv(w, _n=normP, _fp=fp_in, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0.0: return w
            dlt = 1.5e-8 * (1.0 + _n) / wn
            Pp = _P.copy()
            Pp[sl, sl, sl] += dlt * w.reshape(G_inner, G_inner, G_inner)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _fp) / dlt
        A = LinearOperator((n_inner, n_inner), matvec=mv, dtype=np.float64)
        t0 = time.time()
        try:
            dv, info = lgmres(A, F_in, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError:
            dv, info = lgmres(A, F_in, tol=1e-4, maxiter=40, inner_m=25)
        P[sl, sl, sl] += dv.reshape(G_inner, G_inner, G_inner)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)
        phi_P = phi_fn(P)
        F_in = (phi_P - P)[sl, sl, sl].ravel()
        res = float(np.max(np.abs(F_in)))
        log(f"    NK {tag} it={it}  ||F||={res:.3e}  t={time.time()-t0:.0f}s")
    return P, res, max_iter

# ── Quadratic Lagrange predictor ──────────────────────────────────────────────
def quad_predict(history, g_next):
    """history = [(g0,P0), (g1,P1), (g2,P2)] most recent 3."""
    if len(history) < 3:
        return history[-1][1].copy()
    (g0, P0), (g1, P1), (g2, P2) = history[-3], history[-2], history[-1]
    d01 = g1 - g0; d02 = g2 - g0; d12 = g2 - g1
    if abs(d01) < 1e-12 or abs(d12) < 1e-12:
        return P2.copy()
    t = g_next
    L0 = (t - g1) * (t - g2) / (d01 * d02) if abs(d01 * d02) > 1e-20 else 0.0
    L1 = (t - g0) * (t - g2) / ((-d01) * d12) if abs(d01 * d12) > 1e-20 else 0.0
    L2 = (t - g0) * (t - g1) / (d02 * (-d12)) if abs(d02 * d12) > 1e-20 else 0.0
    P_pred = L0 * P0 + L1 * P1 + L2 * P2
    return np.clip(P_pred, 1e-12, 1.0 - 1e-12)

# ── Adaptive continuation ─────────────────────────────────────────────────────
GAMMA_START = 1.0
GAMMA_MIN   = 0.01
TOL         = 1e-12
STEP_INIT   = 0.005
STEP_MIN    = 1e-5
STEP_MAX    = 0.04
GROW        = 1.4
SHRINK      = 0.5
AA_ITERS    = 30
AA_M        = 10
NK_ITERS    = 12
AA_TOL      = 1e-7   # AA target before handing off to Newton

log("="*60)
log("Warming up Numba JIT...")
_ = make_phi(1.0)(P_anchor)
log("JIT warm-up done.")

log("="*60)
log(f"Starting left sweep: gamma {GAMMA_START:.4f} → {GAMMA_MIN:.4f}")

history = [(GAMMA_START, P_anchor.copy())]
g_cur   = GAMMA_START
P_cur   = P_anchor.copy()
step    = STEP_INIT

results = []
n_steps = 0

CKPT_OUT = CKPT  # save checkpoints alongside existing ones

while g_cur > GAMMA_MIN:
    g_next = max(g_cur - step, GAMMA_MIN)
    phi = make_phi(g_next)

    # Predictor
    P_pred = quad_predict(history, g_next)

    # Anderson mixing
    log(f"--- gamma={g_next:.6f}  step={step:.5f} ---")
    P_aa, res_aa, aa_it = anderson_mix(phi, P_pred, m=AA_M,
                                        n_iter=AA_ITERS, tol=AA_TOL,
                                        tag=f"g={g_next:.5f}")

    # Newton-Krylov
    P_new, res_nk, nk_it = newton_jfnk(phi, P_aa, tol=TOL,
                                         max_iter=NK_ITERS,
                                         tag=f"g={g_next:.5f}")

    total_iters = aa_it + nk_it
    converged   = res_nk < TOL

    if converged:
        deficit = float(revelation_deficit(P_new[lo:hi, lo:hi, lo:hi], u_full[lo:hi], tau, 3))
        log(f"  CONVERGED  ||F||={res_nk:.2e}  1-R²={deficit*100:.4f}%  "
            f"iters=AA:{aa_it}+NK:{nk_it}")
        history.append((g_next, P_new.copy()))
        if len(history) > 4:
            history.pop(0)
        g_cur = g_next
        P_cur = P_new.copy()
        results.append({"gamma": g_next, "F_inf": res_nk,
                        "deficit": deficit, "aa_iters": aa_it,
                        "nk_iters": nk_it})
        n_steps += 1

        # Adaptive grow
        if total_iters <= 4:
            step = min(step * GROW, STEP_MAX)
        elif total_iters >= 8:
            step = max(step * SHRINK, STEP_MIN)

        # Periodic checkpoint
        if n_steps % 5 == 0:
            tag = f"g{int(round(g_next*1000)):04d}_t{int(round(tau[0]*100)):04d}_G{G_inner}"
            ckpt_path = CKPT_OUT / f"{tag}.npz"
            P_inner = P_new[lo:hi, lo:hi, lo:hi]
            np.savez_compressed(ckpt_path,
                                P_inner=P_inner, P_full=P_new,
                                u_full=u_full, u_grid_inner=u_full[lo:hi],
                                gamma_vec=np.full(3, g_next),
                                tau_vec=tau, W_vec=W,
                                G_inner=G_inner, pad=pad, K=3,
                                stage_F_inf=res_nk,
                                stage_deficit=deficit)
            log(f"  Checkpoint saved: {ckpt_path.name}")

        # Save running results
        with open(OUT / "sweep_G17_AA_Newton.json", "w") as fh:
            json.dump(results, fh, indent=2)

    else:
        log(f"  FAILED  ||F||={res_nk:.2e}  AA:{aa_it}+NK:{nk_it}  shrinking step")
        step = max(step * SHRINK, STEP_MIN)
        if step <= STEP_MIN * 1.01:
            log(f"  Step at minimum ({STEP_MIN:.1e}) — fold likely near gamma={g_cur:.6f}")
            break

log("="*60)
log(f"Sweep done: {n_steps} converged points, last gamma={g_cur:.6f}")
with open(OUT / "sweep_G17_AA_Newton.json", "w") as fh:
    json.dump(results, fh, indent=2)
log(f"Results saved to {OUT / 'sweep_G17_AA_Newton.json'}")
