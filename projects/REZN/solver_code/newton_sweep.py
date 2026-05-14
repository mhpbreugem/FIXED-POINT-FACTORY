#!/usr/bin/env python3
"""Phase 1 — Newton (JFNK, no Anderson) continuation sweep.

Anchor : g100_t0300_G21.npz  (G_inner=21, pad=4, gamma=1.0, tau=3.0)
Operator: phi_K3_halo_smooth  kernel_h=0.025
Grid   : gamma 1.0 → 0.20, linear step 0.01 (left sweep only)
Start  : nearest converged point (P_prev), no predictor
Newton : JFNK with fixed damping 0.7 (no Armijo), no Anderson
"""
import sys, json, time
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

# ── Anchor ───────────────────────────────────────────────────────────────────
ANCHOR = CKPT / "g100_t0300_G21.npz"
d = np.load(ANCHOR, allow_pickle=True)
G_inner = int(d["G_inner"]); pad = int(d["pad"])
lo, hi  = pad, pad + G_inner
u_full  = d["u_full"].astype(np.float64)
tau     = d["tau_vec"].astype(np.float64)
W       = d["W_vec"].astype(np.float64)
P_anchor = d["P_full"].astype(np.float64)
anchor_gamma = float(d["gamma_vec"][0])
tau_fixed    = float(tau[0])
du = float(u_full[1] - u_full[0])
kernel_h = max(0.005, 0.05 * du)
u_inner = u_full[lo:hi]
n_inner = G_inner ** 3
sl = slice(lo, hi)

import mpmath as mp
mp.mp.dps = 50
if "P_inner_mp_str" in d.files:
    s = d["P_inner_mp_str"]
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_anchor[lo+i, lo+j, lo+l] = float(mp.mpf(str(s[i, j, l])))

log(f"Anchor {ANCHOR.name}: G_inner={G_inner} pad={pad} gamma={anchor_gamma} "
    f"tau={tau_fixed} kernel_h={kernel_h:.4f}  inner u[{u_inner.min():.1f},{u_inner.max():.1f}]")

# ── phi operator ──────────────────────────────────────────────────────────────
def phi_factory(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)
    return fn

# ── JFNK Newton — fixed damping, no Anderson ─────────────────────────────────
def newton_solve(phi_fn, P0, tol=1e-12, max_iter=30, damping=0.7, tag=""):
    """JFNK with fixed step damping. Each step: P += damping * (I-J)^{-1} F."""
    P = P0.copy()
    phi_P = phi_fn(P)
    F_in = (phi_P - P)[sl, sl, sl].ravel()
    res_inf = float(np.max(np.abs(F_in)))
    n_iter = 0
    for it in range(max_iter):
        n_iter = it
        if res_inf < tol:
            log(f"    {tag} newton it={it:2d}  ||F||={res_inf:.4e}  [converged]")
            break
        t_g = time.time()
        normP = np.linalg.norm(P[sl, sl, sl])
        phi_P_in = phi_P[sl, sl, sl].ravel()
        def mv(w, _normP=normP, _phi_P_in=phi_P_in, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0.0: return w
            dlt = 1.5e-8 * (1.0 + _normP) / wn
            Pp = _P.copy()
            Pp[sl, sl, sl] += dlt * w.reshape(G_inner, G_inner, G_inner)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _phi_P_in) / dlt
        A = LinearOperator((n_inner, n_inner), matvec=mv, dtype=np.float64)
        try:
            d_vec, info = lgmres(A, F_in, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError:
            d_vec, info = lgmres(A, F_in, tol=1e-4, maxiter=40, inner_m=25)
        P[sl, sl, sl] += damping * d_vec.reshape(G_inner, G_inner, G_inner)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)
        phi_P = phi_fn(P)
        F_in = (phi_P - P)[sl, sl, sl].ravel()
        res_inf = float(np.max(np.abs(F_in)))
        log(f"    {tag} newton it={it:2d}  ||F||={res_inf:.4e}  "
            f"lgmres_info={info}  t={time.time()-t_g:.0f}s")
    return P, res_inf, n_iter

# ── Grid: linear 1.0 → 0.20 step 0.01 ───────────────────────────────────────
STEP = 0.005
gamma_grid = [round(anchor_gamma - i * STEP, 6)
              for i in range(int(round((anchor_gamma - 0.20) / STEP)) + 1)]
n_pts = len(gamma_grid)
log(f"Grid: {n_pts} points {gamma_grid[0]:.4f} → {gamma_grid[-1]:.4f}  step={STEP}")

# ── Left sweep only (γ decreasing from 1.0 to 0.20) ──────────────────────────
results = []

def solve_point(g, P_prev):
    t0 = time.time()
    F0 = float(np.max(np.abs((phi_factory(g)(P_prev) - P_prev)[sl, sl, sl])))
    log(f"  ← gamma={g:.4f}  [start ||F||={F0:.3e}]")
    phi = phi_factory(g)
    P, res, nit = newton_solve(phi, P_prev, damping=1.0, tag=f"g={g:.3f}")
    F_check = float(np.max(np.abs((phi(P) - P)[sl, sl, sl])))
    r2 = revelation_deficit(P[sl, sl, sl], u_inner, np.full(3, tau_fixed), 3)
    dt = time.time() - t0
    results.append({"gamma": float(g), "one_minus_R2": float(r2),
                    "F_reinsert": F_check, "newton_iters": nit, "wall_s": dt})
    log(f"  ← gamma={g:.4f}  1-R2={r2:.6e}  "
        f"||F||_reinsert={F_check:.3e}  iters={nit}  t={dt:.0f}s")
    return P

t_sweep = time.time()
P_prev = P_anchor.copy()
for g in gamma_grid:
    P_prev = solve_point(g, P_prev)

log(f"sweep done in {time.time()-t_sweep:.0f}s")

# ── Results table ─────────────────────────────────────────────────────────────
log("=" * 70)
log(f"{'gamma':>8} | {'1-R^2':>14} | {'||F|| reinsert':>16} | {'iters':>5} | {'t(s)':>5}")
log("-" * 70)
for r in results:
    log(f"{r['gamma']:>8.4f} | {r['one_minus_R2']:>14.6e} | "
        f"{r['F_reinsert']:>16.3e} | {r['newton_iters']:>5d} | {r['wall_s']:>5.0f}")
log("=" * 70)

# ── Write JSON ────────────────────────────────────────────────────────────────
meta = {
    "generated_at": datetime.now().isoformat(),
    "phase": "1 — Newton JFNK, damping=0.7, step=0.01, left sweep only",
    "anchor_file": ANCHOR.name,
    "operator": f"phi_K3_halo_smooth kernel_h={kernel_h}",
    "grid": f"G_inner={G_inner} pad={pad} gamma[1.0→0.20] step=0.01",
    "tau": tau_fixed,
    "rows": results,
}
out_path = OUT / "newton_sweep.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
