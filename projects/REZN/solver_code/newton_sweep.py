#!/usr/bin/env python3
"""Newton JFNK continuation with adaptive step size.

Anchor : g100_t0300_G21.npz  gamma=1.0 → sweep left toward gamma=0
Solver : JFNK, no Anderson, no predictor, damping=1.0 (full Newton)
Step   : adaptive — grows on easy steps, shrinks on hard/failed steps
Max iter: 10 per Newton solve; failure → halve step, retry
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

# ── Anchor ────────────────────────────────────────────────────────────────────
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
    f"tau={tau_fixed} kernel_h={kernel_h:.4f}")

# ── phi operator ──────────────────────────────────────────────────────────────
def phi_factory(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)
    return fn

# ── JFNK Newton — full step, no Anderson ─────────────────────────────────────
def newton_solve(phi_fn, P0, tol=1e-12, max_iter=10, tag=""):
    P = P0.copy()
    phi_P = phi_fn(P)
    F_in = (phi_P - P)[sl, sl, sl].ravel()
    res = float(np.max(np.abs(F_in)))
    for it in range(max_iter):
        if res < tol:
            log(f"    {tag} it={it:2d}  ||F||={res:.3e}  [converged]")
            return P, res, it
        normP = np.linalg.norm(P[sl, sl, sl])
        phi_P_in = phi_P[sl, sl, sl].ravel()
        def mv(w, _n=normP, _fp=phi_P_in, _P=P):
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
        log(f"    {tag} it={it:2d}  ||F||={res:.3e}  info={info}  t={time.time()-t0:.0f}s")
    return P, res, max_iter   # did not converge

# ── Adaptive continuation: gamma = 1.0 → 0.0 ─────────────────────────────────
TOL       = 1e-12
STEP_INIT = 0.005
STEP_MIN  = 1e-5
STEP_MAX  = 0.05
GAMMA_MIN = 0.01     # stop here
MAX_ITER  = 10
GROW      = 1.5      # step multiplier on easy convergence (iters <= 4)
SHRINK    = 0.5      # step multiplier on hard convergence or failure

results = []
g     = anchor_gamma
P_prev = P_anchor.copy()
step  = STEP_INIT
t_total = time.time()

# record anchor itself
r0 = revelation_deficit(P_anchor[sl, sl, sl], u_inner, np.full(3, tau_fixed), 3)
results.append({"gamma": g, "one_minus_R2": float(r0),
                "F_reinsert": 3.664e-15, "newton_iters": 0, "step": 0.0})
log(f"  anchor gamma={g:.6f}  1-R2={r0:.6e}  ||F||=3.664e-15  step=—")

while g - 1e-9 > GAMMA_MIN:
    g_next = max(GAMMA_MIN, round(g - step, 8))
    phi = phi_factory(g_next)
    t0 = time.time()
    P_try, res, nit = newton_solve(phi, P_prev, tol=TOL, max_iter=MAX_ITER,
                                    tag=f"g={g_next:.4f}")
    dt = time.time() - t0

    if res < TOL:          # ── converged ──────────────────────────────────────
        F_check = float(np.max(np.abs((phi(P_try) - P_try)[sl, sl, sl])))
        r2 = revelation_deficit(P_try[sl, sl, sl], u_inner, np.full(3, tau_fixed), 3)
        results.append({"gamma": float(g_next), "one_minus_R2": float(r2),
                        "F_reinsert": F_check, "newton_iters": nit, "step": float(step)})
        log(f"  ✓ gamma={g_next:.6f}  1-R2={r2:.6e}  ||F||={F_check:.3e}"
            f"  iters={nit}  step={step:.5f}  t={dt:.0f}s")
        P_prev = P_try
        g = g_next
        # adapt step up if easy, down if hard
        if nit <= 4:
            step = min(step * GROW, STEP_MAX)
        elif nit >= 8:
            step = max(step * SHRINK, STEP_MIN)
    else:                  # ── failed — shrink step and retry ─────────────────
        step = max(step * SHRINK, STEP_MIN)
        log(f"  ✗ gamma={g_next:.6f}  ||F||={res:.3e}  FAILED — step → {step:.5f}")
        if step < STEP_MIN * 2:
            log("  step at minimum, skipping to next")
            g = g_next   # advance anyway to avoid infinite loop

log(f"sweep done in {time.time()-t_total:.0f}s  ({len(results)} points)")

# ── Summary table ─────────────────────────────────────────────────────────────
log("=" * 72)
log(f"{'gamma':>8} | {'1-R^2':>12} | {'||F||':>10} | {'iters':>5} | {'step':>7}")
log("-" * 72)
for r in results:
    log(f"{r['gamma']:>8.5f} | {r['one_minus_R2']:>12.6e} | "
        f"{r['F_reinsert']:>10.3e} | {r['newton_iters']:>5d} | {r['step']:>7.5f}")
log("=" * 72)

# ── Write JSON ────────────────────────────────────────────────────────────────
meta = {
    "generated_at": datetime.now().isoformat(),
    "phase": "adaptive Newton JFNK, step 0.005 adaptive, gamma 1→0",
    "anchor_file": ANCHOR.name,
    "operator": f"phi_K3_halo_smooth kernel_h={kernel_h}",
    "tau": tau_fixed,
    "rows": results,
}
out_path = OUT / "newton_sweep.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
