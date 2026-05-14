#!/usr/bin/env python3
"""Phase 1 — plain Newton (JFNK, no Anderson) continuation sweep.

Anchor : g100_t0300_G21.npz  (G_inner=21, pad=4, gamma=1.0, tau=3.0)
Operator: phi_K3_halo_smooth  kernel_h=0.025  (the operator the anchor is a
          ~1e-15 fixed point of; built for Newton's quadratic convergence)
Grid   : gamma in [0.2, 5], 20 log-spaced points
Solver : Jacobian-free Newton-Krylov.  Each Newton step solves
             (I - J) delta = F,   F = phi(P) - P
         with GMRES + finite-difference (I-J)*w products.  NO Anderson.
Mode   : machine precision (float64)

For every converged gamma the residual is re-checked by inserting the
solution back into the fixed-point map: ||phi(P) - P||_inf.
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

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

# high-precision anchor inner values for the best float64 start
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

# ── phi operator (smooth, fixed kernel_h) ────────────────────────────────────
def phi_factory(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)
    return fn

# ── JFNK Newton solver (no Anderson) ─────────────────────────────────────────
def newton_solve(phi_fn, P0, tol=1e-12, max_iter=30,
                 gmres_tol=1e-8, gmres_restart=40, gmres_maxiter=20,
                 tag=""):
    P = P0.copy()
    res = float("inf")
    n_iter = 0
    for it in range(max_iter):
        phi_P = phi_fn(P)
        F_full = phi_P - P
        F_in = F_full[sl, sl, sl].ravel()
        res = float(np.max(np.abs(F_in)))
        n_iter = it
        log(f"    {tag} newton it={it:2d}  ||F||={res:.4e}")
        if res < tol:
            break
        normP = np.linalg.norm(P[sl, sl, sl])
        phi_P_in = phi_P[sl, sl, sl].ravel()

        def matvec(w):
            wn = np.linalg.norm(w)
            if wn == 0.0:
                return w
            delta = 1.5e-8 * (1.0 + normP) / wn
            P_pert = P.copy()
            P_pert[sl, sl, sl] += delta * w.reshape(G_inner, G_inner, G_inner)
            Jw = (phi_fn(P_pert)[sl, sl, sl].ravel() - phi_P_in) / delta
            return w - Jw

        A = LinearOperator((n_inner, n_inner), matvec=matvec, dtype=np.float64)
        try:
            delta, info = gmres(A, F_in, rtol=gmres_tol,
                                restart=gmres_restart, maxiter=gmres_maxiter)
        except TypeError:
            delta, info = gmres(A, F_in, tol=gmres_tol,
                                restart=gmres_restart, maxiter=gmres_maxiter)
        P_new = P.copy()
        P_new[sl, sl, sl] += delta.reshape(G_inner, G_inner, G_inner)
        P = np.clip(P_new, 1e-12, 1.0 - 1e-12)
    return P, res, n_iter

# ── gamma grid [0.2, 5], 20 log points ───────────────────────────────────────
gamma_grid = [float(10**x) for x in np.linspace(np.log10(0.2), np.log10(5.0), 20)]
anchor_idx = int(np.argmin([abs(g - anchor_gamma) for g in gamma_grid]))
log(f"Grid: 20 points {gamma_grid[0]:.4f}..{gamma_grid[-1]:.4f}  "
    f"anchor_idx={anchor_idx} (gamma~{gamma_grid[anchor_idx]:.4f})")

# ── Sweep: Newton continuation outward from the anchor ───────────────────────
results = [None] * 20
P_store = [None] * 20
P_store[anchor_idx] = P_anchor.copy()

t_sweep = time.time()
for direction, idxs in [("→", range(anchor_idx, 20)),
                        ("←", range(anchor_idx - 1, -1, -1))]:
    P_prev = P_anchor.copy()
    for idx in idxs:
        if results[idx] is not None:
            P_prev = P_store[idx]
            continue
        g = gamma_grid[idx]
        t0 = time.time()
        log(f"  {direction} gamma={g:.4f} (idx={idx})  [Newton from neighbour]")
        phi = phi_factory(g)
        P, res, nit = newton_solve(phi, P_prev, tag=f"g={g:.3f}")
        # residual check: re-insert into the fixed-point map
        F_check = float(np.max(np.abs((phi(P) - P)[sl, sl, sl])))
        r2 = revelation_deficit(P[sl, sl, sl], u_inner, np.full(3, tau_fixed), 3)
        dt = time.time() - t0
        results[idx] = {"gamma": float(g), "one_minus_R2": float(r2),
                        "F_reinsert": F_check, "newton_iters": nit, "wall_s": dt}
        P_store[idx] = P
        P_prev = P
        log(f"  {direction} gamma={g:.4f}  1-R2={r2:.6e}  "
            f"||F||_reinsert={F_check:.3e}  iters={nit}  t={dt:.0f}s")

log(f"sweep done in {time.time()-t_sweep:.0f}s")

# ── Results table ────────────────────────────────────────────────────────────
log("=" * 72)
log(f"{'gamma':>10} | {'1-R^2':>14} | {'||F|| reinsert':>16} | {'iters':>6} | {'t(s)':>6}")
log("-" * 72)
for r in results:
    log(f"{r['gamma']:>10.4f} | {r['one_minus_R2']:>14.6e} | "
        f"{r['F_reinsert']:>16.3e} | {r['newton_iters']:>6d} | {r['wall_s']:>6.0f}")
log("=" * 72)

# ── Write JSON ───────────────────────────────────────────────────────────────
meta = {
    "generated_at": datetime.now().isoformat(),
    "phase": "1 — plain Newton (JFNK) continuation, no Anderson",
    "anchor_file": ANCHOR.name,
    "operator": f"phi_K3_halo_smooth kernel_h={kernel_h}",
    "grid": f"G_inner={G_inner} pad={pad} gamma[0.2,5] 20pts",
    "tau": tau_fixed,
    "rows": results,
}
out_path = OUT / "newton_sweep.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
