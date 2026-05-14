#!/usr/bin/env python3
"""Gamma sweep with the EXACT phi_K3_halo_cubic operator.

Replaces the earlier phi_K3_halo_smooth sweeps: the Gaussian-kernel
smoothing in that variant shifts the fixed point and inflates 1-R^2 by
~1000x (spurious ~0.08 plateau). The exact Hermite-cubic root-find
operator gives the true REE, with 1-R^2 ~ 0.

Grid: G_inner=17, pad=4 (G_full=25), inner u in [-5, 5], tau=2.0, K=3.
Grid built directly per staggered_run_K3.py, not loaded from a checkpoint.

Each gamma is solved with Anderson acceleration: the halo is rebuilt
no-learning for that gamma, and the inner region is warm-started from
the previous gamma's solution. Every point is verified by its own |F|.
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np

REPO     = Path("/home/user/FIXED-POINT-FACTORY")
REZN_SRC = Path("/home/user/rezn-src")
OUT      = REPO / "projects/REZN/overnight"
SOLVER   = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SOLVER))
sys.path.insert(0, str(REZN_SRC))

from code.contour_K3_halo import phi_K3_halo_cubic, init_no_learning_K3
from code.metrics import revelation_deficit
from ode_sweep import anderson_solve

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Grid: G_inner=17, pad=4, u_max=5 (per staggered_run_K3.py) ────────────────
G_inner    = 17
pad        = 4
u_inner_max = 5.0
G_full     = G_inner + 2 * pad
inner_lo   = pad
inner_hi   = pad + G_inner
du         = (2.0 * u_inner_max) / (G_inner - 1)
u_full     = np.array([-u_inner_max + (q - pad) * du for q in range(G_full)],
                      dtype=np.float64)
u_inner    = u_full[inner_lo:inner_hi]

K          = 3
tau_fixed  = 2.0
W_fixed    = 1.0
tau        = np.full(K, tau_fixed)
W          = np.full(K, W_fixed)

log(f"G_inner={G_inner} pad={pad} G_full={G_full}  du={du:.4f}")
log(f"inner u in [{u_full[inner_lo]:.3f}, {u_full[inner_hi-1]:.3f}]  "
    f"full u in [{u_full[0]:.3f}, {u_full[-1]:.3f}]")

# ── 50-point gamma grid [0.1, 1000] ──────────────────────────────────────────
gamma_grid = [float(10**x) for x in np.linspace(np.log10(0.10), np.log10(1000.0), 50)]
log(f"Grid: {len(gamma_grid)} points, {gamma_grid[0]:.3f} .. {gamma_grid[-1]:.1f}")

def phi_factory(g):
    gv = np.full(K, g)
    def fn(P):
        return phi_K3_halo_cubic(P, u_full, inner_lo, inner_hi, tau, gv, W)
    return fn

# ── Sweep: per-gamma no-learning halo + warm-start inner ─────────────────────
log("=== cubic (exact) sweep: anderson, warm-start inner ===")
t0 = time.time()
rows = []
P_inner_prev = None
for idx, g in enumerate(gamma_grid):
    gv = np.full(K, g)
    P0 = init_no_learning_K3(u_full, tau, gv, W)        # correct halo for this gamma
    if P_inner_prev is not None:
        P0[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi] = P_inner_prev
    phi = phi_factory(g)
    P, res = anderson_solve(phi, P0, tol=1e-11, max_iter=600, m=5, verbose=False)
    P_inner = P[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
    P_inner_prev = P_inner
    try:
        r2 = revelation_deficit(P_inner, u_inner, np.full(K, tau_fixed), K)
    except Exception:
        r2 = float("nan")
    rows.append({"gamma": float(g), "one_minus_R2": float(r2), "F_f64": float(res)})
    log(f"  [{idx+1:2d}/50] gamma={g:9.4f}  1-R2={r2:.6e}  F={res:.2e}  t={time.time()-t0:.0f}s")

log(f"sweep done in {time.time()-t0:.0f}s")

# ── Write deficits.json ──────────────────────────────────────────────────────
meta = {
    "generated_at": datetime.now().isoformat(),
    "tau":          tau_fixed,
    "operator":     "phi_K3_halo_cubic (exact Hermite-cubic root-find)",
    "grid":         f"G_inner={G_inner} pad={pad} u_max={u_inner_max}",
    "passes":       {"A": rows},
}
out_path = OUT / "deficits.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
