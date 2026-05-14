#!/usr/bin/env python3
"""RK4 ODE sweep — 50-point gamma grid [0.1, 1000], f64-only."""
import os, sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np

REPO     = Path("/home/user/FIXED-POINT-FACTORY")
REZN_SRC = Path("/home/user/rezn-src")
CKPT     = REPO / "projects/REZN/checkpoints"
OUT      = REPO / "projects/REZN/overnight"
SOLVER   = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SOLVER))
sys.path.insert(0, str(REZN_SRC))

from mpmath import mp, mpf
from phi_mp import phi_K3_smooth_mp
from code.contour_K3_halo import phi_K3_halo_smooth
from code.metrics import revelation_deficit
from ode_sweep_rk4 import solve_sweep_rk4

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Anchor ───────────────────────────────────────────────────────────────────
ANCHOR_FILE = CKPT / "g025_t0200.npz"
log(f"Anchor: {ANCHOR_FILE.name}")
d = np.load(ANCHOR_FILE, allow_pickle=True)

G_inner  = int(d["G_inner"])
pad      = int(d["pad"])
inner_lo = pad
inner_hi = pad + G_inner
P_anchor = d["P_full"].astype(np.float64)
u_anc    = d["u_full"].astype(np.float64)
tau_anc  = d["tau_vec"].astype(np.float64)
W_anc    = d["W_vec"].astype(np.float64)
gamma_v  = d["gamma_vec"].astype(np.float64)

du         = float(u_anc[1] - u_anc[0])
kernel_h   = max(0.005, 0.05 * du)
anchor_gamma = float(gamma_v[0])
tau_fixed    = float(tau_anc[0])

mp.dps = 50
if "P_inner_mp_str" in d:
    P_inner_str = d["P_inner_mp_str"]
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_anchor[inner_lo+i, inner_lo+j, inner_lo+l] = \
                    float(mp.mpf(str(P_inner_str[i, j, l])))

log(f"G_inner={G_inner}  tau={tau_fixed}  gamma={anchor_gamma}  kernel_h={kernel_h:.4f}")

# ── 50-point gamma grid [0.1, 1000] ─────────────────────────────────────────
gamma_grid = [float(10**x) for x in np.linspace(np.log10(0.10), np.log10(1000.0), 50)]
anchor_idx = int(np.argmin([abs(g - anchor_gamma) for g in gamma_grid]))
log(f"Grid: {[round(g,3) for g in gamma_grid]}")
log(f"Anchor idx={anchor_idx}  gamma≈{gamma_grid[anchor_idx]:.3f}")

# ── Phi factories ────────────────────────────────────────────────────────────
def phi_f64_factory(gamma_scalar):
    gv = np.full(3, gamma_scalar)
    def phi_fn(P):
        return phi_K3_halo_smooth(P, u_anc, inner_lo, inner_hi,
                                  tau_anc, gv, W_anc, kernel_h)
    return phi_fn

def phi_mp_factory(gamma_scalar):
    gv_mp  = [mp.mpf(str(gamma_scalar))] * 3
    tau_mp = [mp.mpf(str(float(t))) for t in tau_anc]
    W_mp   = [mp.mpf(str(float(w))) for w in W_anc]
    h_mp   = mp.mpf(str(kernel_h))
    def phi_mp_fn(P_mp):
        u_mp = [mp.mpf(str(float(u))) for u in u_anc]
        return phi_K3_smooth_mp(mp, P_mp, u_mp,
                                inner_lo, inner_hi,
                                tau_mp, gv_mp, W_mp, h_mp)
    return phi_mp_fn

# ── Run RK4 sweep ────────────────────────────────────────────────────────────
log("=== RK4 sweep: pure ODE mode (no corrector), 50-point grid [0.1, 1000] ===")
mp.dps = 50
t0 = time.time()
sweep = solve_sweep_rk4(
    phi_f64_fn        = phi_f64_factory,
    phi_mp_fn_factory = phi_mp_factory,
    mp                = mp,
    gamma_grid        = gamma_grid,
    anchor_idx        = anchor_idx,
    P_anchor_full     = P_anchor,
    inner_lo          = inner_lo,
    inner_hi          = inner_hi,
    mp_dps            = 50,
    target_eps        = mpf("1e-40"),
    eps_gamma         = 1e-5,
    gmres_tol         = 1e-4,
    gmres_restart     = 30,
    gmres_maxiter     = 5,
    f64_tol           = 1e-12,
    corrector_max_iter= 0,        # pure ODE mode: accept RK4 prediction directly
    anderson_m        = 5,
    mp_max_iter       = 0,
    verbose           = True,
)
log(f"RK4 sweep done in {time.time()-t0:.0f}s")

# ── 1-R^2 ────────────────────────────────────────────────────────────────────
log("=== 1-R^2 ===")
rows = []
for idx, (g, P_full) in enumerate(zip(sweep["gamma_grid"], sweep["P_outputs"])):
    if P_full is None:
        rows.append({"gamma": float(g), "error": "no solution"})
        log(f"  gamma={g:.4f}  NO SOLUTION")
        continue
    try:
        u_inner = u_anc[inner_lo:inner_hi]
        P_inner = P_full[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
        r2   = revelation_deficit(P_inner, u_inner, np.full(3, tau_fixed), 3)
        F_f64= sweep["F_f64"][idx]
        rows.append({"gamma": float(g), "one_minus_R2": float(r2), "F_f64": F_f64})
        log(f"  gamma={g:.4f}  1-R2={r2:.5f}  F_f64={F_f64:.2e}")
    except Exception as e:
        rows.append({"gamma": float(g), "error": str(e)[:80]})
        log(f"  gamma={g:.4f}  ERROR: {e}")

# ── Write deficits.json ───────────────────────────────────────────────────────
meta = {
    "generated_at": datetime.now().isoformat(),
    "tau":          tau_fixed,
    "anchor_gamma": anchor_gamma,
    "anchor_file":  ANCHOR_FILE.name,
    "passes":       {"A": rows},
}
out_path = OUT / "deficits.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
