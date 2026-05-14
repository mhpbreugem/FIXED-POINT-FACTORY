#!/usr/bin/env python3
"""ODE continuation sweep at RK orders 1 / 2 / 4 — comparison run.

Anchor : g100_t0300_G21.npz  (G_inner=21, pad=4, gamma=1.0, tau=3.0)
Operator: phi_K3_halo_cubic  (EXACT Hermite-cubic root-find — the smooth
          kernel variant is biased, see deficits investigation)
Grid   : gamma in [0.1, 50], 25 log-spaced points
Mode   : machine precision (float64), mp polish skipped

For each RK order the predictor advances the tangent ODE
    (I - J) v = d phi / d gamma
by explicit Euler (1) / midpoint (2) / classic RK4 (4); an Anderson
corrector then snaps back toward the manifold.  NOTE: the exact cubic
operator has root-find kinks, so Anderson stalls around |F| ~ 1e-3 — but
the 1-R^2 regression diagnostic is robust to that high-frequency residual
(smoke test: |F|=7e-3 still gave 1-R^2 ~ 1e-6).

Outputs projects/REZN/overnight/deficits_orders.json with all three orders
plus a per-gamma comparison.
"""
import sys, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import mpmath as mp

REPO     = Path("/home/user/FIXED-POINT-FACTORY")
REZN_SRC = Path("/home/user/rezn-src")
CKPT     = REPO / "projects/REZN/checkpoints"
OUT      = REPO / "projects/REZN/overnight"
SOLVER   = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SOLVER))
sys.path.insert(0, str(REZN_SRC))

from code.contour_K3_halo import phi_K3_halo_cubic
from code.metrics import revelation_deficit
from ode_sweep_rk4 import solve_sweep_rk4

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Anchor: G_inner=21, gamma=1.0, tau=3.0 ───────────────────────────────────
ANCHOR_FILE = CKPT / "g100_t0300_G21.npz"
d = np.load(ANCHOR_FILE, allow_pickle=True)
G_inner  = int(d["G_inner"])
pad      = int(d["pad"])
inner_lo = pad
inner_hi = pad + G_inner
P_anchor = d["P_full"].astype(np.float64)
u_full   = d["u_full"].astype(np.float64)
tau      = d["tau_vec"].astype(np.float64)
W        = d["W_vec"].astype(np.float64)
anchor_gamma = float(d["gamma_vec"][0])
tau_fixed    = float(tau[0])
u_inner  = u_full[inner_lo:inner_hi]

# high-precision anchor inner values
mp.mp.dps = 50
if "P_inner_mp_str" in d.files:
    s = d["P_inner_mp_str"]
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_anchor[inner_lo+i, inner_lo+j, inner_lo+l] = float(mp.mpf(str(s[i, j, l])))

log(f"Anchor {ANCHOR_FILE.name}: G_inner={G_inner} pad={pad} "
    f"gamma={anchor_gamma} tau={tau_fixed}  inner u[{u_inner.min():.2f},{u_inner.max():.2f}]")

# ── Gamma grid [0.1, 50], 25 log points ──────────────────────────────────────
gamma_grid = [float(10**x) for x in np.linspace(np.log10(0.10), np.log10(50.0), 25)]
anchor_idx = int(np.argmin([abs(g - anchor_gamma) for g in gamma_grid]))
log(f"Grid: 25 points {gamma_grid[0]:.3f}..{gamma_grid[-1]:.1f}  "
    f"anchor_idx={anchor_idx} (gamma~{gamma_grid[anchor_idx]:.3f})")

# ── phi factory (exact cubic operator) ───────────────────────────────────────
def phi_f64_factory(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_cubic(P, u_full, inner_lo, inner_hi, tau, gv, W)
    return fn

def phi_mp_dummy(g):          # unused (mp_max_iter=0)
    return None

# ── Run sweep for orders 1, 2, 4 ─────────────────────────────────────────────
orders = [1, 2, 4]
results = {}
for order in orders:
    log(f"================  RK ORDER {order}  ================")
    t0 = time.time()
    sweep = solve_sweep_rk4(
        phi_f64_fn        = phi_f64_factory,
        phi_mp_fn_factory = phi_mp_dummy,
        mp                = mp.mp,
        gamma_grid        = gamma_grid,
        anchor_idx        = anchor_idx,
        P_anchor_full     = P_anchor,
        inner_lo          = inner_lo,
        inner_hi          = inner_hi,
        mp_dps            = 50,
        target_eps        = mp.mpf("1e-30"),
        order             = order,
        eps_gamma         = 1e-4,
        gmres_tol         = 1e-3,
        gmres_restart     = 20,
        gmres_maxiter     = 3,
        f64_tol           = 1e-10,
        corrector_max_iter= 150,
        anderson_m        = 5,
        mp_max_iter       = 0,
        verbose           = True,
    )
    dt = time.time() - t0
    rows = []
    for idx, (g, P_full) in enumerate(zip(sweep["gamma_grid"], sweep["P_outputs"])):
        if P_full is None:
            rows.append({"gamma": float(g), "error": "no solution"})
            continue
        P_in = P_full[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
        try:
            r2 = revelation_deficit(P_in, u_inner, np.full(3, tau_fixed), 3)
        except Exception as e:
            r2 = float("nan")
        rows.append({"gamma": float(g),
                     "one_minus_R2": float(r2),
                     "F_f64": float(sweep["F_f64"][idx])})
    results[order] = {"rows": rows, "wall_s": dt}
    log(f"order {order} done in {dt:.0f}s")
    for r in rows:
        if "one_minus_R2" in r:
            log(f"   gamma={r['gamma']:9.4f}  1-R2={r['one_minus_R2']:.4e}  F={r['F_f64']:.2e}")

# ── Comparison + write JSON ──────────────────────────────────────────────────
comparison = []
for i, g in enumerate(gamma_grid):
    row = {"gamma": float(g)}
    for o in orders:
        r = results[o]["rows"][i]
        row[f"order{o}_1mR2"] = r.get("one_minus_R2", None)
        row[f"order{o}_F"]    = r.get("F_f64", None)
    vals = [row[f"order{o}_1mR2"] for o in orders if row.get(f"order{o}_1mR2") is not None]
    row["spread_1mR2"] = (max(vals) - min(vals)) if len(vals) > 1 else None
    comparison.append(row)

meta = {
    "generated_at": datetime.now().isoformat(),
    "anchor_file":  ANCHOR_FILE.name,
    "operator":     "phi_K3_halo_cubic (exact)",
    "grid":         f"G_inner={G_inner} pad={pad} gamma[0.1,50] 25pts",
    "tau":          tau_fixed,
    "note":         "machine precision; Anderson stalls ~1e-3 on exact cubic "
                    "(root-find kinks) but 1-R^2 is robust to that residual",
    "orders":       {str(o): results[o]["rows"] for o in orders},
    "wall_s":       {str(o): results[o]["wall_s"] for o in orders},
    "comparison":   comparison,
}
out_path = OUT / "deficits_orders.json"
with open(out_path, "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {out_path}")
log("DONE")
