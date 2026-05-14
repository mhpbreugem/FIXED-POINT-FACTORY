#!/usr/bin/env python3
"""
run_sweep.py — Overnight ODE-sweep for GitHub Actions.

Workflow:
  1. Inventory G=17 checkpoints, verify fresh float64 residual.
  2. Verify mp precision at dps=50.
  3. Pick middle anchor.
  4. Gamma-sweep with predictor-corrector (Anderson f64 + mp Newton polish),
     escalating precision: pass A dps=50, B dps=100, C dps=200.
  5. Compute 1-R^2 per point.
  6. Write deficits.json, REPORT.md.

Idempotent: skips passes whose pickle already exists.

Environment variables:
  GITHUB_WORKSPACE   repo root (default: cwd)
  REZN_SRC           path to REZN clone (default: ~/rezn-source, auto-cloned)
  SWEEP_TAU          optional fixed tau override (float)
  SWEEP_PASSES       comma-separated passes to run: A,B,C (default: A,B,C)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────────
REPO      = Path(os.environ.get("GITHUB_WORKSPACE", str(Path.cwd())))
REZN_SRC  = Path(os.environ.get("REZN_SRC", str(Path.home() / "rezn-source")))
CKPT      = REPO / "projects/REZN/checkpoints"
OUT       = REPO / "projects/REZN/overnight"
OUT.mkdir(parents=True, exist_ok=True)
SOLVER    = REPO / "projects/REZN/solver_code"

# Auto-clone REZN if missing
if not REZN_SRC.exists():
    print(f"[sweep] cloning REZN to {REZN_SRC} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/mhpbreugem/REZN.git", str(REZN_SRC)],
        check=True,
    )

sys.path.insert(0, str(SOLVER))
sys.path.insert(0, str(REZN_SRC))

# ── Logging ──────────────────────────────────────────────────────────────────────
LOG = OUT / "sweep.log"

def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")

def section(title):
    log(""); log("=" * 70); log(title); log("=" * 70)

# ── Imports ────────────────────────────────────────────────────────────────────────
section("Importing modules")

from mpmath import mp, mpf
from phi_mp import phi_K3_smooth_mp
from code.contour_K3_halo import phi_K3_halo_smooth, init_no_learning_K3
from code.metrics import revelation_deficit
from ode_sweep import solve_sweep

log("All imports OK")

# ── Args ───────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--passes", default=os.environ.get("SWEEP_PASSES", "A,B,C"))
parser.add_argument("--tau",    default=os.environ.get("SWEEP_TAU",   None), type=float)
args = parser.parse_args()
PASSES_WANTED = [p.strip() for p in args.passes.split(",")]

# ── Phi factories ─────────────────────────────────────────────────────────────────────
def _phi_f64_factory(u_full, inner_lo, inner_hi, tau_vec, W_vec, kernel_h):
    def make(gamma_scalar):
        gv = np.full(3, gamma_scalar)
        def phi_fn(P):
            return phi_K3_halo_smooth(P, u_full, inner_lo, inner_hi,
                                      tau_vec, gv, W_vec, kernel_h)
        return phi_fn
    return make

def _phi_mp_factory(u_full, inner_lo, inner_hi, tau_vec, W_vec, kernel_h):
    def make(gamma_scalar):
        gv_mp  = [mp.mpf(str(gamma_scalar))] * 3
        tau_mp = [mp.mpf(str(float(t))) for t in tau_vec]
        W_mp   = [mp.mpf(str(float(w))) for w in W_vec]
        h_mp   = mp.mpf(str(float(kernel_h)))
        def phi_mp_fn(P_mp):
            u_mp = [mp.mpf(str(float(u))) for u in u_full]
            return phi_K3_smooth_mp(mp, P_mp, u_mp,
                                    inner_lo, inner_hi,
                                    tau_mp, gv_mp, W_mp, h_mp)
        return phi_mp_fn
    return make

# ── Step 1: Checkpoint inventory ───────────────────────────────────────────────────────
section("STEP 1: Checkpoint inventory + fresh float64 residual")

G_TARGET = 17

inventory = []
files = sorted(CKPT.glob("*.npz"))
log(f"Found {len(files)} .npz files in {CKPT}")

for f in files:
    rec = {"file": str(f), "name": f.name}
    try:
        d = np.load(f, allow_pickle=True)
        G_inner = int(d["G_inner"]) if "G_inner" in d else None
        rec["G_inner"] = G_inner
        if G_inner not in (16, 17):
            rec["skip"] = f"G_inner={G_inner}"
            inventory.append(rec)
            continue

        pad         = int(d["pad"])
        gamma_v     = d["gamma_vec"].astype(np.float64)
        tau_v       = d["tau_vec"].astype(np.float64)
        W_v         = d["W_vec"].astype(np.float64)
        P_full      = d["P_full"].astype(np.float64)
        u_full      = d["u_full"].astype(np.float64)
        inner_lo    = pad
        inner_hi    = pad + G_inner
        gamma_scalar = float(gamma_v[0])
        tau_scalar   = float(tau_v[0])

        rec.update({"gamma": gamma_scalar, "tau": tau_scalar,
                    "pad": pad, "inner_lo": inner_lo, "inner_hi": inner_hi})

        du = float(u_full[1] - u_full[0]) if len(u_full) > 1 else 0.1
        kernel_h = max(0.005, 0.05 * du)
        rec["kernel_h"] = kernel_h

        Phi_f64  = phi_K3_halo_smooth(P_full, u_full, inner_lo, inner_hi,
                                       tau_v, gamma_v.copy(), W_v, kernel_h)
        P_inner  = P_full[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
        Ph_inner = Phi_f64[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
        res_f64  = float(np.max(np.abs(Ph_inner - P_inner)))
        rec["fresh_F_f64"] = res_f64

        stage_arr = d["stage_F_inf"] if "stage_F_inf" in d else np.array([float("nan")])
        rec["stage_F_min"] = float(np.nanmin(stage_arr))

        log(f"  {f.name}  G={G_inner}  γ={gamma_scalar:.3f}  τ={tau_scalar:.2f}"
            f"  f64_F={res_f64:.2e}  stage_F={rec['stage_F_min']:.2e}")

        rec["has_mp_str"] = "P_inner_mp_str" in d
        inventory.append(rec)

    except Exception as e:
        rec["error"] = repr(e)[:200]
        inventory.append(rec)
        log(f"  {f.name}: ERROR {e}")

# ── Step 2: Classify solved ────────────────────────────────────────────────────────────────────
section("STEP 2: Verify mp precision + classify solved")

mp.dps = 50
SOLVED_TOL_MP  = 1e-40
USABLE_TOL_F64 = 2e-4

solved_mp  = []
usable_f64 = []

for rec in inventory:
    if rec.get("skip") or rec.get("error"):
        continue
    if not rec.get("has_mp_str"):
        if rec.get("fresh_F_f64", 1.0) < USABLE_TOL_F64:
            usable_f64.append(rec)
        continue

    f = Path(rec["file"])
    try:
        d        = np.load(f, allow_pickle=True)
        P_full   = d["P_full"].astype(np.float64)
        u_full   = d["u_full"].astype(np.float64)
        inner_lo = rec["inner_lo"]; inner_hi = rec["inner_hi"]
        tau_v    = d["tau_vec"].astype(np.float64)
        gamma_v  = d["gamma_vec"].astype(np.float64)
        W_v      = d["W_vec"].astype(np.float64)
        kernel_h = rec["kernel_h"]
        G_inner  = rec["G_inner"]; G_full = P_full.shape[0]

        P_inner_str = d["P_inner_mp_str"]
        P_mp = [[[mp.mpf(str(P_full[i, j, l]))
                  for l in range(G_full)] for j in range(G_full)]
                for i in range(G_full)]
        for i in range(G_inner):
            for j in range(G_inner):
                for l in range(G_inner):
                    P_mp[inner_lo+i][inner_lo+j][inner_lo+l] = mp.mpf(
                        str(P_inner_str[i, j, l]))

        u_mp   = [mp.mpf(str(float(u))) for u in u_full]
        gv_mp  = [mp.mpf(str(float(g))) for g in gamma_v]
        tau_mp = [mp.mpf(str(float(t))) for t in tau_v]
        W_mp   = [mp.mpf(str(float(w))) for w in W_v]
        h_mp   = mp.mpf(str(kernel_h))

        Phi_mp   = phi_K3_smooth_mp(mp, P_mp, u_mp, inner_lo, inner_hi,
                                     tau_mp, gv_mp, W_mp, h_mp)
        max_diff = mp.mpf(0)
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    d2 = abs(Phi_mp[i][j][l] - P_mp[i][j][l])
                    if d2 > max_diff:
                        max_diff = d2
        res_mp = float(max_diff)
        rec["fresh_F_mp50"] = res_mp
        log(f"  {rec['name']}  mp.dps=50  ||F||={res_mp:.3e}")

        if res_mp < SOLVED_TOL_MP:
            rec["P_full"]  = P_full
            rec["u_full"]  = u_full
            rec["tau_vec"] = tau_v
            rec["W_vec"]   = W_v
            solved_mp.append(rec)
        elif rec.get("fresh_F_f64", 1.0) < USABLE_TOL_F64:
            usable_f64.append(rec)

    except Exception as e:
        log(f"  {rec['name']}: mp verify ERROR {e}")
        traceback.print_exc()
        if rec.get("fresh_F_f64", 1.0) < USABLE_TOL_F64:
            usable_f64.append(rec)

log(f"\nMP-solved  (||F||<{SOLVED_TOL_MP}): {len(solved_mp)}")
log(f"F64-usable (||F||<{USABLE_TOL_F64}): {len(usable_f64)}")

candidates = solved_mp if solved_mp else usable_f64
if not candidates:
    log("NO usable checkpoints. Writing failure report.")
    with open(OUT / "REPORT.md", "w") as fh:
        fh.write("# Overnight sweep — no usable checkpoints\n\n")
        fh.write(f"Checked {len(inventory)} checkpoints.\n")
    with open(OUT / "deficits.json", "w") as fh:
        json.dump({}, fh)
    sys.exit(0)

# ── Step 3: Anchor ───────────────────────────────────────────────────────────────────────────
section("STEP 3: Pick anchor")

# Filter by tau if requested
if args.tau is not None:
    subset = [r for r in candidates if abs(r.get("tau", 0) - args.tau) < 0.01]
    if subset:
        candidates = subset
    else:
        log(f"WARNING: no checkpoint at tau={args.tau}, using all candidates")

candidates.sort(key=lambda r: (r.get("tau", 0), r.get("gamma", 0)))
anchor = candidates[len(candidates) // 2]
log(f"ANCHOR: {anchor['name']}  γ={anchor.get('gamma')}  τ={anchor.get('tau')}")

d_anc        = np.load(anchor["file"], allow_pickle=True)
P_anchor     = d_anc["P_full"].astype(np.float64)
u_anc        = d_anc["u_full"].astype(np.float64)
tau_anc      = d_anc["tau_vec"].astype(np.float64)
W_anc        = d_anc["W_vec"].astype(np.float64)
G_anc        = anchor["G_inner"]; pad_anc = anchor["pad"]
inner_lo_anc = pad_anc; inner_hi_anc = pad_anc + G_anc
kernel_h_anc = anchor["kernel_h"]
tau_fixed    = anchor.get("tau", 2.0)
anchor_gamma = anchor.get("gamma", 1.0)

if anchor.get("has_mp_str"):
    P_inner_str = d_anc["P_inner_mp_str"]
    for i in range(G_anc):
        for j in range(G_anc):
            for l in range(G_anc):
                P_anchor[inner_lo_anc+i, inner_lo_anc+j, inner_lo_anc+l] = \
                    float(mp.mpf(str(P_inner_str[i, j, l])))

gamma_grid = [float(10 ** x) for x in np.linspace(np.log10(0.05), np.log10(5.0), 21)]
anchor_idx = int(np.argmin([abs(g - anchor_gamma) for g in gamma_grid]))
log(f"Sweep: τ={tau_fixed}  anchor_idx={anchor_idx}  γ≈{gamma_grid[anchor_idx]:.3f}")

# ── Step 4: Sweep ───────────────────────────────────────────────────────────────────────────
section("STEP 4: Gamma sweep, escalating precision")

f64_factory = _phi_f64_factory(u_anc, inner_lo_anc, inner_hi_anc,
                                tau_anc, W_anc, kernel_h_anc)
mp_factory  = _phi_mp_factory(u_anc, inner_lo_anc, inner_hi_anc,
                               tau_anc, W_anc, kernel_h_anc)

PASS_CONFIGS = {
    "A": {"dps":  50, "eps": mpf("1e-40")},
    "B": {"dps": 100, "eps": mpf("1e-80")},
    "C": {"dps": 200, "eps": mpf("1e-150")},
}

results = {}
for pname in ["A", "B", "C"]:
    if pname not in PASSES_WANTED:
        continue
    p = PASS_CONFIGS[pname]
    section(f"PASS {pname}: dps={p['dps']} target_eps={p['eps']}")
    pkl_path = OUT / f"sweep_pass{pname}.pkl"
    if pkl_path.exists():
        log(f"  Exists — loading {pkl_path}")
        with open(pkl_path, "rb") as fh:
            results[pname] = pickle.load(fh)
        continue

    mp.dps = p["dps"]
    t0 = time.time()
    try:
        sweep = solve_sweep(
            phi_f64_fn        = f64_factory,
            phi_mp_fn_factory = mp_factory,
            mp                = mp,
            gamma_grid        = gamma_grid,
            anchor_idx        = anchor_idx,
            P_anchor_full     = P_anchor,
            inner_lo          = inner_lo_anc,
            inner_hi          = inner_hi_anc,
            mp_dps            = p["dps"],
            target_eps        = p["eps"],
            f64_tol           = 5e-7,
            f64_max_iter      = 500,
            anderson_m        = 7,
            mp_max_iter       = 25,
            verbose           = True,
        )
        elapsed = time.time() - t0
        log(f"  PASS {pname} done in {elapsed:.0f}s")
        results[pname] = sweep
        with open(pkl_path, "wb") as fh:
            pickle.dump(sweep, fh)
    except Exception as e:
        log(f"  PASS {pname} FAILED: {e}")
        log(traceback.format_exc())
        break

# ── Step 5: 1-R^2 ─────────────────────────────────────────────────────────────────────────────
section("STEP 5: Compute 1-R^2 per gamma point")

deficits = {}
tau_vec_3 = np.full(3, tau_fixed)
K = 3
for pname, sweep in results.items():
    gamma_list = sweep["gamma_grid"]
    rows = []
    for idx, (g, P_full) in enumerate(zip(gamma_list, sweep["P_outputs"])):
        if P_full is None:
            rows.append({"gamma": float(g), "error": "no solution"})
            continue
        try:
            u_grid_inner = u_anc[inner_lo_anc:inner_hi_anc]
            P_inner = P_full[inner_lo_anc:inner_hi_anc,
                             inner_lo_anc:inner_hi_anc,
                             inner_lo_anc:inner_hi_anc]
            r2 = revelation_deficit(P_inner, u_grid_inner, tau_vec_3, K)
            rows.append({
                "gamma":       float(g),
                "one_minus_R2": float(r2),
                "F_f64":       sweep["F_f64"][idx],
                "F_mp":        sweep["F_mp"][idx],
            })
        except Exception as e:
            rows.append({"gamma": float(g), "error": str(e)[:120]})
    deficits[pname] = rows
    n_ok = sum(1 for r in rows if "one_minus_R2" in r)
    log(f"  PASS {pname}: 1-R^2 computed for {n_ok}/{len(rows)} points")

# Attach metadata for the website tab
meta = {
    "generated_at": datetime.now().isoformat(),
    "tau":          tau_fixed,
    "anchor_gamma": anchor_gamma,
    "anchor_file":  anchor["name"],
    "passes":       deficits,
}
with open(OUT / "deficits.json", "w") as fh:
    json.dump(meta, fh, indent=2)
log(f"Written {OUT / 'deficits.json'}")

# ── Step 6: REPORT ─────────────────────────────────────────────────────────────────────────────
section("STEP 6: REPORT")

with open(OUT / "REPORT.md", "w") as fh:
    fh.write(f"# ODE-sweep — {datetime.now():%Y-%m-%d %H:%M}\n\n")
    fh.write(f"## Setup\n\n")
    fh.write(f"- Anchor: `{anchor['name']}`  γ={anchor_gamma}  τ={tau_fixed}\n")
    fh.write(f"- Grid: 21 log-spaced γ in [{gamma_grid[0]:.3f}, {gamma_grid[-1]:.3f}]\n")
    fh.write(f"- Passes run: {list(results.keys())}\n\n")
    for pname, sweep in results.items():
        F_mp_vals = [v for v in sweep.get("F_mp", []) if v == v]
        fh.write(f"### Pass {pname}\n")
        if F_mp_vals:
            fh.write(f"- max mp ||F||: {max(F_mp_vals):.3e}\n")
            fh.write(f"- min mp ||F||: {min(F_mp_vals):.3e}\n")
        if pname in deficits:
            fh.write(f"\n| γ | 1-R² | F_mp |\n|---|------|------|\n")
            for r in deficits[pname]:
                if "one_minus_R2" in r:
                    fh.write(f"| {r['gamma']:.4f} | {r['one_minus_R2']:.6f}"
                             f" | {r.get('F_mp','?'):.2e} |\n")
        fh.write("\n")

log("DONE. See deficits.json and REPORT.md")
