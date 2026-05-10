#!/usr/bin/env python3
"""compute_ferr.py — evaluate ||Phi(P) - P||_inf on every on-disk checkpoint.

For K=3 halo checkpoints (have 'halo' key): calls phi_K3_halo_smooth.
For sym checkpoints (have 'P_sorted' key): calls sym_phi.
Updates TASK_QUEUE.json in place (overwrites F_max only, never decreases).
Idempotent.  Run from repo root.
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "projects/REZN/TASK_QUEUE.json"
CKPT_DIR = ROOT / "projects/REZN/checkpoints"
SEED = ROOT / "results/full_ree/seed_g050_t0200_3d.npz"

REZN_SRC = Path(os.environ.get("REZN_SRC", Path.home() / "rezn-source"))
sys.path.insert(0, str(REZN_SRC))
sys.path.insert(0, str(ROOT / "projects/REZN/solver_code"))

from code.contour_K3_halo import phi_K3_halo_smooth  # type: ignore
from code.halo import extract_inner                   # type: ignore
from contour_KN_sym import SymGrid, sym_phi           # type: ignore


def ferr_halo(npz_path: Path) -> float:
    """Evaluate ||Phi(P) - P||_inf for a K=3 halo checkpoint."""
    d = np.load(npz_path, allow_pickle=True)
    P_full      = d["P_full"].astype(np.float64)
    u_full      = d["u_full"].astype(np.float64)
    pad         = int(d["pad"])
    G_inner     = int(d["G_inner"])
    tau_vec     = d["tau_vec"].astype(np.float64)
    gamma_vec   = d["gamma_vec"].astype(np.float64)
    W_vec       = d["W_vec"].astype(np.float64)

    inner_lo, inner_hi = pad, pad + G_inner
    du = u_full[1] - u_full[0]
    kernel_h = max(0.005, 0.05 * du)

    P_new = phi_K3_halo_smooth(
        P_full, u_full, inner_lo, inner_hi,
        tau_vec, gamma_vec, W_vec, kernel_h,
    )
    P_in     = extract_inner(P_full, inner_lo, inner_hi)
    P_in_new = extract_inner(P_new,  inner_lo, inner_hi)
    return float(np.max(np.abs(P_in_new - P_in)))


def ferr_sym(npz_path: Path) -> float:
    """Evaluate ||Phi(P) - P||_inf for a symmetric-K checkpoint."""
    d = np.load(npz_path, allow_pickle=True)
    P_sorted = d["P_sorted"].astype(np.float64)
    u_grid   = d["u_grid"].astype(np.float64)
    K        = int(d["K"])
    G        = int(d["G"])
    gamma    = float(d["gamma"])
    tau      = float(d["tau"])

    sg = SymGrid.build(G, K)
    # W = equal weights (homogeneous case)
    W = np.ones(K, dtype=np.float64) / K

    residual = sym_phi(P_sorted, sg, u_grid, tau, gamma, W) - P_sorted
    return float(np.max(np.abs(residual)))


def main() -> None:
    q = json.loads(QUEUE.read_text())
    by_id = {t["id"]: t for t in q["tasks"]}

    files = sorted(CKPT_DIR.glob("g*.npz"))
    if SEED.exists():
        files.append(SEED)

    updated = 0
    skipped = 0

    for ckpt in files:
        stem = ckpt.stem
        try:
            d = np.load(ckpt, allow_pickle=True)
        except Exception as e:
            print(f"  SKIP {stem}: unreadable ({e})")
            skipped += 1
            continue

        keys = set(d.files)

        # Determine checkpoint type
        if "P_sorted" in keys:
            kind = "sym"
        elif "P_full" in keys and "halo" in keys:
            kind = "halo"
        else:
            print(f"  SKIP {stem}: unknown format (keys={list(keys)[:6]})")
            skipped += 1
            continue

        print(f"  {stem} [{kind}] ...", end=" ", flush=True)
        try:
            if kind == "halo":
                ferr = ferr_halo(ckpt)
            else:
                ferr = ferr_sym(ckpt)
        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1
            continue

        print(f"||F||inf = {ferr:.4e}")

        if stem not in by_id:
            print(f"    (no queue entry for {stem}, skipping queue update)")
            continue

        t = by_id[stem]
        if t.get("result") is None:
            t["result"] = {}

        old = t["result"].get("F_max")
        # Overwrite with freshly measured value (more accurate than stage_F_inf)
        t["result"]["F_max"] = float(f"{ferr:.6e}")
        if old is None:
            print(f"    -> inserted F_max={ferr:.4e}")
        elif abs(float(old) - ferr) / max(abs(ferr), 1e-300) > 0.01:
            print(f"    -> updated F_max {old} -> {ferr:.4e}")
        updated += 1

    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    have_fmax = sum(1 for t in q["tasks"] if (t.get("result") or {}).get("F_max") is not None)
    print(f"\nDone. Updated {updated} tasks, skipped {skipped}.")
    print(f"Queue now has F_max for {have_fmax}/{len(q['tasks'])} tasks.")


if __name__ == "__main__":
    main()
