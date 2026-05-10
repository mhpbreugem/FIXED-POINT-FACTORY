#!/usr/bin/env python3
"""compute_ferr_mp.py — evaluate ||Phi(P)-P||_inf at mpmath precision (dps=200).

For each .npz K=3 halo checkpoint:
  1. Load P_full, convert to mp.mpf.
  2. Call phi_K3_smooth_mp (same phi as the solver, in mpmath).
  3. Compute F = ||phi(P)-P||_inf in mpmath.
  4. Store result['F_max'] in the queue.
  5. If F < 1e-100: mark task 'done' — already at global precision target.
     If F >= 1e-100: leave as 'ready' — workers will polish from checkpoint.

Run from repo root.  Slow (one mpmath phi call per checkpoint, ~1-5 min each).
"""
import json
import sys
import os
import time
from pathlib import Path
from datetime import timezone, datetime

import numpy as np

ROOT = Path(__file__).resolve().parent
QUEUE   = ROOT / "projects/REZN/TASK_QUEUE.json"
CKPT_DIR = ROOT / "projects/REZN/checkpoints"
SEED    = ROOT / "results/full_ree/seed_g050_t0200_3d.npz"

REZN_SRC = Path(os.environ.get("REZN_SRC", Path.home() / "rezn-source"))
sys.path.insert(0, str(REZN_SRC))
sys.path.insert(0, str(ROOT / "projects/REZN/solver_code"))

from phi_mp import phi_K3_smooth_mp, f_inf_mp, np_to_mp  # type: ignore

MP_DPS    = 200
MP_TARGET = "1e-100"   # must match global precision policy in solve.py


def ferr_mp(npz_path: Path, mpmath_mod, mp_ctx) -> object:
    """Return ||Phi(P)-P||_inf as an mpmath.mpf, or raise on failure.

    mpmath_mod : the mpmath module (for mpf(), nstr(), etc.)
    mp_ctx     : mpmath.mp context object (passed to phi_K3_smooth_mp / f_inf_mp)
    """
    d = np.load(npz_path, allow_pickle=True)
    P_full    = d["P_full"].astype(np.float64)
    u_full    = d["u_full"].astype(np.float64)
    pad       = int(d["pad"])
    G_inner   = int(d["G_inner"])
    tau_vec   = d["tau_vec"].astype(np.float64)
    gamma_vec = d["gamma_vec"].astype(np.float64)
    W_vec     = d["W_vec"].astype(np.float64)

    inner_lo, inner_hi = pad, pad + G_inner
    du = u_full[1] - u_full[0]
    kernel_h = max(0.005, 0.05 * du)

    # Convert to mpmath using the module's mpf constructor
    mpf = mpmath_mod.mpf
    P_mp     = np_to_mp(mp_ctx, P_full)
    u_mp     = [mpf(str(x)) for x in u_full]
    tau_mp   = [mpf(str(x)) for x in tau_vec]
    gamma_mp = [mpf(str(x)) for x in gamma_vec]
    W_mp     = [mpf(str(x)) for x in W_vec]
    kh_mp    = mpf(str(kernel_h))

    P_new_mp = phi_K3_smooth_mp(
        mp_ctx, P_mp, u_mp, inner_lo, inner_hi,
        tau_mp, gamma_mp, W_mp, kh_mp,
    )
    return f_inf_mp(mp_ctx, P_new_mp, P_mp, inner_lo, inner_hi)


def main() -> None:
    import mpmath
    mpmath.mp.dps = MP_DPS + 15   # target_dps matches solve.py convention
    mp_ctx = mpmath.mp            # context object expected by phi_K3_smooth_mp
    target = mpmath.mpf(MP_TARGET)

    q = json.loads(QUEUE.read_text())
    by_id = {t["id"]: t for t in q["tasks"]}

    files = sorted(CKPT_DIR.glob("g*.npz"))
    if SEED.exists():
        files.append(SEED)

    marked_done = 0
    left_ready  = 0
    skipped     = 0

    for ckpt in files:
        stem = ckpt.stem
        if stem not in by_id:
            print(f"  {stem}: no queue entry — skip")
            skipped += 1
            continue

        t = by_id[stem]
        if t.get("test"):
            continue

        try:
            d = np.load(ckpt, allow_pickle=True)
        except Exception as e:
            print(f"  {stem}: unreadable ({e}) — skip")
            skipped += 1
            continue

        if "P_full" not in d.files or "halo" not in d.files:
            print(f"  {stem}: not a halo checkpoint — skip")
            skipped += 1
            continue

        print(f"  {stem} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            F = ferr_mp(ckpt, mpmath, mp_ctx)
        except Exception as e:
            print(f"ERROR: {e}")
            skipped += 1
            continue

        elapsed = time.perf_counter() - t0
        F_str = mpmath.nstr(F, 6, strip_zeros=False)
        converged = F < target
        print(f"||F||_mp = {F_str}  ({elapsed:.0f}s)  "
              f"{'✓ done' if converged else '→ ready'}")

        # Store high-precision F_max
        if t.get("result") is None:
            t["result"] = {}
        t["result"]["F_max"] = F_str
        t["result"]["F_max_dps"] = MP_DPS

        converged = bool(F < target)
        if converged:
            t["status"] = "done"
            t["completed_at"] = t.get("completed_at") or datetime.now(timezone.utc).isoformat()
            t["checkpoint"]   = str(ckpt.relative_to(ROOT))
            marked_done += 1
        else:
            # Leave ready so workers polish from this checkpoint
            t["status"]      = "ready"
            t["claimed_by"]  = None
            t["claimed_at"]  = None
            t["completed_at"] = None
            t["checkpoint"]  = str(ckpt.relative_to(ROOT))
            left_ready += 1

    q["updated_at"] = datetime.now(timezone.utc).isoformat()
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")

    print(f"\nDone.")
    print(f"  Marked done (F < {MP_TARGET}): {marked_done}")
    print(f"  Left ready  (F ≥ {MP_TARGET}): {left_ready}")
    print(f"  Skipped: {skipped}")


if __name__ == "__main__":
    main()
