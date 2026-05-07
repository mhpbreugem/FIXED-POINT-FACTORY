#!/usr/bin/env python3
"""
solve.py — REZN solver wrapper for fixed-point-factory.

Wraps the K=3 staggered halo solver from github.com/mhpbreugem/REZN/code/.
Reads task params from TASK_QUEUE.json, runs the fixed-point iteration,
reports live progress, saves a checkpoint, and marks the task done or
bailed via claim_task.py.

Invoked by .github/workflows/solve-tasks.yml (or core/bootstrap.sh):
    python3 projects/REZN/solver_code/solve.py \
        --project REZN --task-id g050_t0030 \
        --branch main --worker-id solver-01

Environment:
    REZN_SRC  path to a clone of github.com/mhpbreugem/REZN
              (default: ~/rezn-source; auto-cloned if missing)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths — make core/ importable regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]   # fixed-point-factory repo root
sys.path.insert(0, str(ROOT / "core"))

from progress import ProgressReporter  # noqa: E402

# ---------------------------------------------------------------------------
# Make REZN code/ importable
# ---------------------------------------------------------------------------
REZN_SRC = Path(os.environ.get("REZN_SRC", Path.home() / "rezn-source"))
if not REZN_SRC.exists():
    print(f"[solve] cloning REZN to {REZN_SRC} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/mhpbreugem/REZN.git", str(REZN_SRC)],
        check=True,
    )
sys.path.insert(0, str(REZN_SRC))

from code.contour_K3_halo import (   # type: ignore
    init_no_learning_K3, phi_K3_halo_smooth,
)
from code.halo import extract_inner, replace_inner  # type: ignore
from code.staggered import staggered_solve          # type: ignore
from code.f128 import revelation_deficit_f128       # type: ignore

# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def load_queue(project: str) -> dict:
    return json.loads((ROOT / "projects" / project / "TASK_QUEUE.json").read_text())


def find_task(queue: dict, task_id: str) -> dict:
    for t in queue["tasks"]:
        if t["id"] == task_id:
            return t
    raise SystemExit(f"[solve] task {task_id!r} not found in queue")


def dps_to_tol(dps) -> float:
    """Map legacy mpmath dps to a float64-realistic Newton tolerance."""
    d = int(dps) if dps is not None else 50
    if d <= 50:
        return 1.0e-7
    if d <= 100:
        return 1.0e-9
    return 1.0e-11


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------

def load_warm_start(
    project: str, task: dict,
    u_full: np.ndarray, tau_vec: np.ndarray,
    gamma_vec: np.ndarray, W_vec: np.ndarray,
    inner_lo: int, inner_hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (halo_full, P_inner).  Falls back to no-learning if no usable checkpoint."""
    queue = load_queue(project)
    by_id = {t["id"]: t for t in queue["tasks"]}

    for dep_id in task.get("depends_on") or []:
        dep = by_id.get(dep_id)
        if not dep or dep.get("status") != "done":
            continue
        ckpt = dep.get("checkpoint")
        if not ckpt:
            continue
        ckpt_path = ROOT / ckpt
        if not ckpt_path.exists() or ckpt_path.suffix != ".npz":
            continue
        try:
            arr = np.load(ckpt_path)
            if "P_inner" in arr and "halo" in arr:
                P_inner = arr["P_inner"].astype(np.float64)
                halo = arr["halo"].astype(np.float64)
                print(f"[solve] warm-start from {ckpt_path.name}", flush=True)
                return halo, P_inner
        except Exception as e:
            print(f"[solve] warm-start load failed ({e}), falling back to cold start",
                  flush=True)

    print("[solve] cold start (no-learning init)", flush=True)
    halo = init_no_learning_K3(u_full, tau_vec, gamma_vec, W_vec)
    P_inner = extract_inner(halo, inner_lo, inner_hi)
    return halo, P_inner


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_checkpoint(project: str, task_id: str,
                    P_inner: np.ndarray, halo: np.ndarray,
                    P_full: np.ndarray, u_full: np.ndarray,
                    u_grid_inner: np.ndarray, gamma_vec: np.ndarray,
                    tau_vec: np.ndarray, W_vec: np.ndarray,
                    G_inner: int, pad: int, history) -> str:
    out_dir = ROOT / "projects" / project / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.npz"
    stage_F = np.array([r.F_inner_inf for r in history.stages], dtype=np.float64)
    stage_d = np.array([r.deficit_f128 for r in history.stages], dtype=np.float64)
    np.savez_compressed(
        path,
        P_inner=P_inner, halo=halo, P_full=P_full,
        u_full=u_full, u_grid_inner=u_grid_inner,
        gamma_vec=gamma_vec, tau_vec=tau_vec, W_vec=W_vec,
        G_inner=G_inner, pad=pad, K=3,
        stage_F_inf=stage_F, stage_deficit=stage_d,
    )
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------------------
# claim_task.py helpers (call via subprocess so git ops run in correct cwd)
# ---------------------------------------------------------------------------

def claim_done(project: str, task_id: str, branch: str,
               checkpoint: str, result: dict) -> None:
    subprocess.run(
        [sys.executable, "core/claim_task.py", "done",
         "--project", project,
         "--task-id", task_id,
         "--branch", branch,
         "--checkpoint", checkpoint,
         "--result", json.dumps(result)],
        check=False, cwd=str(ROOT),
    )


def claim_bail(project: str, task_id: str, branch: str, reason: str) -> None:
    subprocess.run(
        [sys.executable, "core/claim_task.py", "bail",
         "--project", project,
         "--task-id", task_id,
         "--branch", branch,
         "--reason", reason],
        check=False, cwd=str(ROOT),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="REZN K=3 staggered solver")
    ap.add_argument("--project",   required=True)
    ap.add_argument("--task-id",   required=True)
    ap.add_argument("--branch",    default="main")
    ap.add_argument("--worker-id", required=True)
    # Grid overrides (defaults match staggered_run_K3 paper settings)
    ap.add_argument("--G-inner",       type=int,   default=12)
    ap.add_argument("--pad",           type=int,   default=4)
    ap.add_argument("--u-inner-max",   type=float, default=3.0)
    ap.add_argument("--max-stages",    type=int,   default=6)
    ap.add_argument("--presmooth",     type=int,   default=15)
    ap.add_argument("--presmooth-alpha", type=float, default=0.05)
    ap.add_argument("--inner-max-iter", type=int,  default=30)
    args = ap.parse_args()

    queue = load_queue(args.project)
    task  = find_task(queue, args.task_id)

    gamma = float(task.get("gamma") or 0.5)
    tau   = float(task.get("tau")   or 2.0)
    tol   = dps_to_tol(task.get("dps") or queue.get("params", {}).get("dps"))
    K     = 3

    gamma_vec = np.full(K, gamma, dtype=np.float64)
    tau_vec   = np.full(K, tau,   dtype=np.float64)
    W_vec     = np.ones(K,        dtype=np.float64)

    G_inner  = args.G_inner
    pad      = args.pad
    G_full   = G_inner + 2 * pad
    du       = (2.0 * args.u_inner_max) / (G_inner - 1)
    u_full   = np.array([-args.u_inner_max + (q - pad) * du
                          for q in range(G_full)], dtype=np.float64)
    inner_lo, inner_hi = pad, pad + G_inner
    u_grid_inner = u_full[inner_lo:inner_hi].copy()

    # Auto kernel bandwidth (mirrors staggered_run_K3 heuristic)
    kernel_h = max(0.005, 0.05 * du)

    # ------------------------------------------------------------------
    # CI smoke-test: task flagged "test": true — just verify the full
    # import chain + one phi evaluation, then mark done immediately.
    # ------------------------------------------------------------------
    if task.get("test"):
        print("[solve] TEST TASK — smoke-test mode, skipping full solve", flush=True)
        reporter = ProgressReporter(
            project=args.project, task_id=args.task_id,
            worker_id=args.worker_id, branch=args.branch, interval=30,
            repo_root=ROOT,
        )
        reporter.start()
        try:
            halo = init_no_learning_K3(u_full, tau_vec, gamma_vec, W_vec)
            P_full_test = phi_K3_halo_smooth(
                halo, u_full, inner_lo, inner_hi,
                tau_vec, gamma_vec, W_vec, kernel_h,
            )
            P_inner_test = extract_inner(P_full_test, inner_lo, inner_hi)
            F_inf = float(np.max(np.abs(extract_inner(halo, inner_lo, inner_hi) - P_inner_test)))
            deficit = revelation_deficit_f128(P_inner_test, u_grid_inner, tau_vec, K)
            print(f"[solve] smoke: phi(P_init) OK  ||F||inf={F_inf:.4e}  1-R²={deficit:.6e}", flush=True)
            claim_done(args.project, args.task_id, args.branch, checkpoint="", result={
                "smoke": True,
                "1-R2": round(deficit, 8),
                "F_max": float(f"{F_inf:.4e}"),
                "note": "CI smoke test — one phi eval, no convergence required",
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            claim_bail(args.project, args.task_id, args.branch, f"smoke test failed: {e}")
            sys.exit(2)
        finally:
            reporter.stop(delete=True)
        sys.exit(0)

    print(f"[solve] task={args.task_id}  γ={gamma}  τ={tau}  "
          f"G_inner={G_inner} pad={pad} G_full={G_full}  tol={tol:.0e}",
          flush=True)

    # --- progress reporter ------------------------------------------------
    reporter = ProgressReporter(
        project=args.project, task_id=args.task_id,
        worker_id=args.worker_id, branch=args.branch, interval=30,
        repo_root=ROOT,
    )
    reporter.start()

    t_start = time.perf_counter()
    exit_code = 0

    try:
        halo, P_inner_seed = load_warm_start(
            args.project, task,
            u_full, tau_vec, gamma_vec, W_vec,
            inner_lo, inner_hi,
        )

        # phi closure — wrap to update reporter on each evaluation
        phi_calls = {"n": 0}

        def phi_full_fn(P_full: np.ndarray) -> np.ndarray:
            out = phi_K3_halo_smooth(
                P_full, u_full, inner_lo, inner_hi,
                tau_vec, gamma_vec, W_vec, kernel_h,
            )
            phi_calls["n"] += 1
            # Light residual estimate for live dashboard (float64, cheap)
            P_in = extract_inner(P_full, inner_lo, inner_hi)
            P_in_new = extract_inner(out, inner_lo, inner_hi)
            F_inf = float(np.max(np.abs(P_in - P_in_new)))
            reporter.update(iter=phi_calls["n"], ftol=F_inf)
            return out

        P_inner_final, history = staggered_solve(
            phi_full_fn, u_full, inner_lo, inner_hi,
            u_grid_inner=u_grid_inner, tau_vec=tau_vec, K=K,
            halo_initial=halo, inner_initial=P_inner_seed,
            max_stages=args.max_stages,
            stage_tol=1.0e-3,
            inner_method="lgmres",
            inner_max_iter=args.inner_max_iter,
            inner_tol=tol,
            inner_outer_k=40,
            inner_inner_maxiter=80,
            inner_rdiff=1.0e-4,
            presmooth_steps=args.presmooth,
            presmooth_alpha=args.presmooth_alpha,
            halo_update="no_learning",
            heartbeat_s=30.0,
        )

        # Final diagnostics
        P_full_final = replace_inner(halo, P_inner_final, inner_lo, inner_hi)
        F_full = phi_full_fn(P_full_final) - P_full_final
        F_inner = extract_inner(F_full, inner_lo, inner_hi)
        F_inf_final = float(np.max(np.abs(F_inner)))
        deficit = revelation_deficit_f128(P_inner_final, u_grid_inner, tau_vec, K)
        wall_s = time.perf_counter() - t_start

        print(f"[solve] done  ||F_inner||inf={F_inf_final:.4e}  "
              f"1-R²={deficit:.6e}  wall={wall_s:.0f}s", flush=True)

        ckpt_rel = save_checkpoint(
            args.project, args.task_id,
            P_inner_final, halo, P_full_final,
            u_full, u_grid_inner,
            gamma_vec, tau_vec, W_vec,
            G_inner, pad, history,
        )

        result = {
            "1-R2":        round(deficit, 8),
            "F_max":       float(f"{F_inf_final:.4e}"),
            "n_stages":    len(history.stages),
            "phi_calls":   phi_calls["n"],
            "wall_s":      round(wall_s, 1),
        }

        # Bail if residual is too large (solver stalled)
        BAIL_THRESHOLD = max(tol * 1000, 1.0e-4)
        if F_inf_final > BAIL_THRESHOLD:
            claim_bail(args.project, args.task_id, args.branch,
                       f"||F||inf={F_inf_final:.3e} > bail threshold {BAIL_THRESHOLD:.0e}")
            exit_code = 1
        else:
            claim_done(args.project, args.task_id, args.branch, ckpt_rel, result)
            exit_code = 0

    except Exception as e:
        import traceback
        reason = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        claim_bail(args.project, args.task_id, args.branch, reason)
        exit_code = 2

    finally:
        reporter.stop(delete=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
