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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # phi_mp.py lives here

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

def _try_load_npz(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (halo, P_inner) from an .npz checkpoint, or None on failure."""
    if not ckpt_path.exists() or ckpt_path.suffix != ".npz":
        return None
    try:
        arr = np.load(ckpt_path)
        if "P_inner" in arr and "halo" in arr:
            return arr["halo"].astype(np.float64), arr["P_inner"].astype(np.float64)
    except Exception as e:
        print(f"[solve] npz load failed ({e}): {ckpt_path.name}", flush=True)
    return None


def load_warm_start(
    project: str, task: dict,
    u_full: np.ndarray, tau_vec: np.ndarray,
    gamma_vec: np.ndarray, W_vec: np.ndarray,
    inner_lo: int, inner_hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (halo_full, P_inner).  Falls back to no-learning if no usable checkpoint."""
    # 1. Try the task's own previous checkpoint (re-solve to higher precision)
    own_ckpt = task.get("checkpoint")
    if own_ckpt:
        result = _try_load_npz(ROOT / own_ckpt)
        if result is not None:
            halo, P_inner = result
            print(f"[solve] warm-start from own checkpoint: {Path(own_ckpt).name}",
                  flush=True)
            return halo, P_inner

    # 2. Try the task's dependency checkpoints
    queue = load_queue(project)
    by_id = {t["id"]: t for t in queue["tasks"]}

    for dep_id in task.get("depends_on") or []:
        dep = by_id.get(dep_id)
        if not dep or dep.get("status") != "done":
            continue
        ckpt = dep.get("checkpoint")
        if not ckpt:
            continue
        result = _try_load_npz(ROOT / ckpt)
        if result is not None:
            halo, P_inner = result
            print(f"[solve] warm-start from dep {dep_id}: {Path(ckpt).name}", flush=True)
            return halo, P_inner

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
# Symmetric-K solver path
# ---------------------------------------------------------------------------

def _save_sym_checkpoint(project: str, task_id: str,
                         P_sorted: np.ndarray, sg, u_grid: np.ndarray,
                         gamma: float, tau: float, metrics: dict) -> str:
    out_dir = ROOT / "projects" / project / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.npz"
    np.savez_compressed(
        path,
        P_sorted=P_sorted,
        u_grid=u_grid,
        K=sg.K, G=sg.G,
        gamma=gamma, tau=tau,
        one_minus_r2=metrics["1-R2"],
    )
    return str(path.relative_to(ROOT))


def _try_load_sym_npz(ckpt_path: Path, K: int, G: int):
    """Return P_sorted from a symmetric checkpoint, or None."""
    if not ckpt_path.exists() or ckpt_path.suffix != ".npz":
        return None
    try:
        arr = np.load(ckpt_path)
        if "P_sorted" in arr and int(arr["K"]) == K and int(arr["G"]) == G:
            return arr["P_sorted"].astype(np.float64)
    except Exception as e:
        print(f"[solve] sym npz load failed ({e}): {ckpt_path.name}", flush=True)
    return None


def _run_sym_task(args, task: dict, gamma: float, tau: float) -> None:
    from contour_KN_sym import (  # noqa: PLC0415
        SymGrid, sym_phi, sym_weighted_R2, sym_init_no_learning,
    )

    K = int(task.get("K", 3))
    sp = task.get("solver_params") or {}
    G_inner = int(sp.get("G_inner", sp.get("G", 15)))
    pad     = int(sp.get("pad", 4))
    G_full  = G_inner + 2 * pad
    u_max   = float(sp.get("u_max", 3.0))
    alpha   = float(sp.get("alpha", 0.3))
    max_iters = int(sp.get("max_iters", 5000))
    tol     = float(sp.get("tol", 5e-7))
    W       = float(sp.get("W", 1.0))

    # Build grid: inner spans [-u_max, u_max] with G_inner points;
    # halo extends pad cells beyond each side with uniform spacing du.
    du = 2.0 * u_max / (G_inner - 1)
    u_grid = np.array([-u_max + (q - pad) * du for q in range(G_full)])
    sg = SymGrid.build(G_full, K)

    reporter = ProgressReporter(
        project=args.project, task_id=args.task_id,
        worker_id=args.worker_id, branch=args.branch, interval=30,
        repo_root=ROOT,
    )
    reporter.start()

    t_start = time.perf_counter()
    exit_code = 0

    try:
        # Warm-start: own checkpoint; dep checkpoint only if same K and G_full.
        P_sorted = None
        own_ckpt = task.get("checkpoint")
        if own_ckpt:
            P_sorted = _try_load_sym_npz(ROOT / own_ckpt, K, G_full)
            if P_sorted is not None:
                print(f"[solve] sym warm-start from own checkpoint", flush=True)

        if P_sorted is None:
            queue = load_queue(args.project)
            by_id = {t["id"]: t for t in queue["tasks"]}
            for dep_id in task.get("depends_on") or []:
                dep = by_id.get(dep_id)
                if not dep or dep.get("status") != "done":
                    continue
                ckpt = dep.get("checkpoint")
                if not ckpt:
                    continue
                P_sorted = _try_load_sym_npz(ROOT / ckpt, K, G_full)
                if P_sorted is not None:
                    print(f"[solve] sym warm-start from dep {dep_id}", flush=True)
                    break

        if P_sorted is None:
            print("[solve] sym cold start (no-learning init)", flush=True)
            P_sorted = sym_init_no_learning(sg, u_grid, tau, gamma, W)

        print(f"[solve] sym K={K} G_inner={G_inner} pad={pad} G_full={G_full} "
              f"γ={gamma} τ={tau} alpha={alpha} max_iters={max_iters} tol={tol:.0e}",
              flush=True)

        inner_mask = np.array([
            all(pad <= int(j) < pad + G_inner for j in sg.tuples[s])
            for s in range(sg.n)
        ], dtype=bool)

        F_inf = float("inf")
        for i in range(max_iters):
            P_new = sym_phi(P_sorted, sg, u_grid, tau, gamma, W, pad=pad, G_inner=G_inner)
            F_inf = float(np.max(np.abs((P_new - P_sorted)[inner_mask])))
            P_sorted = (1.0 - alpha) * P_sorted + alpha * P_new
            reporter.update(iter=i + 1, ftol=F_inf)
            if i % 100 == 0:
                print(f"[solve] sym iter {i:5d}  ||F||={F_inf:.4e}", flush=True)
            if F_inf < tol:
                print(f"[solve] sym converged at iter {i+1}  ||F||={F_inf:.4e}", flush=True)
                break
        else:
            print(f"[solve] sym reached max_iters={max_iters}  ||F||={F_inf:.4e}", flush=True)

        metrics = sym_weighted_R2(P_sorted, sg, u_grid, tau, pad=pad, G_inner=G_inner)
        wall_s = time.perf_counter() - t_start
        print(f"[solve] sym done  1-R²={metrics['1-R2']:.6e}  "
              f"||F||={F_inf:.4e}  wall={wall_s:.0f}s", flush=True)

        ckpt_rel = _save_sym_checkpoint(
            args.project, args.task_id, P_sorted, sg, u_grid, gamma, tau, metrics
        )

        result = {
            "1-R2":      round(metrics["1-R2"], 8),
            "slope":     round(metrics["slope"], 6),
            "F_max":     float(f"{F_inf:.4e}"),
            "n_cells":   metrics["n_cells"],
            "K":         K,
            "G_inner":   G_inner,
            "pad":       pad,
            "wall_s":    round(wall_s, 1),
        }

        BAIL_THRESHOLD = 1e-4
        if F_inf > BAIL_THRESHOLD:
            claim_bail(args.project, args.task_id, args.branch,
                       f"sym ||F||={F_inf:.3e} > {BAIL_THRESHOLD:.0e}")
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

    # Per-task solver parameter overrides (task["solver_params"] wins over CLI defaults)
    sp = task.get("solver_params") or {}
    G_inner      = int(sp.get("G_inner",        args.G_inner))
    pad          = int(sp.get("pad",             args.pad))
    u_inner_max  = float(sp.get("u_inner_max",   args.u_inner_max))
    max_stages   = int(sp.get("max_stages",      args.max_stages))
    presmooth    = int(sp.get("presmooth",        args.presmooth))
    presmooth_alpha = float(sp.get("presmooth_alpha", args.presmooth_alpha))
    inner_max_iter  = int(sp.get("inner_max_iter",    args.inner_max_iter))
    inner_rdiff  = float(sp.get("inner_rdiff",    1.0e-4))
    noise_level  = float(sp.get("noise_level",    0.0))

    G_full   = G_inner + 2 * pad
    du       = (2.0 * u_inner_max) / (G_inner - 1)
    u_full   = np.array([-u_inner_max + (q - pad) * du
                          for q in range(G_full)], dtype=np.float64)
    inner_lo, inner_hi = pad, pad + G_inner
    u_grid_inner = u_full[inner_lo:inner_hi].copy()

    # Auto kernel bandwidth (mirrors staggered_run_K3 heuristic)
    kernel_h = max(0.005, 0.05 * du)

    # ------------------------------------------------------------------
    # Reject tasks with an unrecognised kind (e.g. meta-tasks).
    # ------------------------------------------------------------------
    task_kind = task.get("kind")
    if task_kind is not None and task_kind not in ("ree", "ree_k3", "solver"):
        print(f"[solve] task {args.task_id} has kind={task_kind!r} — not a solver task, skipping",
              flush=True)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Symmetric-K dispatch: tasks with "symmetric": true use the
    # contour_KN_sym solver for K = 3..8 in sorted-tuple storage.
    # ------------------------------------------------------------------
    if task.get("symmetric"):
        _run_sym_task(args, task, gamma, tau)
        return

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

        # Optional noise perturbation to escape saddle points / Newton traps
        if noise_level > 0.0:
            rng = np.random.default_rng(seed=abs(hash(args.task_id)) % (2**31))
            noise = rng.normal(0.0, noise_level * float(np.std(P_inner_seed)), P_inner_seed.shape)
            P_inner_seed = np.clip(P_inner_seed + noise, 1e-6, 1.0 - 1e-6).astype(np.float64)
            halo_noise = rng.normal(0.0, noise_level * float(np.std(halo)), halo.shape)
            halo = np.clip(halo + halo_noise, 1e-6, 1.0 - 1e-6).astype(np.float64)
            print(f"[solve] noise perturbation applied: level={noise_level}", flush=True)

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

        print(f"[solve] params: presmooth={presmooth} alpha={presmooth_alpha} "
              f"max_stages={max_stages} inner_rdiff={inner_rdiff:.0e} "
              f"inner_max_iter={inner_max_iter}", flush=True)

        if sp.get("pure_picard"):
            # Bypass staggered_solve entirely — pure damped Picard, no Newton.
            # Avoids the oscillation where Newton accepts a bad iterate and
            # stalls the solver in an indefinite presmooth/Newton cycle.
            max_picard_iters = int(sp.get("picard_iters", 200000))
            picard_tol = float(sp.get("picard_tol", tol))
            print(f"[solve] pure_picard mode: alpha={presmooth_alpha} "
                  f"max_iters={max_picard_iters} tol={picard_tol:.0e}", flush=True)

            P_full_cur = replace_inner(halo, P_inner_seed, inner_lo, inner_hi)
            F_inf_cur = float("inf")
            for _i in range(max_picard_iters):
                P_new = phi_full_fn(P_full_cur)
                F_inf_cur = float(np.max(np.abs(
                    extract_inner(P_new, inner_lo, inner_hi)
                    - extract_inner(P_full_cur, inner_lo, inner_hi)
                )))
                P_full_cur = (1.0 - presmooth_alpha) * P_full_cur + presmooth_alpha * P_new
                if F_inf_cur < picard_tol:
                    print(f"[solve] pure_picard converged at iter={_i+1} F={F_inf_cur:.4e}",
                          flush=True)
                    break
            else:
                print(f"[solve] pure_picard reached max_iters={max_picard_iters} "
                      f"F={F_inf_cur:.4e}", flush=True)

            P_inner_final = extract_inner(P_full_cur, inner_lo, inner_hi)

            class _Stage:
                def __init__(self, F, d):
                    self.F_inner_inf = F
                    self.deficit_f128 = d

            class _History:
                def __init__(self, F):
                    self.stages = [_Stage(F, 0.0)]

            history = _History(F_inf_cur)
        else:
            P_inner_final, history = staggered_solve(
                phi_full_fn, u_full, inner_lo, inner_hi,
                u_grid_inner=u_grid_inner, tau_vec=tau_vec, K=K,
                halo_initial=halo, inner_initial=P_inner_seed,
                max_stages=max_stages,
                stage_tol=1.0e-3,
                inner_method="lgmres",
                inner_max_iter=inner_max_iter,
                inner_tol=tol,
                inner_outer_k=40,
                inner_inner_maxiter=80,
                inner_rdiff=inner_rdiff,
                presmooth_steps=presmooth,
                presmooth_alpha=presmooth_alpha,
                halo_update="no_learning",
                heartbeat_s=30.0,
            )

        # Final diagnostics (float64)
        P_full_final = replace_inner(halo, P_inner_final, inner_lo, inner_hi)
        F_full = phi_full_fn(P_full_final) - P_full_final
        F_inner = extract_inner(F_full, inner_lo, inner_hi)
        F_inf_final = float(np.max(np.abs(F_inner)))
        deficit = revelation_deficit_f128(P_inner_final, u_grid_inner, tau_vec, K)

        # ----------------------------------------------------------------
        # Optional mpmath polish phase
        # Activated when solver_params contains mp_dps (e.g. 100).
        # Runs pure Picard in mpmath starting from the float64 result.
        # ----------------------------------------------------------------
        mp_dps = int(sp.get("mp_dps", 0))
        if mp_dps > 0:
            from phi_mp import phi_newton_mp  # noqa: PLC0415
            mp_tol       = str(sp.get("mp_tol", "1e-50"))
            mp_iters     = int(sp.get("mp_iters", 20))
            lgmres_tol   = float(sp.get("lgmres_tol", 1e-10))
            lgmres_inner = int(sp.get("lgmres_inner_m", 30))
            lgmres_outer = int(sp.get("lgmres_outer", 10))
            print(f"[solve] mpmath Newton: dps={mp_dps} tol={mp_tol} "
                  f"max_newton={mp_iters} lgmres_tol={lgmres_tol:.0e}", flush=True)

            # Raw float64 phi for LGMRES Jacobian-vector products (no reporter calls)
            def _phi64_raw(P_full: np.ndarray) -> np.ndarray:
                return phi_K3_halo_smooth(
                    P_full, u_full, inner_lo, inner_hi,
                    tau_vec, gamma_vec, W_vec, kernel_h,
                )

            P_inner_mp, F_inf_mp_val, n_mp = phi_newton_mp(
                P_inner_final, halo, u_full,
                inner_lo, inner_hi,
                tau_vec, gamma_vec, W_vec,
                kernel_h,
                phi_float64_fn=_phi64_raw,
                dps=mp_dps,
                tol_str=mp_tol,
                max_newton=mp_iters,
                lgmres_tol=lgmres_tol,
                lgmres_inner_m=lgmres_inner,
                lgmres_outer=lgmres_outer,
                reporter=reporter,
            )
            P_inner_final = P_inner_mp
            F_inf_final   = F_inf_mp_val
            P_full_final  = replace_inner(halo, P_inner_final, inner_lo, inner_hi)
            deficit = revelation_deficit_f128(P_inner_final, u_grid_inner, tau_vec, K)
            print(f"[solve] mpmath done  ||F||={F_inf_final:.4e}  "
                  f"1-R²={deficit:.6e}  mp_iters={n_mp}", flush=True)

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

        # Bail if residual is too large (solver stalled).
        # When mp_dps is set, the acceptance criterion is mp_tol (e.g. 1e-50).
        # Without mp_dps, accept anything below 1e-4 (float64 realistic).
        if mp_dps > 0:
            BAIL_THRESHOLD = float(sp.get("mp_tol", "1e-50"))
        else:
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
