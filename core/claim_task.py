#!/usr/bin/env python3
"""
claim_task.py — Git-race locking for fixed-point-factory task queues.

Usage (CLI):
    python3 core/claim_task.py claim   --project REZN --task-id g400_t1000
    python3 core/claim_task.py done    --project REZN --task-id g400_t1000 \
                                       --checkpoint results/full_ree/task3_g400_t1000_mp50.json \
                                       --result '{"1-R2": 0.012}'
    python3 core/claim_task.py bail    --project REZN --task-id g400_t1000 \
                                       --reason "did not converge"
    python3 core/claim_task.py release --project REZN --worker-id hostname:1234
    python3 core/claim_task.py status  --project REZN

Python API:
    from core.claim_task import find_ready_task, try_claim, mark_done, mark_failed, release_stale_claims
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> str:
    root = os.environ.get("REPO_ROOT", "")
    if not root:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True
        )
        root = result.stdout.strip() if result.returncode == 0 else "."
    return root


def queue_path(project: str) -> str:
    return os.path.join(repo_root(), "projects", project, "TASK_QUEUE.json")


# ---------------------------------------------------------------------------
# Queue I/O
# ---------------------------------------------------------------------------

def load_queue(project: str) -> dict:
    with open(queue_path(project)) as f:
        return json.load(f)


def save_queue(project: str, queue: dict) -> None:
    path = queue_path(project)
    queue["updated_at"] = _now()
    with open(path, "w") as f:
        json.dump(queue, f, indent=2)
        f.write("\n")


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def find_ready_task(project: str, worker_id: str | None = None) -> dict | None:
    """Return the best ready task for this worker, or None."""
    queue = load_queue(project)
    done_ids = {t["id"] for t in queue["tasks"] if t["status"] == "done"}
    default_mode = queue.get("deps_semantics", {}).get("default", "all")

    def deps_ok(t):
        deps = set(t.get("depends_on", []))
        mode = t.get("deps_satisfy", default_mode)
        if not deps:
            return True
        if mode == "any":
            return bool(deps & done_ids)
        return deps <= done_ids

    ready = [t for t in queue["tasks"]
             if t["status"] == "ready" and deps_ok(t)]

    if not ready:
        return None

    wid = worker_id or _default_worker_id()

    def priority(t):
        return hashlib.sha256(f"{wid}|{t['id']}".encode()).hexdigest()

    return min(ready, key=priority)


def _default_worker_id() -> str:
    return os.environ.get("WORKER_ID",
                          socket.gethostname() + ":" + str(os.getpid()))


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + list(args),
                          capture_output=True, text=True, check=check,
                          cwd=repo_root())


def _pull_rebase(branch: str | None = None) -> None:
    branch = branch or _current_branch()
    _git("pull", "--rebase", "origin", branch)


def _push(branch: str | None = None, retries: int = 4) -> bool:
    branch = branch or _current_branch()
    backoff = 2
    for attempt in range(retries + 1):
        result = _git("push", "origin", branch, check=False)
        if result.returncode == 0:
            return True
        # non-fast-forward → another worker pushed → caller handles
        if "rejected" in result.stderr or "non-fast-forward" in result.stderr:
            return False
        # network error → retry
        if attempt < retries:
            time.sleep(backoff)
            backoff *= 2
    return False


def _current_branch() -> str:
    return os.environ.get("BRANCH",
                          _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip())


def _stage_queue(project: str) -> None:
    rel = os.path.relpath(queue_path(project), repo_root())
    _git("add", rel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_claim(project: str, task_id: str, worker_id: str | None = None,
              branch: str | None = None) -> bool:
    """
    Attempt to claim task_id. Returns True if claim landed on origin,
    False if another worker beat us (caller should pick another task).
    """
    _pull_rebase(branch)

    queue = load_queue(project)
    task = _find_by_id(queue, task_id)
    if task is None or task["status"] != "ready":
        return False  # already claimed/done by another worker after the pull

    task["status"] = "claimed"
    task["claimed_by"] = worker_id or _default_worker_id()
    task["claimed_at"] = _now()

    save_queue(project, queue)
    _stage_queue(project)
    _git("commit", "-m", f"claim {task_id}")

    if _push(branch):
        return True

    # Push rejected — rebase back to origin state and report failure
    _git("rebase", "--abort", check=False)
    _git("reset", "--hard", f"origin/{_current_branch() if not branch else branch}",
         check=False)
    return False


def mark_done(project: str, task_id: str, checkpoint: str | None,
              result: dict, branch: str | None = None) -> bool:
    """Flip task to done, stage checkpoint, commit, push. Returns True on success."""
    queue = load_queue(project)
    task = _find_by_id(queue, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    task["status"] = "done"
    task["checkpoint"] = checkpoint
    task["result"] = result
    task["completed_at"] = _now()
    task.pop("claimed_by", None)
    task.pop("claimed_at", None)

    save_queue(project, queue)
    _stage_queue(project)

    if checkpoint:
        cp_abs = os.path.join(repo_root(), checkpoint)
        if os.path.exists(cp_abs):
            _git("add", checkpoint)

    metric_str = _result_summary(result)
    _git("commit", "-m", f"{task_id}: {metric_str} done")

    for attempt in range(5):
        if _push(branch):
            return True
        _pull_rebase(branch)
        # Re-apply our done status after rebase (preserve all other done entries)
        queue = load_queue(project)
        task = _find_by_id(queue, task_id)
        if task:
            task["status"] = "done"
            task["checkpoint"] = checkpoint
            task["result"] = result
            task["completed_at"] = _now()
            task.pop("claimed_by", None)
            task.pop("claimed_at", None)
            save_queue(project, queue)
            _stage_queue(project)
            if checkpoint:
                cp_abs = os.path.join(repo_root(), checkpoint)
                if os.path.exists(cp_abs):
                    _git("add", checkpoint)
            _git("commit", "-m", f"{task_id}: {metric_str} done")
    return False


def _auto_requeue_bailed(queue: dict, task: dict) -> bool:
    """
    After a bail, try to find a better warm-start or insert a ladder task.
    Mutates queue and task in place. Returns True if task was re-queued.

    Strategy:
      1. Find done tasks with .npz checkpoints not yet in depends_on.
         If any is closer (in log-γ/τ space) than what was already tried,
         add it as a dep and reset to ready.
      2. Otherwise, if the gap to the closest same-γ done checkpoint exceeds
         a 1.25× ratio in τ, insert a new intermediate task at the log-midpoint
         and reset the bailed task to depend on it.
      3. Give up after 3 auto-requeue attempts to avoid infinite loops.
    """
    requeue_count = task.get("requeue_count", 0)
    if requeue_count >= 3:
        return False

    gamma = float(task.get("gamma") or 1.0)
    tau   = float(task.get("tau")   or 1.0)

    # Only .npz checkpoints are loadable by the warm-start path in solve.py
    done_npz = [
        t for t in queue["tasks"]
        if t["status"] == "done"
        and str(t.get("checkpoint") or "").endswith(".npz")
        and t["id"] != task["id"]
    ]
    if not done_npz:
        return False

    current_deps = set(task.get("depends_on") or [])

    def log_dist(t: dict) -> float:
        g2 = float(t.get("gamma") or 1.0)
        t2 = float(t.get("tau")   or 1.0)
        return math.hypot(math.log(gamma / g2), math.log(tau / t2))

    done_npz_by_dist = sorted(done_npz, key=log_dist)
    untried = [t for t in done_npz_by_dist if t["id"] not in current_deps]

    if untried:
        best = untried[0]
        d_best = log_dist(best)
        tried_dists = [log_dist(t) for t in done_npz if t["id"] in current_deps]
        if not tried_dists or d_best < min(tried_dists) - 0.05:
            task["status"] = "ready"
            task["result"] = None
            task["checkpoint"] = None
            task["depends_on"] = sorted(current_deps | {best["id"]})
            task["deps_satisfy"] = "any"
            task["requeue_count"] = requeue_count + 1
            task["note"] = (f"Auto-requeued (attempt {task['requeue_count']}): "
                            f"warm-start from {best['id']} (log-dist={d_best:.2f}).")
            return True

    # No better untried checkpoint — try inserting a τ-ladder task
    same_gamma = [t for t in done_npz
                  if abs(float(t.get("gamma") or 0) - gamma) < 0.01 * gamma]
    if not same_gamma:
        return False

    closest = min(same_gamma, key=lambda t: abs(math.log(tau / max(float(t.get("tau") or 1.0), 1e-9))))
    tau_prev = float(closest.get("tau") or 1.0)
    ratio = max(tau, tau_prev) / min(tau, tau_prev)
    if ratio < 1.25:
        return False  # gap already small; a ladder won't help

    # Midpoint in log-τ space, rounded to 2 significant figures
    tau_mid = math.exp((math.log(tau) + math.log(max(tau_prev, 1e-9))) / 2.0)
    mag = 10 ** math.floor(math.log10(tau_mid))
    tau_mid = round(tau_mid / mag, 1) * mag

    g_tag = f"g{int(round(gamma * 100)):03d}"
    t_tag = f"t{int(round(tau_mid * 100)):04d}"
    new_id = f"{g_tag}_{t_tag}"

    existing_ids = {t["id"] for t in queue["tasks"]}
    if new_id in existing_ids:
        # Ladder task already exists — add it as a dep if it's done
        existing = _find_by_id(queue, new_id)
        if existing and existing["status"] == "done" and new_id not in current_deps:
            task["status"] = "ready"
            task["result"] = None
            task["checkpoint"] = None
            task["depends_on"] = sorted(current_deps | {new_id})
            task["deps_satisfy"] = "any"
            task["requeue_count"] = requeue_count + 1
            task["note"] = (f"Auto-requeued (attempt {task['requeue_count']}): "
                            f"dep on existing ladder {new_id}.")
            return True
        return False

    # Insert a new ladder task immediately before the bailed task
    ladder: dict = {
        "id": new_id,
        "gamma": gamma,
        "tau": tau_mid,
        "depends_on": [closest["id"]],
        "deps_satisfy": "any",
        "status": "ready",
        "checkpoint": None,
        "result": None,
        "note": (f"Auto-inserted ladder between {closest['id']} (τ={tau_prev}) "
                 f"and {task['id']} (τ={tau})."),
    }
    idx = next(i for i, t in enumerate(queue["tasks"]) if t["id"] == task["id"])
    queue["tasks"].insert(idx, ladder)

    task["status"] = "ready"
    task["result"] = None
    task["checkpoint"] = None
    task["depends_on"] = [new_id]
    task["deps_satisfy"] = "any"
    task["requeue_count"] = requeue_count + 1
    task["note"] = (f"Auto-requeued via new ladder {new_id} (τ={tau_mid}, "
                    f"attempt {task['requeue_count']}).")
    return True


def mark_failed(project: str, task_id: str, reason: str,
                branch: str | None = None) -> bool:
    """Flip task to bailed with a reason note, then try to auto-requeue."""
    queue = load_queue(project)
    task = _find_by_id(queue, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    task["status"] = "bailed"
    task["result"] = {"reason": reason}
    task["checkpoint"] = None
    task.pop("claimed_by", None)
    task.pop("claimed_at", None)

    requeued = _auto_requeue_bailed(queue, task)
    action = "requeued" if requeued else "bailed"

    save_queue(project, queue)
    _stage_queue(project)
    _git("commit", "-m", f"{task_id}: {action}")
    _push(branch)
    return True


def release_stale_claims(project: str, max_age_hours: float = 6.0,
                          branch: str | None = None) -> list[str]:
    """Release any claimed tasks older than max_age_hours. Returns released ids."""
    queue = load_queue(project)
    released = []
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)

    for task in queue["tasks"]:
        if task["status"] != "claimed":
            continue
        claimed_at_str = task.get("claimed_at")
        if not claimed_at_str:
            continue
        claimed_at = datetime.datetime.fromisoformat(claimed_at_str.rstrip("Z"))
        if claimed_at < cutoff:
            task["status"] = "ready"
            task.pop("claimed_by", None)
            task.pop("claimed_at", None)
            released.append(task["id"])

    if released:
        save_queue(project, queue)
        _stage_queue(project)
        _git("commit", "-m", f"release stale claims: {', '.join(released)}")
        _push(branch)

    return released


def release_worker_claims(project: str, worker_id: str,
                           branch: str | None = None) -> list[str]:
    """Release all claims held by a specific worker (clean exit)."""
    queue = load_queue(project)
    released = []

    for task in queue["tasks"]:
        if task["status"] == "claimed" and task.get("claimed_by") == worker_id:
            task["status"] = "ready"
            task.pop("claimed_by", None)
            task.pop("claimed_at", None)
            released.append(task["id"])

    if released:
        save_queue(project, queue)
        _stage_queue(project)
        _git("commit", "-m", f"release claims for {worker_id}: {', '.join(released)}")
        _push(branch)

    return released


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_by_id(queue: dict, task_id: str) -> dict | None:
    for t in queue["tasks"]:
        if t["id"] == task_id:
            return t
    return None


def _result_summary(result: dict) -> str:
    if not result:
        return ""
    parts = []
    for key in ("1-R2", "slope", "F_max"):
        if key in result and result[key] is not None:
            parts.append(f"{key}={result[key]}")
    return " ".join(parts) if parts else str(result)[:60]


def print_status(project: str) -> None:
    queue = load_queue(project)
    tasks = queue["tasks"]
    by_status: dict[str, list] = {}
    for t in tasks:
        by_status.setdefault(t["status"], []).append(t)

    for status in ("ready", "claimed", "done", "bailed", "blocked", "skip"):
        group = by_status.get(status, [])
        if not group:
            continue
        print(f"\n{status.upper()} ({len(group)}):")
        for t in group:
            line = f"  {t['id']}"
            if status == "claimed":
                line += f"  [by {t.get('claimed_by', '?')} at {t.get('claimed_at', '?')}]"
            elif status == "done":
                r = t.get("result") or {}
                metric = _result_summary(r)
                line += f"  {metric}"
            elif status == "bailed":
                r = t.get("result") or {}
                line += f"  reason={r.get('reason', '?')[:60]}"
            print(line)

    print(f"\nTotal: {len(tasks)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="fixed-point-factory task manager")
    parser.add_argument("command", choices=["claim", "done", "bail", "release", "status"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--worker-id")
    parser.add_argument("--checkpoint")
    parser.add_argument("--result", help="JSON string")
    parser.add_argument("--reason")
    parser.add_argument("--branch")
    parser.add_argument("--max-age-hours", type=float, default=6.0)
    args = parser.parse_args()

    if args.command == "claim":
        if not args.task_id:
            parser.error("--task-id required for claim")
        ok = try_claim(args.project, args.task_id, args.worker_id, args.branch)
        print("claimed" if ok else "failed (another worker beat us)")
        sys.exit(0 if ok else 1)

    elif args.command == "done":
        if not args.task_id:
            parser.error("--task-id required for done")
        result = json.loads(args.result) if args.result else {}
        ok = mark_done(args.project, args.task_id, args.checkpoint, result, args.branch)
        print("done" if ok else "push failed after retries")
        sys.exit(0 if ok else 1)

    elif args.command == "bail":
        if not args.task_id:
            parser.error("--task-id required for bail")
        mark_failed(args.project, args.task_id, args.reason or "no reason given", args.branch)
        print("bailed")

    elif args.command == "release":
        if args.worker_id:
            released = release_worker_claims(args.project, args.worker_id, args.branch)
            print(f"released: {released}")
        else:
            released = release_stale_claims(args.project, args.max_age_hours, args.branch)
            print(f"released stale: {released}")

    elif args.command == "status":
        print_status(args.project)


if __name__ == "__main__":
    main()
