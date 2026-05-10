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
import base64 as _b64
import datetime
import hashlib
import json
import math
import os
import random
import re as _re
import socket
import subprocess
import sys
import time
import urllib.error as _urllib_err
import urllib.request as _urllib_req


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


def _pull_rebase(branch: str | None = None) -> bool:
    """Rebase onto origin. Returns True on success, False on conflict/error."""
    branch = branch or _current_branch()
    result = _git("pull", "--rebase", "origin", branch, check=False)
    if result.returncode == 0:
        return True
    # Conflict or other error — abort rebase and hard-reset to origin
    _git("rebase", "--abort", check=False)
    _git("reset", "--hard", f"origin/{branch}", check=False)
    return False


def _push(branch: str | None = None, retries: int = 4) -> bool:
    branch = branch or _current_branch()
    backoff = 2
    for attempt in range(retries + 1):
        result = _git("push", "origin", branch, check=False)
        if result.returncode == 0:
            return True
        # network error → retry with backoff
        stderr = result.stderr or ""
        is_network = not ("rejected" in stderr or "non-fast-forward" in stderr
                          or "fetch first" in stderr)
        if is_network and attempt < retries:
            time.sleep(backoff)
            backoff *= 2
            continue
        # non-fast-forward or other push rejection → caller should rebase
        return False
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


# ---------------------------------------------------------------------------
# REST API helpers — used by mark_done to avoid git-push race conditions.
# Progress commits only touch progress/taskid.json, NOT TASK_QUEUE.json, so
# its SHA is stable between done/claim events.  Optimistic locking via SHA
# means 409 conflicts are rare and resolve immediately on retry.
# ---------------------------------------------------------------------------

def _gh_token() -> str:
    t = os.environ.get("GITHUB_TOKEN", "")
    if t:
        return t
    result = _git("remote", "get-url", "origin", check=False)
    m = _re.search(r"https://([A-Za-z0-9_]+)@github\.com", result.stdout.strip())
    return m.group(1) if m else ""


def _gh_repo() -> tuple[str, str]:
    result = _git("remote", "get-url", "origin", check=False)
    m = _re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", result.stdout.strip())
    return (m.group(1), m.group(2)) if m else ("", "")


def _api_get(token: str, owner: str, repo: str, path: str, branch: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    req = _urllib_req.Request(url, headers={"Authorization": f"token {token}"})
    with _urllib_req.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _api_put(token: str, owner: str, repo: str, path: str,
             message: str, content_bytes: bytes, sha: str | None, branch: str) -> int:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    payload: dict = {"message": message,
                     "content": _b64.b64encode(content_bytes).decode(),
                     "branch": branch}
    if sha:
        payload["sha"] = sha
    req = _urllib_req.Request(url, data=json.dumps(payload).encode(), method="PUT",
                              headers={"Authorization": f"token {token}",
                                       "Content-Type": "application/json"})
    try:
        with _urllib_req.urlopen(req, timeout=25) as r:
            return r.status
    except _urllib_err.HTTPError as e:
        return e.code


def mark_done(project: str, task_id: str, checkpoint: str | None,
              result: dict, branch: str | None = None) -> bool:
    """
    Flip task to done and upload checkpoint via GitHub REST API.

    Uses SHA-based optimistic locking on TASK_QUEUE.json.  Progress commits
    never touch TASK_QUEUE.json so its SHA is stable; 409 conflicts only occur
    when two tasks finish simultaneously, and resolve in 1–2 retries.
    """
    branch = branch or _current_branch()
    token = _gh_token()
    owner, repo_name = _gh_repo()
    metric_str = _result_summary(result)
    queue_api_path = f"projects/{project}/TASK_QUEUE.json"

    for attempt in range(20):
        # Fetch live queue + its SHA (the optimistic-lock key).
        # Progress commits never touch TASK_QUEUE.json, so SHA only changes on
        # done/claim events — conflicts are rare and resolve in 1-2 retries.
        try:
            meta = _api_get(token, owner, repo_name, queue_api_path, branch)
        except Exception as exc:
            print(f"[mark_done] GET attempt {attempt}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 30))
            continue

        file_sha = meta["sha"]
        queue = json.loads(_b64.b64decode(meta["content"].replace("\n", "")))
        task = _find_by_id(queue, task_id)
        if task is None:
            return False
        if task.get("status") == "done":
            return True  # another worker already landed it

        task["status"] = "done"
        task["checkpoint"] = checkpoint
        task["result"] = result
        task["completed_at"] = _now()
        task.pop("claimed_by", None)
        task.pop("claimed_at", None)

        newly_ready = _unblock_downstream(queue)
        _update_summary(queue)
        unblock_str = f" unblocked:{','.join(newly_ready)}" if newly_ready else ""

        status = _api_put(
            token, owner, repo_name, queue_api_path,
            f"{task_id}: {metric_str} done{unblock_str}",
            json.dumps(queue, indent=2).encode(), file_sha, branch,
        )

        if status in (200, 201):
            # Upload checkpoint via REST API (best-effort; non-critical)
            if checkpoint:
                cp_abs = os.path.join(repo_root(), checkpoint)
                if os.path.exists(cp_abs):
                    with open(cp_abs, "rb") as fh:
                        cp_bytes = fh.read()
                    try:
                        cp_meta = _api_get(token, owner, repo_name, checkpoint, branch)
                        cp_sha: str | None = cp_meta["sha"]
                    except Exception:
                        cp_sha = None
                    rc = _api_put(token, owner, repo_name, checkpoint,
                                  f"{task_id}: checkpoint", cp_bytes, cp_sha, branch)
                    if rc not in (200, 201):
                        print(f"[mark_done] checkpoint upload returned {rc} (non-fatal)",
                              flush=True)
            return True

        if status == 409:
            # SHA changed: another done/claim commit landed between GET and PUT
            print(f"[mark_done] 409 conflict attempt {attempt}, retrying", flush=True)
            time.sleep(random.uniform(0, min(2 ** attempt, 8)))
            continue

        print(f"[mark_done] PUT returned {status} attempt {attempt}", flush=True)
        time.sleep(min(2 ** attempt, 15))

    return False


def _update_summary(queue: dict) -> None:
    counts: dict[str, int] = {}
    for t in queue["tasks"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    queue["summary"] = {
        "total":   len(queue["tasks"]),
        "done":    counts.get("done", 0),
        "ready":   counts.get("ready", 0),
        "claimed": counts.get("claimed", 0),
        "blocked": counts.get("blocked", 0),
        "bailed":  counts.get("bailed", 0),
        "skip":    counts.get("skip", 0),
    }


def _unblock_downstream(queue: dict) -> list[str]:
    """Flip blocked tasks whose deps are now fully satisfied to ready."""
    done_ids = {t["id"] for t in queue["tasks"] if t["status"] == "done"}
    default_mode = queue.get("deps_semantics", {}).get("default", "all")
    unblocked = []
    for t in queue["tasks"]:
        if t["status"] != "blocked":
            continue
        deps = set(t.get("depends_on", []))
        mode = t.get("deps_satisfy", default_mode)
        if not deps:
            satisfied = True
        elif mode == "any":
            satisfied = bool(deps & done_ids)
        else:
            satisfied = deps <= done_ids
        if satisfied:
            t["status"] = "ready"
            unblocked.append(t["id"])
    return unblocked


def _make_ladder_id(gamma: float, tau: float) -> str:
    return f"g{int(round(gamma * 100)):03d}_t{int(round(tau * 100)):04d}"


def _insert_ladder(queue: dict, task: dict, ladder: dict,
                   requeue_count: int, reason: str) -> bool:
    new_id = ladder["id"]
    existing_ids = {t["id"] for t in queue["tasks"]}
    if new_id in existing_ids:
        existing = _find_by_id(queue, new_id)
        if existing and existing["status"] == "done" and new_id not in set(task.get("depends_on") or []):
            task["status"] = "ready"
            task["result"] = None
            task["checkpoint"] = None
            task["depends_on"] = sorted(set(task.get("depends_on") or []) | {new_id})
            task["deps_satisfy"] = "any"
            task["requeue_count"] = requeue_count + 1
            task["note"] = f"Auto-requeued (attempt {task['requeue_count']}): dep on existing {new_id}. {reason}"
            return True
        return False
    idx = next(i for i, t in enumerate(queue["tasks"]) if t["id"] == task["id"])
    queue["tasks"].insert(idx, ladder)
    task["status"] = "ready"
    task["result"] = None
    task["checkpoint"] = None
    task["depends_on"] = [new_id]
    task["deps_satisfy"] = "any"
    task["requeue_count"] = requeue_count + 1
    task["note"] = f"Auto-requeued (attempt {task['requeue_count']}): via new ladder {new_id}. {reason}"
    return True


def _auto_requeue_bailed(queue: dict, task: dict) -> bool:
    """
    After a bail, try to find a better warm-start or insert a ladder task.
    Mutates queue and task in place. Returns True if task was re-queued.

    Strategy (in order):
      1. Add the closest untried .npz checkpoint as a dep if it's closer than
         what's already been tried.
      2. Insert a τ-ladder at the log-midpoint between the bailed task and the
         closest same-γ done .npz.
      3. Insert a γ-ladder at the log-midpoint between the bailed task and the
         closest done .npz (any γ) — bridges the γ-gap.
      4. Re-queue with a noise perturbation (solver_params.noise_level) as a
         last resort before giving up.
      Gives up after 5 attempts.
    """
    requeue_count = task.get("requeue_count", 0)
    if requeue_count >= 5:
        return False

    gamma = float(task.get("gamma") or 1.0)
    tau   = float(task.get("tau")   or 1.0)

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

    # ── Step 1: untried closer checkpoint ────────────────────────────────────
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

    # ── Step 2: τ-ladder (same γ) ────────────────────────────────────────────
    same_gamma = [t for t in done_npz
                  if abs(float(t.get("gamma") or 0) - gamma) < 0.01 * gamma]
    if same_gamma:
        closest_tau = min(same_gamma,
            key=lambda t: abs(math.log(tau / max(float(t.get("tau") or 1.0), 1e-9))))
        tau_prev = float(closest_tau.get("tau") or 1.0)
        ratio = max(tau, tau_prev) / min(tau, tau_prev)
        if ratio >= 1.25:
            tau_mid = math.exp((math.log(tau) + math.log(max(tau_prev, 1e-9))) / 2.0)
            mag = 10 ** math.floor(math.log10(tau_mid))
            tau_mid = round(tau_mid / mag, 1) * mag
            new_id = _make_ladder_id(gamma, tau_mid)
            ladder = {
                "id": new_id, "gamma": gamma, "tau": tau_mid,
                "depends_on": [closest_tau["id"]], "deps_satisfy": "any",
                "status": "ready", "checkpoint": None, "result": None,
                "note": f"τ-ladder between {closest_tau['id']} (τ={tau_prev}) and {task['id']} (τ={tau}).",
            }
            if _insert_ladder(queue, task, ladder, requeue_count,
                              f"τ-ladder at τ={tau_mid}"):
                return True

    # ── Step 3: γ-ladder (cross-γ, same τ or nearest) ───────────────────────
    # Find the closest done .npz and insert a γ-midpoint task at the same τ
    if done_npz_by_dist:
        closest_any = done_npz_by_dist[0]
        g2 = float(closest_any.get("gamma") or 1.0)
        t2 = float(closest_any.get("tau")   or 1.0)
        g_ratio = max(gamma, g2) / min(gamma, g2)
        if g_ratio >= 1.25:
            gamma_mid = math.exp((math.log(gamma) + math.log(g2)) / 2.0)
            # Round gamma_mid to 2 sig figs
            mag = 10 ** math.floor(math.log10(gamma_mid))
            gamma_mid = round(gamma_mid / mag, 1) * mag
            # Ladder is at the midpoint γ, same τ as the bailed task
            new_id = _make_ladder_id(gamma_mid, tau)
            ladder = {
                "id": new_id, "gamma": gamma_mid, "tau": tau,
                "depends_on": [closest_any["id"]], "deps_satisfy": "any",
                "status": "ready", "checkpoint": None, "result": None,
                "note": (f"γ-ladder between {closest_any['id']} (γ={g2}) "
                         f"and {task['id']} (γ={gamma}) at τ={tau}."),
            }
            if _insert_ladder(queue, task, ladder, requeue_count,
                              f"γ-ladder at γ={gamma_mid}"):
                return True

    # ── Step 4: noise perturbation as last resort ────────────────────────────
    existing_sp = dict(task.get("solver_params") or {})
    current_noise = float(existing_sp.get("noise_level", 0.0))
    next_noise = max(0.002, current_noise * 2.0) if current_noise > 0 else 0.002
    if next_noise <= 0.05:
        task["status"] = "ready"
        task["result"] = None
        task["checkpoint"] = None
        existing_sp["noise_level"] = next_noise
        existing_sp.setdefault("presmooth", 100)
        existing_sp.setdefault("presmooth_alpha", 0.02)
        task["solver_params"] = existing_sp
        task["requeue_count"] = requeue_count + 1
        task["note"] = (f"Auto-requeued (attempt {task['requeue_count']}): "
                        f"noise perturbation noise_level={next_noise:.3f}.")
        return True

    return False


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

    _update_summary(queue)
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

