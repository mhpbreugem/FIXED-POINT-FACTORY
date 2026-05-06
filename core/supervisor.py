#!/usr/bin/env python3
"""
supervisor.py — Local monitoring script for fixed-point-factory.

Run from your laptop (or any machine with the repo pulled) to see
VM health, claimed tasks, and stale workers at a glance.

Usage:
    python3 core/supervisor.py --project REZN
    python3 core/supervisor.py --project REZN --auto-release
    python3 core/supervisor.py --project REZN --pull
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True
    )
    return Path(result.stdout.strip() if result.returncode == 0 else ".")


def queue_path(project: str) -> Path:
    return repo_root() / "projects" / project / "TASK_QUEUE.json"


def heartbeat_dir(project: str) -> Path:
    return repo_root() / "projects" / project / "heartbeats"


def progress_dir(project: str) -> Path:
    return repo_root() / "projects" / project / "progress"


def load_progress(project: str) -> dict:
    """Return {task_id: progress_dict} from progress/*.json files."""
    out = {}
    pdir = progress_dir(project)
    if not pdir.is_dir():
        return out
    for f in pdir.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            continue
    return out


def show_progress(project: str) -> None:
    progress = load_progress(project)
    if not progress:
        return
    print_separator(f"LIVE PROGRESS  ({len(progress)} active solves)")
    now = datetime.datetime.now(datetime.timezone.utc)
    for task_id, p in sorted(progress.items()):
        worker = p.get("worker_id", "?")
        it = p.get("iter")
        ftol = p.get("ftol")
        last = p.get("last_update", "?")
        try:
            last_dt = datetime.datetime.fromisoformat(last.replace("Z", "+00:00"))
            age = (now - last_dt).total_seconds()
            age_s = f"{int(age)}s ago" if age < 120 else f"{int(age/60)}m ago"
            stale = "  STALE" if age > 180 else ""
        except Exception:
            age_s = "?"
            stale = ""
        print(f"  {task_id:<25}  by={worker:<14}  iter={it}  ftol={ftol}  "
              f"({age_s}){stale}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queue(project: str) -> dict:
    p = queue_path(project)
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def load_heartbeats(project: str) -> dict[str, datetime.datetime]:
    """Returns {worker_id: last_heartbeat_utc}."""
    hb_dir = heartbeat_dir(project)
    result = {}
    if not hb_dir.exists():
        return result
    for hb_file in hb_dir.glob("*.txt"):
        worker_id = hb_file.stem
        try:
            ts_str = hb_file.read_text().strip()
            ts = datetime.datetime.fromisoformat(ts_str.rstrip("Z"))
            result[worker_id] = ts
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def age_str(dt: datetime.datetime) -> str:
    now = datetime.datetime.utcnow()
    delta = now - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs//60}m ago"
    return f"{secs//3600}h{(secs%3600)//60}m ago"


def is_stale_heartbeat(dt: datetime.datetime, warn_minutes: int = 15) -> bool:
    now = datetime.datetime.utcnow()
    return (now - dt).total_seconds() > warn_minutes * 60


def is_stale_claim(claimed_at_str: str, max_hours: float = 6.0) -> bool:
    try:
        claimed_at = datetime.datetime.fromisoformat(claimed_at_str.rstrip("Z"))
        delta = datetime.datetime.utcnow() - claimed_at
        return delta.total_seconds() > max_hours * 3600
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
RED   = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{RESET}"
    return text


def print_separator(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {BOLD}{title}{RESET}")
    print('─'*60)


def show_vm_status(project: str) -> dict[str, datetime.datetime]:
    hbs = load_heartbeats(project)
    print_separator("ACTIVE VMs  (from heartbeat files)")
    if not hbs:
        print("  (no heartbeat files found)")
        return hbs

    for worker_id, last_hb in sorted(hbs.items()):
        stale = is_stale_heartbeat(last_hb)
        status_str = colored("STALE", RED) if stale else colored("alive", GREEN)
        print(f"  {worker_id:<30}  last={age_str(last_hb)}  [{status_str}]")

    return hbs


def show_queue_status(project: str, hbs: dict) -> None:
    queue = load_queue(project)
    tasks = queue["tasks"]

    by_status: dict[str, list] = {}
    for t in tasks:
        by_status.setdefault(t["status"], []).append(t)

    # Summary line
    counts = {s: len(v) for s, v in by_status.items()}
    total = len(tasks)
    summary_parts = [f"{s}={counts.get(s,0)}" for s in
                     ["done","ready","claimed","blocked","bailed","skip"]]
    print_separator(f"TASK QUEUE  ({total} total)")
    print("  " + "  ".join(summary_parts))

    # Claimed tasks (most important to show)
    claimed = by_status.get("claimed", [])
    if claimed:
        print(f"\n  CLAIMED ({len(claimed)}):")
        for t in claimed:
            cby = t.get("claimed_by", "?")
            cat = t.get("claimed_at", "?")
            stale = is_stale_claim(cat) if cat != "?" else False
            age = ""
            try:
                age = age_str(datetime.datetime.fromisoformat(cat.rstrip("Z")))
            except Exception:
                age = cat
            stale_str = colored(" [STALE CLAIM]", RED) if stale else ""
            alive_str = ""
            if cby in hbs:
                hb_stale = is_stale_heartbeat(hbs[cby])
                alive_str = colored(" [VM STALE]", YELLOW) if hb_stale else colored(" [VM alive]", GREEN)
            print(f"    {t['id']:<30}  by={cby}  {age}{stale_str}{alive_str}")

    # Ready tasks
    ready = by_status.get("ready", [])
    if ready:
        print(f"\n  READY ({len(ready)}):")
        for t in ready:
            print(f"    {t['id']}")

    # Bailed (problems)
    bailed = by_status.get("bailed", [])
    if bailed:
        print(f"\n  BAILED ({len(bailed)}):")
        for t in bailed:
            r = t.get("result") or {}
            reason = r.get("reason", "")[:70]
            print(f"    {t['id']:<30}  {reason}")


def show_done_summary(project: str) -> None:
    queue = load_queue(project)
    done = [t for t in queue["tasks"] if t["status"] == "done"]
    if not done:
        return
    print_separator(f"DONE ({len(done)}) — recent results")
    # Show last 10 done tasks
    for t in done[-10:]:
        r = t.get("result") or {}
        metric = ""
        for key in ("1-R2", "slope", "F_max"):
            if key in r and r[key] is not None:
                metric += f"  {key}={r[key]}"
        print(f"  {t['id']:<30}{metric}")


# ---------------------------------------------------------------------------
# Auto-release
# ---------------------------------------------------------------------------

def auto_release(project: str, max_age_hours: float = 6.0) -> None:
    sys.path.insert(0, str(repo_root()))
    from core.claim_task import release_stale_claims  # noqa: E402
    released = release_stale_claims(project, max_age_hours)
    if released:
        print(f"\n[auto-release] released: {released}")
    else:
        print("\n[auto-release] no stale claims to release")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="fixed-point-factory supervisor")
    parser.add_argument("--project", required=True, help="Project name (e.g. REZN)")
    parser.add_argument("--pull", action="store_true", help="git pull before showing status")
    parser.add_argument("--auto-release", action="store_true",
                        help="Release stale claims (> 6 h)")
    parser.add_argument("--max-age-hours", type=float, default=6.0,
                        help="Stale claim threshold in hours (default: 6)")
    args = parser.parse_args()

    if args.pull:
        print("[supervisor] pulling latest...")
        subprocess.run(["git", "pull", "--rebase"], check=False)

    print(f"\n{'='*60}")
    print(f"  fixed-point-factory supervisor   project={args.project}")
    print(f"  {datetime.datetime.utcnow().isoformat()}Z")
    print('='*60)

    hbs = show_vm_status(args.project)
    show_progress(args.project)
    show_queue_status(args.project, hbs)
    show_done_summary(args.project)

    if args.auto_release:
        auto_release(args.project, args.max_age_hours)

    print()


if __name__ == "__main__":
    main()
