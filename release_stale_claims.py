#!/usr/bin/env python3
"""release_stale_claims.py — reset claimed tasks whose worker is no longer alive.

A task is considered stale if it has been claimed for more than STALE_MINUTES
with no heartbeat from the worker. Run from repo root when no workers are active.

Idempotent. Skips tasks that are genuinely in-progress (check heartbeat files).
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

QUEUE = Path("projects/REZN/TASK_QUEUE.json")
HEARTBEAT_DIR = Path("projects/REZN/heartbeats")
STALE_MINUTES = 30


def is_heartbeat_alive(worker_id: str) -> bool:
    hb = HEARTBEAT_DIR / f"{worker_id}.txt"
    if not hb.exists():
        return False
    mtime = datetime.fromtimestamp(hb.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) < timedelta(minutes=STALE_MINUTES)


def main():
    q = json.loads(QUEUE.read_text())
    now = datetime.now(timezone.utc)
    count = 0
    for t in q["tasks"]:
        if t.get("status") != "claimed":
            continue
        worker = t.get("claimed_by") or ""
        claimed_at = t.get("claimed_at")

        # Check heartbeat if available
        if is_heartbeat_alive(worker):
            print(f"  LIVE   {t['id']:35s}  worker={worker}")
            continue

        # Check claim age if no heartbeat
        if claimed_at:
            age = now - datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))
            if age < timedelta(minutes=STALE_MINUTES):
                print(f"  RECENT {t['id']:35s}  age={int(age.total_seconds()//60)}m")
                continue

        old_worker = worker
        t["status"] = "ready"
        t["claimed_by"] = None
        t["claimed_at"] = None
        note = f"RESET: stale claim from {old_worker}"
        t["note"] = ((t.get("note") or "") + " | " + note).strip(" |")
        count += 1
        print(f"  reset  {t['id']:35s}  was claimed by {old_worker}")

    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    print(f"\nReset {count} stale claims → ready. Push to trigger workflow.")


if __name__ == "__main__":
    main()
