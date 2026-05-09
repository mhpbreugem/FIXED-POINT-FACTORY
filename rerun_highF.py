#!/usr/bin/env python3
"""rerun_highF.py — requeue done tasks whose F_max exceeds 1e-25.

These tasks converged to a checkpoint but not to the required Newton
tolerance. Reset to ready so the solver warm-starts from the checkpoint
and continues Newton iterations.

Idempotent. Run from repo root.
"""
import json
from pathlib import Path

QUEUE = Path("projects/REZN/TASK_QUEUE.json")
F_THRESH = 1e-25
SKIP = {"test_ci_smoke"}


def main():
    q = json.loads(QUEUE.read_text())
    count = 0
    for t in q["tasks"]:
        if t.get("status") != "done":
            continue
        if t["id"] in SKIP:
            continue
        f = (t.get("result") or {}).get("F_max")
        if not f or f <= F_THRESH:
            continue
        old_f = f
        t["status"] = "ready"
        t["claimed_by"] = None
        t["claimed_at"] = None
        t["completed_at"] = None
        # Keep checkpoint for warm-start; clear result so solver records fresh one
        t["result"] = None
        note = f"RERUN-NEWTON: F_max={old_f:.3e} > {F_THRESH:.0e}; warm-start from {t.get('checkpoint') or 'ckpt'}."
        t["note"] = ((t.get("note") or "") + " | " + note).strip(" |")
        count += 1
        print(f"  reset {t['id']:35s}  F_max was {old_f:.3e}  ckpt={t.get('checkpoint')}")
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    print(f"\nReset {count} tasks. Push to trigger workflow.")


if __name__ == "__main__":
    main()
