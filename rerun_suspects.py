#!/usr/bin/env python3
"""rerun_suspects.py — rerun stale γ=1.0 results and the γ=1.4 τ=2 anomaly.

The 8 γ=1.0 tasks at τ ∈ {0.3,0.5,0.8,1.0,1.5,2.0,3.0,4.0} all point
to legacy mp50 JSON checkpoints that no longer exist in the repo. Their
cached 1-R² values are from a previous solver path and must be discarded.

The g140_t0200 task (γ=1.4, τ=2) yielded 1-R²=0.084, which sits at the
γ=2.0 result (0.083) — non-monotone in γ for fixed τ. Rerun to verify.

Idempotent. Run from repo root.
"""
import json
from pathlib import Path

QUEUE = Path("projects/REZN/TASK_QUEUE.json")

# Anchors (currently done, valid .npz checkpoints):
ANCHORS_G100 = ["g100_t0500", "g050_t0200"]
ANCHORS_G140 = ["g140_t0240", "g050_t0200", "g200_t0200"]

stale_g100 = [
    "g100_t0030", "g100_t0050", "g100_t0080", "g100_t0100",
    "g100_t0150", "g100_t0200", "g100_t0300", "g100_t0400",
]
suspect_g140 = ["g140_t0200"]


def main():
    q = json.loads(QUEUE.read_text())
    by_id = {t["id"]: t for t in q["tasks"]}
    count = 0
    for tid in stale_g100 + suspect_g140:
        t = by_id.get(tid)
        if t is None:
            print(f"  SKIP {tid}: not in queue")
            continue
        if t.get("status") in ("ready", "claimed"):
            print(f"  SKIP {tid}: already {t['status']}")
            continue
        old_r2 = (t.get("result") or {}).get("1-R2")
        old_ckpt = t.get("checkpoint")
        t["status"] = "ready"
        t["claimed_by"] = None
        t["claimed_at"] = None
        t["completed_at"] = None
        t["result"] = None
        t["checkpoint"] = None
        t["depends_on"] = ANCHORS_G100 if tid in stale_g100 else ANCHORS_G140
        t["deps_satisfy"] = "any"
        rerun_note = (f"RERUN: previous 1-R²={old_r2} from "
                      f"{old_ckpt or 'no ckpt'}; warm-start any-of {t['depends_on']}.")
        t["note"] = ((t.get("note") or "") + " | " + rerun_note).strip(" |")
        t.pop("requeue_count", None)
        count += 1
        print(f"  reset {tid}  γ={t.get('gamma')} τ={t.get('tau')}")
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    print(f"\nReset {count} tasks. Push to trigger workflow.")


if __name__ == "__main__":
    main()
