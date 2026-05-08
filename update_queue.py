#!/usr/bin/env python3
"""
update_queue.py - Idempotent queue patch.

Actions (all idempotent, safe to re-run):
1. Reactivate suspect points for re-run
2. Fill gamma=4.0 tau sub-sweep
3. Add gamma=0.1 tau sub-sweep
4. Drop g050_t2000 (outlier), add g050_t1200
5. Rebuild all extraction tasks for all 7 gamma values with deps_satisfy=all
"""
import json
from collections import Counter
from pathlib import Path

QUEUE_PATH = Path("projects/REZN/TASK_QUEUE.json")


def load():
    return json.loads(QUEUE_PATH.read_text())


def save(q):
    QUEUE_PATH.write_text(json.dumps(q, indent=2))


def find_task(tasks, tid):
    for t in tasks:
        if t["id"] == tid:
            return t
    return None


def main():
    q = load()
    tasks = q["tasks"]
    ids = lambda: {t["id"] for t in tasks}

    # ------------------------------------------------------------------
    # 1. Reactivate suspect points to ready (clear result/checkpoint)
    # ------------------------------------------------------------------
    suspects = ["g140_t0200", "g400_t0200", "g100_t0500"]
    for sid in suspects:
        t = find_task(tasks, sid)
        if t and t["status"] == "done":
            print(f"  reactivate {sid}: done -> ready")
            t["status"] = "ready"
            t["checkpoint"] = None
            t["result"] = None
            t.pop("completed_at", None)

    # g050_t0400 - new solver task
    if "g050_t0400" not in ids():
        print("  add g050_t0400")
        tasks.append({"id": "g050_t0400", "gamma": 0.5, "tau": 4.0,
                       "status": "ready", "checkpoint": None, "result": None})

    # ------------------------------------------------------------------
    # 2. Fill gamma=4.0 tau sub-sweep (idempotent)
    # ------------------------------------------------------------------
    for tid, tau in [("g400_t0030", 0.3), ("g400_t0050", 0.5),
                     ("g400_t0100", 1.0), ("g400_t0300", 3.0), ("g400_t0500", 5.0)]:
        if tid not in ids():
            print(f"  add {tid}")
            tasks.append({"id": tid, "gamma": 4.0, "tau": tau,
                           "status": "ready", "checkpoint": None, "result": None})

    # ------------------------------------------------------------------
    # 3. Add gamma=0.1 tau sub-sweep
    # ------------------------------------------------------------------
    for tid, tau in [("g010_t0050", 0.5), ("g010_t0100", 1.0), ("g010_t0150", 1.5),
                     ("g010_t0200", 2.0), ("g010_t0300", 3.0), ("g010_t0500", 5.0)]:
        if tid not in ids():
            print(f"  add {tid}")
            tasks.append({"id": tid, "gamma": 0.1, "tau": tau,
                           "status": "ready", "checkpoint": None, "result": None})

    # ------------------------------------------------------------------
    # 4. Drop g050_t2000 (outlier), add g050_t1200
    # ------------------------------------------------------------------
    t2000 = find_task(tasks, "g050_t2000")
    if t2000 is None:
        print("  add g050_t2000 as skip (outlier)")
        tasks.append({"id": "g050_t2000", "gamma": 0.5, "tau": 20.0,
                       "status": "skip", "note": "outlier - excluded from fit", "result": None})
    elif t2000["status"] not in ("skip",):
        print(f"  set g050_t2000 -> skip (outlier)")
        t2000["status"] = "skip"
        t2000.setdefault("note", "")
        t2000["note"] += " [outlier - excluded]"

    if "g050_t1200" not in ids():
        print("  add g050_t1200")
        tasks.append({"id": "g050_t1200", "gamma": 0.5, "tau": 12.0,
                       "status": "ready", "checkpoint": None, "result": None})

    # ------------------------------------------------------------------
    # 5. Rebuild extraction tasks for all 7 gamma values
    # ------------------------------------------------------------------
    GAMMAS = {0.1: "g010", 0.25: "g025", 0.5: "g050",
              1.0: "g100", 1.4: "g140", 2.0: "g200", 4.0: "g400"}

    def is_solver(t):
        tid = t["id"]
        return (not tid.startswith("test_")
                and not tid.startswith("extract_")
                and not tid.startswith("nl_r2_")
                and t.get("status") != "skip")

    # Compute deps per gamma BEFORE removing old extraction tasks
    gamma_deps = {g: [] for g in GAMMAS}
    for t in tasks:
        if is_solver(t) and t.get("gamma") in gamma_deps:
            gamma_deps[t["gamma"]].append(t["id"])
    for g in gamma_deps:
        gamma_deps[g] = sorted(gamma_deps[g])

    # Remove old extraction tasks
    extract_prefixes = ("extract_volume_", "extract_value_info_")
    removed = [t["id"] for t in tasks if any(t["id"].startswith(p) for p in extract_prefixes)]
    if removed:
        print(f"  removing old extraction tasks: {removed}")
    tasks[:] = [t for t in tasks if not any(t["id"].startswith(p) for p in extract_prefixes)]

    # Add rebuilt extraction tasks
    for gamma, code in GAMMAS.items():
        deps = gamma_deps[gamma]
        for kind, fig in (("volume", "7"), ("value_info", "8")):
            tid = f"extract_{kind}_{code}"
            print(f"  rebuild {tid} ({len(deps)} deps)")
            tasks.append({
                "id": tid,
                "gamma": gamma,
                "tau": None,
                "status": "blocked",
                "deps_satisfy": "all",
                "depends_on": deps,
                "note": f"Fig {fig} extraction for gamma={gamma}",
                "result": None,
            })

    save(q)

    counts = Counter(t["status"] for t in tasks)
    print(f"\nDone. {len(tasks)} total tasks:")
    for s, n in sorted(counts.items()):
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()
