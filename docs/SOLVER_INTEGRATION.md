# How a project's solve.py integrates progress reporting

Three lines of integration; the rest is pre-built.

---

## Minimal example

```python
# projects/REZN/solver_code/solve.py
import argparse
from pathlib import Path
import sys

# Make core importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core"))
from progress import ProgressReporter

def solve(project, task_id, branch, worker_id):
    reporter = ProgressReporter(
        project=project,
        task_id=task_id,
        worker_id=worker_id,
        branch=branch,
        interval=60,           # flush every minute
    )
    reporter.start()
    try:
        # ----- your existing convergence loop -----
        for it in range(MAX_ITER):
            ftol = newton_step(...)        # whatever your inner step does
            reporter.update(iter=it, ftol=ftol)
            if float(ftol) < TARGET_TOL:
                break
        # ----- end of solver loop -----
    finally:
        reporter.stop()        # delete progress file, push deletion

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--branch", default="main")
    p.add_argument("--worker-id", required=True)
    args = p.parse_args()
    solve(args.project, args.task_id, args.branch, args.worker_id)
```

---

## Notes

**`update()` is non-blocking.** The git push happens in a background
thread every `interval` seconds. Calling `update()` 1000 times per
second is fine — only the latest snapshot is pushed.

**`ftol` accepts any of:** `float`, `mpmath.mpf`, or `string`. mpmath values
are converted to scientific-notation strings to preserve digits like
`7.4e-119` that would underflow a float.

**`extra` kwargs** are persisted under `extra` in the JSON. Use this
for solver-specific fields (damping coefficient, Picard step number,
etc.):

```python
reporter.update(iter=it, ftol=err, picard_step=k, alpha=0.3)
```

**If the solver crashes** without calling `stop()`, the progress file
remains in the repo. The supervisor flags it as `STALE` after 3
minutes. The next worker that picks up the task (after stale-claim
reclaim) will overwrite it.
