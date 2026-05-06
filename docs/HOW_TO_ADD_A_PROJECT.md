# How to add a new project

A "project" is a fixed-point problem with its own parameter sweep, solver code,
and checkpoint format. Adding one takes about 30 minutes.

---

## Step 1 — Create the project directory

```bash
mkdir -p projects/MYPROJECT/heartbeats
mkdir -p projects/MYPROJECT/checkpoints
mkdir -p projects/MYPROJECT/solver_code
mkdir -p projects/MYPROJECT/progress
touch projects/MYPROJECT/heartbeats/.gitkeep
touch projects/MYPROJECT/checkpoints/.gitkeep
touch projects/MYPROJECT/solver_code/.gitkeep
touch projects/MYPROJECT/progress/.gitkeep
```

Workers will write live `iter`/`ftol` telemetry to
`projects/MYPROJECT/progress/$TASK_ID.json` once per minute.
See `core/PROGRESS_FORMAT.md` for the file schema and lifecycle.

---

## Step 2 — Write EQUATIONS.md

Document the mathematical problem your solver will solve. Must include:

1. **Primitives** — state space, agents, signal structure, parameters.
2. **The fixed point** — `μ* = Φ(μ*)` or equivalent. Define Φ precisely.
3. **Convergence quality metric** — how you measure whether a solution is
   "good" (not just technically converged). This goes in the `result` field.
   Example: weighted 1-R² of a regression on a sufficient statistic.
4. **Warm-start rules** — when parameters change, how to initialise from
   a nearby checkpoint (interpolation strategy, grid recomputation, etc.).

See `projects/REZN/EQUATIONS.md` for a complete example.

---

## Step 3 — Write CHECKPOINT_FORMAT.md

Document the JSON schema for solver output files. Must specify:

- File naming convention.
- Required fields and their types (use string representation for high-precision numbers).
- Invariants that a worker must verify before committing a checkpoint.
- How to read the checkpoint back in Python.

See `projects/REZN/CHECKPOINT_FORMAT.md` for a complete example.

---

## Step 4 — Write TASK_QUEUE.json

Create the initial task queue following `core/TASK_SCHEMA.md`. Key decisions:

- **Task granularity**: one task per (parameter set). Tasks that can warm-start
  from each other should be chained via `depends_on`.
- **deps_satisfy**: use `"any"` when a task can warm-start from any of several
  nearby checkpoints (enables more parallel starts).
- **Seed task**: the first task in the chain either has no dependencies (it's a
  cold start) or depends on a manually computed checkpoint.
- **Parallel fanout**: expose multiple independent `ready` tasks so parallel
  VMs don't all race for the same task. See REZN's `fanout` block for an example.

Initial status for all tasks: `"ready"` (if no deps) or `"blocked"`.

```json
{
  "queue_version": 1,
  "updated_at": "2026-01-01T00:00:00Z",
  "params": { ... },
  "tasks": [
    {
      "id": "param_set_1",
      "status": "ready",
      "depends_on": [],
      "checkpoint": null,
      "result": null
    }
  ],
  "summary": { "total": 1, "ready": 1, "done": 0, "blocked": 0 },
  "deps_semantics": { "default": "all" },
  "notes": []
}
```

---

## Step 5 — Write the solver (solver_code/solve.py)

The entry point called by `core/bootstrap.sh`:

```bash
python3 projects/MYPROJECT/solver_code/solve.py \
    --project MYPROJECT \
    --task-id param_set_1 \
    --branch main \
    --worker-id solver-1
```

The solver must:

1. Load the task from TASK_QUEUE.json (use `core/claim_task.py` Python API).
2. Load the warm-start checkpoint if the task has a dependency.
3. Run the fixed-point iteration.
4. Verify checkpoint invariants (don't commit a bad result).
5. Call `mark_done(project, task_id, checkpoint_path, result_dict)`.

The solver should also handle the `bailed` path (call `mark_failed` if it
cannot converge after retries).

```python
import sys, os
sys.path.insert(0, os.environ.get("REPO_ROOT", "."))
from core.claim_task import mark_done, mark_failed

# ... solve ...

mark_done(
    project="MYPROJECT",
    task_id=task_id,
    checkpoint="projects/MYPROJECT/checkpoints/task_001.json",
    result={"metric": 0.042, "F_max": "1.2e-26"}
)
```

---

## Step 6 — Write SOLVER_INSTRUCTIONS.md (optional but recommended)

A project-specific addendum to `core/SOLVER_INSTRUCTIONS.md`. Covers:

- Project-specific parameters (grid sizes, tolerances, dps).
- Warm-start interpolation details.
- What "bailed" means for this project.
- Checkpoint naming convention.

---

## Step 7 — Write README.md

Brief description of the project:

- What problem it solves and why.
- Current task status (X done, Y ready, Z blocked).
- Key results so far.
- Seed checkpoint location (if any).

---

## Step 8 — Test locally

```bash
export PROJECT=MYPROJECT
export BRANCH=main
export WORKER_ID=test-local

# Check that the queue parses cleanly
python3 -c "import json; q=json.load(open('projects/MYPROJECT/TASK_QUEUE.json')); print(q['summary'])"

# Check that claim_task.py can find a ready task
python3 core/claim_task.py status --project MYPROJECT

# Run bootstrap locally (will install deps and enter the worker loop)
bash core/bootstrap.sh
```

---

## Step 9 — Spin up VMs

```bash
GITHUB_TOKEN=ghp_xxx \
PROJECT=MYPROJECT \
BRANCH=main \
VM_NAME=solver-1 \
bash core/create_gcp_vm.sh
```

Monitor:

```bash
python3 core/supervisor.py --project MYPROJECT --pull
```
