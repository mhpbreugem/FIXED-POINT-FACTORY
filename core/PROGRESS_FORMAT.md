# Worker progress files

Each worker that is actively solving a task writes its live state to:

    projects/$PROJECT/progress/$TASK_ID.json

This file is overwritten approximately every 60 seconds by the worker.
It is **not** the canonical state of the task — that lives in
`TASK_QUEUE.json`. The progress file is purely live telemetry, the
solver-side analogue of `heartbeats/$WORKER_ID.txt`.

## File format

```json
{
  "task_id": "g400_t1000",
  "worker_id": "solver-1",
  "started_at": "2026-05-06T18:00:00Z",
  "last_update": "2026-05-06T18:07:00Z",
  "iter": 5,
  "ftol": "1.234e-12",
  "extra": {}
}
```

| Field         | Meaning |
|---------------|---------|
| `task_id`     | Task this progress refers to. Matches the JSON filename. |
| `worker_id`   | VM that is currently solving the task. |
| `started_at`  | ISO-8601 UTC time when the worker began the task. |
| `last_update` | ISO-8601 UTC time of the latest `update()` call. |
| `iter`        | Current outer iteration count (e.g. Newton step #). |
| `ftol`        | Current residual norm as a string (preserves mp precision). |
| `extra`       | Solver-defined extra fields (e.g. damping, picard step). |

`ftol` is a string because solvers using mpmath may produce values
below float underflow (e.g. `"7.4e-119"`). Supervisors should treat
`ftol` as opaque text unless they explicitly parse it.

---

## Lifecycle

1. Worker claims a task via `claim_task.py claim`.
2. Worker starts a `ProgressReporter` and calls `start()`.
3. Worker calls `reporter.update(iter=k, ftol=err)` from its inner loop.
4. Background thread flushes the latest snapshot every `interval` seconds
   (default 60). Flush = write file → commit → push (with rebase retry).
5. On task completion (`done` or `bailed`), worker calls `reporter.stop()`,
   which deletes the progress file and pushes the deletion.

---

## Why a separate file (and not a field on the task)?

Updating the task in `TASK_QUEUE.json` every minute would:

- conflict with `claim_task.py` operations (the task queue is the lock file),
- create a heavy commit stream from N workers × every minute,
- slow down `git pull` for every supervisor and worker.

A per-task file is a single-writer object: only the claiming worker
writes it. This eliminates contention and keeps the canonical task
queue stable.

---

## Reading progress

The supervisor (`core/supervisor.py`) reads progress files alongside
heartbeats and prints a unified status table. Anything that needs
live telemetry should read these files; anything that needs the
canonical state of a task should read `TASK_QUEUE.json`.
