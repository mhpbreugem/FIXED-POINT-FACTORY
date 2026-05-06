# TASK SCHEMA

Every `projects/<PROJECT>/TASK_QUEUE.json` must follow this schema.
The file is the single source of truth for task state — it also acts as
the distributed lock (via the git-race protocol in `claim_task.py`).

---

## Top-level structure

```json
{
  "queue_version": 1,
  "updated_at": "<ISO-8601 timestamp>",
  "params": { ... },
  "tasks": [ ... ],
  "summary": { ... },
  "deps_semantics": { ... },
  "notes": [ ... ]
}
```

`params` is project-specific. Suggested fields: `G`, `dps`, `F_tol`,
`weighting`. Workers may read `params` to configure themselves.

---

## Task fields

| Field          | Type                | Required | Description |
|----------------|---------------------|----------|-------------|
| `id`           | string              | yes      | Unique within the queue. Use a slug (e.g. `g050_t0400`). |
| `status`       | enum (see below)    | yes      | Current lifecycle state. |
| `depends_on`   | array of ids        | yes      | Empty array if no dependencies. |
| `deps_satisfy` | `"all"` or `"any"` | no       | Default `"all"`. Use `"any"` when the task can warm-start from any one of several checkpoints. |
| `checkpoint`   | string or null      | yes      | Relative path to output checkpoint within `projects/<PROJECT>/` (or `results/` at the repo root). Null until done. |
| `result`       | object or null      | yes      | Metric summary written by the worker on completion. Schema is project-specific. Null until done. |
| `claimed_by`   | string or null      | no       | VM hostname that holds the claim. Set during `claimed`; cleared on done/failed. |
| `claimed_at`   | ISO-8601 or null    | no       | When the claim was made. Used to detect stale claims (> 6 h). |
| `completed_at` | ISO-8601 or null    | no       | When the worker flipped to `done`. |
| `note`         | string              | no       | Free text. Explain unusual warm-start choices, bail reasons, etc. |

---

## Status lifecycle

```
    ready ──► claimed ──► done
                │
                ▼
             bailed
```

| Status    | Meaning |
|-----------|---------|
| `ready`   | Dependencies are satisfied. Any worker may claim this task. |
| `claimed` | A worker has claimed it and is (or was) solving. Push has landed on origin before the solve starts. |
| `done`    | Converged. `checkpoint` and `result` are filled. |
| `bailed`  | Failed to converge despite retries. `checkpoint` is null; `result.reason` explains. Downstream tasks remain `blocked`. |
| `blocked` | A required dependency is not yet `done`. Workers skip blocked tasks. |
| `skip`    | Intentionally excluded (e.g. superseded by a better run). Workers skip these. |

**Transition rules (enforced by `claim_task.py`):**
- `ready → claimed`: set `claimed_by`, `claimed_at`.
- `claimed → done`: set `checkpoint`, `result`, `completed_at`; clear `claimed_by`, `claimed_at`.
- `claimed → bailed`: set `result.reason`; clear `claimed_by`, `claimed_at`.
- A worker MUST NOT mark a task `done` without a passing checkpoint invariant check (see project's `CHECKPOINT_FORMAT.md`).

---

## Stale claim recovery

If `claimed_at` is more than 6 hours ago and the task is still
`claimed`, any worker (or `supervisor.py --auto-release`) may flip it
back to `ready`. This handles silent VM death without any central
coordinator.

---

## deps_semantics block

```json
"deps_semantics": {
  "default": "all",
  "any": "Used when a task can warm-start from any of several checkpoints."
}
```

---

## summary block

Informational; not enforced by workers. Recompute manually when the
queue changes significantly.

```json
"summary": {
  "total":   51,
  "done":    25,
  "ready":    6,
  "blocked": 18,
  "bailed":   2
}
```
