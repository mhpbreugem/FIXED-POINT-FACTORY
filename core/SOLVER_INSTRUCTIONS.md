# SOLVER INSTRUCTIONS — fixed-point-factory worker protocol

This file is the contract for any worker VM that wants to claim and
solve tasks from `projects/$PROJECT/TASK_QUEUE.json`.

Read it end-to-end before doing anything.

Project-specific references (model equations, convergence metric,
checkpoint format) live in:

- `projects/$PROJECT/EQUATIONS.md`       — model and fixed-point system
- `projects/$PROJECT/CHECKPOINT_FORMAT.md` — JSON schema for checkpoints
- `projects/$PROJECT/TASK_QUEUE.json`    — list of tasks to solve
- `projects/$PROJECT/SOLVER_INSTRUCTIONS.md` — project-specific addenda

The schema for TASK_QUEUE.json is documented in `core/TASK_SCHEMA.md`.

---

## 1. ENVIRONMENT

The following environment variables are expected:

| Variable          | Required | Default         | Purpose |
|-------------------|----------|-----------------|---------|
| `PROJECT`         | yes      | —               | Which project to work on (e.g. `REZN`). |
| `WORKER_ID`       | no       | hostname:pid    | Unique identifier for this VM/process. |
| `REPO_ROOT`       | no       | current dir     | Absolute path to the repo. |
| `MAX_RUN_HOURS`   | no       | unlimited       | Exit cleanly after this many hours. |
| `BRANCH`          | no       | current branch  | Git branch to push claims/results to. |
| `GITHUB_TOKEN`    | if needed | —              | For authenticated git push. |

---

## 2. MAIN LOOP

```
while True:
    1. git pull --rebase origin $BRANCH
    2. release any stale claims (> 6 h old) — see claim_task.py
    3. find a READY task (status=ready, deps satisfied)
    4. if none: sleep 60 s, continue
    5. try_claim(task) — git commit + push; if push fails, continue
    6. [claim landed] run the solver (project-specific)
    7. verify checkpoint invariants
    8. mark_done(task, results) — git commit + push result + queue update
    9. if MAX_RUN_HOURS elapsed: release any partial claim, exit
```

The claim push (step 5) is the synchronisation barrier. **Do not start
the solver until the push has landed on origin.**

See `core/claim_task.py` for the implementation of steps 2-5, 7-8.

---

## 3. FINDING A READY TASK

A task is READY when:
- `status == "ready"`, AND
- its dependencies are satisfied per `deps_satisfy`:
  - `"all"` (default): every id in `depends_on` must be `done`
  - `"any"`: at least one id in `depends_on` must be `done`

```python
import json, hashlib, os, socket

with open(f"projects/{PROJECT}/TASK_QUEUE.json") as f:
    queue = json.load(f)

done = {t["id"] for t in queue["tasks"] if t["status"] == "done"}
default_mode = queue.get("deps_semantics", {}).get("default", "all")

def deps_ok(t):
    deps = set(t.get("depends_on", []))
    mode = t.get("deps_satisfy", default_mode)
    if mode == "any":
        return bool(deps & done) or not deps
    return deps <= done

ready = [t for t in queue["tasks"] if t["status"] == "ready" and deps_ok(t)]
```

**Worker-specific tiebreak** — prevents all workers racing for the
same task when multiple are ready simultaneously:

```python
worker_id = os.environ.get("WORKER_ID",
                socket.gethostname() + ":" + str(os.getpid()))

def pick_priority(t):
    return hashlib.sha256(f"{worker_id}|{t['id']}".encode()).hexdigest()

task = min(ready, key=pick_priority)
```

---

## 4. CLAIMING A TASK (git-race locking)

```bash
git pull --rebase origin $BRANCH

# edit TASK_QUEUE.json: status "ready" → "claimed", set claimed_by + claimed_at
python3 core/claim_task.py claim --project $PROJECT --task-id $TASK_ID

git add projects/$PROJECT/TASK_QUEUE.json
git commit -m "claim $TASK_ID"
git push origin $BRANCH
# if push fails → rebase + try another task
```

If push is rejected (non-fast-forward), another worker beat you.
Rebase, drop your claim, and pick a different task. If no other task is
ready, sleep and retry the main loop.

**Never start the solver before the claim push succeeds.**

---

## 5. LOADING THE WARM-START

Each task has a `depends_on` list. The warm-start checkpoint comes from
the dependency's `checkpoint` field in TASK_QUEUE.json. Read the
project's `CHECKPOINT_FORMAT.md` for the exact loading procedure.

General warm-start rules:
- If only the search parameter changes (e.g. τ changes), the grid may
  need to be recomputed and μ* interpolated onto the new grid.
- If only the preference parameter changes (e.g. γ changes at fixed τ),
  the grid is often identical and μ* transfers directly.
- If a task has no dependencies, it is a seed task — no warm-start.

---

## 6. CONVERGENCE QUALITY METRIC

Each project defines its own convergence metric in `EQUATIONS.md`.
Write the metric value(s) into the task's `result` field on completion.

The metric is used to:
1. Verify the solution is acceptable (not just technically converged).
2. Populate the paper figures.
3. Alert the supervisor when a solution is suspicious.

---

## 7. SAVING A CHECKPOINT

Write the output file to the path implied by the task (see project's
`CHECKPOINT_FORMAT.md` for naming convention and schema). Verify all
invariants before committing. A worker MUST refuse to commit a
checkpoint that fails invariant checks.

---

## 8. MARKING A TASK DONE

```bash
python3 core/claim_task.py done \
    --project $PROJECT \
    --task-id $TASK_ID \
    --checkpoint $CHECKPOINT_PATH \
    --result '{"1-R2": 0.085, "slope": 0.543, "F_max": "1.2e-26"}'

git add projects/$PROJECT/TASK_QUEUE.json $CHECKPOINT_PATH
git commit -m "$TASK_ID: 1-R2=0.085 done"
git push origin $BRANCH
```

On push conflict: rebase, re-resolve TASK_QUEUE.json (preserve all
`done` entries from origin), re-push.

---

## 9. WHAT TO DO WHEN A TASK FAILS

1. Try the remedies listed in the project's `SOLVER_INSTRUCTIONS.md`
   (e.g. smaller step size, reduced grid).
2. If still no convergence: mark `bailed` with a short reason.

```bash
python3 core/claim_task.py bail \
    --project $PROJECT \
    --task-id $TASK_ID \
    --reason "did not converge below F_max=1e-3 after 200 iters"
```

A bailed task blocks all downstream tasks in its dependency chain.

---

## 10. PARALLEL SAFETY

- Two workers running different tasks write disjoint checkpoint files —
  no conflict there.
- Both update the same `TASK_QUEUE.json`. The claim push is the sync
  primitive: whichever lands first wins.
- Never resolve a TASK_QUEUE.json merge conflict by overwriting — always
  preserve every `done` and `bailed` entry from origin.

---

## 11. CLEAN EXIT

Before exiting (whether by `MAX_RUN_HOURS` elapsed, signal, or error),
release any task that is `claimed` by this worker but not yet `done`:

```bash
python3 core/claim_task.py release --project $PROJECT --worker-id $WORKER_ID
```

This flips the task back to `ready` so another worker can pick it up.
