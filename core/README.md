# core/

Project-agnostic infrastructure for fixed-point-factory.

| File                   | Purpose |
|------------------------|---------|
| `TASK_SCHEMA.md`       | JSON schema and lifecycle rules for TASK_QUEUE.json |
| `SOLVER_INSTRUCTIONS.md` | Generic worker protocol (claim → solve → done) |
| `claim_task.py`        | Git-race locking: claim, done, bail, release, status |
| `bootstrap.sh`         | VM startup: install deps, clone repo, start worker loop |
| `create_gcp_vm.sh`     | Spin up a GCP spot VM with 1-hour max-run-duration |
| `heartbeat.sh`         | Background process: write timestamp every 5 min, push |
| `supervisor.py`        | Local monitoring: VM health, stale claims, queue status |

None of these files mention any specific project. Project-specific math,
task queues, solver code, and checkpoint formats live in `projects/<PROJECT>/`.
