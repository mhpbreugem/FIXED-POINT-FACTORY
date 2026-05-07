# fixed-point-factory

A reusable platform for running parallel fixed-point solvers on cloud VMs.

Each project defines a mathematical fixed-point problem and a queue of
parameter-sweep tasks. The platform handles claiming tasks (git-race locking),
distributing work across spot VMs, heartbeating, and detecting stale workers.

---

## Architecture

```
fixed-point-factory/
├── core/                          # Project-agnostic infrastructure
│   ├── TASK_SCHEMA.md             # JSON schema and lifecycle rules
│   ├── SOLVER_INSTRUCTIONS.md     # Generic worker protocol
│   ├── claim_task.py              # Git-race locking (claim/done/bail/release)
│   ├── bootstrap.sh               # VM startup script
│   ├── create_gcp_vm.sh           # GCP spot VM provisioner
│   ├── heartbeat.sh               # Background heartbeat writer
│   └── supervisor.py              # Local monitoring script
│
└── projects/                      # One directory per project
    └── REZN/
        ├── EQUATIONS.md           # The math (model, fixed point, metric)
        ├── CHECKPOINT_FORMAT.md   # JSON schema for solver output
        ├── TASK_QUEUE.json        # Task list with statuses (the lock file)
        ├── SOLVER_INSTRUCTIONS.md # Project-specific solver addendum
        ├── README.md              # Project description and status
        ├── solver_code/           # Solver implementation (solve.py)
        ├── checkpoints/           # Local checkpoint storage
        └── heartbeats/            # VM heartbeat files
```

**Adding a new project**: see `docs/HOW_TO_ADD_A_PROJECT.md`.

---

## How it works

### Locking via git race

The `TASK_QUEUE.json` file IS the distributed lock. To claim a task:

1. Edit `status: "ready"` → `"claimed"`, add `claimed_by` and `claimed_at`.
2. `git commit` + `git push`.
3. If push is rejected (non-fast-forward), another worker beat you.
   Pull, rebase, pick a different task.

No external coordinator needed. Stale claims (VM died silently) are
auto-released after 6 hours by any worker on startup.

### Worker lifecycle

1. `.github/workflows/solve-tasks.yml` runs every 15 minutes.
2. The `schedule` job claims up to N ready tasks atomically and pushes the claims.
3. Parallel `solve` jobs each run `projects/$PROJECT/solver_code/solve.py` end-to-end.
4. Results are committed and pushed. Stale claims (jobs that fail mid-run) are
   released by `.github/workflows/cleanup.yml`, which runs hourly at :17.

---

## Running workers

Workers run as GitHub Actions jobs — no external accounts needed.

- **Automatic**: cron every 15 min via `.github/workflows/solve-tasks.yml`
- **Manual**: GitHub UI → Actions → "Solve tasks" → Run workflow

Live status: https://raw.githack.com/mhpbreugem/fixed-point-factory/main/docs/status.html

Hourly cleanup of stale claims: `.github/workflows/cleanup.yml`

The GCP-based scripts in `core/bootstrap.sh`, `core/create_gcp_vm.sh`,
and `core/heartbeat.sh` are deprecated but kept for the case where
we need to scale beyond Actions concurrency limits (currently 20
parallel jobs on public repos).

### Monitor from your laptop

```bash
python3 core/supervisor.py --project REZN --pull
```

To release stale claims automatically:

```bash
python3 core/supervisor.py --project REZN --auto-release
```

### Check task status

```bash
python3 core/claim_task.py status --project REZN
```

### Run a solver locally (without a VM)

```bash
export PROJECT=REZN
export BRANCH=main
export WORKER_ID=local-$(hostname)
bash core/bootstrap.sh
```

---

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [REZN](projects/REZN/README.md) | REE posterior fixed point, CRRA preference sweep | Active |

---

## Retry / network failure policy

`core/claim_task.py` retries pushes up to 4 times with exponential backoff
(2s, 4s, 8s, 16s) on network errors. A rejected push (non-fast-forward) is
NOT retried — it means another worker claimed the task first.

---

## Requirements

- Python 3.10+ (for `|` union type hints)
- `git` (for locking)
- For solver VMs: `mpmath`, `numpy`, `scipy` (installed by bootstrap.sh)
- For GCP provisioning: `gcloud` CLI, authenticated to your project
