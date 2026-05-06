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

### VM lifecycle

1. `core/create_gcp_vm.sh` spins up a GCP spot VM with `max-run-duration=1h`.
2. On startup, the VM runs `core/bootstrap.sh`:
   - Installs Python + dependencies
   - Clones this repo
   - Starts `core/heartbeat.sh` in background (writes a timestamp every 5 min)
   - Enters the worker loop: claim → solve → done → repeat
3. After 1 hour, GCP terminates the VM. Any claimed task is recovered by
   the 6-hour stale claim mechanism.

### Heartbeat

Each VM writes `projects/<PROJECT>/heartbeats/<VM_ID>.txt` every 5 minutes
and pushes. The supervisor reads these files to detect live vs. stuck VMs.

---

## Quick start

### Spin up a solver VM (GCP)

```bash
GITHUB_TOKEN=ghp_xxx \
PROJECT=REZN \
BRANCH=main \
VM_NAME=solver-1 \
bash core/create_gcp_vm.sh
```

Spin up multiple VMs by changing `VM_NAME` (e.g. `solver-2`, `solver-3`).
Each will automatically claim different tasks due to the worker-specific tiebreak.

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
