# HANDOFF — fixed-point-factory

Read this whole file before doing anything. It captures the state of
the project and the supervisor's mental model so you can pick up
directly from here.

## What this repo is

A reusable platform for running parallel fixed-point solvers on cloud
VMs. `core/` is project-agnostic infrastructure (locking, heartbeats,
progress, GCP provisioning). `projects/` contains one subdirectory per
project with that project's math, task queue, and solver code.

Currently one project: `REZN`. It computes rational-expectations
equilibria for the paper "On the Possibility of Informationally
Inefficient Markets Without Noise" (Matthijs Breugem, posted to SSRN
2 May 2026, abstract id 6693198). The paper repo is separate:
`github.com/mhpbreugem/REZN`.

## Live dashboard

Status of the queue, active workers, and live `iter`/`ftol` progress
is rendered at:

    https://raw.githack.com/mhpbreugem/fixed-point-factory/main/docs/status.html

Auto-refreshes every 60 s. Implemented as a static HTML file calling
GitHub's public API. Source: `docs/status.html`.

## Architecture

```
core/
  bootstrap.sh           VM startup; clones repo, runs main worker loop
  create_gcp_vm.sh       GCP spot VM provisioner
  heartbeat.sh           Background liveness ping (every 5 min)
  progress.py            Background iter/ftol reporter (every 60 s)
  claim_task.py          Git-race locking primitives (claim/done/bail/release)
  supervisor.py          Local CLI: status table, stale-claim detection
  SOLVER_INSTRUCTIONS.md Generic protocol every worker follows
  PROGRESS_FORMAT.md     Schema for progress/$TASK_ID.json
  TASK_SCHEMA.md         Schema and lifecycle for tasks

projects/REZN/
  EQUATIONS.md           The fixed-point system F(μ*) = 0
  CHECKPOINT_FORMAT.md   JSON schema for converged-μ* outputs
  TASK_QUEUE.json        Canonical task list (the lock file)
  SOLVER_INSTRUCTIONS.md Project-specific addendum
  WELFARE_METRICS.md     Definitions for Fig 7 and Fig 8 extractions
  FIG9_GS_NOTE.md        Why Fig 9 is deferred (needs two-population REE)
  solver_code/           ← EMPTY: needs posterior_v3.py from REZN paper repo
  checkpoints/           ← Output: converged μ* JSONs
  heartbeats/            ← VM liveness files (one per worker)
  progress/              ← Live iter/ftol files (one per active task)

docs/
  status.html            Live dashboard
  SOLVER_INTEGRATION.md  How a project's solve.py plugs in ProgressReporter
  HOW_TO_ADD_A_PROJECT.md Guide for adding a new project
```

## Current solver state (last checked: 2026-05-06)

About 51 tasks, ~25 done, 6+ ready, ~18 blocked, 2 bailed. Always
consult `projects/REZN/TASK_QUEUE.json` or the live dashboard for the
truth — this paragraph drifts.

### Bailed tasks (need attention)
- `g050_t0400` (γ=0.5, τ=4.0): boundary issue at G=20 UMAX=5;
  posteriors pin at 0/1 in tail rows. Retry strategy proposed:
  cross-warm from γ=1.0 τ=4.0 + reduce UMAX to 4. Was set to "ready"
  with this strategy in an earlier patch but may need re-applying.
- `g100_t0500` (γ=1.0, τ=5.0): did not converge below ||F||=1e-3.

### Welfare and value-of-information extraction tasks
Six tasks `extract_volume_g{050,100,400}` and
`extract_value_info_g{050,100,400}` should compute Fig 7 and Fig 8
from already-converged μ*. Math is in `projects/REZN/WELFARE_METRICS.md`.
If these tasks are missing from `TASK_QUEUE.json`, run
`add_welfare_tasks.py` (script may live in repo root or need to be
recreated; see git log for "Add welfare and value-of-information
extraction tasks").

### Fig 9 (Grossman–Stiglitz) is deferred
Requires a two-population REE solver (informed + uninformed agents).
See `projects/REZN/FIG9_GS_NOTE.md`. The proof of Proposition 8 in
the paper does not depend on numerical computation, so SSRN v2 can
ship without it.

## Critical conventions

### Weighted 1-R² (use this everywhere)
Unweighted regression of logit(p) on T* is wrong; it depends on grid
choice. Always weight by ex-ante probability:

```python
def signal_density(u, v, tau):
    return np.sqrt(tau/(2*np.pi)) * np.exp(-tau/2 * (u-(v-0.5))**2)

# For each triple (i,j,l) on the G³ signal grid:
w = 0.5 * (f0(ui)*f0(uj)*f0(ul) + f1(ui)*f1(uj)*f1(ul))

slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))
pred = slope*Tstar + intercept
mean_lp = np.average(logit_p, weights=weights)
var_tot = np.average((logit_p-mean_lp)**2, weights=weights)
var_res = np.average((logit_p-pred)**2, weights=weights)
weighted_1mR2 = var_res / var_tot
```

### Lock file is `TASK_QUEUE.json`
Workers claim by setting `status: "claimed"` + `claimed_by` +
`claimed_at`, then push. If push fails (race), pull-rebase and try
another task. Use `core/claim_task.py`.

### Live progress is a separate file per task
Workers write `projects/REZN/progress/$TASK_ID.json` every 60 s with
latest `iter` and `ftol`. Heartbeats (`heartbeats/$WORKER_ID.txt`)
every 5 min are independent. The dashboard reads both. Don't put
live telemetry in `TASK_QUEUE.json` — it would create commit storms
and conflict with claim operations.

## Spinning up VMs (operator workflow)

```bash
GITHUB_TOKEN=ghp_xxx ANTHROPIC_API_KEY=sk-ant-xxx \
  PROJECT=REZN VM_NAME=solver-1 bash core/create_gcp_vm.sh
```

Spot e2-medium, max 1h run duration. Repeat with different `VM_NAME`
for parallel workers. Each VM independently claims different tasks
(locking prevents collisions). When `MAX_RUN_HOURS` expires or the VM
is preempted, claimed-but-not-done tasks return to ready after the
6-hour stale-claim timeout.

## Outstanding gaps

1. **`projects/REZN/solver_code/` is empty.** Workers will bail with
   "solver_code/solve.py not found" until the actual solver code from
   the REZN paper repo (`github.com/mhpbreugem/REZN/python/posterior_v3*.py`)
   is ported here and wrapped to take `--task-id` and `--worker-id` CLI
   args. See `docs/SOLVER_INTEGRATION.md` for the wrapping pattern.
2. **Bailed tasks may need the cross-warm-start retry applied.**
3. **Welfare extraction tasks may not yet be in the queue.**

## Useful one-liners for the next session

```bash
# Status from CLI
python3 core/supervisor.py --project REZN

# Validate task queue JSON
python3 -c "import json; json.load(open('projects/REZN/TASK_QUEUE.json'))"

# Find bailed/failed tasks
python3 -c "
import json
for t in json.load(open('projects/REZN/TASK_QUEUE.json'))['tasks']:
    if t['status'] == 'bailed': print(t['id'], '—', t.get('note',''))"

# Release a stale claim
python3 core/claim_task.py release --project REZN --worker-id solver-X

# Pull and rebuild dashboard preview locally
git pull origin main && python3 -m http.server 8000
# then open http://localhost:8000/docs/status.html?project=REZN
```

## Tone for follow-up sessions

- Use Sonnet at low effort. The work is mechanical; orchestration
  over invention.
- Always pull before editing `TASK_QUEUE.json`. Always push immediately
  after.
- When bailed tasks recur, propose a retry strategy (different
  warm-start, different UMAX, smaller dps) rather than giving up.
- The paper-side work (LaTeX, figures, SSRN uploads) lives in the
  separate REZN repo, not here. Don't conflate the two.
