# SOLVER INSTRUCTIONS — Posterior-Method v3 τ-sweep workers

This file is the contract for any worker that wants to advance the
COMPUTE/TASK_QUEUE.json. Read it end-to-end before doing anything.
Reference docs:
- `COMPUTE/EQUATIONS.md`        — model and fixed-point system
- `COMPUTE/CHECKPOINT_FORMAT.md` — JSON schema for checkpoints
- `COMPUTE/TASK_QUEUE.json`      — list of (γ, τ) tasks
- `POSTERIOR_METHOD_V2.md`       — algorithm derivation (sections A-E)
- `FIGURES_TODO.md`              — weighted 1-R² formula

---

## 1. THE MODEL (one paragraph)

A binary asset has payoff v ∈ {0, 1} with prior 1/2. K = 3 CRRA agents
with common risk aversion γ each observe a private signal

    s_k = v + ε_k,   ε_k ~ N(0, 1/τ),     u_k := s_k − ½

so under v the centered signal u_k is Gaussian with mean v − ½ and
precision τ. There are NO noise traders, NO supply shocks, and NO
random endowments. The price is set by market clearing among the
three CRRA traders. CRRA demand at posterior μ and price p:

    R = exp((logit μ − logit p) / γ)
    x(μ, p) = W · (R − 1) / ((1 − p) + R · p)

The unknown of the REE is the posterior function μ*(u, p) = P(v = 1 |
own signal u, observed price p). It satisfies the fixed point

    μ* = Φ(μ*)

where Φ is the Bayes-update-through-the-market-clearing-contour map
defined in EQUATIONS.md. Φ is implemented by contour integration: at
fixed (u, p) the contour {(u₂, u₃) : Σ x_k = 0} is traced via a
vectorised np.interp inversion of the demand column, then the density
ratio Σ f₁(u₂) f₁(u₃*) / Σ f₀(u₂) f₀(u₃*) is fed into Bayes' rule.

---

## 2. FIXED PARAMETERS (DO NOT CHANGE)

```
G        = 20                 # signal grid points
UMAX     = 5.0                # u ∈ [-5, +5]
trim     = 0.0                # full per-row p-range, no trimming
K        = 3                  # number of agents
W        = 1                  # wealth normalization
dps      = 50                 # mpmath precision in decimal digits
F_tol    = 1e-25              # ||F||_inf target on Bayes residual
i_max    = 200                # iterations cap (Picard + Anderson)
weighting = ex-ante 0.5*(f0³+f1³)
```

The seed reference is
`results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json`
(γ = 0.5, τ = 2, ||F|| = 7.4e-119). All later checkpoints inherit the
same u_grid and per-row p_grid layout.

---

## 3. WORKFLOW (one task = one commit)

```
┌─────────────────────────────────────────────────────────────┐
│ for task in COMPUTE/TASK_QUEUE.json                         │
│     1. git fetch + check-out the assigned branch            │
│     2. select READY task (deps satisfied, status=ready)     │
│     3. claim it via git (atomic single-commit status flip)  │
│     4. load warm-start checkpoint = depends_on result       │
│     5. solve μ* = Φ(μ*) with posterior-method v3 (mp50)     │
│     6. measure weighted 1-R² and slope                      │
│     7. save checkpoint JSON to results/full_ree/            │
│     8. update TASK_QUEUE.json (status=done, fill results)   │
│     9. commit (checkpoint + queue) and push                 │
└─────────────────────────────────────────────────────────────┘
```

Each step is detailed below.

### 3.1 Find a ready task

A task is READY when its own status is `"ready"` AND its dependencies
are satisfied per its `deps_satisfy` field:

- `deps_satisfy: "all"` (default) — every id in `depends_on` must be `done`.
- `deps_satisfy: "any"` — at least one id in `depends_on` must be `done`.

```python
import json, hashlib, os, socket
with open("COMPUTE/TASK_QUEUE.json") as f:
    queue = json.load(f)

done = {t["id"] for t in queue["tasks"] if t["status"] == "done"}

def deps_ok(t):
    deps = set(t.get("depends_on", []))
    mode = t.get("deps_satisfy", queue.get("deps_semantics", {}).get("default", "all"))
    return (deps & done) if mode == "any" else (deps <= done)

ready = [t for t in queue["tasks"]
         if t["status"] == "ready" and deps_ok(t)]
```

If `ready` is empty, exit cleanly — there is nothing to do.

**Pick deterministically with a worker-specific tiebreak.**
If you simply pick `ready[0]`, every worker that starts at the same
moment chooses the same task and they all race for one claim — N−1
workers waste a round-trip. Instead, hash `(worker_id, task_id)` and
pick the task with the lowest hash:

```python
worker_id = os.environ.get("WORKER_ID", socket.gethostname() + ":" + str(os.getpid()))

def pick_priority(t):
    h = hashlib.sha256(f"{worker_id}|{t['id']}".encode()).hexdigest()
    return h    # lexicographic tiebreak

task = min(ready, key=pick_priority)
```

This shuffles the ready list per worker so two workers starting
simultaneously usually pick different tasks. They only collide when
just one task is ready — and that case is handled by §3.2 below
(the loser exits cleanly).

### 3.2 Claim it via git

The queue file is the single source of truth. To avoid two workers
solving the same task, claim atomically:

```bash
# pull the latest queue
git pull --rebase origin <branch>

# flip status: "ready" -> "claimed", add claimed_by + claimed_at
python3 -c '
import json, datetime, socket, os
q = json.load(open("COMPUTE/TASK_QUEUE.json"))
tid = "g400_t1000"   # the one you picked
for t in q["tasks"]:
    if t["id"] == tid and t["status"] == "ready":
        t["status"] = "claimed"
        t["claimed_by"] = socket.gethostname()
        t["claimed_at"] = datetime.datetime.utcnow().isoformat()+"Z"
json.dump(q, open("COMPUTE/TASK_QUEUE.json","w"), indent=2)
'
git add COMPUTE/TASK_QUEUE.json
git commit -m "claim $tid"
git push origin <branch>     # if push fails -> someone else claimed,
                             # rebase + pick another task
```

If the push is rejected non-fast-forward, another worker beat you to
it. `git pull --rebase`, drop the claim, and select a different ready
task. If after rebasing there is no other ready task, EXIT — do not
re-attempt the same task that another worker is already solving. Two
workers solving the same task is the failure mode this whole protocol
exists to prevent.

**Hard rule: do NOT begin §3.4 (the actual solve) until the claim
push has landed on origin.** The claim push is the synchronization
barrier. If you start computing optimistically and the claim later
fails, you've burned compute on a task somebody else is also doing.

### 3.3 Load the warm-start

Each task has exactly one `depends_on` entry (or none, for the seed).
Open that task's `checkpoint` file and read:

- `u_grid`           — same for all tasks (G = 20, UMAX = 5)
- `p_grid[i]`        — per-row achievable price range at depends_on (γ, τ)
- `mu_strings[i][j]` — converged μ*(u_i, p_j) at depends_on (γ, τ)

Walk-up rules:
- If γ changes (γ_old → γ_new at fixed τ), keep the same u_grid and
  the same per-row p_grid. The μ* values transfer directly: same (u, p)
  cell, new γ. The fixed point migrates; that's what we re-solve.
- If τ changes (τ_old → τ_new at fixed γ), the per-row p_grid
  CHANGES because the no-learning price range depends on τ. Recompute
  p_lo[i], p_hi[i] from scratch (see §3.4 step a). Then interpolate
  μ*(u_i, ·) in the p-direction onto the new p-grid as the initial guess.

### 3.4 Run posterior-method v3

The algorithm is in POSTERIOR_METHOD_V2.md §C, in mpmath at dps = 50.
Skeleton:

```python
import mpmath as mp
mp.mp.dps = 50

def init_p_range(u_grid, tau, gamma):
    """Step a: no-learning lens, p_lo[i] and p_hi[i] for each row."""
    # solve sum_k x(Lambda(tau*u_k), p) = 0 with
    #   (u_i, u_min, u_min) and (u_i, u_max, u_max)
    # using mp.findroot. Return G arrays p_lo, p_hi.
    ...

def column_extract(mu, p_grids, p):
    """Step A: μ_col[i] = interp1d(mu[i,:], p_grids[i], p)."""
    ...

def demand_column(mu_col, p, gamma):
    """Step B: vectorised d[i] = x(mu_col[i], p) on G_u grid."""
    ...

def contour(d, u_grid, i, tau):
    """Step C+D+E: vectorised np.interp inversion + density dot products.
       Returns A0, A1 for own-signal index i."""
    ...

def phi_step(mu, p_grids, u_grid, tau, gamma):
    mu_new = mp.matrix(G, G)
    for j, p_j in enumerate(p_grids_per_j(...)):
        mu_col = column_extract(mu, p_grids, p_j)
        d      = demand_column(mu_col, p_j, gamma)
        for i in active_rows(j):
            A0, A1 = contour(d, u_grid, i, tau)
            f1 = signal_density(u_grid[i], 1, tau)
            f0 = signal_density(u_grid[i], 0, tau)
            mu_new[i, j] = f1 * A1 / (f0 * A0 + f1 * A1)
    return mu_new

# Picard with PAVA monotonicity projection
mu = warm_start
for n in range(i_max):
    mu_raw  = phi_step(mu, p_grids, u_grid, tau, gamma)
    mu_mono = pava_uv(mu_raw)             # PAVA in u then in p
    F_inf   = max_abs(mu_mono - mu)       # over active cells only
    if F_inf < F_tol: break
    mu = alpha * mu_mono + (1 - alpha) * mu          # alpha = 0.15-0.30
# optional Anderson polish window m = 5-8
```

Notes:
- ALL arithmetic inside Φ uses `mp.mpf`. `np.exp / np.log` are NOT
  precise enough at dps = 50.
- The contour inversion is the only step where np.interp is faster
  than a manual mp loop; a clean implementation re-does the search
  in mpmath because the precision matters at the boundary cells.
- The PAVA projection uses logit-space ordering (sklearn's
  IsotonicRegression on numpy floats is fine — it only fixes
  ordering, not values, so the loss of precision is bounded by the
  raw monotonicity gap, which is at most ~1e-14).
- Convergence target is `F_max < 1e-25`. Tighter than that fits
  interpolation noise.

### 3.5 Measure weighted 1-R²

After convergence, reconstruct the price for all G³ = 8000 triples and
regress logit(p_REE) on T* = τ(u₁ + u₂ + u₃) with the ex-ante weight

    w(i, j, l) = ½ · (f₀(u_i)·f₀(u_j)·f₀(u_l)
                    + f₁(u_i)·f₁(u_j)·f₁(u_l))

where f_v(u) = sqrt(τ / 2π) · exp(−τ/2 · (u − v + ½)²). Code:

```python
import numpy as np

def signal_density(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2*np.pi)) * np.exp(-tau/2 * (u - mean)**2)

# After computing T*, logit_p, weights for all triples:
sw = np.sqrt(weights)
slope, intercept = np.polyfit(Tstar, logit_p, 1, w=sw)
pred = slope * Tstar + intercept

mean_lp     = np.average(logit_p,             weights=weights)
var_total   = np.average((logit_p - mean_lp)**2, weights=weights)
var_residual= np.average((logit_p - pred)**2,    weights=weights)

R2          = 1.0 - var_residual / var_total    # this is the R^2
one_minus_R2= var_residual / var_total          # report this
```

Report `1-R²`, `slope`, `n_triples = 8000`, and `F_max`.

### 3.6 Save the checkpoint

Write a JSON file matching the schema in CHECKPOINT_FORMAT.md to

    results/full_ree/task3_g{γ×100:03d}_t{τ×100:04d}_mp50.json

For example: γ = 4.0, τ = 10.0 → `task3_g400_t1000_mp50.json`.

### 3.7 Update TASK_QUEUE.json

Flip the entry from `claimed` to `done` and write the measured numbers
back into the `result` field:

```json
{
  "id": "g400_t1000",
  "status": "done",
  "checkpoint": "results/full_ree/task3_g400_t1000_mp50.json",
  "result": {"1-R2": 0.0XXX, "slope": 0.XXX, "F_max": 1.2e-26,
             "n_iters": 23, "n_triples": 8000},
  "completed_at": "2026-05-06T..."
}
```

### 3.8 Commit and push

```bash
git add results/full_ree/task3_g400_t1000_mp50.json
git add COMPUTE/TASK_QUEUE.json
git commit -m "g400_t1000: 1-R²=... done"
git push origin <branch>           # see Git Operations §retry policy
```

If the push fails on a non-network error, rebase, re-resolve any
conflicts in TASK_QUEUE.json (preserve the other worker's `done`
entries), and re-push.

---

## 4. WHAT TO DO IF A TASK BAILS

A τ-step "bails" when, after `i_max` iterations, ||F||_inf > 1e-3 or
the PAVA-projected μ has more than 2% of its cells pinned at 0/1.

1. Halve α (try α = 0.075) and restart from the warm-start.
2. If still no progress, drop UMAX from 5 to 4 (fewer wasted tail cells).
3. If still no progress, mark the task `bailed` (NOT `done`) with a
   short note in the `result` field. Push the queue update, do not
   push a checkpoint.

A bailed τ-step BLOCKS every downstream task because the warm-start
chain is broken. The next worker will see those downstream tasks
remain in `blocked` indefinitely.

---

## 5. PARALLEL SAFETY

- Two workers running on different (γ, τ) tasks at the same time is
  fine — they write disjoint files in `results/full_ree/`.
- Both must update the SAME `COMPUTE/TASK_QUEUE.json`. The `claim`
  commit (§3.2) is the synchronization primitive: whichever worker's
  push lands first wins. Losers rebase and pick a different task.
- Never resolve a TASK_QUEUE.json merge conflict by overwriting —
  always preserve every `done` entry from `origin/<branch>`.
