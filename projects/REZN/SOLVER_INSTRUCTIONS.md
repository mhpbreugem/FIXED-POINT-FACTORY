# SOLVER INSTRUCTIONS — REZN project

This is the REZN-specific addendum to `core/SOLVER_INSTRUCTIONS.md`.
Read `core/SOLVER_INSTRUCTIONS.md` first for the generic worker protocol.

---

## REZN at a glance

**Paper**: "On the Possibility of Informationally Inefficient Markets Without Noise"  
**SSRN**: abstract_id=6693198  
**Claim**: CARA utility is the unique preference class giving full price revelation.
Every other preference (CRRA, log, etc.) gives partial revelation — without noise
traders, supply shocks, or random endowments.

The numerics require solving a posterior-function fixed point on a 2-D grid.
See `projects/REZN/EQUATIONS.md` for the complete mathematical specification.

---

## Parameters (DO NOT CHANGE)

```
G        = 20          signal grid points
UMAX     = 5.0         u ∈ [-5, +5]
trim     = 0.0         full per-row p-range, no trimming
K        = 3           number of agents
W        = 1           wealth normalization
dps      = 50          mpmath precision (decimal digits)
F_tol    = 1e-25       ||Φ(μ) − μ||_inf convergence target
i_max    = 200         iteration cap
weighting = ex-ante 0.5*(f0³+f1³)
```

Seed reference checkpoint (γ = 0.5, τ = 2, ||F|| = 7.4e-119, dps = 300):

    results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json

All subsequent checkpoints inherit the same u_grid layout.

---

## Solver entry point

    projects/REZN/solver_code/solve.py

The solver is called by `core/bootstrap.sh` as:

```bash
python3 projects/REZN/solver_code/solve.py \
    --project REZN \
    --task-id g400_t1000 \
    --branch <branch> \
    --worker-id <worker_id>
```

The solver is responsible for:
1. Loading the warm-start checkpoint from the dependency's `checkpoint` path.
2. Running posterior-method v3 at the task's (γ, τ) values.
3. Measuring weighted 1-R² (see EQUATIONS.md §7).
4. Calling `core/claim_task.py done` with the results.

---

## Warm-start rules

- **τ changes at fixed γ**: recompute `p_grid[i]` from the no-learning
  market clearing at the new τ (using `mp.findroot`). Interpolate
  `μ[i, ·]` from the old p_grid onto the new p_grid.
- **γ changes at fixed τ**: p_grid is unchanged. Copy `μ` directly and
  re-iterate Φ at the new γ.
- **`deps_satisfy: "any"`**: warm-start from the first available dependency
  (whichever checkpoint file exists on disk).

---

## Checkpoint naming

```
results/full_ree/task3_g{γ×100:03d}_t{τ×100:04d}_mp{dps}.json
```

Examples: γ=4.0, τ=10.0 → `task3_g400_t1000_mp50.json`

Full schema: see `projects/REZN/CHECKPOINT_FORMAT.md`.

---

## Convergence metric

The metric reported in `result` is **weighted 1-R²** of logit(p_REE)
regressed on T* = τ(u₁ + u₂ + u₃), weighted by ex-ante probability.

Formula: see EQUATIONS.md §7.

Verified anchor: γ=0.5, τ=2 → 1-R² ≈ 0.085, slope ≈ 0.543.

Result fields to populate:

```json
{
  "1-R2": 0.085,
  "slope": 0.543,
  "F_max": "1.2e-26",
  "n_iters": 23,
  "n_triples": 8000
}
```

---

## What to do if a task bails

A task bails when after `i_max` iterations `||F||_inf > 1e-3` OR more
than 2% of cells are pinned at 0/1.

1. Halve α (try α = 0.075) and restart from the warm-start.
2. If still failing, drop UMAX from 5 to 4.
3. If still failing, mark `bailed` with a short reason.

A bailed task blocks all downstream tasks in its dependency chain.
Known bail cases: γ=0.5 τ≥4 (boundary issue at high τ), γ=1.0 τ≥5.

---

## Parallel safety

Two workers running different (γ, τ) tasks write disjoint checkpoint
files. Both update `projects/REZN/TASK_QUEUE.json` — the claim push is
the synchronisation primitive (see `core/SOLVER_INSTRUCTIONS.md §4`).

Never overwrite a `done` or `bailed` entry when resolving a merge conflict.
