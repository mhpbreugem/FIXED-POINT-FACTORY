# COMPUTE/ — Distributed REE solver harness

This folder is the coordination layer for the posterior-method v3
solver runs that produce Fig 4A (1-R² vs τ at γ ∈ {0.5, 1.0, 4.0})
in the REZN paper. Multiple workers can pick up tasks in parallel
without stepping on each other; the git history is the audit log.

## Files

| File                     | Purpose |
|--------------------------|---------|
| `README.md`              | this file |
| `SOLVER_INSTRUCTIONS.md` | exact protocol a worker must follow |
| `EQUATIONS.md`           | model + fixed-point system Φ(μ) = μ |
| `TASK_QUEUE.json`        | list of (γ, τ) tasks, status, dependencies |
| `CHECKPOINT_FORMAT.md`   | JSON schema for `results/full_ree/*.json` |

## Worker loop in three lines

```
1. parse TASK_QUEUE.json, find a task with status="ready" whose
   depends_on are all "done"
2. load that task's depends_on checkpoint, run the v3 solver to
   ||F||_∞ < 1e-25, measure weighted 1-R²
3. write a new checkpoint, flip the task to "done" in the queue,
   commit + push as a single atomic unit
```

If two workers race on a claim, git rejects the second push as
non-fast-forward. The losing worker rebases and picks the next ready
task.

## Status snapshot (as of the commit that introduced this folder)

```
done:    25 / 51 tasks
ready:    6   (g400_t1000, g025_t0200, g200_t0200,
               figR2_lognormal_g{050,100,400})
blocked: 18
bailed:   2   (g050_t0400, g100_t0500 — boundary issue at high τ)
```

The five ready tasks are independent — workers running in parallel
will hash-pick different ones (see SOLVER_INSTRUCTIONS.md §3.1) and
only collide when exactly one task is ready, in which case all but
one worker exit cleanly.

The accepted Fig 4A data are:
- γ = 0.5: 7 points,  τ ∈ {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0}
- γ = 1.0: 8 points,  τ ∈ {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0}
- γ = 4.0: 11 points once g400_t1000 lands; 12 if g400_t1500 lands

## Where to start reading

1. `EQUATIONS.md` — what the solver computes (one page)
2. `CHECKPOINT_FORMAT.md` — what a checkpoint looks like
3. `SOLVER_INSTRUCTIONS.md` — worker contract (read all of it before
   committing anything to TASK_QUEUE.json)

For deeper context: `POSTERIOR_METHOD_V2.md` (algorithm) and
`FIGURES_TODO.md` (where the metric formulas come from).
