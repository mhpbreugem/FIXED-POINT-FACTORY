# REZN solver wrapper

`solve.py` wraps the K=3 staggered halo solver from
`github.com/mhpbreugem/REZN/code/`.

## How it works

1. Reads task `gamma`, `tau` from `TASK_QUEUE.json`.
2. Builds homogeneous K=3 parameter vectors (`gamma_vec = [γ,γ,γ]`, `tau_vec = [τ,τ,τ]`).
3. Grid: G_inner=12, pad=4 → G_full=20, u_inner ∈ [-3, +3], u_outer ∈ [-6, +6].
4. Warm-start: loads `.npz` checkpoint from the first `done` dependency if present; otherwise cold-starts from the no-learning equilibrium.
5. Runs `staggered_solve` (Newton-Krylov with halo boundary).
6. Measures `revelation_deficit_f128` (weighted 1-R², longdouble precision).
7. Saves `.npz` checkpoint to `projects/REZN/checkpoints/$TASK_ID.npz`.
8. Calls `core/claim_task.py done` or `bail`.

## Precision

The REZN code is float64. Legacy task-queue `dps` fields are mapped:

| dps       | Newton tol |
|-----------|-----------|
| ≤ 50      | 1e-7      |
| ≤ 100     | 1e-9      |
| ≥ 200     | 1e-11     |

## Checkpoint format

`.npz` files contain:
- `P_inner` — converged price grid on inner cells, shape `(G_inner,)*3`
- `halo` — no-learning boundary, shape `(G_full,)*3`
- `P_full` — full grid (`halo` with `P_inner` inserted)
- `u_full`, `u_grid_inner`, `gamma_vec`, `tau_vec`, `W_vec`
- `stage_F_inf`, `stage_deficit` — per-stage diagnostics

## Dependencies

Requires `numpy`, `scipy`, `numba` (for `@njit` kernels in REZN).
Bootstrap / workflow clones REZN to `~/rezn-source` and sets `REZN_SRC`.
