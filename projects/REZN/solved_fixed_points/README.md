# Solved Fixed Points

Each `.npz` file is one converged CRRA REE fixed point with full metadata.

## Filename convention

`g{gamma}_t{tau}_A{A_logit}_G{G}_Gp{Gp}.npz`

E.g.  `g100.0_t2.0_A5.5_G10_Gp17.npz`

## Stored arrays

- `mu_vals` (G × Gp): converged µ at solver grid (ξ_i, p_j)
- `xi_grid` (G,): solver ξ-grid
- `p_grid` (Gp,): solver p-grid

## Stored metadata (as scalar arrays)

- `gamma`, `tau`: model parameters
- `G`, `Gp`: grid sizes
- `A_logit`: half-range of logit-uniform p-grid
- `F_final`: final ‖F‖∞ achieved
- `newton_iters`: total Newton iterations to converge
- `tol`: convergence tolerance used
- `solver_interp`: PCHIP basis (e.g. "logit_logit_pchip")
- `wall_seconds`: wall-clock time for the solve
- `timestamp_utc`: ISO-8601 timestamp at save
- `commit_hash`: git commit hash at solve time
- `script`: name of script that produced this
- `note`: free-text note

## Loading

```python
import numpy as np
fp = np.load("g100.0_t2.0_A5.5_G10_Gp17.npz", allow_pickle=False)
mu_vals = fp['mu_vals']                  # shape (G, Gp)
gamma = float(fp['gamma'])
tau = float(fp['tau'])
print(f"γ={gamma}, τ={tau}, F={float(fp['F_final']):.2e}")
```
