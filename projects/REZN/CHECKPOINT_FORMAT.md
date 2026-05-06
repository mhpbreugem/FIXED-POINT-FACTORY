# CHECKPOINT FORMAT

Every converged (γ, τ) point is stored as a single JSON file under
`results/full_ree/`. The format mirrors the existing seed file
`posterior_v3_G20_umax5_notrim_mp300.json` so that a worker can read
ANY checkpoint as a warm-start without special-casing.

---

## 1. FILE NAMING

```
results/full_ree/task3_g{γ×100:03d}_t{τ×100:04d}_mp{dps}.json
```

Examples:
| γ    | τ     | filename                                   |
|------|-------|--------------------------------------------|
| 0.5  |  2.0  | task3_g050_t0200_mp50.json                 |
| 1.0  |  4.0  | task3_g100_t0400_mp50.json                 |
| 4.0  | 10.0  | task3_g400_t1000_mp50.json                 |
| 4.0  | 15.0  | task3_g400_t1500_mp50.json                 |

The seed checkpoint is the exception, kept under its historical name:
`posterior_v3_G20_umax5_notrim_mp300.json` (γ=0.5, τ=2, dps=300).

---

## 2. SCHEMA

All numerical fields with dynamic precision (μ values, p values,
F_max) are stored as DECIMAL STRINGS, not JSON numbers. JSON numbers
in standard parsers are float64, which loses ~35 digits of mp50 work.
The u_grid is also stored as strings for symmetry.

```json
{
  "G":     20,
  "UMAX":  5.0,
  "tau":   10.0,
  "gamma": 4.0,
  "trim":  0.0,
  "dps":   50,
  "K":     3,

  "F_max":  "1.234e-26",
  "F_med":  "5.678e-30",

  "u_grid": ["-5.0", "-4.473684210526316", ..., "5.0"],

  "p_grid": [
    ["1.43e-3", "2.13e-3", ..., "1.0e-2"],
    ["...", ..., "..."],
    ...
  ],

  "mu_strings": [
    ["0.000123...", "0.000456...", ..., "0.999..."],
    ["...", ..., "..."],
    ...
  ],

  "warm_start_from": "posterior_v3_G20_umax5_notrim_mp300.json",
  "n_iters":         23,
  "alpha":           0.20,
  "anderson_m":      6,
  "wall_seconds":    1234.5,

  "metrics": {
    "1-R2":     0.0123,
    "slope":    0.612,
    "intercept": 0.0,
    "n_triples": 8000,
    "weighting": "ex-ante 0.5*(f0^3+f1^3)"
  }
}
```

### Field meanings

| Field                | Type                | Notes |
|----------------------|---------------------|-------|
| `G`                  | int                 | signal grid size, 20 |
| `UMAX`               | float               | u_grid extent, ±5 |
| `tau`                | float               | signal precision |
| `gamma`              | float               | CRRA risk aversion |
| `trim`               | float               | per-row p-range trim margin (0.0 = no trim) |
| `dps`                | int                 | mpmath decimal digits used |
| `K`                  | int                 | number of agents, 3 |
| `F_max`              | string (mp scalar)  | maximum |Φ(μ) − μ| over active cells |
| `F_med`              | string (mp scalar)  | median |Φ(μ) − μ|, sanity diagnostic |
| `u_grid`             | array[G] of strings | signal grid in [-UMAX, UMAX] |
| `p_grid`             | array[G][G_p] str   | per-row achievable price range |
| `mu_strings`         | array[G][G_p] str   | converged posterior values |
| `warm_start_from`    | string (filename)   | the depends_on checkpoint |
| `n_iters`            | int                 | iterations used |
| `alpha`              | float               | Picard damping |
| `anderson_m`         | int or null         | Anderson window |
| `wall_seconds`       | float               | wallclock time |
| `metrics.1-R2`       | float (json number) | weighted 1-R² (regular precision) |
| `metrics.slope`      | float               | weighted slope of logit p on T* |
| `metrics.n_triples`  | int                 | usually G^3 = 8000 |

The metrics block is a regular JSON-number summary intended for
plotting; the high-precision values live in mu_strings.

---

## 3. INVARIANTS (verified at write time)

A worker MUST refuse to commit a checkpoint that fails any of these:

1. `len(u_grid) == G` and the grid is symmetric about 0 with spacing
   `2 * UMAX / (G - 1)`.
2. `len(p_grid) == G` and `len(p_grid[i]) == G_p` for all i (constant
   G_p across rows; inherited from the warm-start).
3. Each `p_grid[i]` is strictly increasing.
4. Each `mu_strings[i]` is monotone non-decreasing in p (within
   numerical tolerance ≤ 1e-15 in the absolute logit gap).
5. For each j, `mu_strings[:, j]` is monotone non-decreasing in u.
   These two are the PAVA invariants and must hold.
6. `F_max < 1e-25` for an mp50 checkpoint (looser at lower dps).
7. `metrics.1-R2 ≥ 0` and `metrics.1-R2 ≤ 1`.

---

## 4. READING A CHECKPOINT

```python
import json, mpmath as mp
mp.mp.dps = 50

with open(path) as f:
    d = json.load(f)

u_grid = mp.matrix([mp.mpf(s) for s in d["u_grid"]])
p_grid = [[mp.mpf(s) for s in row] for row in d["p_grid"]]
mu     = [[mp.mpf(s) for s in row] for row in d["mu_strings"]]
```

When warm-starting at NEW τ, recompute p_grid from the no-learning
market clearing at the new τ and then interpolate `mu[i, ·]` from
the OLD p_grid[i] onto the NEW p_grid[i] in the p-direction.

When warm-starting at NEW γ at fixed τ, the p_grid is unchanged. Just
copy `mu` over and re-iterate Φ at the new γ.

---

## 5. THE METRICS BLOCK IS DERIVED — DO NOT TRUST WITHOUT REPRODUCING

The `metrics` block is convenience-level. If you depend on a number
for a paper figure, recompute 1-R² from `mu_strings` using the
formula in `EQUATIONS.md` §7. The mp-strings are the authoritative
representation; everything else is decoration.
