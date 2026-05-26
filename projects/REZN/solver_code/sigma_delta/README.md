# (u_1, Σ, δ) Picard solver — sum/diff coordinates with exact boundary

## Method

Solves the K=3 symmetric h=0 CRRA equilibrium on a **rotated grid**:
- Axis 1: u_1 (agent 1's own signal)
- Axis 2: **Σ = u_2 + u_3** (sum of the other two agents' signals)
- Axis 3: **δ = u_2 − u_3** (difference of the other two agents' signals)

With u_2 = (Σ+δ)/2 and u_3 = (Σ−δ)/2. The grid is a square in (Σ, δ).

### Why this coordinate change

Σ is the **sufficient-statistic direction** for agent 1's inference about v
given the other agents' signals (S = u_1+Σ is fully sufficient for v).
δ is the orthogonal direction that carries information beyond sufficiency —
the Jensen-wedge content of the equilibrium.

In FR REE: P depends only on S = u_1+Σ, so should be **δ-flat**. Any
δ-dependence is a Jensen-wedge signature.

## Grid (ξ-coords)

All three axes use ξ ∈ (−1, +1) with u = TOT · atanh(ξ):
- ξ_u1 axis: TOT_u = 2.0
- ξ_Σ  axis: TOT_Σ = 3.0
- ξ_δ  axis: TOT_δ = 3.0

Inner cells uniformly in ξ (G=10 per axis); boundary nodes at ξ = ±1.

## Boundary conditions (EXACT)

| Boundary | Value | Justification |
|---|---|---|
| u_1 = ±∞ | P = 0 / 1 | FR limit Λ(τS)→0 or 1 |
| Σ = +∞   | P = 1     | S = u_1 + Σ → +∞ ⇒ Λ → 1 (mathematically exact) |
| Σ = −∞   | P = 0     | Same, opposite sign |
| δ = ±∞   | zero-order extrap | δ direction is "free" — no FR limit applies |

## Φ algorithm

For each interior cell (i, j, k) = (u_1, Σ, δ):

- **Agent 1**: own signal u_1 fixed → contour scan in (Σ, δ), axis-aligned
  in the new grid. Standard linear-interp crossings.
- **Agent 2**: own signal u_2 = (Σ+δ)/2. Slice is **oblique** in (Σ, δ);
  we use Σ-interpolation: for each (u_1, δ), read P at the required Σ.
  Then 2D contour scan in (u_1, δ).
- **Agent 3**: own signal u_3 = (Σ−δ)/2. Mirror of agent 2.

Posteriors μ_k combined via symmetric CRRA bisection clearing.

## Result at γ=0.1, τ=2, K=3, G=10 per axis (float64)

| IC | ferr | d_FR | 1−R²(T*) | slope | intercept | Notes |
|---|---|---|---|---|---|---|
| FR_ansatz | 5.2e-3 | 4.24e-3 | 1.0e-4 | — | — | Limit cycle, doesn't converge |
| two_step  | 1.86e-2 | 2.18e-2 | 4.21e-3 | 1.0129 | −0.0022 | Limit cycle at distinct attractor |

Both ICs end in **limit-cycle/chaotic regimes** rather than crisp fixed
points — characteristic of the oblique-slice machinery's added error
versus the axis-aligned (u_1, u_2, u_3) Φ.

## Files

- `phi_sigma_delta.py` — numba-JIT Φ operator with oblique slices
- `picard_sigma_delta.py` — 200-iter Picard runner, reports every 5 iter
- `run_log.txt` — execution log of the most recent run
- `P_FR_ansatz_final.npy`, `P_two_step_final.npy` — final P (inner block, shape (G,)*3)
- `summary.json` — numerical summary

## Status

**Experimental.** The (u_1, Σ, δ) frame doesn't give cleaner convergence
than the standard (u_1, u_2, u_3) frame at the same G — likely because the
Σ-interpolation needed for agents 2,3 introduces error. Useful mainly as
an analytical lens (P should be δ-flat under FR) rather than a faster solver.
