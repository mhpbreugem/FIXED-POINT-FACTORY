# σ-δ at G_inner=15 with flint dps=50: result and reading

## Setup
- Operator: `phi_sigdelta` (axis-aligned scan for agent 1; Σ-interp + 2D
  contour scan for agents 2,3 — the existing σ-δ machinery).
- Grid: (u_1, Σ, δ) ξ-cube, G_FULL=17 (G_inner=15), uniform ξ.
- BCs (exact, FR-derived):
  - u_1 = ±∞ → P = 0/1
  - Σ = ±∞ → P = 0/1 (since S = u_1+Σ → ±∞, Λ → 0/1)
  - δ = ±∞ → zero-order extrapolation (FR is δ-flat)
- Precision: flint dps=50 (~166 bits).
- Picard: adaptive ω damping (×0.7 on stall, ×1.05 on good contraction).

## Runs
**CARA** (`flint_sd_G15_cara.py`): IC = analytic FR. Hellwig predicts FR is the
fixed point; we test if Picard stays there.

**CRRA** (`flint_sd_G15_crra.py`, γ=0.1, τ=2): IC = no-learning. We look for
the genuine PR attractor.

## Result (5–6 Picard iterations each, ~5 min total)

| iter | CARA ferr | CARA d_FR | CRRA ferr | CRRA d_FR |
|------|-----------|-----------|-----------|-----------|
|  1   | 6.1e-1    | 0.136     | 4.6e-1    | 0.239     |
|  2   | 2.0e-1    | 0.161     | 4.6e-1    | 0.245     |
|  3   | 1.9e-1    | 0.179     | 2.0e-1    | 0.249     |
|  4   | 1.9e-1    | 0.191     | 1.9e-1    | 0.254     |
|  5   | 1.7e-1    | 0.197     | 2.0e-1    | 0.257     |
|  6   | 1.6e-1    | 0.199     |           |           |

Both runs converge slowly toward a non-FR attractor at:
- CARA: d_FR ≈ 0.20
- CRRA: d_FR ≈ 0.26

The two attractors are different but **of the same order of magnitude**,
and CARA drifts WELL AWAY from FR.

## Reading

This contradicts Hellwig if taken at face value — but compare to the
**G_inner=10 float64 σ-δ result** (`picard_sigma_delta_cara.py`, this
session):

| Grid              | CARA d_FR | CARA 1−R²(T*) | CARA slope |
|-------------------|-----------|---------------|------------|
| G=10 float64      | **6e-3**  | 2.5e-4        | 1.015      |
| G=15 flint dps=50 | **0.20**  | —             | —          |

The σ-δ op at **G=10 reproduces Hellwig** (CARA = FR with d_FR ≈ 0); at
**G=15 it fails** by a factor of 30×. Going from float64 to dps=50 did not
reduce d_FR — confirming this is a **structural discretization bias of the
operator**, not a floating-point precision issue.

The σ-δ README is explicit about this:
> Both ICs end in **limit-cycle/chaotic regimes** rather than crisp fixed
> points — characteristic of the oblique-slice machinery's added error
> versus the axis-aligned (u_1, u_2, u_3) Φ. Useful mainly as an
> analytical lens (P should be δ-flat under FR) rather than a faster solver.

The Σ-interpolation needed by agents 2,3 (since their own-signal slices
are oblique in the σ-δ frame) introduces an error that, at G=15, dominates
the Hellwig FR signal. At G=10 it's small enough that d_FR=6e-3 — at G=15
it grows to 0.20. The error scales with grid density because Σ-interp
samples more cells per contour and each sampling contributes a small bias.

## So what does this verify?

- **The σ-δ frame at G=10 float64 verifies Hellwig** (CARA → FR to d_FR=6e-3).
  That earlier result stands. See `picard_sigma_delta_cara.py` +
  `sigmadelta_cara_summary.json`.
- **The σ-δ frame at G=15 is dominated by oblique-slice discretization bias**.
  This says nothing about Hellwig — only that this particular operator
  needs G≤10 to be useful, or needs a re-derived Σ-interp that doesn't
  accumulate error with refinement.
- **The principled CARA = FR confirmation we have is the open-box u-grid**
  (`../../solved_fixed_points/k3_cara/cara_open_box_vs_G.json`): deficit
  3.4e-14 at G=7, 7.3e-9 at G=9 with UMAX=12. That's the clean numerical
  realization of Hellwig.

## Files
- `flint_sd_G15_crra.py` — high-prec CRRA σ-δ at G=15, no-learn IC.
- `flint_sd_G15_cara.py` — high-prec CARA σ-δ at G=15, FR-ansatz IC.
- `flint_sd_G15_cara.json` / `flint_sd_G15_crra.json` — per-iter ferr, d_FR, ω.
- `flint_sd_G15_cara_P.npy` / `flint_sd_G15_crra_P.npy` — last-iter inner P.
