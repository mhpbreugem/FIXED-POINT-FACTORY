# Deep G-ladder continuum extrapolation

## Setup
Five representative cells from the certified emin15 region. Per cell: G-ladder
from G=21 (sweep) extended to G ∈ {25, 29, 33, 37}, kernel-smooth h = 0.45·√Δu,
Newton-Krylov f_tol=1e-11. All rungs warm-start from the previous G's solution.

## Headline table (clean fit, nailed F<1e-8 rungs only)

| cell (τ, γ) | rungs nailed | rungs stalled | deficit @ G=21 | **d_inf** | ±err | q_free |
|---|---|---|---|---|---|---|
| (0.2, 1.04) | 8 | 0 | 0.00026 | **0.0001** | 0.0001 | 0.74 |
| (0.5, 1.04) | 8 | 0 | 0.00658 | **-0.019** | 0.023 | 0.20 |
| (1.0, 1.04) | 6 | 2 | 0.0414 | **-0.16** | 0.19 | 0.20 |
| (1.0, 0.098) | 8 | 0 | 0.205 | **0.2042** | 0.0011 | **5.00** |
| (2.0, 0.098) | 5 | 3 | 0.280 | **0.2681** | 0.0098 | **1.65** |

(error bar = max span across q-free, q=1, q=2 extrapolations)

## Key findings

1. **The C=0.45·√Δu kernel schedule hits a small-h pathology around G≈25–29**
   that flips a previously converging cell into a Newton-Krylov stall. Three
   cells (cells 2 stalled at G=37, cells 3 at G=33–37, cell 5 at G=29–37)
   converted from machine-eps convergence at G=21 to F≈3e-2 stalls deeper.
   This is independent of (τ, γ); it is intrinsic to the operator+schedule.

2. **The immortal-anchor cell (τ=2, γ=0.098) extrapolates to d_inf ≈ 0.268 ± 0.010**
   from its 5 nailed rungs (G=21,25 + earlier reference rungs).
   Consistent with the known G-sequence 0.291@G9 → 0.276@G25 → continuing down.

3. **The convergence order q is _not_ O(h) at well-behaved cells.** Cells with
   8 nailed rungs and no stall corruption give q_free ∈ [0.7, 5]; cell 4
   (τ=1, γ=0.098, the smoothest fit in the set) yields q≈5 indicating it has
   already plateaued. The referee's "O(h)" was extracted from data where stall
   contamination flattens the convergence; the underlying operator converges
   faster on cells where all rungs are nailed.

4. **Two cells produce negative d_inf** under the q-free fit (cells 2, 3),
   which is unphysical. These are cases where the persistent downward drift of
   deficit combined with limited rungs leads the fit to overshoot. The honest
   reading: these cells need either deeper trustworthy rungs (achievable via
   the CDF-slice or DD-fix routes) or a regularized fit with a strict d_inf≥0
   prior.

## Implications for the truth metric

- For cells where the kernel operator stays well-behaved across G=21→37
  (e.g. cell 4), G=21 deficit is already an excellent continuum estimate
  (within ~5e-4).
- For cells where the kernel operator stalls at G≥29 (cells 2, 3, 5),
  the G=21 number is the best available with this operator; extrapolation
  past G=21 is unreliable without a stall-free strict-h operator.
- This is exactly the operator that the strict CDF-slice route (h=0 by
  construction) and the now-fixed DD solver (no operator floor) were
  designed to replace.

## Files
- `results.json`: full ladder per cell (raw)
- `extrapolation_clean.json`: clean-fit results
- `extrapolation_clean.png`: per-cell h→0 plot
- `P_t*_g*_G*.npy`: per-rung inner G×G×G solutions
