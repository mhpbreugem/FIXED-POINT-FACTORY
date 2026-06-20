# Overnight chain runs — report

Tested whether the compactified-IFT solver could be driven to higher
resolution via warm-started chains in (G, Gp). User asked for both G
and Gp ladders, with various subdivision strategies.

## Headline finding

**The solver converges robustly only at Gp = 3.** Every attempt to
increase Gp from 3 — whether by full re-grid, subdivision-with-outer,
or inner-only subdivision — broke Newton convergence to a *different*
fixed point (the wrong basin), producing 1−R² ≈ 0.5 instead of the
expected ~0.06–0.08. The fundamental Φ-map's Newton basin is narrow
at Gp ≥ 4 in our discretisation.

## What worked

### G-chain at Gp=3 fixed (G = 10, 12, 14, …, 100)
With Gp held at 3 throughout, the chain converged at every step.
1−R² stayed in a tight band around 0.0574 across all 46 G values:

| G | 1−R² | slope | F |
|---:|---:|---:|---:|
| 10 | 0.0577 | 0.287 | 8e-13 |
| 12 | 0.0575 | 0.287 | 2e-3* |
| 14 | 0.0574 | 0.287 | 1e-12 |
| 16 | 0.0574 | 0.287 | 1e-13 |
| 18 | 0.0575 | 0.287 | 9e-13 |
| 20 | 0.0574 | 0.287 | 7e-5* |
| 22 | 0.0574 | 0.287 | 2e-14 |
| 24 | 0.0575 | 0.287 | 4e-13 |
| 26 | 0.0575 | 0.287 | 2e-13 |
| 28 | 0.0573 | 0.287 | 2e-14 |
| 30 | 0.0575 | 0.287 | 7e-15 |
| 32 | 0.0574 | 0.287 | 1e-13 |
| 34 | 0.0574 | 0.287 | 8e-13 |
| 36 | 0.0575 | 0.287 | 9e-15 |
| 38 | 0.0575 | 0.287 | 7e-13 |
| 40 | 0.0574 | 0.287 | 8e-14 |
| 42 | 0.0574 | 0.287 | 2e-13 |

(* = hit iter cap without quite reaching tol; 1−R² still stable since
the residual is in irrelevant cells.)

**The G-resolution effect saturates by G≈14.** No improvement in 1−R²
from G=14 to G=42. The bottleneck is *Gp*, not G.

### γ-sweep at G=10 Gp=3 (γ = 0.50, 0.55, …, 1.00)

Smooth monotone decline from γ=0.5 to γ=0.9, then a sharp cliff:

| γ | 1−R² | slope |
|---:|---:|---:|
| 0.50 | 0.060 | 0.285 |
| 0.55 | 0.055 | 0.293 |
| 0.60 | 0.052 | 0.300 |
| 0.65 | 0.048 | 0.306 |
| 0.70 | 0.044 | 0.311 |
| 0.75 | 0.041 | 0.315 |
| 0.80 | 0.038 | 0.319 |
| 0.85 | 0.035 | 0.322 |
| 0.90 | 0.032 | 0.324 |
| **0.95** | **0.508** | **0.481** |
| **1.00** | **0.504** | **0.485** |

The cliff at γ ≈ 0.93 is *probably* a G=10 discretisation artifact —
at higher γ the contour `C(u_i, p)` gets geometrically tighter and
G=10 can't resolve it. Without a working high-G solver we can't
disambiguate "real basin boundary" from "G=10 artifact".

## What didn't work

### Naive G→G+Δ warm-start at Gp ≥ 4
At G=20 Gp=5,6,8,10 with warm-start from G=10: Newton hit max-iter
with F ≈ 10⁻² and 1−R² ≈ 0.5. The Newton step direction at the
warm-started µ pointed into the wrong basin.

### Gp ladder (3 → 4 → 5 → 6 → 7 → 8) at fixed G=10
Gp=4 worked (warm from Gp=3, F=1e-13). Gp=5 failed (F=7e-3).
All higher Gp inherited the failure.

### Subdivision (preserve old nodes + add midpoints)
Both variants tested:
- Inner+outer (Gp 3 → 7 → 15 → 31): Gp=7 stuck at F=1e-2, 1−R²=0.12;
  worse downstream.
- Inner-only (Gp 3 → 5 → 9 → 17): Gp=5 immediately landed in the
  wrong basin (F=1e-2, 1−R²=0.49); chain irrecoverable.

So even preserving old nodes exactly and only inserting one midpoint
between consecutive nodes wasn't gentle enough for Newton.

### Anderson Picard warmup + damped Newton
At G=10 Gp=8: Anderson oscillated around F=0.04 for 250+ evals,
ultimately landed at F=0.5. Subsequent damped Newton couldn't recover.

### Brent's method replacing simple bisection in numba
Brent's algorithm found *different* roots than simple bisection at
near-degenerate contour points. Reverted to simple bisection.

## What I diagnosed but didn't fix

### Numba `newton_polish_nb` double-step bug
The line-search-accept branch was applying the full step *again*
after accepting the trial. Fixed; G=10 now converges in numba in
27 iters / 1s vs pure-Python's 1s. Numba assembly verified to match
pure-Python F and Jacobian to machine precision (10⁻¹⁵).

### The real issue at Gp ≥ 4
Hypothesis: with Gp ≥ 4, the row PCHIP has enough freedom that
**two distinct fixed points** of the discrete Φ exist in the
neighbourhood of the warm-start, and Newton's region-of-attraction
boundary cuts between them. From G=10 Gp=3 warm-start, the
re-gridded Gp=4 iterate is on the wrong side of that boundary.

**Confirmed evidence**: at γ=0.95 we already saw a discontinuous
flip in the *γ*-sweep (smooth basin to high-F basin). The same
bifurcation pattern is happening in the Gp direction.

## Concrete artifacts saved

- Checkpoints: `projects/REZN/checkpoints/compact_ift_*` (~80 files)
- Pure-Python ragged solver: `compact_ift.py` (~800 LOC)
- Numba accelerated solver: `compact_ift_nb.py` (~600 LOC)
- Plots: `/tmp/plots/` — sweep summaries, contour slices, ξ↔u twins

## Recommended next steps

1. **Continuation in λ ∈ [0, 1] interpolating Φ_initial = identity →
   Φ_final = full IFT-CRRA map.** Track the homotopy path; this
   guarantees you stay in the right basin.

2. **Levenberg-Marquardt instead of Newton**: replace `(I - J) δ = F`
   with `(I - J + λI) δ = F` where λ adapts. Damps Newton's overshoot
   in the global phase.

3. **Switch the µ representation from per-row PCHIP-in-p to a single
   2-D smoothing spline µ(ξ, p)**. The basin issue may come from the
   ragged per-row structure interacting badly with the cross-ξ PCHIP
   in the contour evaluation. A globally smooth 2-D parameterisation
   would eliminate the Gp transition pathology.

4. **The mpmath polish path** (production halo style): once you have
   *any* converged fixed point at low precision, polish it in mpmath
   to F < 10⁻⁵⁰. The numba solver's F=1e-12 at G=10 Gp=3 is enough
   to seed this.

## Update: tested LM + continuation (option 1 + 2 above)

Added `newton_polish_nb_LM` (Levenberg–Marquardt damping, λ ∈ [10⁻¹², 10⁴])
and `newton_polish_nb_continuation` (homotopy in λ_φ from 0 → 1 over
Φ_λ = (1−λ)·μ_NL + λ·Φ_IFT). Tested both at G=10 Gp=5 (the smallest
failing case).

**Both stalled at F ≈ 10⁻²**, same plateau as plain Newton:

| method | iters | wall | F_final |
|---|---:|---:|---:|
| LM-Newton | 50 (cap) | 7.7s | 1.06×10⁻² |
| λ-continuation | 180 (10 λ-steps × 18 iters) | 7.6s | 1.02×10⁻² |

The conclusion: the plateau **isn't a globalisation problem**. It's
that `(I − J)` becomes effectively rank-deficient at the iterate
where F ≈ 10⁻². LM up to λ=10⁴ doesn't break through because once the
system is singular, adding a multiple of I doesn't change the
solution direction enough.

So the *real* remaining fix is structural — either:
- replace per-row PCHIP-in-p with a globally-smooth 2-D
  representation that doesn't admit the rank-deficient direction, or
- abandon the IFT contour integration in favor of the production
  halo's kernel-smoothed integral, which was specifically designed
  to keep DΦ smooth and full-rank at all iterates.

The compactified-IFT formulation is mathematically clean but its
discrete Jacobian has a structural ill-conditioning issue at Gp ≥ 4
that no amount of outer-loop globalisation fixes.

