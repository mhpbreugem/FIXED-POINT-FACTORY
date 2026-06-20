# Edge-accuracy and symmetry fixes for the (Σ̂, δ̂) contour-Φ

## What the baseline does wrong

The contour-scan-based Φ on the (u_1, Σ̂, δ̂) cube has three failure modes near grid edges:

1. **Crossings near ξ=±1 are silently dropped.** When a level set {P=p_target}
   approaches the boundary face, the linear-interp position can land in the
   last interval [ξ_{G-2}, ξ_{G-1}=±1]. The baseline code rejects such
   crossings (the `if abs(xi_off) < 1 - 1e-15` guard), biasing A_v down
   for p_target near 0 or 1.

2. **Hard P=0/1 at boundary cells creates a discrete "cliff".** The true
   smooth FR limit Λ(τS)→0/1 is asymptotic; in the discrete grid we slap
   the value at ξ=±1 to exactly 0/1, but the LAST INTERIOR cell has a finite
   logit that doesn't smoothly extrapolate to the boundary value. Contour
   crossings interpolated through this discontinuity sit at an unphysical
   ξ_off (jumping straight from interior value to the saturation).

3. **Numerical symmetry drift.** The model has two exact symmetries —
   δ̂-sym (P(i,j,k)=P(i,j,G-1-k)) and v↔1-v sym
   (P(i,j,k)=1−P(G-1-i,G-1-j,k)). The discrete contour scan implements
   them only approximately because crossings on +δ̂ and −δ̂ are computed
   independently with independent rounding. Symmetry drift accumulates
   at ~1e-3/iter (measured in the baseline run).

## Fixes

### Fix A — tail-aware boundary

Replace the hard P=0/1 at boundary nodes (ξ=±1) with **logit-linear
extrapolation** from the two nearest interior cells:

    logit P[boundary] = 2·logit P[nearest interior] − logit P[2nd-nearest]
    P[boundary]       = sigmoid(logit P[boundary])

This naturally saturates to 0 or 1 because logit is unbounded; the FR
limit at ±∞ is preserved as an asymptote, but the local logit-P gradient
is continuous across the boundary. The contour scan no longer sees a
discrete cliff at the edge.

### Fix B — symmetry projection

After each Φ step, project P onto the symmetric subspace:

    P_proj = (1/4) [P + flip_δ(P) + (1 − flip_uΣ(P)) + (1 − flip_all(P))]

This bit-exactly enforces both symmetries (measured asymmetry drops to
machine eps), at the cost of one extra slice + arithmetic per iter
(<1% overhead).

### Fix C — folded storage / symmetric scan

See `phi_sigma_delta_folded.py`. Stores only one quadrant
(û ∈ full, Σ̂≥0, δ̂≥0); reads outside the stored domain go through a
reflection-aware accessor `P_at(i,j,k)`. The crossings at +δ̂ and −δ̂ are
exact mirrors by construction (the reflection is applied to indices, not
re-computed). Gives ~4× speedup as a side benefit.

## Comparison: G=31 uniform, FR-ansatz IC, 20 Picard iters at γ=0.1

| Variant      | iter 1 ferr | iter 20 ferr | iter 20 1−R² | iter 20 d_FR | iter 20 asymmetry |
|--------------|-------------|--------------|---------------|---------------|--------------------|
| baseline     | 9.09e-2     | **4.00e-2**  | 8.34e-6       | 1.01e-3       | 1.23e-2            |
| +A only      | 9.09e-2     | 3.99e-2      | 8.36e-6       | 1.01e-3       | 2.76e-2            |
| +B only      | 9.09e-2     | **1.10e-2**  | 5.29e-6       | 7.83e-4       | **1.1e-16**        |
| +A+B         | 9.09e-2     | 1.12e-2      | 5.30e-6       | 7.82e-4       | **6.9e-16**        |

Findings:

* **Fix B (symmetry projection) is the big win.** ferr at iter 20 drops by
  **3.6×** (4.00e-2 → 1.10e-2). 1-R² drops by 1.6×, d_FR by 1.3×.
  Asymmetry collapses from 1.23e-2 to machine epsilon.

* **Fix A (tail BC) alone has no visible effect** at this γ and grid.
  Counter-intuitively, +A actually slightly INCREASES asymmetry (2.76e-2
  vs baseline 1.23e-2) because the logit-linear extrapolation introduces
  a small additional asymmetric term at the boundary that the baseline's
  hard 0/1 didn't have. The δ̂ direction's zero-order extrap and ferr-scale
  oscillation isn't sensitive to this.

* **Fix A+B ≈ Fix B alone** (within 2% on ferr). The tail BC neither helps
  nor hurts once symmetry is projected.

## Recommendation

Always run with **Fix B (symmetry projection)**. It's a 5-line wrapper, no
perf cost, fixes the dominant numerical error in the iteration, and
preserves the model's exact symmetry by construction.

For grid resolutions where the level set genuinely runs into the boundary
(large γ, small G, or near-saturated p), Fix A is conceptually right but
its empirical effect is dwarfed by Fix B at the parameters we tested.

Fix C (folded) is independently valuable as a 4× speedup AND it has Fix B
built in (symmetry by construction).

## Plot

`figures/edge_fixes_compare.png` — 2×2 panel showing the four diagnostics
across all four variants over 20 Picard iters.

## Reproduce

```bash
python projects/REZN/solver_code/sigma_delta/edge_fixes.py
```
