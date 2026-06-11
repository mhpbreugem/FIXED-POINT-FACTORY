# Stage 6b: graph/height-function strict-h=0 solver — results

**Target:** convergent strict-h=0 solver for the K=3 CRRA REE at (tau, gamma) = (2.0, 0.098),
height-function representation H_m(a,b) of M price-level surfaces, then revelation deficit vs
d_inf = 0.268 +/- 0.010.

**Code:** `projects/REZN/solver_code/cmm_endogenous/cmm_stage6b_solver.py`
(+ `cmm_stage6b_analyze.py`, `cmm_stage6b_probe.py`).
**Artifacts:** `stage6b_results.json`, `stage6b_H.npy` (M=8 x 15 x 15 final heights),
`stage6b_masks.npy` (final well-posed vertex masks), logs `stage6b_run.log` (first attempt),
`stage6b_run2.log` (final run).

**VERDICT: NOT CONVERGED — and the reason is now precisely diagnosed.** The strict-h=0
equations are structurally singular on a fat region of every surface (the one-signal-dominance
band); no Newton-type method can drive max|r| to 1e-6 there. On the well-posed subset the graph
formulation gets far below the mesh's plateau (medians 0.0075–0.079 vs mesh 0.130) — the wall
is in the *problem*, not the mesh bookkeeping.

---

## T1 — slice-root validation (PASS)

Bugs fixed vs the 6a sketch: `np.trapz` -> uniform trapezoid (numba); dF/dc re-derived
(the sketch's cyclic form `E_A[j]+E_A[l]` equals `-E_A[k]` since the e_a/e_b components sum
to zero — it was correct but opaque; clean form `dF/dc = 2/sqrt3 + H_a e_a[k] + H_b e_b[k]`
is in the module docstring); c'(s) analytic via implicit differentiation
`c' = (H_a DAS_k + H_b DBS_k) / dF/dc` (no `np.gradient`).

5 random vertices, initial H (kernel FP), root equation residual and curve-on-surface check
(P interpolated on the warm-start cube along the recovered curve):

| m | vertex (a,b) | k | max\|F\| | \|P-p_m\| max | \|P-p_m\| med |
|---|---|---|---|---|---|
| 3 | (-0.64, +1.29) | 2 | 8.3e-14 | 7.5e-03 | 2.6e-03 |
| 0 | (0.00, +2.57)  | 2 | 9.6e-14 | 4.1e-02 | 1.6e-02 |
| 5 | (+1.93, +3.21) | 2 | 7.4e-14 | 1.9e-06 | 6.8e-08 |
| 6 | (0.00, +1.29)  | 0 | 9.6e-14 | 1.2e-02 | 2.8e-04 |
| 2 | (-2.57, -1.29) | 2 | 9.4e-14 | 2.5e-02 | 1.1e-02 |

Roots to 1e-13; curve-vs-cube deviation is O(cube trilinear interpolation error) on the
du=0.4 grid (median few 1e-3, worst 4e-2 in high-curvature regions). PASS.

## T2 — initial residual, gate V1 (PASS)

M=8 quantile surfaces, 15x15 H grid on [-4.5,4.5]^2, 13x13 vertices/surface (1352 total),
3.6 s full evaluation (numba; target <60 s beaten by 16x).

- Graph-in-s residual: **max 0.3285, med 0.1618, p90 0.2796**
- Mesh stage-3b reference at the same FP: max 0.324, med 0.180.
- Gate V1 (med within factor 3 of 0.18): **PASS** — the two strict-h=0 formulations see the
  same disequilibrium.
- Fold-aware tracer residual (see below): max 0.3333, med 0.1619 — consistent.

## T3 — solver: what happened and why

### First LM attempt (graph-in-s residual, plain Marquardt) — FAILED, diagnostically
Medians collapsed (1e-4 per surface) while max froze; the LM null space drifted heights to
|t|~28 (sign-flip diag 1.1e-2 -> 2.1e+2), and the per-surface medians were branch-tracked
artifacts. FD-vs-direct probe on m=3: ||J||~1e5 spurious entries while true directional cost
derivatives are O(1-10) — the FD Jacobian was invalid.

### Root cause (the central finding)
The slice-curve graph premise — unique root c(s) — requires the surface's directional tilt
along (e_a[k], e_b[k]) to stay below 2/sqrt3 / sqrt(2/3) = **sqrt(2)**. A directional slope of
exactly sqrt(2) corresponds to a surface locally of the form {u_k = const}: price pinned by a
single trader's signal. Tilt audit of the kernel-FP surfaces (61x61 sample):

| m | p | max\|gradH\| | min Fc | frac(Fc<0.05) |
|---|------|------|--------|------|
| 0 | 0.162 | 2.51 | -0.86 | 0.54 |
| 1 | 0.328 | 2.29 | -0.70 | 0.58 |
| 2 | 0.333 | 2.14 | -0.57 | 0.36 |
| 3 | 0.417 | 1.65 | -0.19 | 0.68 |
| 4 | 0.583 | 1.65 | -0.19 | 0.68 |
| 5 | 0.667 | 2.14 | -0.57 | 0.36 |
| 6 | 0.672 | 2.29 | -0.70 | 0.58 |
| 7 | 0.838 | 2.51 | -0.86 | 0.54 |

The middle surfaces sit AT the critical slope sqrt(2)=1.414 through the high-density center
(node-slope check: central max slope exactly 1.414 for m=3/4). Physics: with tau=2 a large
part of the cube has two of three posteriors saturated; CRRA clearing then gives the 1/3-2/3
"one-dissenter" plateaus (visible as atoms in the P distribution — the M=8 quantile levels
came out 0.328 AND 0.333!). Near those plateaus the level surface follows
{u_dissenter = const}; for the dissenter agent, conditioning on P=p_m degenerates (the level
set in their slice plane is near-empty-or-everything). Consequences at exact h=0:
- slice curves fold (no graph in s), evidence integrals acquire sqrt-singular and
  discontinuous dependence on H (topology flips of the traced component — the graph
  twin of stage 5's TOPO_FLIP storms);
- a band of vertices has exponentially saturated posteriors: residual rows with near-zero
  Jacobian but O(0.1-0.3) residual — *unsatisfiable equations* within reach of any solver.

### Fixes implemented (second attempt)
1. **Pseudo-arclength tracer** for the slice curve through the vertex (the vertex lies
   exactly on the curve): no parametrization singularity at folds, handles supercritical
   tilt exactly, C1-true unclamped linear extrapolation of H beyond the grid (the tilt-clamped
   version kinks at the grid edge and stalls the corrector), adaptive step halving,
   weight-cutoff termination (1e-13). Validated against the graph-in-s residual
   (med diff 3e-5 on mild surfaces).
2. **Tikhonov anchor** alpha||H-H0||^2 (alpha=1e-3) pinning the LM null space; step cap
   ||d||_inf <= 1.
3. **Well-posedness filter** (frozen at warm start): keep vertices with (i) ex-ante density
   >= 1e-8 at the lift (the cube-halo analog), (ii) smooth rows (no initial |J|>100 —
   tangency-singular), (iii) sensitive rows (max row |J| >= 0.05 — not saturated).
   Plus phantom suppression |J|>100 -> 0 during the solve.

### Final run (M=8, 15x15, 40 LM iters/surface, ~5 min total)

| m | p | density-active | ill-posed dropped | well-posed | final max\|r\| | final med\|r\| |
|---|------|----|----|----|--------|---------|
| 0 | 0.162 | 50 | 33 | 17 | 0.1305 | 0.0075 |
| 1 | 0.328 | 72 | 39 | 33 | 0.0791 | 0.0269 |
| 2 | 0.333 | 110 | 31 | 79 | 0.2691 | 0.2470 |
| 3 | 0.417 | 115 | 50 | 65 | 0.2503 | 0.0794 |
| 4 | 0.583 | 115 | 53 | 62 | 0.2501 | 0.0669 |
| 5 | 0.667 | 110 | 33 | 77 | 0.0367 | 0.0083 |
| 6 | 0.672 | 72 | 39 | 33 | 0.0681 | 0.0235 |
| 7 | 0.838 | 50 | 32 | 18 | 0.0880 | 0.0111 |

Overall (well-posed vertices): **max 0.2691, med 0.0439** — `converged(<1e-6) = False`.

### Plateau localization
- The dropped ill-posed band: 30-50% of density-active vertices on every surface — the
  saturation band, mid-radius |w| in [1.5, 4] for extreme surfaces, reaching the center
  for m=3/4 (the critical-tilt band runs through w=0 along the a=0 / pair-symmetric lines).
- Surviving worst rows: m=2 stalls at 0.269 immediately (iteration 1) at central vertices
  (|a|~1.3-1.9, b~0.6) — the p~1/3 atom level is degenerate in the bulk; m=3/4 stall at
  0.25 at (|a|~2.6-3.2, |b|~1.3-1.9); m=0/7 at (|a|~1.3, b=0).
- Mirror-pair asymmetry (m=2: 0.269 vs m=5: 0.037) is LM path dependence on a jagged
  landscape — another symptom of non-smoothness, not of an asymmetric problem
  (initial sign-flip diag 9e-3).
- **Resolution test:** m=3 re-solved on 25x25 over [-3.6,3.6] (spacing 0.30 vs 0.64):
  182/425 rows singular (vs 49/115), LM cannot take any step. Refinement resolves MORE
  singular structure -> the plateau is intrinsic to strict h=0, not a resolution artifact.
- Comparison to mesh: stage-5 mesh LM plateaued at med 0.130 / max 0.331 globally. The graph
  formulation reaches med 0.0075-0.079 on 6 of 8 surfaces (overall med 0.0439) — well below
  the mesh wall — but cannot touch the singular band that the mesh also could not pass.

## T4 — deficit

Not meaningfully computable: no converged strict-h=0 equilibrium exists to measure.
- Reconstruction machinery verified: warm-start cube unweighted deficit = **0.2796**
  (exactly the kernel G=21 value); reconstruction from initial H gives **0.2653**
  (M=8 logit-linear ladder bias -0.014).
- Reconstruction from final H gives 0.916 — an artifact: the per-surface independent partial
  solves moved heights by up to ~2 in the solved regions and broke the global p-ladder
  ordering (5297 pointwise crossings vs 1030 initially). This number is reported only as a
  diagnostic; it is NOT an equilibrium deficit.
- d_inf = 0.268 +/- 0.010 comparison: **cannot be made at strict h=0 with this method.**

## Conclusions / recommendation

1. The graph (height-function) formulation is correctly implemented and validated (T1, V1).
   It eliminates the mesh pathologies it was designed to eliminate (no topology bookkeeping,
   exact root-recovered evidence domains, 16x faster residuals).
2. It thereby EXPOSES the real wall: at (tau=2, gamma=0.098) the strict-h=0 fixed-point
   equations are singular/ill-posed on the one-signal-dominance band (critical surface tilt
   sqrt(2), CRRA 1/3-2/3 clearing plateaus, exponentially saturated evidence). The kernel
   operator's h-blur is not a numerical convenience there — it is the regularization that
   makes the problem well-posed.
3. Recommendation for Stage 7: h-continuation (solve the kernel-regularized problem on a
   decreasing h-ladder and Richardson-extrapolate the deficit to h->0) instead of exact h=0;
   or reformulate the evidence conditioning in the saturated band (e.g. conditioning on the
   price PLATEAU event {P in [1/3-eps,1/3+eps]}, which has positive measure, rather than on
   the measure-zero level set).
