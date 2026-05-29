# K=3 / K=4 CRRA Rational-Expectations Equilibrium — Investigation Findings

**Branch:** `claude/study-fixed-point-economics-y12PB`
**Model:** K traders, noisy private signals u_i about a binary asset value v∈{0,1}; signal
precision τ; CRRA risk aversion γ; market mass W_i. Market-clearing price P(u_1,…,u_K) is a
function of all signals. Rational expectations: each trader reads the price as an extra signal.
Equilibrium = fixed point Φ(P)=P of the "infer-from-price → Bayes-update → CRRA-clear" operator.

This document is the human-readable index + extended description of every study directory and the
corrected conclusions. Full method writeup with figures: `methodology/methodology.pdf` (9 pp);
runnable reference: `methodology/reference_operator.py`.

---

## LATEST (no-kernel / smoothness chapter — resolves "is the smoothing just noise?")

**Question pursued:** can the deterministic PR equilibrium be nailed with NO smoothing parameter
(no kernel h), to settle whether the kernel is "noise in disguise"?

**Answer — sharp and honest:**
1. **A no-kernel co-area operator exists and nails the PR at coarse G** (`k3_hfree_fast`,
   `k3_hfree_morse`): zero kernel/bandwidth, only grid + Gauss quadrature (standard
   discretizations). Nails G=9 to ‖F‖≈8e-15, deficit≈0.17. Proves a smooth deterministic fixed
   point exists with no regularization parameter. The Morse-robust variant is also C0 (continuous):
   the naive contour-scan's O(0.1) tie-jumps were a discretization artifact (jump 1115→1.18, shrinks
   with step).
2. **But it FAILS at high G** (G≥13: ‖F‖ floors ~2.6e-3, Newton stalls). **Diagnosed and PROVEN
   (`k3_hfree_morse/c0_not_c1.{json,png}`):** the deterministic co-area evidence A_v(p) is
   **C0 but NOT C1** at Morse-critical prices (where ∇P=0 on a level set). On a controlled saddle
   surface, A_v(p) spikes and its derivative A_v'(p) blows up **128×** at the critical price, while
   the kernel-band A_h'(p) stays bounded. Newton needs C1; at finer grids more cells sit near
   critical prices → the no-kernel operator is fundamentally not Newton-nailable at high resolution.
3. **The kernel is the C∞ fix, and it is NUMERICS not noise:** the Gaussian band is a CONSISTENT
   co-area quadrature (→ the exact integral as h→0, verified to 1.5e-5; the missing 1/|∇P| weight
   makes the *naive* scan biased ~19%, `k3_verify_coarea`). It regularizes the intrinsic
   C0-not-C1 singularity. So the kernel parameter is a numerical convergence knob (deficit is
   INVARIANT to it once resolved), NOT economic noise (whose deficit would be a real h-curve).

**Net:** partial revelation is a genuine **deterministic** CRRA equilibrium — **no noise traders
needed** (the contribution vs the 1990s noise-trader requirement). Existence is solid: continuous
operator ⇒ Brouwer fixed point; nailed at G=9 (no-kernel) and to G=31 by the kernel method
(continuum deficit ≈ 0.26, `k3_coarea_limit`). High-RESOLUTION computation requires a C∞
(kernel/consistent-quadrature) operator because the bare co-area operator is C0-not-C1 — a
numerical fact, stated honestly, not a re-introduction of noise.

**Definitive (γ,τ) deficit map** (`k3_coarea_2dsweep`, consistent kernel co-area, G=17, 31/32 cells
nailed to 1e-12): deficit rises with τ, falls with γ — the Jensen-gap law on genuine equilibria;
replaces the artifact strict-scan maps. **CARA = γ→∞ CRRA limit** (`k3_cara`, deficit ~γ^-0.94 → 0):
the no-gap benchmark — confirming the Jensen gap (hence partial revelation) is exactly the CRRA
wealth-curvature effect, absent under CARA.

---

## HEADLINE RESULT (corrected)

The deterministic (noiseless, h=0) K=3 CRRA economy has a **genuine, smooth, partially-revealing
(PR) equilibrium** — a true fixed point with a positive information deficit (1−R² ≈ 0.26–0.28 at
γ=0.1, τ=2), **nailable to machine/100-dec precision** (residual ≤ 1e-12 … 1e-14) with a *consistent*
numerical operator. Fully-revealing (FR) is the only fixed point when there is no belief dispersion.

### The correction that mattered
For most of this investigation the strict operator appeared to have an *intrinsic non-smooth
residual floor* (~0.06–0.15) — it oscillated, would not nail, and the floor got **worse** under grid
refinement. We attributed this to "PR is not a fixed point / PR needs noise."

**That was a numerical artifact.** The continuum inference is a level-set (co-area) integral
A_v(p) = ∫_{P=p} f_v/|∇P| dσ, which is a **smooth** function of p (the level set deforms smoothly).
The naive grid "contour scan" (detect sign-changes of P−p on grid edges) is an *inconsistent*
discretization in two ways:
  1. **Binary edge include/exclude** → discontinuous O(0.1) jumps as p crosses a grid node (the
     "tie" problem; an active-set discontinuity). Verified: naive scan's evidence jump does NOT
     shrink as the price step halves, while the proper contour integral's does (`k3_strict_h0_exact`).
  2. **Missing the 1/|∇P| co-area weight** → it integrates the *arc-length* measure, a definite
     ~19% bias. Verified on a known analytic surface: naive → 2.07 vs true co-area 1.748
     (`k3_verify_coarea`).

A *consistent* operator removes both: the Gaussian-band/kernel evidence ∫K_h(P−p)f du → the co-area
integral automatically (with the correct 1/|∇P| weight). Taken in the **joint limit** (bandwidth
h→0 AND grid spacing Δx→0 with Δx≪h — the interior-point "central path"), it nails the genuine PR
equilibrium to ~1e-14 with a positive grid-independent deficit (`k3_coarea_limit`). Doing it strictly
at h=0 with no kernel at all — exact marching-squares contour + 1/|∇P| weight — gives a continuous
operator (`k3_strict_h0_exact`).

---

## THE JENSEN GAP (mechanism, closed form)

Expanding CRRA demand in log-odds (m=logit μ, π=logit p, δ=(m−π)/γ):
  x_i ≈ W_i[ δ_i + (½−p)δ_i² ].
The quadratic term (absent for CARA) is the whole story. Market clearing gives, to 2nd order:

  **π = m̄_W + (½ − p)·Var_W(m)/γ**        ← the price = average belief + a Jensen/curvature wedge.

Consequences (all verified, `k3_jensen_gap`):
  - The gap is **bound to disagreement** Var_W(m): it is identically zero under full revelation (so
    FR is always a clean fixed point; gap ≡ partial revelation).
  - **Sign (½−p):** the price is *compressed toward ½* (under-reaction; measured slope β<τ).
  - **∝ 1/γ:** more risk aversion → smaller gap → more revealing.
  - **∝ Var(m):** sharper signals (higher τ) → more dispersed beliefs → larger gap.
  - Exact at large γ / small disagreement; non-perturbatively larger in the low-γ regime where PR is
    strong (the regime is intrinsically beyond 2nd order).
  - Explains the K=3→K=4 collapse: more agents → more aggregation → less disagreement → gap → 0 → FR.

CARA is the γ→∞ limit (linear demand, no curvature term) — the no-gap benchmark (`k3_cara`).

---

## K=4

  - **FR is the unique, globally attracting equilibrium** under homogeneous or moderately
    heterogeneous (τ,γ): every seed — no-learning, fully-revealing, non-revealing constant, anti-FR,
    random, extreme — flows to full revelation; no period-2 oscillation (`k4_sym`, `k4_basin`,
    `k4_extreme`, `k4_het`). Contrast K=3, whose deterministic learning dynamics settle on the PR
    object.
  - **Extreme risk-aversion heterogeneity** γ=[g,g,1,1], g→∞: the two risk-averse agents stop
    trading (demand→0), so the price reveals only the active traders → genuine partial revelation
    (deficit → ~0.33; β_active rise, β_averse → 0) — an *effective-agent-count* channel
    (`k4_riskaversion`). At g=300 it floors ~2.6e-3 (`k4_pr_nail`); whether this nails fully under
    the consistent co-area operator is open.

---

## DIRECTORY INDEX

### Current / authoritative
  - `methodology/` — **full method writeup (PDF, 9pp) + runnable reference_operator.py.** Start here.
  - `k3_coarea_limit/` — joint co-area h→0 ladder: genuine smooth PR nailed to 1e-14, deficit→~0.28
    (positive, grid-independent). **The key result.**
  - `k3_verify_coarea/` — verification: h→0 ladder = co-area-WEIGHTED contour ≠ naive scan (19% bias);
    6-point numerical-issues audit.
  - `k3_strict_h0_exact/` — strict h=0, NO kernel, exact marching-squares co-area, flint ~130-dec,
    nail toward 1e-100; confirms the strict operator is *continuous* (naive scan's jumps were the
    artifact).
  - `k3_coarea_sweep/` — (γ,τ) deficit map on the genuine equilibrium (all cells nail); deficit
    rises with τ, falls with γ.
  - `k3_sweep100/` — 15×15 (γ,τ)∈[0.1,100] log grid at 100-dec; high-τ cells flagged under-resolved.
  - `k3_cara/` — CARA operator + test that CARA is the γ→∞ CRRA limit (no-gap benchmark).
  - `k3_jensen_gap/` — verifies the closed-form gap = (½−p)Var(m)/γ on the genuine equilibrium.
  - `k4_sym/`, `k4_basin/`, `k4_extreme/`, `k4_het/` — K=4: unique globally-attracting FR.
  - `k4_riskaversion/`, `k4_pr_nail/` — K=4 partial revelation via extreme-γ agent dropout.

### Superseded / artifact-contaminated (kept for the record)
  - `flint_pr_nail*/`, `flint_clamp_solve/`, `flint_sym_tie/`, `flint_smooth*/`, `flint_g21/`,
    `flint_gamma_floor/`, `flint_strict_200/`, `k3_plateau_table/`, `k3_plateau_highgamma/`,
    `k3_wide_plateau/`, `flint_nl_picard/`, `flint_osc_vs_spike/`, `k3_strict_cycle/`,
    `k3_noisy_pr/`, `k3_noisy_grossman/` — these used the *naive contour scan* or the
    fixed-grid h→0 (which stalls when the band drops below grid spacing). Their "non-smooth floor /
    oscillation / needs-noise" findings are the discretization artifact, not the economics. Retained
    to document the path and the diagnosis.
  - `flint_persist2/` — original Picard checkpoint (warm-start source).

---

## NUMERICAL CAVEATS (honest)
  1. **Co-area weight (HIGH):** the naive contour scan is biased ~19% (arc-length vs co-area measure).
  2. **Deficit measure-dependence (MED):** 1−R² depends on how states are weighted (uniform vs
     signal-weighted vs stretched grid; spread ~0.12). The price *surface* is invariant; report the
     measure with any scalar deficit.
  3. **Bandwidth exponent (MED):** limit ~0.28 robust; use matched-grid raw values, not naive
     h→0 extrapolation of a single grid.
  4. **High-τ resolution:** signal width ~1/√τ; at τ≳16 a feasible grid under-resolves — flagged.
  5. **Morse-critical prices (LOW):** mild integrable 1/|∇P| singularities where level-set topology
     changes; do not prevent nailing at generic fixed points.
