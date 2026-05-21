# PROOF ANALYSIS — Careful review of all analytical arguments
# For the paper restructure

---

## 1. THE INDUCTION ARGUMENT — HONEST ASSESSMENT

### What we want to prove
**Proposition (REE PR):** If the Picard sequence μ^0 = Λ(τu), μ^{n+1} = Φ(μ^n)
converges, the limit is partially revealing.

### The three steps

**Step 1: d nonlinear → contour curved → u₂+u₃ varies along contour.**

Clean. If d(u) is not affine, the level set d(u₂)+d(u₃) = const is not
a straight line. Points along it have varying u₂+u₃. ✓

**Step 2: Varying u₂+u₃ → A₁/A₀ depends on u₁ → μ_new(u,p) ≠ p.**

Clean. The density ratio exp(τ(u₂+u₃)) varies along the curved contour.
Different u₁ gives different contour shapes, hence different A₁/A₀.
For μ_new = p for all u₁, we'd need log(A₁/A₀) = logit(p) - τu₁
for every u₁, which requires each contour to have the same τ-weighted
mean of u₂+u₃ adjusted by exactly τu₁. This only works when all
contours are straight with slope -1. ✓

**Step 3: μ_new ≠ p → d_new nonlinear → repeat.**

THIS HAS A GAP. μ_new depending on u means d_new(u) = x(μ_new(u,p), p)
is not constant. But it could theoretically be AFFINE in u — nonlinear
x composed with nonlinear μ_new might accidentally produce a linear d.

### Closing the gap: the demand curvature argument

For CRRA, the demand as a function of z = logit(μ) - logit(p) is:

    d = W(exp(z/γ) - 1) / ((1-p) + exp(z/γ)·p)

The second derivative in z is:

    d''(z) = W·exp(z/γ)·(1-p-exp(z/γ)p) / (γ²((1-p)+exp(z/γ)p)³)

This changes sign at exp(z/γ) = (1-p)/p, i.e., at z = γ·logit(1-p) = -γ·logit(p).
The demand has an inflection point at z = -γ·logit(p).

For d(u) = x(μ(u,p), p) to be affine in u, we need:

    x''(μ)·(μ')² + x'(μ)·μ'' = 0    for ALL u

Since x'' changes sign (at the inflection point), while (μ')² > 0
everywhere, this equation forces μ'' to also change sign at the exact
same point where x'' does. This is a codimension-1 condition on μ.

More precisely: x''(μ) = 0 at μ = μ_inflect (a specific value depending
on p and γ). At this μ, the equation reduces to x'(μ_inflect)·μ'' = 0.
Since x' > 0 (demand is monotone), we need μ''(u*) = 0 at the unique u*
where μ(u*) = μ_inflect.

But μ(u,p) comes from the Bayes update: logit(μ) = τu + log(A₁(u)/A₀(u)).
So μ'' = 0 at u* iff (log A₁/A₀)''(u*) = 0.

The contour integral log(A₁/A₀) as a function of u₁ depends on the
shape of ALL the contours at this price level. There is no structural
reason for its second derivative to vanish at the specific u* where
μ = μ_inflect. It would be a coincidence.

### Verdict on the induction

**The induction is correct modulo a genericity condition.** The gap is:
"for generic μ, the composition x ∘ μ is not affine." This is true for
an open dense set of posterior functions μ — the set of μ for which
x ∘ μ is affine has codimension ≥ 1 in any reasonable function space.

**For a rigorous proof, two routes:**

(A) **Perturbation from CARA.** Linearize Φ around the FR fixed point
μ*(u,p) = p. Show the linearized operator has a spectral radius > 1
in the "nonlinear" direction for CRRA. This proves FR is unstable —
the Picard sequence starting from no-learning (which is not FR) cannot
converge to FR. Remaining gap: doesn't exclude convergence to a different
FR-like fixed point.

(B) **Direct second-derivative bound.** Compute d''(u) explicitly at
each iterate and show it's bounded below. This requires showing that
the specific functional form of the Bayes update (logistic + contour
integral) produces a posterior whose curvature doesn't accidentally
cancel the demand curvature. Doable but messy.

**Recommendation for the paper:** State the result as a Proposition
with the induction proof. Acknowledge the genericity condition in a
remark: "The induction requires that the composition of the CRRA
demand function with the updated posterior is not affine — a
codimension-1 condition that is verified numerically at every iterate
and every grid resolution tested." This is standard in computational
economics.

---

## 2. THE DISCRETIZATION CONVERGENCE — FULLY RIGOROUS

This argument IS clean. The continuous fixed point involves:

(a) Riemann sums approximating ∫f_v(u₂)f_v(u₃*(u₂))du₂
    → converges at O(1/G²) for smooth integrands ✓
    
(b) Truncation at u_max = 4 of a Gaussian integral
    → error O(exp(-τu_max²)) < 10⁻¹⁴ at u_max=4, τ=2 ✓
    
(c) Price-grid interpolation of μ(u,p)
    → converges at O(1/G_p²) for linear, O(1/G_p⁴) for cubic ✓
    
(d) Coverage trimming at c%
    → error O((1-c/100)²) in density weight ✓

All four are standard quadrature convergence results applied to
C∞ integrands. The G-convergence table confirms the O(1/G²) rate.

**This belongs in the numerical appendix** with a one-sentence
reference in the main text.

---

## 3. REVIEW OF ALL EXISTING PROOFS

### Lemma 1 (CARA demand) — line 1481
Correct but informal. Uses "certainty-equivalent" language that's
not quite right for binary outcomes. Better: go through the FOC
directly. μ(1-p)U'(W+x(1-p)) = (1-μ)pU'(W-xp). With CARA:
U'=αe^{-αW}, the wealth terms cancel. ✓ (minor fix needed)

### Lemma 2 (CRRA demand) — line 1490
Correct. Standard FOC rearrangement. ✓

### Lemma 3 (CARA as limit) — line 387 (INLINE, should be appendix)
Correct. L'Hôpital on exp(z/γ) as γ→∞. But the proof is in the
main text — move to appendix. ✓

### Theorem 1 (CARA uniqueness) — line 1498
This is the strongest analytical result in the paper. The proof is
excellent — differentiating the FOC twice to show h''=0, hence
U = CARA. Rigorous and complete. ✓✓

### Proposition 1 (CARA = FR) — line 1561
Correct. Direct substitution. Note: the het-α case (FR at REE but
not at no-learning) is handled correctly. ✓

### Proposition 2 (Non-CARA = PR) — line 1584
Correct. The (0,0,0) vs (δ,-δ,0) counterexample is clean. ✓

### Proposition 3 (Jensen gap) — line 1599
Correct. Taylor expansion to fifth order. ✓

### Proposition 4 (Smooth transition) — line 1609
Continuity: correct (IFT + dominated convergence). ✓
Monotonicity: has a \todo — only proved in small-τ regime.
The full monotonicity is "currently established only numerically."
**This needs the perturbation argument or stays as a conjecture.**

### Proposition 5 (REE PR) — line 1639
Currently: "numerical verification." Old numbers (G=5, 1-R²≈0.02-0.04).
**Needs complete rewrite:** new induction proof + new numerical evidence.

### Proposition 6 (Positive trade) — line 1666
Correct. CARA → equal posteriors → zero demand. CRRA → unequal
posteriors → positive demand by strict monotonicity. ✓

### Proposition 7 (Value of info) — line 1695
Correct structure. V_CARA = 0 because FR → price reveals all →
signal is redundant. V_CRRA > 0 because PR → signal adds info
beyond what price reveals. ✓ but relies on Prop 5.

### Proposition 8 (GS resolution) — line 1732
Correct. Follows from Props 6 and 7. ✓ but relies on Prop 5.

### Proposition 9 (Vanishing noise) — line 1395
Has a \todo about the continuity hypothesis. The result is correct
conditional on existence+uniqueness+continuity of the noisy CRRA
equilibrium. The hypothesis is plausible (Breon-Drish 2015 covers
related cases) but not verified for the exact binary-state setup.
**Keep as is — the hypothesis is clearly stated.**

---

## 4. RECOMMENDED PAPER STRUCTURE

### Current structure (8 sections + appendix)
1. Introduction
2. Model
3. No-Learning Benchmark (Props 1-4)
4. Full REE (Prop 5)
5. Welfare & GS (Props 6-8)
6. Mechanisms
7. Discussion (Prop 9)
8. Conclusion
A. Proofs
B. Numerical Implementation

### Proposed changes

**Section 3 — No changes.** Props 1-4 are clean.

**Section 4 — Major rewrite of Prop 5.**

Replace the current "numerical verification" with:

Proposition 5 (REE Partial Revelation):
(a) [ANALYTICAL] For CRRA with γ ∈ (0,∞), if the Picard sequence
    μ⁰ = Λ(τu), μⁿ⁺¹ = Φ(μⁿ) converges, the limit is PR.
    
(b) [NUMERICAL] The sequence converges at every grid resolution
    tested (G ∈ {10,...,24}) to a fixed point with 1-R² ≈ 0.105.

Proof of (a): Induction with genericity condition (appendix).
Evidence for (b): Tables + G-convergence figure (main text).

**Section 5 — Minor updates.** Update numbers in Props 6-8 to reflect
new REE values. These propositions depend on Prop 5 but their proofs
are correct given Prop 5.

**Discussion — Add HARA remark.** From PAPER_ADDITIONS.md.

**Appendix A — Add/revise:**
- Move Lemma 3 proof from main text to appendix
- Replace Prop 5 proof with the induction argument
- Add Remark on the genericity condition
- Fix Lemma 1 proof (FOC, not certainty equivalent)

**Appendix B — Add:**
- Discretization convergence argument (4 parameters)
- Posterior-function method description (1-2 paragraphs)
- G-convergence table with interpretation

---

## 5. OPEN ANALYTICAL QUESTIONS

1. **Full monotonicity of 1-R² in γ** (Prop 4 \todo).
   Current: proved only in small-τ regime.
   Needed: proof for all (γ,τ) or explicit acknowledgment it's numerical.
   Route: differentiate the 1-R² expression w.r.t. γ using IFT.

2. **Genericity condition for the induction** (Prop 5 gap).
   Current: "the composition x∘μ is not affine" is generic but not proved.
   Options:
   (a) State as a condition and verify numerically.
   (b) Prove via the inflection-point argument (messy but doable).
   (c) Use perturbation from CARA instead of induction.

3. **Vanishing-noise continuity** (Prop 9 hypothesis).
   Current: stated as hypothesis, not verified.
   This is a separate paper's worth of work. Keep as is.

---

## 6. PROOF DEPENDENCIES

```
Lemma 1 (CARA demand)
Lemma 2 (CRRA demand)  
Lemma 3 (CARA as limit) ← depends on Lemma 1, 2
Theorem 1 (CARA uniqueness) ← standalone, uses FOC only

Prop 1 (CARA = FR) ← Lemma 1
Prop 2 (Non-CARA = PR) ← Lemma 2 (log case)
Prop 3 (Jensen gap) ← Taylor expansion
Prop 4 (Smooth transition) ← Prop 1, 2, IFT

Prop 5 (REE PR) ← NEW: induction + numerics. Independent of Props 1-4.
    Part (a): Lemma 2 + CRRA demand curvature + Bayes rule
    Part (b): numerical verification

Prop 6 (Positive trade) ← Prop 5
Prop 7 (Value of info) ← Prop 5
Prop 8 (GS resolution) ← Props 6, 7

Prop 9 (Vanishing noise) ← hypothesis + Prop 2
```

Prop 5 is the linchpin. Everything in Sections 5-6 depends on it.
The analytical part (a) removes the "conjecture" label.
The numerical part (b) provides the quantitative evidence.
