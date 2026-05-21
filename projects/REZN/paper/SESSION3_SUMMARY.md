# SESSION 3 SUMMARY — 2026-05-02/03
# For ChatGPT or future Claude: read this entire file before doing anything.

## PROJECT
Author: Matthijs Breugem, Nyenrode Business University
Paper: "On the Possibility of Informationally Inefficient Markets Without Noise"
Repo: github.com/mhpbreugem/REZN
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6693198 (posted May 2, 2026)
arXiv: pending endorsement (econ.GN, arXiv perpetual non-exclusive license)

## WHAT THE PAPER PROVES
CARA (exponential) utility is the UNIQUE preference class that gives full revelation
of information through prices. ANY non-CARA preference (including CRRA) gives partial
revelation WITHOUT noise traders, random endowments, supply shocks, or any substitute.
The mechanism: CARA demand is linear in log-odds (= Bayesian sufficient statistic),
so market clearing aggregates in the right space. Non-CARA aggregates in a different
space → Jensen gap → partial revelation. This resolves the Grossman-Stiglitz paradox.

## PAPER STATE (as of session end)
- 41 pages, 12pt, 0.92in margins
- 0 undefined references, 0 TODOs
- "Preliminary and Incomplete" on title page
- Extended proofs (337→561 lines) with full intermediate steps
- CARA FR proof generalized to arbitrary payoff distributions (Remark 2)
- Convexity conjecture added to appendix

## STRUCTURE
1. Introduction
2. Model (info, preferences, equilibrium)
3. The Knife-Edge (aggregation space, CARA→FR, non-CARA→PR, Jensen gap, smooth transition)
4. Full REE (contour mechanism, why straight/curved, existence, magnitudes)
5. Economic Implications (volume, V(τ), GS, mechanisms, robustness)
6. Related Literature and Extensions
7. Conclusion
A. Proofs (extended)
B. Numerical Implementation

## KEY NUMBERS IN PAPER (UPDATED THIS SESSION)
- 1-R² = 0.085 (was 0.108, corrected via WEIGHTED regression)
- slope = 0.543 (was 0.341)
- G = 20 (was G=14)
- Precision: ||F|| = 7.4e-119 (mp300)
- "prices capture roughly half of T*" (was "a third")

## CRITICAL DISCOVERY THIS SESSION: WEIGHTED 1-R²
Unweighted 1-R² gives WRONG values that depend on umax:
- G=15 umax=4 unweighted: 0.191
- G=20 umax=5 unweighted: 0.230
WEIGHTED by ex-ante probability w = 0.5*(f₀³ + f₁³):
- G=15 umax=4 weighted: 0.078
- G=18 umax=4 weighted: 0.083
- G=20 umax=5 weighted: 0.085
Top 10% of triples carry 99.7% of probability mass.
Formula: np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))

## FIGURES STATUS

### WHITE BACKGROUND (real data):
| Fig | Description | Source |
|-----|-------------|--------|
| 1   | Knife-edge 1-R² vs τ | G=15 no-learning, γ=0.5/1/4 |
| 3A  | CARA contours (straight) | Analytical + solver endpoints |
| 3B  | CRRA contours (curved) | G=18 mp300, convexity-enforced |
| 4B  | REE vs NL vs γ | G=20 mp50 weighted, 4 of 6 γ values |
| 6A  | CARA posteriors | Analytical Λ(T*/3) |

### GRAY BACKGROUND (placeholder or pending):
| Fig | Description | Status |
|-----|-------------|--------|
| 4A  | REE 1-R² vs τ (3 curves) | γ=0.5: 7pts ✅, γ=1.0: 8pts ✅, γ=4.0: REDO |
| 5   | Price vs T* | Terminal 4 extracted (asymmetric triples) ✅ |
| 6B  | CRRA posteriors | Terminal 4 extracted from G=18 ✅ |
| 7   | Trade volume | Skip for SSRN |
| 8   | Value of info | Skip for SSRN |
| 9   | GS resolution | Skip for SSRN |
| 10  | Convergence | Terminal 4 extracted ✅ |
| 11  | Mechanisms bar chart | Done ✅ |
| R1  | K agents | Done ✅ |
| R2  | Lognormal | Terminal 4 extracted ✅ |

### FIGURE STYLE
- 5-color scheme: red(p=0.2)/gold(p=0.3)/black(p=0.5)/green(p=0.7)/blue(p=0.8)
- BCred(0.7,0.11,0.11), BCgold(0.72,0.53,0.04), BCgreen(0.11,0.35,0.02), BCblue(0,0.20,0.42)
- 5 dash patterns: dotted/dashed/solid/dashdotted/loosely dashed
- Legend in upper right (Fig 3B only, applies to both panels)
- Gray placeholders: black!8
- Gray table rows: rowcolor{black!8} or {black!15}
- Linear interpolation (no pgfplots smooth keyword)
- Data preprocessed via scipy CubicSpline + arc-length resampling

## SOLVER STATUS (4 parallel terminals)

### Terminal 1 (main branch): γ=0.5 high-τ
- τ=0.3-3.0: 7 points converged ✅
- τ≥4: stuck (boundary issue, posteriors hit 0/1)
- STOPPED

### Terminal 2 (branch claude/task3-gamma1-KSNBo): γ=1.0 τ-sweep
- τ=0.3-3.0: 7 points converged (||F|| < 1e-12) ✅
- τ=4.0: marginal (||F|| = 2.3e-5) ⚠️
- τ≥5: NOT converged, DISCARD
- DONE — 8 usable points

### Terminal 3 (branch claude/task3-gamma4-fDgbB): γ=4.0 τ-sweep
- Used WRONG SOLVER (K3 staggered NK float64 instead of posterior-method v3 mp50)
- ALL 12 points have ||F|| > 0.001 — NONE converged
- NEEDS COMPLETE REDO with correct solver
- γ=4.0 is closest to CARA, should converge fastest

### Terminal 4 (branch claude/extractions-oC85I): Figure extractions
- Fig 5 (asymmetric triples): ✅ DONE — three distinct curves
- Fig 6B (G=18 posteriors): ✅ DONE
- Fig R2 (lognormal): ✅ DONE — peaks at τ≈7-8
- Fig 10 (convergence): ✅ DONE — 9 iters, quadratic convergence
- ALL DONE

## COMPLETED γ-SWEEP (Fig 4B) — ALL 6 VALUES

| γ   | REE 1-R² | NL 1-R² | Amplification | REE slope |
|-----|----------|---------|---------------|-----------|
| 0.1 | 0.257    | 0.141   | 1.8×          | 0.41      |
| 0.25| 0.133    | 0.098   | 1.4×          | 0.51      |
| 0.5 | 0.088    | 0.062   | 1.4×          | 0.52      |
| 1.0 | 0.047    | 0.029   | 1.6×          | 0.55      |
| 2.0 | 0.025    | 0.011   | 2.3×          | 0.59      |
| 4.0 | 0.016    | 0.003   | 4.7×          | 0.61      |

KEY FINDING: REE exceeds NL at EVERY γ. Learning from curved contours
AMPLIFIES partial revelation, doesn't reduce it. Amplification ratio
INCREASES with γ (paradoxically, preferences closer to CARA show more
amplification).

## PARTIAL τ-SWEEP (Fig 4A)

| τ   | γ=0.5  | γ=1.0  | γ=4.0  |
|-----|--------|--------|--------|
| 0.3 | 0.037  | 0.030  | REDO   |
| 0.5 | 0.038  | 0.026  | REDO   |
| 0.8 | 0.047  | 0.031  | REDO   |
| 1.0 | 0.052  | 0.033  | REDO   |
| 1.5 | 0.064  | 0.040  | REDO   |
| 2.0 | 0.088  | 0.046  | 0.016  |
| 3.0 | 0.138  | 0.066  | REDO   |
| 4.0 | stuck  | 0.090⚠️| REDO   |

## IMMEDIATE NEXT STEPS
1. Redo Terminal 3 (γ=4.0) with correct solver (posterior-method v3 mp50)
2. Build Fig 4A from available data (γ=0.5 + γ=1.0, 15 points)
3. Build Figs 5, 6B, R2, 10 from Terminal 4 extractions into paper
4. Update Fig 4B with all 6 γ values
5. Update SSRN to v2

## HIGH-PRECISION CHECKPOINTS IN REPO
- G=15 umax=4 mp100: results/full_ree/posterior_v3_G15_mp100_final.json
- G=15 umax=4 mp200: results/full_ree/posterior_v3_G15_mp200.json
- G=18 umax=4 mp300: results/full_ree/posterior_v3_G18_mp300_notrim.json
- G=20 umax=5 mp300: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json

## KEY FILES
- main.tex — the paper (41 pages)
- FIGURES_TODO.md — solver task list with weighted 1-R² formula
- PARALLEL_SOLVER.md — instructions for 4 parallel terminals
- ANNOTATED_PROOFS.md — detailed proof explanations (personal reference)
- python/convex_contour.py — convexity-constrained contour integration
- python/build_fig3B_G18.py — builds Fig 3B from G=18 data
- python/verify_convexity.py — verifies contour curvature sign
- proofs_extended.tex — extended proofs source

## THINGS TO REMEMBER
- Standard gammas: γ = 0.5, 1.0, 4.0
- ALWAYS use weighted 1-R² (formula in FIGURES_TODO.md)
- Tolerance: ||F|| < 1e-25 for mp50, 1e-100 for mp300
- Do NOT converge to machine precision (fits interpolation artifacts)
- Fig 3B uses style 29: 5 colors, 5 dashes, legend upper right
- Gray backgrounds: black!8 for figures, rowcolor{black!8/15} for tables
- G=25 is stuck on singular Jacobian — accept G=18 for Fig 3B
- All contour data preprocessed: CubicSpline + arc-length resample → linear interp in pgfplots
