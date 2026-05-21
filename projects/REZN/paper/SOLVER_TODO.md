# SOLVER TODO — Posterior Method v3
# Priority-ordered tasks for Claude Code
# Read POSTERIOR_METHOD_V2.md (sections A-E) for method details
# Current best results: G=14 strict (max≤1e-14) at γ=0.3,0.5,1,2 and τ=0.5,1,2,4,8

---

## STATUS SUMMARY

### What works (strict convergence, max≤1e-14)
- G=10, 12, 14 at baseline (γ=0.5, τ=2)
- G=14 γ-ladder: γ=0.3, 0.5, 1.0, 2.0
- G=14 τ-ladder: τ=0.5, 1.0, 2.0, 4.0, 8.0

### What doesn't work yet
- G≥16: NK-monotone trade-off. Picard+PAVA converges median cells but
  max residual stuck at ~0.08-0.20. NK polish breaks p-monotonicity.
- γ=0.1: NK breaks p-monotonicity. Fallback 1-R²=0.154 but max=0.05.
- γ≥5: not attempted at strict level.

---

## P0: CRITICAL FOR PAPER

### 1. Run at paper figure gammas: γ = 0.25, 1.0, 4.0
The paper figures use γ=0.5, 1, 4 (not 0.3, 0.5, 2). Need strict
convergence at G=14 for all three at τ=2.
- γ=0.5: between 0.3 (done, 1-R²=0.119) and 0.1 (fallback). Should converge.
- γ=1.0: already done (1-R²=0.100). Verify.
- γ=4.0: between 2.0 (done, 1-R²=0.079) and 10 (not done). Should converge.

### 2. CARA baseline at each parameter point
Run explicit CARA (γ=100 or γ=50) at G=14, τ=2. Must give 1-R²≈0
(machine precision). This is the ground truth — if CARA doesn't give
FR, the method has a bug. Currently only verified informally.
Report: converged μ*(u,p) = p (or close), residual, 1-R².

### 3. REE survival ratios
For each strictly converged (γ, τ):
- Compute no-learning 1-R² at the same G=14 (this is fast, no iteration)
- Report: REE 1-R² / no-learning 1-R² = survival ratio
- Current puzzle: REE 1-R² (0.108) > no-learning 1-R² (0.062) at γ=0.5, τ=2.
  This needs verification. Is the no-learning number computed at G=14 or G=20?
  Recompute both at G=14 with identical 1-R² methodology.

### 4. Posteriors table for Fig 4
At the converged μ* (G=14, γ=0.5, τ=2), reconstruct P at realization
(u₁,u₂,u₃) = (1, -1, 1):
- Solve market clearing: F(p) = Σ d(u_k; p) = 0 for p
- Extract posteriors: μ*(u_k, p) for k=1,2,3
- Report table:

| | Prior μ_k | CARA posterior | CRRA posterior |
|---|---|---|---|
| Agent 1 (u=+1) | | | |
| Agent 2 (u=-1) | | | |
| Agent 3 (u=+1) | | | |
| Price | | | |

---

## P1: HIGH PRIORITY

### 5. Break through G≥16
The NK-monotone trade-off is the main blocker. Three approaches to try:

a) **Gap reparametrization** (POSTERIOR_METHOD_V2.md §E): Instead of
   PAVA projection, reparametrize as logit(μ_k) = logit(μ_{k-1}) + exp(c_k).
   Monotonicity by construction, smooth, nonsingular Jacobian. Run NK
   in (base, c) space. May avoid the NK-vs-monotone conflict entirely.

b) **Warmer warm-start**: Interpolate converged G=14 onto G=16 grid
   with high-order interpolation (cubic, not linear). The G=14
   solution is strict — should be a good starting point if interpolated
   carefully.

c) **Hybrid**: Picard-PAVA to get close (max~0.01), then gap-reparam
   NK to polish.

### 6. Knife-edge figure data at full REE
At G=14, sweep τ = 20-30 log-spaced points from 0.1 to 10.
Three curves: γ = 0.25, 1.0, 4.0.
Output as pgfplots coordinates.
This is the REE version of the no-learning knife-edge figure.
Compare: are the curves similar? Does learning change the shape?

### 7. Contour figure data (Fig 3)
From the converged μ* at G=14, γ=0.5, τ=2:
- Reconstruct the price surface P(u₁, u₂, u₃) at u₁=1
- Extract the level set at p₀ = P(1, -1, 1)
- Report: list of (u₂, u₃, T*) crossing points
- Also do the same for CARA (should be a straight line)
Side-by-side: straight (CARA) vs curved (CRRA)

### 8. Convergence figure data (Fig 5)
From the iteration history at G=14, γ=0.5, τ=2:
- Report: ||Φ-I||∞ at each iteration
- If Picard was used first then NK, report both stages
- Output as pgfplots coordinates for a log-y plot

---

## P2: MEDIUM PRIORITY

### 9. Extended γ-ladder
Fill in: γ = 0.1 (strict, not fallback), γ = 3, 5, 10, 20.
The γ→∞ limit should give 1-R²→0. Currently only γ≤2 strictly converged.
The full curve γ ∈ [0.1, 20] at τ=2 would be a powerful figure.

### 10. Trade volume (Fig 7)
At each converged μ*:
- E[|x_k|] = Σ w(u₁,u₂,u₃) · |x_k(μ*(u_k, P(u₁,u₂,u₃)), P)|
- Requires reconstructing P at each grid triple (expensive but doable)
- Report one number per (γ, τ) pair

### 11. Value of information (Fig 8)
- E[U_informed] = Σ_{v,i,j,l} 0.5·f_v(i)f_v(j)f_v(l) · U(W + x_k·(v-P))
- E[U_uninformed] = U(W) (since x=0 when μ=1/2 and p=1/2)
- V(τ) = E[U_informed] - E[U_uninformed]
- Report V(τ) for each converged (γ, τ)

### 12. Two-branch investigation
At G=14, seed from FR array. Does it stay at FR? (It should.)
Report the basin structure: how far from FR can you perturb before
falling to the PR branch?

---

## P3: LOW PRIORITY

### 13. K=4 extension
The side-projects branch has K=4 code. Can the posterior method handle K=4?
μ(u, p) is still 2D (same for each symmetric agent). The contour becomes
3D: d(u₁)+d(u₂)+d(u₃)+d(u₄)=0. Sweep u₂,u₃, root-find u₄. Harder but
the posterior method still applies. Try at G_u=10, G_p=10.

### 14. Heterogeneous γ at full REE
The REZN branch showed het γ=(5,3,1) gives 1-R²≈0.32 at no-learning.
What happens at full REE? Need separate μ_k(u, p) per agent type.
State: K × G_u × G_p. At K=3, G=14: 3×14×14 ≈ 600 unknowns. Feasible.

### 15. Gauss-Hermite quadrature
Replace equally-spaced u₂ sweep with Gauss-Hermite nodes weighted by
the signal density. Should improve accuracy at extreme signals without
increasing sweep count. Compare 1-R² at N_GH=20 vs N_uniform=50.

### 16. Publication table
Once all P0+P1 tasks are done, produce a single table:

| γ | τ | No-learning 1-R² | REE 1-R² | Survival | Slope | E[|x|] | V(τ) |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.5 | | | | | | |
| 0.25 | 1.0 | | | | | | |
| ... | ... | | | | | | |
| CARA | all | 0 | 0 | — | 1.0 | 0 | 0 |

---

## TECHNICAL NOTES

### The survival ratio puzzle
REE 1-R² (0.108) appears LARGER than no-learning 1-R² (0.062) at
γ=0.5, τ=2. This could be:
a) Different G used (no-learning was G=20, REE is G=14)
b) Different UMAX (no-learning uses [-4,4], posterior method may differ)
c) Real: learning from curved contours amplifies Jensen gap
d) The posterior method's 1-R² includes the contour-integration
   approximation error in both directions

Resolution: recompute no-learning at G=14 with IDENTICAL methodology
(same u-grid, same weight computation, same regression). If still
REE > no-learning, it's finding (c) and should be investigated.

### Strict convergence definition
max(|Φ(μ) - μ|) ≤ 1e-14 over ALL active cells.
Zero u-monotonicity violations. Zero p-monotonicity violations.
This is machine precision — the answer cannot improve further at
float64.

### File naming convention
Save converged μ arrays as:
  posterior_v3_strict_G{G}_gamma{γ}_tau{τ}.npz
Save summary JSON as:
  posterior_v3_strict_results.json (append to existing)
Push to results/full_ree/ on main branch.

---

## P0.5: URGENT DIAGNOSTIC — CARA FLOOR

The CARA baseline at γ=50 gives 1-R²=0.037 (status: no_strict).
This could be:
  (a) γ=50 is not large enough — the CRRA demand at γ=50 still has
      a tiny Jensen gap that gets amplified by the REE loop
  (b) Convergence artifact — the solver didn't reach the true fixed point
  (c) Method artifact — the posterior-function discretization introduces
      nonlinearity that even true CARA can't escape

### Tests to discriminate:

**Test 1: Higher γ sweep.**
Run γ = 50, 100, 200, 500 at G=14, τ=2. All with same solver settings.
- If 1-R² keeps falling toward zero: (a) confirmed. Report the decay rate.
- If 1-R² plateaus at ~0.037: (b) or (c).

**Test 2: Explicit CARA demands.**
Replace the CRRA demand formula x = W(R-1)/((1-p)+Rp) with the exact
CARA formula x = (logit(μ)-logit(p))/γ_CARA (with some fixed γ_CARA
for scaling). Everything else identical: same posterior method, same
PAVA, same grid. Run at G=14, τ=2.
- If 1-R² = 0 (machine precision): the method works perfectly for true
  CARA, and the 0.037 at γ=50 is genuine CRRA residual → finding (a).
  This would mean: the REE amplifies even the tiniest Jensen gap by
  1000×, making CARA an even sharper knife-edge than no-learning suggests.
- If 1-R² > 0: the posterior-function discretization introduces artifacts
  even for true CARA → finding (c). Must fix the method.

**Test 3: Convergence check at γ=50.**
Run γ=50 with more iterations (double), tighter damping, Anderson with
larger window. Check if 1-R² is still falling when stopped, or truly flat.
Report: 1-R² at iteration 100, 200, 500. Is it still decreasing?

### Why this matters

If finding (a): the paper's story is even stronger. The no-learning
knife-edge table shows γ=50 as "effectively CARA" (1-R²=0.0000). But
the REE amplifies that invisible gap to 0.037 — detectable and
economically significant. The knife-edge is SHARPER at REE than at
no-learning. Add a paragraph to the paper about this.

If finding (b) or (c): need to subtract the floor from all CRRA numbers.
The NET deficits from CHAT_MEMORY are still valid but the raw numbers
in the paper's tables need a baseline correction column.

### Expected result

Most likely (a). Reason: the no-learning 1-R² at γ=50 is 0.00003.
The survival ratio at γ=2 is 3.8×. If the ratio continues growing
(which the data suggests — it's monotone in γ), then at γ=50 the
ratio could be 100-1000×, giving 0.003-0.03 for the REE — consistent
with the observed 0.037. The high-γ CRRA demand is nearly linear but
not exactly linear, and the REE loop amplifies the residual nonlinearity.

Test 2 (explicit CARA) is the definitive discriminator.

---

## P0.6: TRIM p-GRID TO 95% FEASIBLE RANGE

### The problem
At G≥16, 1-2 cells at the extreme edges of the lens domain prevent
strict convergence. These cells have:
- 1-2 contour crossings (noisy A_v)
- Signal density weight ~1e-8 (invisible in 1-R²)
- Max residual ~0.1 while median is ~1e-13
They dominate ||F||∞ but contribute nothing to the answer.

### The fix
Trim each row's p-grid to the central 95% of the achievable range:

```python
# Current:
p_lo[i] = P(u_i, u_min, u_min)
p_hi[i] = P(u_i, u_max, u_max)

# New:
margin = 0.025
p_range = p_hi[i] - p_lo[i]
p_lo_trim[i] = p_lo[i] + margin * p_range
p_hi_trim[i] = p_hi[i] - margin * p_range
```

Or equivalently in the signal quantile: instead of using u_min and
u_max (the 0th and 100th percentile of the grid), use the 2.5th
and 97.5th percentile.

### Why it's safe
The trimmed 5% corresponds to other-agent signal pairs that are
jointly extreme: both agents at their 2.5th or 97.5th percentile.
Joint probability: (0.025)² ≈ 6e-4. The density weight on these
configurations is negligible. They don't affect 1-R², slope, or
any weighted metric.

### For off-grid prices
If a self-consistency check or contour tracing needs μ(u, p) at a
price outside the trimmed range, extrapolate:

```python
# Linear extrapolation in logit space from the two nearest interior points
if p < p_lo_trim[i]:
    logit_mu = logit(mu[i, 0]) + (logit(p) - logit(p_lo_trim[i])) * slope_lo[i]
elif p > p_hi_trim[i]:
    logit_mu = logit(mu[i, -1]) + (logit(p) - logit(p_hi_trim[i])) * slope_hi[i]
```

where slope_lo, slope_hi are the logit-space slopes at the boundary.

### Expected result
- G=16-24 should reach strict convergence (max<1e-14)
- 1-R² should be unchanged (same answer, different domain)
- Confirms that the G=14-15 strict results are the true answer

### Implementation
1. Change p_lo, p_hi computation to use 2.5/97.5% margins
2. Add linear-logit extrapolation for off-grid prices
3. Re-run G=16, 18, 20, 24 at γ=0.5, τ=2
4. Verify 1-R² matches the untrimmed values
5. If strict at G≥16: run the full knife-edge sweep at G=16

---

## P0.7: CUTOFF LADDER — HOMOTOPY IN COVERAGE PERCENTAGE

### The idea
Use the coverage percentage as a continuation parameter. Start with
a heavy cutoff (easy convergence), widen gradually, warm-start each
step from the previous converged solution.

### The ladder

```
coverage = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
```

At coverage c%: the p-grid for row i covers the central c% of
[p_lo(u_i), p_hi(u_i)]:

```python
margin = (100 - c) / 200  # e.g. c=90 → margin=0.05 on each side
p_lo_c[i] = p_lo[i] + margin * (p_hi[i] - p_lo[i])
p_hi_c[i] = p_hi[i] - margin * (p_hi[i] - p_lo[i])
```

### The algorithm

```
Step 0: Solve at coverage=90%, G=target (e.g. G=20).
        Initialize from no-learning μ = Λ(τu).
        Should converge easily — no boundary cells.
        
Step 1: Expand to coverage=91%.
        New cells on the boundary ring.
        Initialize new cells by extrapolation from interior.
        Warm-start interior from Step 0 solution.
        Run Picard + NK to strict convergence.
        
Step 2: Expand to 92%. Same procedure.
        ...
        
Step N: coverage=100% (full domain).
        If converges: done, full strict solution.
        If stalls at some c*: report "converged on central c%*
        of the price distribution (weight > 1 - 2·(1-c*/100)²)."
```

### Why it works
- At 90%: the outermost 10% of prices are removed. These are
  configurations where both other agents have extreme signals
  (top/bottom 5%). The domain is clean, contours have many
  crossings, convergence is easy.
  
- At each step: only ~2% of cells are new (the thin ring).
  They're initialized from the converged interior. The
  perturbation is small → NK converges in 1-2 steps.
  
- At 100%: if reachable, you have the full solution. If not,
  the stalling point c* is diagnostic — tells you exactly
  which price quantile is problematic.

### Reporting
At whatever coverage the ladder achieves strict convergence, report:

    "Strict convergence at G=20 on the central c% of the equilibrium
     price distribution (covering > X% of the density-weighted
     probability mass). The revelation deficit 1-R² = Y is invariant
     to the coverage level from 90% onward."

The density weight of the excluded region at coverage c:
    excluded_weight ≈ 2 · Φ(-(100-c)/200 · range_in_sigmas)²
    At c=95: ~6e-4. At c=99: ~2e-5. Negligible.

### Also use for the G-ladder
Combine with the G-ladder:

```
For G in [14, 15, 16, 18, 20, 24]:
    For c in [90, 91, ..., 100]:
        Solve (G, c). Warm-start from (G, c-1) or (G-1, 100).
        If strict: record and continue.
        If stalls: record c* for this G and move to next G.
```

This gives a 2D convergence map (G × coverage). The answer is
in the interior where it's constant across both dimensions.

### Quick test
Before the full ladder, just try:
1. G=20, coverage=90% → expect strict
2. G=20, coverage=95% → expect strict  
3. G=20, coverage=100% → does it reach strict?

If step 2 works strict, skip the fine ladder.

---

## P1.5: FIGURE QUALITY — ALL FIGURES MUST BE FIXED BEFORE SUBMISSION

### Style requirements
- ALL figures must be pgfplots (standalone .tex → .pdf)
- BC20 color scheme: green(0.7,0.11,0.11), red(0,0.20,0.42), blue(0.11,0.35,0.02)
  Wait — CORRECTION: red=(0.7,0.11,0.11), blue=(0,0.20,0.42), green=(0.11,0.35,0.02)
- Line order: green solid, red dashed, blue dotted, black dashdotted
- very thick curves, ultra thick CARA, smooth
- 8cm × 8cm, legend draw=none footnotesize
- ymin=-0.001 for 1-R² plots
- Paper gammas: γ = 0.25, 1.0, 4.0
- No matplotlib figures in the final paper

### Figure-by-figure status and fix needed

**1. fig_knife_edge.tex/pdf — WRONG GAMMAS**
Current: γ = 0.2, 1.0, 5.0 (old values)
Fix: Recompute no-learning 1-R² at γ = 0.25, 1.0, 4.0
Data: 30 log-spaced τ from 0.1 to 10, G=20
Output: pgfplots coordinates for three curves + CARA at zero

**2. fig3_contour.tex/pdf — DONE ✓**
Updated with G=14 REE contour data from solver. Pgfplots, real data.

**3. fig_ree_vs_nolearning.pdf — MATPLOTLIB, NEEDS PGFPLOTS CONVERSION**
Current: solver's matplotlib PNG/PDF. Correct data but wrong style.
Fix: Extract the raw (T*, p) data from the converged μ* at G=14.
     For each triple (u_i, u_j, u_l) on the G=14 grid:
       - Compute T* = τ(u_i + u_j + u_l)
       - Solve market clearing F(p) = Σ d(u_k; p) = 0 for p using converged μ*
       - Also compute no-learning price (using μ = Λ(τu))
       - Also compute FR price p = Λ(T*/3)
     Bin by T* (40 bins from -12 to 12), compute mean p in each bin.
     Output as three pgfplots \addplot coordinate lists.
     Clip to T* ∈ [-10, 10] to avoid edge wobbles.

**4. fig4_posteriors.pdf — WRONG DATA, REMOVE OR REPLACE**
Current: yellow-bg table showing G=5 numbers where μ₂ = 0.8808 (= CARA!).
The posteriors TABLE in the main text has the correct numbers (μ₂ = 0.667).
Options:
  a) REMOVE this figure entirely (the table is sufficient)
  b) Replace with a plot of μ_k vs T* showing CARA (identity) vs CRRA (spread)
If (b): use converged μ* at G=14. For a range of T*, plot the three
posteriors μ₁, μ₂, μ₃ and the price p. Show how they fan out under CRRA
vs collapse to a single line under CARA.

**5. fig5_convergence.pdf — WRONG DATA**
Current: old G=20 price-grid convergence (Picard vs Anderson).
Fix: Generate from the posterior method iteration history at G=14, γ=0.5, τ=2.
     Plot ||F||∞ vs iteration number. Show Picard phase and NK polish phase.
     If iteration history not saved, re-run and save it.
Output: pgfplots with log-y axis.

**6. fig_knife_edge_K.pdf — CHECK GAMMAS**
May use old gamma values. Verify and regenerate if needed.
Should show 1-R² vs K for γ = 0.25, 1, 4 at fixed τ=2.

**7. fig_knife_edge_lognormal.pdf — CHECK GAMMAS**
Same issue. Verify gamma values match 0.25, 1, 4.

**8. fig7_volume.pdf — PLACEHOLDER OK ✓**
Gray background, invented data, correct style. Will be replaced
when trade volume computation is done (P2 task 10).

**9. fig8_value_info.pdf — PLACEHOLDER OK ✓**
Gray background, correct gammas (0.25, 1, 4). Will be replaced
when V(τ) computation is done (P2 task 11).

**10. fig9_GS.pdf — PLACEHOLDER OK ✓**
Gray background, correct style. Will be replaced when V(τ)-c
computation is done.

**11. fig6_mechanisms.pdf — YELLOW-BG TABLE**
This is a table rendered as a figure. The data is in the paper's
Table (mechanisms). Consider:
  a) Remove the figure, keep only the table
  b) Replace with a bar chart of 1-R² by mechanism
If (b): simple grouped bar chart, one bar per configuration,
colored by mechanism type. pgfplots ybar.

### Priority order for fixes
1. fig_knife_edge (wrong gammas — this is Fig 1, most important)
2. fig_ree_vs_nolearning (matplotlib → pgfplots)
3. fig5_convergence (wrong data)
4. fig4_posteriors (wrong data — remove or replace)
5. fig_knife_edge_K, fig_knife_edge_lognormal (check gammas)
6. fig6_mechanisms (cosmetic)

### Data extraction recipe for fig_ree_vs_nolearning

```python
# After loading converged mu_star[G_u, G_p] at G=14:
import numpy as np
from scipy.optimize import brentq

# For each grid triple (i, j, l):
Tstar_list, p_ree_list, p_nl_list, p_fr_list = [], [], [], []
for i in range(G):
    for j in range(G):
        for l in range(G):
            u1, u2, u3 = u_grid[i], u_grid[j], u_grid[l]
            Tstar = tau * (u1 + u2 + u3)
            
            # FR price
            p_fr = expit(Tstar / K)
            
            # No-learning price
            def nl_excess(p):
                return sum(crra_demand(expit(tau*u), p) for u in [u1,u2,u3])
            p_nl = brentq(nl_excess, 0.001, 0.999)
            
            # REE price (using converged mu*)
            def ree_excess(p):
                return sum(crra_demand(interp_mu(u, p), p) for u in [u1,u2,u3])
            p_ree = brentq(ree_excess, 0.001, 0.999)
            
            Tstar_list.append(Tstar)
            p_fr_list.append(p_fr)
            p_nl_list.append(p_nl)
            p_ree_list.append(p_ree)

# Bin by T* and output pgfplots coordinates
```
