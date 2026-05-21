# COMPLETE SESSION SUMMARY — Noise Traders Are Not a Primitive
# For Claude Code: read this entire file before doing anything.

## PROJECT
Author: Matthijs Breugem, Nyenrode Business University. Target: Econometrica.
Core claim: CRRA preferences produce partial revelation (PR) of information through prices WITHOUT noise traders. CARA is a knife-edge: only exponential utility gives full revelation (FR). The paper needs a converged numerical REE solution to confirm PR survives rational expectations learning.

## THE MODEL
- Binary asset v ∈ {0,1}, prior P(v=1) = 1/2
- K = 3 agent groups
- Signal: s_k = v + ε_k, ε_k ~ N(0, 1/τ_k)
- Centered signal: u_k = s_k − 1/2
  - v=1: u_k ~ N(+1/2, 1/τ)
  - v=0: u_k ~ N(−1/2, 1/τ)
- Signal densities: f_v(u) = sqrt(τ/(2π)) exp(−τ/2 (u − v + 1/2)²)
- Private likelihood ratio: f_1/f_0 = exp(τ u)
- Private posterior: μ_prior = Λ(τu) where Λ is the logistic function
- CRRA utility: U(W) = W^(1−γ)/(1−γ)
- CRRA demand: x_k = W(R−1)/((1−p)+Rp) where R = exp((logit(μ)−logit(p))/γ)
- CARA demand: x_k = (logit(μ)−logit(p))/γ (linear in log-odds)
- Log demand (γ=1): x_k = W(μ−p)/(p(1−p))
- Zero net supply, NO noise traders

## THE CONTOUR METHOD — FINAL AGREED APPROACH

### The unknown
P[i,j,l] for i,j,l = 1,...,G. One price array, G³ scalars.

### System of equations for ONE signal realization (i,j,l)

| # | Name | Equation | Unknown | Count |
|---|---|---|---|---|
| D1 | Root-find (pass A, all agents) | P(..., u_jc, u_lc, ...) = p | u_lc | 6G |
| D2 | Root-find (pass B, all agents) | P(..., u_jc, u_lc, ...) = p | u_jc | 6G |
| D3-D5 | Contour integrals | A_v^(k) = ½[Σ_jc f_v f_v + Σ_lc f_v f_v] | A_v^(k) | 6 |
| D6 | Bayes' rule | μ_k = f1_own·A1/(f0_own·A0 + f1_own·A1) | μ_k | 3 |
| E1 | Market clearing | Σ x_k(μ_k, p) = 0 | P[i,j,l] | 1 |

Superscript c = conjectured signal. (i,j,l) = actual realization.

### Full system: G³ equations, G³ unknowns. Total root-finds: 6G⁴.

### How each agent computes her contour

For agent 1 at actual signal u_i, observing price p = P[i,j,l]:
- Her slice is P[i, :, :] — a G×G matrix
- Pass A: sweep u_jc over G grid points, for each root-find u_lc (off grid) such that P[i, u_jc, u_lc] = p
- Pass B: sweep u_lc over G grid points, for each root-find u_jc (off grid) such that P[i, u_jc, u_lc] = p
- Average the two passes: A_v = ½(A_v^A + A_v^B)
- Each crossing contributes f_v(u_jc) · f_v(u_lc) to A_v — the ex ante signal likelihood

For agent 2 at signal u_j: slice P[:, j, :], sweep/root-find on axes (u_ic, u_lc)
For agent 3 at signal u_l: slice P[:, :, l], sweep/root-find on axes (u_ic, u_jc)

Each agent sees her OWN slice of P. No transposition trick.

### The root-finding
For the no-learning initialization, P is defined analytically: solve market clearing for any (u1,u2,u3). No grid interpolation needed, no boundary issues. The market clearing condition IS the sub-fixed-point.

For subsequent iterations, P is stored on a grid. Use linear interpolation along one axis to find the off-grid crossing. Linear extrapolation beyond grid edges.

### Two solver strategies

**Picard iteration (intuitive, mimics learning):**
1. Initialize P⁰ from no-learning prices
2. Given P^(n), compute contour integrals → posteriors → market clearing → P^(n+1)
3. Damped update: P^(n+1) = α·P_new + (1−α)·P^(n)
4. Symmetrize: average over all 6 permutations of (i,j,l)
5. Repeat until ||P^(n+1) − P^(n)||∞ < tol

**Newton (Anderson acceleration):**
Stack P − Φ(P) = 0 and solve simultaneously. Anderson builds a quasi-Newton Jacobian from past iterates. One Φ evaluation per step. Converges much faster than Picard but stalls at the interpolation floor.

In principle, Newton can also include D1-D6 as explicit unknowns (the full 12G⁴ + 10G³ system) instead of substituting them out.

## WHAT WE BUILT AND TESTED

### Python solvers (all in /home/claude/)
- solve_v3.py: Binning method + Anderson. Fast but has noise floor ~0.012
- solve_v5.py: Same, cleaner. Converges but CARA=CRRA at same floor
- solve_v6.py: Projection method (parametric). Too restrictive, converges to FR
- solve_v7.py: Contour with cubic interpolation. Artifacts grow over iterations
- solve_v8.py: γ-sweep diagnostic. Cleanest test of no-learning PR
- solve_v10.py: Logit-space binning. Same floor
- solve_exact.py: Exact discrete prices. No smoothing → both converge identically
- solve_kernel.py: Gaussian kernel smoothing. h-sweep shows signal but floor persists
- solve_final.py: Binned contour + Anderson (cleanest converging code)
- solve_net.py: NET = converged CRRA − converged CARA baseline
- solve_2pass.py: The 2-pass contour method (your design)
- solve_2pass_jac.py: Same with natural Jacobian from interpolation slope
- anderson.py: Anderson acceleration at (1,−1,1), G=5
- anderson2.py: Same with α coefficients displayed
- anderson_cara.py: Explicit CARA with analytical benchmark
- one_point.py, one_point2.py, one_point3.py: Detailed walkthrough at (1,−1,1)
- picard10.py: 30 Picard iterations at (1,−1,1)
- value_info.py: Value of private information decomposition

### React artifacts (in /mnt/user-data/outputs/)
- contour_viz.jsx: Interactive CARA vs CRRA contour with γ slider
- contour_5x5x5.jsx: 5×5×5 step-by-step walkthrough
- contour_7x7x7.jsx: 7×7×7 with 50 iterations, all 3 agents, proper slicing, Anderson+Picard comparison
- cara_vs_crra.jsx: Side-by-side CARA/CRRA heatmaps at (1,−1,1) with iteration slider

### Key instruction document
- INSTRUCTIONS.md: Full Julia implementation guide (in /mnt/user-data/outputs/)
- BB_contour_method_numerical.md: Detailed equations document

## NUMERICAL RESULTS

### No-learning smooth transition table (EXACT, no artifacts, G=20)
| γ    | τ=0.5  | τ=1.0  | τ=2.0  |
|------|--------|--------|--------|
| 0.1  | 0.146  | 0.145  | 0.137  |
| 0.3  | 0.044  | 0.070  | 0.090  |
| 0.5  | 0.016  | 0.038  | 0.062  |
| 1.0  | 0.004  | 0.013  | 0.029  |
| 3.0  | 0.000  | 0.002  | 0.006  |
| 10.0 | 0.000  | 0.000  | 0.001  |
| 100  | 0.000  | 0.000  | 0.000  |
Matches the paper's Table 1.

### Converged fixed point at (u₁,u₂,u₃) = (1,−1,1), G=5, τ=2

**CARA (explicit, analytical: p = Λ(T*/3) = Λ(2/3) = 0.6608):**
Converged price = 0.8808 = Λ(T*) = Λ(2). All posteriors equal 0.8808. Full revelation.

**CRRA (γ=0.5):**
Converged price = 0.9077. μ₁ = 0.9185, μ₂ = 0.8889, μ₃ = 0.9185.
Agents disagree. Partial revelation.

| | Prior | CARA FP | CRRA FP |
|---|---|---|---|
| μ₁ (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| μ₂ (u=−1) | 0.1192 | 0.8808 | 0.8889 |
| μ₃ (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| price | 0.648 | 0.8808 | 0.9077 |

### Anderson convergence at this point
- Gets to ||G||∞ ~ 0.002 in ~6 iterations (vs 30+ for Picard)
- Stalls at ~0.002 floor — this is the G=5 interpolation limit
- Values stable from iteration ~10 onward

### Picard convergence at this point (α=0.3, 30 iterations)
- Price: 0.648 → 0.909 (still moving by 0.0004/step)
- Posteriors barely change across iterations (μ₂ shifts by 0.001 total)
- Price overshoots at iteration 16 (Δp flips sign), then oscillates down

## KEY NUMERICAL LESSONS LEARNED

1. **Binning creates a noise floor (~0.012 at G=15-20)** that is identical for CARA and CRRA. Cannot distinguish PR from artifacts.

2. **Interpolation (cubic/linear) creates fake curvature** that grows over iterations. Even CARA develops artificial PR.

3. **Exact discrete grouping** makes every price unique → singleton groups → trivial FR for both.

4. **Kernel smoothing** shows signal but same floor issues. h-sweep: as h→0, CARA→0 but CRRA signal also vanishes at finite G.

5. **The 2-pass contour with root-finding** is the correct method. No Jacobian needed inside — the Jacobian only matters at the outer Newton level.

6. **The NET approach** (CRRA minus CARA baseline) isolates genuine PR. At G=15: NET positive but tiny (~0.001 at γ=0.5).

7. **For publication quality:** Need G≥100 in Julia. The PR signal scales with grid resolution because finer grids resolve the contour curvature that distinguishes CRRA from CARA.

## WHAT NEEDS TO BE BUILT NEXT

1. **Julia implementation** of the 2-pass contour method with:
   - G = 100-200
   - Anderson acceleration (or full Newton with Broyden Jacobian)
   - All three agents computing their own slices (no transposition)
   - Root-finding via market clearing (no grid interpolation for initialization)
   - Linear interpolation with extrapolation for subsequent iterations
   
2. **The full Newton system** as an alternative: stack all 12G⁴ + 10G³ equations and unknowns, solve F(x)=0 in one shot. No sub-iteration, no damping.

3. **Convergence verification:** Run CARA baseline at same G to measure noise floor, subtract from CRRA.

4. **Publication outputs:**
   - Smooth transition table (no-learning, already done)
   - Converged REE survival ratios by (γ, τ)
   - Level-set figure showing curved vs straight contours
   - Contour plot colored by T* variation

## THE CORE INSIGHT (for the contour plot)

Under CARA: the price is logit(p) = (1/K) Σ τu_k. Level sets are STRAIGHT lines: w₂τu₂ + w₃τu₃ = const. Every point on the contour has the same T*. The agent learns T* perfectly from the price. → Full revelation.

Under CRRA: the price aggregates in probability space, not log-odds. Level sets of Σ Λ(τu_k) = const are CURVED (because Λ is nonlinear). Different points on the contour have different T*. The agent cannot extract T* from the price. → Partial revelation.

The curvature is the Jensen gap: Σ Λ(τu_k)/K ≠ Λ(Σ τu_k/K). It vanishes as γ→∞ (CARA limit) and grows as γ→0.

## FILE LOCATIONS
- Project summary: /mnt/project/project_summary.txt (read-only)
- All outputs: /mnt/user-data/outputs/
- Working scripts: /home/claude/
- Julia instructions: /mnt/user-data/outputs/INSTRUCTIONS.md

## PARAMETER DEFAULTS
- τ = 2.0
- γ = 0.5 (CRRA main case), 100.0 (CARA baseline)
- W = 1.0 (equal wealth)
- K = 3 agents
- G = 5 (debug), 15-20 (quick test), 100-200 (publication)
- u_grid = [−4, +4] for G≥15, [−2, +2] for G=5
- Anderson window m = 6-8
- Picard damping α = 0.15-0.3
