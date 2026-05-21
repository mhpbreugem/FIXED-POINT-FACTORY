# POSTERIOR-FUNCTION METHOD v2
# Fixes the domain problem from v1
# Key change: adaptive p-range per signal, active-cell detection

---

## 0. THE PROBLEM WITH v1

v1 stores μ(u, p) on a rectangular grid (u × p). But not every (u, p)
is realizable. If u₁ = +3 and p = 0.05, no combination of other
agents' signals produces that price. The contour is empty, the Bayes
update saturates, and the fixed point oscillates.

The fix: for each u_i, only define μ on the achievable price range.
The domain of μ is not a rectangle — it's a lens-shaped region in
(u, p) space.

---

## 1. THE ACHIEVABLE PRICE RANGE

For a given own signal u₁, the equilibrium price depends on the
other agents' signals (u₂, u₃). The extreme prices are:

    p_lo(u₁) = P(u₁, u_min, u_min)    [other agents maximally bearish]
    p_hi(u₁) = P(u₁, u_max, u_max)    [other agents maximally bullish]

where u_min and u_max are the edges of the signal grid.

### At initialization (no-learning)

μ⁰(u, p) = Λ(τu). The price doesn't enter the posterior.
Market clearing at (u₁, u₂, u₃):

    Σ x_k(Λ(τu_k), p) = 0    → solve for p

This is a standard 1D root-find. For extreme configs:

    p_lo(u_i) = solve Σ x_k(Λ(τu_k), p) = 0 with (u₁, u_min, u_min)
    p_hi(u_i) = solve Σ x_k(Λ(τu_k), p) = 0 with (u₁, u_max, u_max)

### At subsequent iterations

μ^(n)(u, p) depends on p. Market clearing becomes:

    F(p) = x(μ^(n)(u₁, p), p) + x(μ^(n)(u₂, p), p) + x(μ^(n)(u₃, p), p) = 0

Still a 1D root-find in p, but now requires interpolating μ^(n)
in the p-direction at each trial p. F is monotone in p (higher price
→ lower total demand) so the root-find is clean.

    p_lo^(n)(u_i) = solve F(p) = 0 with (u_i, u_min, u_min)
    p_hi^(n)(u_i) = solve F(p) = 0 with (u_i, u_max, u_max)

Cost: 2 × G_u root-finds per iteration. Negligible.

---

## 2. THE ADAPTIVE p-GRID

For each u_i, define a local p-grid:

    p_j(u_i) ∈ [p_lo(u_i), p_hi(u_i)],   j = 1, ..., G_p

Option A: G_p equally spaced in p within the range.
Option B: G_p equally spaced in logit(p) within the range.
Option C: G_p points concentrated near the center of the range
(where most realizations fall) via the distribution of prices
at this u_i.

Recommendation: Option B (logit spacing). The posterior is smoother
in logit(p) and the sufficient statistic lives in logit space.

The p-grid is u-dependent. At extreme u_i (e.g. u = ±3), the price
range is narrow (most of the information is in the own signal). At
moderate u_i (near 0), the range is wide (the price carries
substantial information). The grid adapts automatically.

### Fixed vs adaptive across iterations

Option 1 (simplest): Compute the price range at initialization.
Add 5% margin on each side. Fix the p-grid for all iterations.
Rationale: the price range shouldn't change dramatically as μ updates.

Option 2 (safer): Recompute the price range every N iterations
(say N=5). Remap μ onto the new grid via interpolation.

Option 3 (most robust): Recompute every iteration. More expensive
but guarantees no out-of-domain issues.

Start with Option 1. Fall back to Option 2 if convergence stalls.

---

## 3. STORAGE LAYOUT

Store μ as a 2D array with RAGGED p-dimension:

    μ[i, j]  for  i = 1,...,G_u  and  j = 1,...,G_p

But each row i has its own p-range [p_lo(u_i), p_hi(u_i)].

Equivalently, store three arrays:
    μ_vals[G_u, G_p]     — the posterior values
    p_lo[G_u]            — lower price bound per signal
    p_hi[G_u]            — upper price bound per signal

The p-grid for row i is: p_j = p_lo[i] + j/(G_p-1) * (p_hi[i] - p_lo[i]).
Or in logit: logit(p_j) = logit(p_lo[i]) + j/(G_p-1) * (logit(p_hi[i]) - logit(p_lo[i])).

---

## 4. INTERPOLATION

### 4.1 Interpolating μ in the u-direction at fixed p

Given p_j (a grid point for row i), we need μ(u₂, p_j) where u₂
may be off the u-grid. But p_j is a grid point for row i, not
necessarily for row i' corresponding to u₂.

Solution: for each target (u₂, p), find the two rows i', i'+1
that bracket u₂, then in each row find the p-grid points that
bracket p, interpolate μ in p, then interpolate the two resulting
values in u. This is bilinear interpolation on the irregular grid.

BUT: this defeats the purpose. The whole point was to avoid 2D
interpolation.

### 4.2 Better: interpolate in u only, at a COMMON p value

Here's the key insight. When we sweep u₂ at fixed price p₀,
we need μ(u₂, p₀) for many u₂ values. The price p₀ is the SAME
for all of them (it's the price the agent observes).

So: extract the column μ(·, p₀) — the posterior as a function of u
at this specific price. This requires interpolating each row's
μ[i, ·] to the common price p₀.

Step 1: For each row i, interpolate μ[i, ·] (over that row's
p-grid) to evaluate μ[i, p₀]. This gives a 1D array μ_at_p0[i].

Step 2: μ_at_p0[i] is now a function of u_i on the u-grid.
Interpolate in u to get μ(u₂, p₀) at any off-grid u₂.

Total: G_u 1D p-interpolations (one per row) + a 1D u-interpolation
per sweep point. Clean and fast.

HOWEVER: this only works if p₀ is within the achievable range for
every row. If p₀ = 0.5 but p_hi(u_min) = 0.3, then μ(u_min, 0.5)
is undefined — u_min can never see price 0.5.

Fix: for rows where p₀ is out of range, extrapolate linearly or
set μ to the boundary value. Since these rows correspond to
extreme signals with vanishing density f_v(u_min), they contribute
negligibly to the contour integral.

---

## 5. THE ITERATION

### 5.0 Initialization

    For each u_i on the signal grid:
        p_lo[i], p_hi[i] = price range from no-learning market clearing
        For each p_j in [p_lo[i], p_hi[i]]:
            μ[i, j] = Λ(τ · u_i)    # no-learning: ignore price

### 5.1 One step of Φ_μ

For each grid point (u_i, p_j):

    1. EXTRACT μ-COLUMN AT THIS PRICE
       For each row i' = 1,...,G_u:
           If p_j ∈ [p_lo[i'], p_hi[i']]:
               μ_col[i'] = interpolate μ[i', ·] at p_j
           Else:
               μ_col[i'] = boundary value (Λ(τu_{i'}) or extrapolate)

       Now μ_col[·] is the posterior as a function of u at fixed price p_j.

    2. OWN-AGENT DEMAND
       R₁ = exp((logit(μ[i, j]) - logit(p_j)) / γ)
       x₁ = W(R₁ - 1) / ((1-p_j) + R₁·p_j)
       D₁ = -x₁

    3. SWEEP u₂ AND ROOT-FIND u₃
       Choose N_sweep points for u₂ (Gauss-Hermite or uniform).
       
       For each u₂ sweep point:
           μ₂ = interpolate μ_col at u₂  [1D interpolation]
           x₂ = demand(μ₂, p_j)
           target = D₁ - x₂
           
           Root-find u₃ such that demand(μ_col(u₃), p_j) = target
           
           The function demand(μ_col(u₃), p_j) is:
               μ₃ = interpolate μ_col at u₃
               R₃ = exp((logit(μ₃) - logit(p_j)) / γ)
               x₃ = W(R₃ - 1) / ((1-p_j) + R₃·p_j)
           
           This is monotone in u₃ (because μ_col is monotone in u
           and demand is monotone in μ). So brentq works cleanly.
           
           If root-find succeeds (u₃ ∈ [u_min, u_max]):
               A_v += f_v(u₂) · f_v(u₃)
               n_crossings += 1
           Else:
               Skip this sweep point (contour exits domain)
       
       If n_crossings < 2:
           Mark (u_i, p_j) as degenerate. Keep μ[i,j] unchanged.
           Continue to next grid point.

    4. BAYES UPDATE
       μ_new[i, j] = f₁(u_i)·A₁ / (f₀(u_i)·A₀ + f₁(u_i)·A₁)

### 5.2 Damping / acceleration

After computing μ_new for all active cells:

    μ^(n+1) = α · μ_new + (1-α) · μ^(n)      [Picard, α=0.1-0.3]

Or Anderson acceleration on the flattened vector of active cells.

Convergence: ||μ^(n+1) - μ^(n)||∞ < tol over active cells only.

---

## 6. GAUSS-HERMITE QUADRATURE FOR THE SWEEP

The contour integral weights each crossing by f_v(u₂) · f_v(u₃).
Under state v, u₂ ~ N(v-½, 1/τ). So the density-weighted integral
is naturally a Gaussian quadrature problem.

Use Gauss-Hermite nodes and weights for the u₂ sweep:

    nodes: u₂^(n) = (v-½) + z_n / √τ     for Gauss-Hermite nodes z_n
    weights: w_n = Gauss-Hermite weights

Since we don't know v, average over v=0 and v=1:

    A_v = Σ_n w_n · f_v(u₃^(n))    [where u₃^(n) is root-found]

with appropriate transformation of the Gauss-Hermite weights
to account for the density f_v(u₂).

In practice, just use N_sweep = 30-50 equally-spaced u₂ points
on [-4, 4] and multiply by f_v(u₂) explicitly. The Gaussian
density kills anything beyond ±3/√τ anyway. More sweep points
in the center, fewer in the tails, is slightly more efficient
but not necessary.

---

## 7. SELF-CONSISTENCY CHECK AND 1-R² MEASUREMENT

At convergence, verify self-consistency and measure revelation:

### 7.1 Reconstruct the price function

For a sample of realizations (u₁, u₂, u₃) — either on-grid or random:

    Solve F(p) = x(μ*(u₁,p),p) + x(μ*(u₂,p),p) + x(μ*(u₃,p),p) = 0

This gives P(u₁, u₂, u₃) = p. The root-find interpolates μ* in
the p-direction at each trial p.

### 7.2 Check posterior consistency

At the reconstructed price p:
    μ₁ = μ*(u₁, p)
    μ₂ = μ*(u₂, p)
    μ₃ = μ*(u₃, p)

Verify that demands at these posteriors sum to zero (they should by
construction). Report the posterior disagreement: max|μ_k - μ_l|.

Under CARA: all three posteriors should be equal = p (FR).
Under CRRA: posteriors should differ (PR).

### 7.3 Compute 1-R²

Compute T* = τ(u₁ + u₂ + u₃) and logit(p) for each realization.
Weight by w = ½(Πf₁ + Πf₀). Regress logit(p) on T*.
Report 1-R².

---

## 8. CARA BENCHMARK

Run the same method with CARA demands:
    x_k = (logit(μ(u_k, p)) - logit(p)) / α

Should converge to μ*(u, p) = p for all (u, p). Verify:
    - All posteriors equal the price at every realization
    - 1-R² = 0

This is the ground truth test. If CARA doesn't converge to FR,
the method has a bug.

---

## 9. WHY THE JACOBIAN CANCELS

The contour is {(u₂, u₃) : x₁ + x₂ + x₃ = 0} at fixed (u₁, p).

The proper line integral is:

    A_v = ∫ f_v(u₂) f_v(u₃(u₂)) · |J|^{-1} du₂

where J = ∂(x₂+x₃)/∂u₃ = (∂x₃/∂μ₃)(∂μ₃/∂u₃).

In the posterior ratio:

    μ_new = f₁(u₁) A₁ / (f₀(u₁) A₀ + f₁(u₁) A₁)

The Jacobian |J| depends on the demand derivatives and on
∂μ/∂u, but NOT on v. The same (u₂, u₃(u₂)) crossing appears
in both A₁ and A₀ with the same Jacobian. In the discrete
sum with equally-spaced sweep points, each term has the same
implicit Δu₂ weight. The Jacobian rescales A₁ and A₀ by the
same factor at each crossing, so it cancels in the ratio.

Strictly: the Jacobian cancellation is exact in the ratio
A₁/A₀ only if we include it consistently. When we omit it
from both, we get:

    A₁/A₀ ≈ [Σ_n f₁(u₂ⁿ)f₁(u₃ⁿ)] / [Σ_n f₀(u₂ⁿ)f₀(u₃ⁿ)]

This is the correct density ratio IF the sweep points are
equally spaced in u₂ (so the Jacobian from the change of
variables is the same for each term). Since the Jacobian
|du₃/du₂| varies along the contour, this introduces a small
approximation error — but it's the SAME error in both A₁
and A₀ and cancels in the ratio. The error vanishes as the
number of sweep points grows.

---

## 10. EXPECTED BEHAVIOR

### At initialization
μ⁰(u, p) = Λ(τu). All posteriors ignore the price.
Market-clearing prices are the no-learning prices.
1-R² matches the no-learning table exactly.

### After first iteration
μ¹(u, p) incorporates information from the contour.
Agents who observe a price inconsistent with their signal
adjust their posterior toward the price. 1-R² decreases
(more revelation from learning).

### At convergence
μ*(u, p) is the REE posterior. Under CARA: μ*(u,p) = p
(full revelation, all info in price). Under CRRA: μ*(u,p)
depends on both u and p, with the dependence on u reflecting
the residual private information not captured by the price.

### Convergence rate
Picard with α = 0.1-0.2 should converge in 20-50 iterations.
Anderson with m = 5-8 should converge in 10-20 iterations.
The small state space (G_u × G_p vs G³) makes each iteration
much faster.

---

## 11. IMPLEMENTATION CHECKLIST

1. [ ] Define signal grid u ∈ [-4, 4], G_u = 50
2. [ ] Compute no-learning price ranges p_lo(u_i), p_hi(u_i)
3. [ ] Define per-row p-grid in logit space, G_p = 50
4. [ ] Initialize μ = Λ(τu)
5. [ ] Implement column extraction (§4.2): μ_col(u) at fixed p
6. [ ] Implement sweep + root-find (§5.1 step 3)
7. [ ] Implement Bayes update (§5.1 step 4)
8. [ ] Implement degenerate-cell detection (skip if < 2 crossings)
9. [ ] Implement damped Picard iteration
10. [ ] Run CARA benchmark — verify μ* = p
11. [ ] Run CRRA — measure 1-R² at convergence
12. [ ] Compare with price-grid result at same parameters
13. [ ] Anderson acceleration
14. [ ] Scale to G_u = G_p = 100

---

# ADDENDUM: v3 OPTIMIZATIONS

## A. VECTORIZED CONTOUR TRACING (eliminates all root-finding)

### The key insight

At fixed price p₀, all agents use the same demand function:

    d(u) = x(μ_col(u, p₀), p₀)

This is monotone in u. Precompute it once on the u-grid — just G_u
evaluations. Store the array.

### The contour becomes interpolation search

For agent 1 at signal u_i:
    D₁ = -d[i]
    targets[:] = D₁ - d[:]      # vector of targets for agent 3
    u₃*[:] = np.interp(targets, d, u_grid)   # vectorized inversion

No brentq. No iterative μ evaluation. Just np.interp on a monotone
array. O(G_u log G_u) for the binary search, fully vectorized.

### The integral becomes a dot product

    valid = (u₃* >= u_min) & (u₃* <= u_max)
    A_v = np.dot(f_v_sweep[valid], f_v(u₃*[valid]))

### The loop structure

```python
for j in range(G_p):                        # outer: price levels
    # Step A: extract column (shared across all u_i at this price)
    mu_col = np.array([interp_p(mu[i,:], p_grids[i], p_j) for i in range(G_u)])
    
    # Step B: precompute demand array (shared)
    R = np.exp((logit(mu_col) - logit(p_j)) / gamma)
    d = W * (R - 1) / ((1 - p_j) + R * p_j)
    
    for i in active_rows_at_pj:              # inner: own-signal
        # Step C: vectorized contour tracing
        targets = -d[i] - d
        u3_star = np.interp(targets, d, u_grid)  # if d increasing
        # OR: np.interp(targets, d[::-1], u_grid[::-1]) if d decreasing
        
        valid = (u3_star >= u_grid[0]) & (u3_star <= u_grid[-1])
        
        # Step D+E: density integrals
        A1 = np.sum(f1[valid] * f1_at(u3_star[valid]))
        A0 = np.sum(f0[valid] * f0_at(u3_star[valid]))
        
        # Step F: Bayes
        mu_new[i, j] = f1_own[i] * A1 / (f0_own[i] * A0 + f1_own[i] * A1)
```

### Cost comparison per iteration

| Method                | Root-finds        | Cost at G=100     |
|-----------------------|-------------------|-------------------|
| Price-grid (original) | 6G⁴ brentq        | 600,000,000       |
| Posterior v2           | G_u·G_p·N brentq  | 1,000,000         |
| Posterior v3 (this)    | 0 (vectorized)    | ~50,000 effective |

Speedup: 12,000× over price-grid, 20× over v2.

---

## B. MONOTONICITY PROJECTION (eliminates edge oscillation)

### Economic monotonicity constraints

At any REE, the posterior μ(u, p) must satisfy:

    ∂μ/∂u > 0    (higher own signal → more bullish)
    ∂μ/∂p > 0    (higher price → market more bullish → agent more bullish)

These are not assumptions — they are consequences of Bayesian updating
with positively-correlated signals. If the numerical μ violates either
monotonicity, it has left the equilibrium set.

### What causes violations

At the edges of the (u, p) domain:
- The contour barely enters the signal grid → few crossings → noisy A_v
- The Bayes update produces a μ that's slightly non-monotone
- Next iteration: the non-monotone μ creates a kinked demand function
- The kinked demand traces through wrong signal space → wrong posterior
- The kink amplifies → oscillation

This is exactly what causes the quirky red points at the edges of the
two-branches plot: the solver produces transient non-monotone iterates
that bounce between the PR and FR basins.

### The fix: project onto the monotone cone

After each Bayes update, project μ_new onto the set of bivariate
monotone functions using the Pool-Adjacent-Violators Algorithm (PAVA):

```python
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression()

# After computing mu_new for all active cells:

# Pass 1: enforce monotonicity in u at each price level
for j in range(G_p):
    active = active_mask[:, j]
    if np.sum(active) >= 2:
        mu_new[active, j] = iso.fit_transform(
            u_grid[active], mu_new[active, j])

# Pass 2: enforce monotonicity in p at each signal level
for i in range(G_u):
    active = active_mask[i, :]
    if np.sum(active) >= 2:
        mu_new[i, active] = iso.fit_transform(
            p_grid_logit[i, active], mu_new[i, active])
```

PAVA is O(n), exact, and projects onto the nearest monotone function
in L². Standard tool in shape-constrained estimation.

### Why it doesn't bias the answer

If the true fixed point μ* is monotone (which it must be on economic
grounds), then proj(μ*) = μ*. The projection is the identity at the
solution. It only constrains the PATH of iteration — killing
oscillations without moving the destination.

Formally: if μ* = Φ(μ*) and μ* is monotone, then
μ* = proj(Φ(μ*)) = proj(μ*) = μ*. The projected map
Ψ(μ) = proj(Φ(μ)) has the same fixed point as Φ, but a smaller
basin of bad iterates.

### For the price-grid method too

P(u₁, u₂, u₃) must be monotonically increasing in each u_k.
After each Picard step, enforce via separable PAVA along each axis:

```python
# For each fiber along axis 0:
for j in range(G):
    for l in range(G):
        P[:, j, l] = pava(P[:, j, l])

# Same for axes 1 and 2
```

Three passes, each O(G) per fiber, G² fibers. Total O(G³) — negligible
compared to the O(G⁴) Φ evaluation.

### Expected effect on the two-branches plot

The scattered red points at τ > 3.5 where the PR gap drops to zero
then jumps back — those are non-monotone transients. PAVA would:

1. Suppress the oscillation: the solver stays on the PR branch or
   transitions smoothly to FR, no bouncing.
2. Clean the edges: the fold bifurcation at τ ≈ 2.87 becomes a
   clean endpoint, not a noisy fade.
3. Allow tighter convergence criteria: without transients, Anderson
   can converge to tighter tolerance.

### When to apply

Apply AFTER the Bayes update, BEFORE damping/Anderson:

```
μ_raw = Φ(μⁿ)           # Bayes update
μ_mono = proj(μ_raw)     # PAVA projection
μⁿ⁺¹ = α·μ_mono + (1-α)·μⁿ   # damped update
```

Or with Anderson: flatten μ_mono (not μ_raw) into the Anderson
history. Anderson sees only monotone iterates.

---

## C. COMPLETE v3 ALGORITHM

```
Initialize:
    For each u_i: compute p_lo[i], p_hi[i] from no-learning
    μ[i, j] = Λ(τ·u_i) for all active (i, j)

Iterate until convergence:

    For each price level p_j:                           [G_p iterations]
        
        1. COLUMN EXTRACTION
           mu_col[i] = interp(μ[i, :], p_grids[i], p_j)
                                                        [G_u 1D interps]
        
        2. DEMAND ARRAY
           R[:] = exp((logit(mu_col) - logit(p_j)) / γ)
           d[:] = W·(R - 1) / ((1 - p_j) + R·p_j)
                                                        [G_u vectorized]
        
        3. For each own-signal u_i in active range:     [≤ G_u per p_j]
           
           a. CONTOUR (vectorized)
              targets = -d[i] - d
              u3_star = np.interp(targets, d, u_grid)
              valid = (u3_star ≥ u_min) & (u3_star ≤ u_max)
           
           b. DENSITIES (vectorized)
              A₁ = Σ f₁(u_sweep[valid]) · f₁(u3_star[valid])
              A₀ = Σ f₀(u_sweep[valid]) · f₀(u3_star[valid])
           
           c. BAYES
              μ_raw[i, j] = f₁(u_i)·A₁ / (f₀(u_i)·A₀ + f₁(u_i)·A₁)
    
    4. MONOTONICITY PROJECTION
       For each j: PAVA on μ_raw[:, j] in u-direction
       For each i: PAVA on μ_raw[i, :] in p-direction
    
    5. DAMPING / ANDERSON
       μⁿ⁺¹ = α · μ_mono + (1-α) · μⁿ
       or Anderson on flattened μ_mono vector
    
    6. CONVERGENCE CHECK
       ||μⁿ⁺¹ - μⁿ||∞ < tol over active cells
```

---

## D. IMPLEMENTATION CHECKLIST (updated)

1. [ ] Column extraction with per-row p-grid
2. [ ] Vectorized demand array d(u; p)
3. [ ] Vectorized contour via np.interp (no brentq)
4. [ ] Vectorized density integrals (dot products)
5. [ ] Bayes update
6. [ ] PAVA projection (u-direction and p-direction)
7. [ ] Active-cell detection (skip degenerate cells)
8. [ ] Damped Picard iteration
9. [ ] Anderson acceleration
10. [ ] CARA benchmark: verify μ* = p
11. [ ] CRRA at G_u = G_p = 50: measure 1-R²
12. [ ] Compare with price-grid at same parameters
13. [ ] Test at G_u = G_p = 100
14. [ ] Edge stability test: does PAVA eliminate quirky points?

---

## E. ALTERNATIVE TO PAVA: GAP REPARAMETRIZATION

### The idea

Instead of projecting onto the monotone cone after each update (PAVA),
reparametrize so monotonicity is automatic. Store the GAPS between
adjacent values instead of the values themselves.

### Along the u-direction at fixed price p_j

Order the signal grid: u_1 < u_2 < ... < u_{G_u}.
Monotonicity requires μ(u_1, p_j) ≤ μ(u_2, p_j) ≤ ... ≤ μ(u_{G_u}, p_j).

Reparametrize:

    base_j     = logit(μ(u_1, p_j))           [unconstrained real]
    c_k^j      = unconstrained real,  k = 2, ..., G_u
    
    logit(μ(u_1, p_j)) = base_j
    logit(μ(u_k, p_j)) = logit(μ(u_{k-1}, p_j)) + exp(c_k^j)

Since exp(c_k) > 0 always, the logit-posteriors are strictly increasing
in u, hence the posteriors are strictly increasing in u. Monotonicity
is guaranteed for ANY values of (base_j, c_2^j, ..., c_{G_u}^j).

### Why exp(c) instead of a²

The original suggestion was μ_{k+1} = μ_k + a_k². This works but has
a vanishing Jacobian problem: ∂μ_{k+1}/∂a_k = 2a_k → 0 when
a_k → 0 (nearly equal posteriors). Newton steps blow up.

With exp(c_k): ∂μ_{k+1}/∂c_k = exp(c_k) > 0 everywhere. The
Jacobian is always full rank. Near-equal values correspond to
c_k → -∞, which is numerically stable (exp(-20) ≈ 2e-9, not zero).

### Working in logit space

The reparametrization works best in logit space because:
1. The sufficient statistic T* lives in logit space
2. CARA posteriors are linear in logit space
3. The gaps in logit space are more uniform than in probability space

Store: base_j and c_k^j for k = 2,...,G_u at each price level j.
Total unknowns: G_p × G_u (same as before — one base + (G_u-1) gaps
= G_u values per price level).

### Reconstruction

To get μ from the stored (base, c) representation:

```python
def reconstruct_mu(base, c_array, G_u):
    """base: scalar, c_array: length G_u - 1"""
    logit_mu = np.zeros(G_u)
    logit_mu[0] = base
    for k in range(1, G_u):
        logit_mu[k] = logit_mu[k-1] + np.exp(c_array[k-1])
    return expit(logit_mu)  # logistic transform back to (0,1)
```

Vectorized:

```python
logit_mu = base + np.cumsum(np.concatenate([[0], np.exp(c_array)]))
mu = expit(logit_mu)
```

### The iteration

Option A — Reparametrize the fixed point:

    1. Given current (base^n, c^n), reconstruct μ^n
    2. Run one step of Φ_μ to get μ_raw (the Bayes update)
    3. Convert μ_raw back to (base_new, c_new):
           base_new = logit(μ_raw[0])
           c_new[k] = log(logit(μ_raw[k+1]) - logit(μ_raw[k]))
       But: if logit(μ_raw[k+1]) < logit(μ_raw[k]) (non-monotone),
       the log of a negative number is undefined.
    4. Fix: c_new[k] = log(max(logit(μ_raw[k+1]) - logit(μ_raw[k]), ε))
       with ε = 1e-12.
    5. Damped update in (base, c) space:
           base^{n+1} = α·base_new + (1-α)·base^n
           c^{n+1}    = α·c_new    + (1-α)·c^n

The damped update in (base, c) space automatically produces a
monotone μ^{n+1}. Even if μ_raw is non-monotone, the clamped c_new
with damping toward the previous (monotone) c^n recovers monotonicity.

Option B — Newton in (base, c) space:

    Define F(base, c) = (base, c) - T(Φ_μ(reconstruct(base, c)))
    where T is the transform to (base, c) coordinates.
    
    Solve F = 0 with Newton. The Jacobian ∂F/∂(base, c) exists and
    is nonsingular (since ∂reconstruct/∂c_k = exp(c_k) > 0).

### The 2D coupling problem

For full bivariate monotonicity (increasing in both u and p), we
cannot independently reparametrize along u and along p — the two
directions couple.

Recommended approach: enforce u-monotonicity via the gap
reparametrization (strong constraint, this is what breaks at edges),
and leave p-monotonicity to either:
  a) Soft penalty: add λ·Σ max(0, μ(u,p_j) - μ(u,p_{j+1}))² to
     the fixed-point residual.
  b) Post-hoc PAVA in the p-direction only (mild, rarely needed).
  c) Ignore — p-monotonicity violations are typically small and
     don't cause oscillation.

The u-direction monotonicity is the critical one. The contour sweep
is along u, and non-monotone d(u) = demand(μ(u,p), p) makes the
np.interp inversion fail (need strictly monotone array). The
p-direction only enters through interpolation when extracting
μ_col at off-grid prices, which is more forgiving.

### Advantages over PAVA

1. Smooth — C^∞ map from (base, c) to μ. Newton needs this.
2. No projection step — iterate never leaves the monotone cone.
3. Damping in gap space is more natural — mixing two monotone
   functions in (base, c) space always gives a monotone result.
4. Newton in (base, c) space has a well-defined nonsingular Jacobian.

### Disadvantages vs PAVA

1. More complex implementation (cumsum + exp + log transforms).
2. The log(max(..., ε)) clamping in step 4 is a mild hack.
3. 2D monotonicity requires a hybrid approach.

### Summary: when to use which

- **PAVA**: simpler, works for both directions, good for Picard.
- **Gap reparametrization**: better for Newton/Anderson, guarantees
  smoothness, stronger convergence properties.
- **Hybrid**: gap reparam in u-direction + soft penalty or PAVA
  in p-direction. Best of both worlds.
