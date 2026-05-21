# SYSTEM OF EQUATIONS AND UNKNOWNS — Posterior-Function Method v3

---

## 1. THE UNKNOWN

A single function on a 2D grid:

    μ : {u₁,...,u_{G_u}} × {p₁(u),...,p_{G_p}(u)} → (0, 1)

    μ[i, j] = P(v=1 | own signal = u_i, observed price = p_j(u_i))

Total unknowns: G_u × G_p scalars.
At G_u = G_p = 100: 10,000 unknowns.

The p-grid is per-row: p_j(u_i) ∈ [p_lo(u_i), p_hi(u_i)].

---

## 2. DERIVED QUANTITIES (not unknowns — functions of μ)

### 2.1 The demand column at fixed price p

Given the current μ and a specific price level p₀:

    d(u; p₀) = x(μ(u, p₀), p₀)

where x is the CRRA demand:

    x(μ, p) = W(R-1) / ((1-p) + Rp)
    R = exp((logit(μ) - logit(p)) / γ)

d(u; p₀) is a monotone function of u, computed on the u-grid
by interpolating μ[i, ·] to p₀ in the p-direction for each row i.

Cost to construct: G_u interpolations (one per row).

### 2.2 The contour at (u_i, p₀)

Agent 1 at signal u_i observing price p₀. The contour is:

    C(u_i, p₀) = { (u₂, u₃) : d(u₁; p₀) + d(u₂; p₀) + d(u₃; p₀) = 0 }

Parametrize by u₂:

    u₃*(u₂) = d⁻¹(-d(u_i; p₀) - d(u₂; p₀))

where d⁻¹ is the functional inverse of d(·; p₀) — a monotone
function, so the inverse exists and is unique.

Because d is precomputed on the u-grid, d⁻¹ is just interpolation
search on the array. No root-finding.

### 2.3 The contour integral

    A_v(u_i, p₀) = Σ_{u₂ sweep} f_v(u₂) · f_v(u₃*(u₂))

where f_v(u) = √(τ/(2π)) exp(-τ/2 (u - v + ½)²) is the signal
density under state v ∈ {0,1}.

The sum runs over sweep points u₂. For each, u₃* is obtained from
d⁻¹ as above.

A₀ and A₁ are the contour integrals under v=0 and v=1.

---

## 3. THE EQUATION (one per grid point)

For each active grid point (u_i, p_j):

    μ[i,j] = f₁(u_i) · A₁(u_i, p_j) / (f₀(u_i) · A₀(u_i, p_j) + f₁(u_i) · A₁(u_i, p_j))

This is Bayes' rule. The left side is the unknown. The right side
depends on the unknown through:

    μ → d(u; p_j) → u₃*(u₂) → A_v → Bayes

That's the fixed point: μ appears on both sides.

---

## 4. THE FULL SYSTEM

    μ[i, j] = Φ(μ)[i, j]     for all active (i, j)

where Φ is the map:

    μ  ──→  d(u; p_j) for each p_j    [column extraction + demand]
       ──→  u₃*(u₂) for each (u_i, p_j, u₂)    [contour tracing]
       ──→  A_v(u_i, p_j)    [density integration]
       ──→  μ_new(u_i, p_j)    [Bayes]

### Equation count
- Active grid points ≈ G_u × G_p (minus degenerate boundary cells)
- One equation per active point
- One unknown per active point
- System is square

### Dimensions at G_u = G_p = 100
- ~10,000 equations in ~10,000 unknowns

### Comparison with price-grid method
- Price grid: G³ = 1,000,000 equations in 1,000,000 unknowns
- Posterior grid: ~10,000 equations in ~10,000 unknowns
- Reduction factor: 100x

---

## 5. EXPANDED VIEW: ALL INTERMEDIATE STEPS

To make the chain of computation explicit at a single grid point
(u_i, p_j), here is every equation in order:

### Step A: Column extraction (G_u operations per price level)

For each row i' = 1, ..., G_u:

    (A1)  μ_col[i'] = interp₁ᴰ(μ[i', ·], p_j)
    
    This interpolates μ along row i' to the price value p_j.
    If p_j ∉ [p_lo(u_{i'}), p_hi(u_{i'})], set μ_col[i'] = Λ(τu_{i'}).

### Step B: Demand array (G_u operations per price level)

For each grid point i' = 1, ..., G_u:

    (B1)  R[i'] = exp((logit(μ_col[i']) - logit(p_j)) / γ)
    (B2)  d[i'] = W · (R[i'] - 1) / ((1 - p_j) + R[i'] · p_j)

d is now a length-G_u array: the demand at each signal value,
at this price level, incorporating learning from prices.

### Step C: Contour tracing (vectorized, per own-signal u_i)

    (C1)  D_i = -d[i]                    [own-agent demand, scalar]
    (C2)  targets[i'] = D_i - d[i']      [residual for agent 3, vector]
    (C3)  u₃*[i'] = interp_invert(d, u_grid, targets[i'])
                                          [invert d to get u₃, vector]

interp_invert: given monotone array d on u_grid, find u such that
d(u) = target. This is np.interp with arguments swapped:

    u₃*[i'] = np.interp(targets[i'], d, u_grid)

if d is increasing, or with flipped arrays if decreasing.

Validity mask: u₃*[i'] ∈ [u_min, u_max]. Outside → crossing doesn't
exist → exclude from integral.

### Step D: Density computation (vectorized)

    (D1)  f₁_sweep[i'] = √(τ/2π) · exp(-τ/2 · (u_{i'} - ½)²)    [f₁ at u₂ = u_{i'}]
    (D2)  f₀_sweep[i'] = √(τ/2π) · exp(-τ/2 · (u_{i'} + ½)²)    [f₀ at u₂ = u_{i'}]
    (D3)  f₁_root[i']  = √(τ/2π) · exp(-τ/2 · (u₃*[i'] - ½)²)   [f₁ at u₃*]
    (D4)  f₀_root[i']  = √(τ/2π) · exp(-τ/2 · (u₃*[i'] + ½)²)   [f₀ at u₃*]

### Step E: Contour integrals (dot products)

    (E1)  A₁ = Σ_{valid i'} f₁_sweep[i'] · f₁_root[i']
    (E2)  A₀ = Σ_{valid i'} f₀_sweep[i'] · f₀_root[i']

### Step F: Bayes' rule (scalar)

    (F1)  L₁ = f₁(u_i) · A₁
    (F2)  L₀ = f₀(u_i) · A₀
    (F3)  μ_new[i, j] = L₁ / (L₀ + L₁)

---

## 6. THE LOOP STRUCTURE

```
For each price level p_j:                           # outer loop: G_p
    
    Step A: extract μ_col[i'] for all i'            # G_u interps
    Step B: compute d[i'] for all i'                # G_u evals
    
    For each own-signal u_i where p_j is in range:  # inner loop: ≤ G_u
        
        Step C: targets = -d[i] - d[:]              # vectorized
                u₃* = interp_invert(d, targets)     # vectorized
        
        Step D: f_v at sweep and root points         # vectorized
        
        Step E: A_v = dot products                   # 2 dot products
        
        Step F: μ_new[i, j] = Bayes                  # scalar
```

### Operation count per iteration

| Step | Count | Per-step cost | Total |
|------|-------|---------------|-------|
| A: column extraction | G_p | G_u interps | G_u · G_p |
| B: demand array | G_p | G_u evals | G_u · G_p |
| C: contour tracing | G_p · G_u | G_u interp (vectorized) | G_u² · G_p |
| D: densities | G_p · G_u | G_u evals (vectorized) | G_u² · G_p |
| E: dot products | G_p · G_u | G_u mults + sum | G_u² · G_p |
| F: Bayes | G_p · G_u | O(1) | G_u · G_p |

Dominant cost: Steps C-E at O(G_u² · G_p) per iteration.

At G_u = G_p = 100: 100² × 100 = 1,000,000 operations per iteration.
Compare price-grid: 6 × G⁴ = 600,000,000 root-finds at G=100.

---

## 7. CONVERGENCE

### Fixed-point iteration (Picard)

    μ^(n+1) = α · Φ(μ^(n)) + (1-α) · μ^(n)

α ∈ [0.1, 0.3]. Convergence monitored by:

    ||μ^(n+1) - μ^(n)||_∞  < tol    (over active cells)

### Anderson acceleration

Flatten μ[i,j] over active cells into a vector x ∈ ℝ^N where
N ≈ G_u × G_p. The residual is g(x) = Φ(x) - x. Anderson builds
a quasi-Newton update from the last m residuals.

Memory window m = 5-8. Expected convergence in 10-20 iterations.

---

## 8. SELF-CONSISTENCY VERIFICATION

After convergence of μ*, verify by reconstructing P and measuring 1-R²:

### 8.1 Reconstruct P at a realization (u₁, u₂, u₃)

Solve for price p* such that market clearing holds:

    (V1)  F(p) = d(u₁; p) + d(u₂; p) + d(u₃; p) = 0

where d(u; p) = x(μ*(u, p), p) uses the converged μ*.

This is a 1D root-find in p. For each trial p, we need μ*(u_k, p)
for k=1,2,3, obtained by interpolating μ* in the p-direction.

### 8.2 Check posteriors at the price

    μ₁* = μ*(u₁, p*)
    μ₂* = μ*(u₂, p*)
    μ₃* = μ*(u₃, p*)

Under CARA: μ₁* = μ₂* = μ₃* = p* (FR).
Under CRRA: μ₁* ≠ μ₂* ≠ μ₃* (PR).

### 8.3 Measure 1-R²

Sample M realizations (u₁, u₂, u₃) on the grid or randomly.
For each, solve (V1) for p*. Compute:

    T* = τ(u₁ + u₂ + u₃)
    Y = logit(p*)
    w = ½(f₁(u₁)f₁(u₂)f₁(u₃) + f₀(u₁)f₀(u₂)f₀(u₃))

Weighted regression of Y on T*. Report 1-R².

---

## 9. WHY LEARNING FROM PRICES IS CAPTURED

The μ at iteration 0 is Λ(τu) — the private prior.
The μ at convergence is the REE posterior.

The learning enters through the contour integral:

    A_v(u_i, p₀) = Σ f_v(u₂) · f_v(u₃*(u₂))

This tells agent 1: "Given that I see price p₀ and my signal is u_i,
what signal pairs (u₂, u₃) are consistent with this price?"

The set of consistent pairs depends on μ — because the demands
depend on μ, which determines which (u₂, u₃, p₀) triples clear
the market. At convergence, the beliefs that determine the contour
ARE the beliefs that the contour produces. That's the fixed point.

Without learning: μ = Λ(τu), the contour reflects no-learning
market clearing, and A_v encodes no price information.

With learning: μ(u, p) tilts toward the price, the contour shifts,
and A_v encodes the information that was available in the price.
The tilt is self-consistent at the fixed point.

---

## 10. COMPARISON OF THE THREE METHODS

| | Price grid | Posterior v2 | Posterior v3 |
|---|---|---|---|
| Unknown | P[i,j,l] | μ[i,j] | μ[i,j] |
| Dimension | G³ | G_u × G_p | G_u × G_p |
| At G=100 | 10⁶ | 10⁴ | 10⁴ |
| Contour tracing | interpolate P surface | brentq per sweep point | interp_invert d array |
| Root-finds/iter | 6G⁴ | G_u·G_p·N_sweep | 0 (vectorized interp) |
| P interpolation | 2D on rough surface | none | none |
| μ interpolation | none (μ computed fresh) | 1D in u | 1D in u (via d⁻¹) |
| Domain issues | none | rectangular fails | adaptive p-range |
| Jacobian | cancels | cancels | cancels |
| Symmetrization | average 6 permutations | automatic | automatic |
| CARA benchmark | must give 1-R²=0 | must give μ*=p | must give μ*=p |
