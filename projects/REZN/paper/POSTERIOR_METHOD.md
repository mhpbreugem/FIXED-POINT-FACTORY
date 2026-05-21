# THE POSTERIOR-FUNCTION METHOD
# Alternative to the price-grid contour method
# Key idea: make μ(u, p) the unknown, not P[i,j,l]

---

## 1. MOTIVATION

The standard contour method stores P[i,j,l] — a G³ array — and traces
contours by interpolating this array. All grid artifacts (binning noise,
interpolation curvature, boundary issues) come from storing and
interpolating P.

The posterior-function method stores μ(u, p) — a G_u × G_p array — and
traces contours analytically through market clearing. No interpolation
of P at any step. The unknown shrinks from G³ to G_u × G_p, which is
two orders of magnitude smaller at G=100.

---

## 2. THE UNKNOWN

With symmetric agents (same γ, τ, W), there is a single posterior
function:

    μ : ℝ × (0,1) → (0,1)
    μ(u, p) = P(v=1 | own signal = u, observed price = p)

This is a 2D function. We discretize it on a grid:

    u ∈ {u₁, ..., u_{G_u}}     (signal grid, e.g. linspace(-4, 4, G_u))
    p ∈ {p₁, ..., p_{G_p}}     (price grid, e.g. linspace(0.01, 0.99, G_p))

Total unknowns: G_u × G_p scalars.

At G_u = G_p = 100: 10,000 unknowns instead of 1,000,000.

---

## 3. HOW THE CONTOUR FALLS OUT

Given the current μ(u, p), the contour at a specific (u₁, p₀) is
traced as follows.

### 3.1 The demand of each agent

Agent k at signal u_k observing price p has demand:

    x_k = W · (R_k - 1) / ((1-p) + R_k · p)
    R_k = exp((logit(μ(u_k, p)) - logit(p)) / γ)

Note: each agent uses the SAME function μ(·, ·) (symmetric agents)
evaluated at her own signal.

### 3.2 Market clearing defines the contour

Fix agent 1 at signal u₁ and price p₀. The contour is:

    { (u₂, u₃) : x₁(u₁, p₀) + x₂(u₂, p₀) + x₃(u₃, p₀) = 0 }

where x_k(u_k, p₀) ≡ x_k(μ(u_k, p₀), p₀).

Define the residual demand from agents 2 and 3:

    D₁ ≡ -x₁(u₁, p₀)

Then the contour is:

    x₂(u₂, p₀) + x₃(u₃, p₀) = D₁

### 3.3 Tracing the contour by sweeping

For each sweep point u₂:
- Compute x₂(u₂, p₀) using the current μ(u₂, p₀)
- The target for agent 3 is: x₃(u₃, p₀) = D₁ - x₂(u₂, p₀) ≡ T₃
- Root-find u₃* such that x₃(u₃*, p₀) = T₃
- This is a 1D root-find in u₃: given p₀ and γ, find u₃ such that
  the CRRA demand at posterior μ(u₃, p₀) equals T₃

### 3.4 The root-find

The demand x₃ = W(R₃-1)/((1-p₀)+R₃p₀) with R₃ = exp((logit(μ(u₃,p₀))-logit(p₀))/γ).

This is monotone in μ(u₃, p₀), and μ(u₃, p₀) is monotone in u₃ (at
least at no-learning, where μ = Λ(τu)). So x₃ is monotone in u₃,
and the root-find is clean.

To evaluate μ(u₃, p₀) at an off-grid u₃: interpolate the stored
μ array in the u-direction. This is 1D interpolation of a smooth
function — much better behaved than 2D interpolation of the price surface.

### 3.5 What goes INTO the root-find

At no-learning (iteration 0): μ(u, p) = Λ(τu). The root-find becomes:
solve x₃(Λ(τu₃), p₀) = T₃ for u₃. This is purely analytical — no
grid needed.

At subsequent iterations: μ(u, p) is stored on the (u, p) grid. To
evaluate μ(u₃*, p₀) at off-grid u₃*, interpolate in the u-direction
at the known p₀ grid point. If p₀ is also off-grid, use bilinear
interpolation of μ — but this is interpolation of a SMOOTH 2D function,
not a ROUGH 3D price surface.

---

## 4. THE CONTOUR INTEGRAL

At each crossing (u₂, u₃*), compute the signal density:

    f_v(u₂) · f_v(u₃*)

Sum over all sweep points:

    A_v(u₁, p₀) = Σ_{u₂ sweep} f_v(u₂) · f_v(u₃*(u₂))

(With two-pass averaging if desired, though for smooth μ the single
pass may suffice.)

No Jacobian needed — same argument as in contour.md. The Jacobian
|∂P/∂u₃| enters both A₁ and A₀ identically and cancels in the
posterior ratio.

---

## 5. BAYES' RULE

The updated posterior at (u₁, p₀) is:

    μ_new(u₁, p₀) = f₁(u₁) · A₁(u₁, p₀) / (f₀(u₁) · A₀(u₁, p₀) + f₁(u₁) · A₁(u₁, p₀))

This is the new value of μ at this grid point.

---

## 6. THE FIXED POINT

Iterate:

    μ^(n+1)(u, p) = Φ_μ(μ^(n))(u, p)

where Φ_μ is the map:
1. For each (u_i, p_j) on the grid:
   a. Compute D₁ = -x(μ^(n)(u_i, p_j), p_j)
   b. Sweep u₂ over N_sweep points
   c. For each u₂, root-find u₃* via market clearing
   d. Sum A_v = Σ f_v(u₂) · f_v(u₃*)
   e. Compute μ_new = Bayes(f_v(u_i), A₁, A₀)
2. Update: μ^(n+1) = α · μ_new + (1-α) · μ^(n) (damped Picard)
   or use Anderson acceleration on the flattened μ vector.

Convergence: ||μ^(n+1) - μ^(n)||∞ < tol.

---

## 7. SELF-CONSISTENCY CHECK

At convergence, the method must be self-consistent:

For any realization (u₁, u₂, u₃), the market-clearing price p* satisfies:

    x(μ*(u₁, p*), p*) + x(μ*(u₂, p*), p*) + x(μ*(u₃, p*), p*) = 0

This implicitly defines P(u₁, u₂, u₃) = p*. The price function is
NOT stored — it's derived from the posterior function on the fly.

To measure 1-R²: sample many realizations (u₁, u₂, u₃), solve for
p* at each, compute T*, regress logit(p*) on T*.

---

## 8. INITIALIZATION

Iteration 0 (no-learning):

    μ^(0)(u, p) = Λ(τu)

The posterior ignores the price entirely. The first iteration of Φ_μ
will update this: the contour integral at (u, p) tells the agent
what other signals are consistent with price p, and the posterior
adjusts.

---

## 9. COMPUTATIONAL COST

Per iteration of Φ_μ:
- G_u × G_p grid points to update
- At each: N_sweep root-finds (each is 1D, fast)
- Total root-finds per iteration: G_u × G_p × N_sweep

Compare with the price-grid method:
- G³ grid points to update
- At each: 3 agents × 2 passes × G sweeps = 6G root-finds
- Total root-finds per iteration: 6G⁴

Example at G = G_u = G_p = N_sweep = 100:
- Posterior method: 100 × 100 × 100 = 1,000,000 root-finds
- Price-grid method: 6 × 100⁴ = 600,000,000 root-finds

The posterior method is 600x faster at G=100.

---

## 10. THE p-GRID

What prices should we grid on?

Option A: Uniform on (0.01, 0.99). Safe but wasteful — many price
grid points will never be realized in equilibrium.

Option B: Adaptive. Run the no-learning case first, find the range
of realized prices [p_min, p_max], grid densely there.

Option C: Use logit(p) as the grid variable. This is natural because
the sufficient statistic T* lives in logit space, and the posterior
function μ(u, p) is smoother in (u, logit(p)) than in (u, p).

Recommendation: Option C. Grid on logit(p) ∈ [-L, +L] with L chosen
so that Λ(L) covers the realized price range. For τ=2, K=3:
L ≈ 6 suffices (covers p from 0.0025 to 0.9975).

---

## 11. WHAT THE p-GRID MEANS ECONOMICALLY

Each grid point (u_i, p_j) represents a scenario: "I received signal u_i
and I observe price p_j." The posterior μ(u_i, p_j) answers: "Given my
signal and this price, what do I believe about v?"

Under CARA (FR): μ(u, p) = p for all u. The agent ignores her own signal
because the price already reveals everything.

Under CRRA (PR): μ(u, p) depends on BOTH u and p. The agent combines her
private signal with the (imperfect) information in the price.

The fixed point μ* is the rational-expectations posterior: the beliefs
that are self-consistent with the price function they generate.

---

## 12. SYMMETRY

With symmetric agents, the single function μ(u, p) suffices. Each agent
evaluates it at her own signal. No need for separate μ₁, μ₂, μ₃.

With heterogeneous agents (different γ_k or τ_k): store K separate
posterior functions μ_k(u, p). The grid becomes K × G_u × G_p. Still
much smaller than G^K for K ≥ 3.

---

## 13. ADVANTAGES OVER THE PRICE-GRID METHOD

1. **No P interpolation.** The contour is traced through market clearing
   using μ directly. No 2D/3D interpolation of a rough surface.

2. **Smaller state space.** G_u × G_p vs G³. Two orders of magnitude
   at G=100.

3. **Smooth interpolation.** When we do interpolate (μ at off-grid u),
   it's 1D interpolation of a smooth function, not 2D interpolation of
   a price surface with potential kinks.

4. **Natural economic interpretation.** The unknown IS the agent's
   belief function — exactly what we're trying to characterize.

5. **No boundary issues.** The sweep points u₂ can go as far as needed.
   The Gaussian density kills the tails. No grid edge to hit.

6. **No symmetrization needed.** With symmetric agents, μ is automatically
   symmetric. No need to average over 6 permutations of (i,j,l).

7. **Easy to measure 1-R².** Sample realizations → solve for prices →
   regress. The price function is implicit, derived from μ.

---

## 14. DISADVANTAGES

1. **The p-grid is a modeling choice.** Unlike the price-grid method
   where P[i,j,l] naturally lives on the signal grid, the p-grid
   requires choosing a range and density. If the equilibrium price
   range shifts during iteration, the grid may need adapting.

2. **Market clearing at each realization.** To compute 1-R² or the
   converged posteriors table, you need to solve market clearing at
   each (u₁, u₂, u₃) to get the price. This is cheap (one 1D root-find)
   but adds a step.

3. **The root-find in step 3.4 uses μ interpolation.** If μ has
   steep gradients in u (which it does when τ is large), the
   interpolation needs care. But this is 1D interpolation of a
   sigmoid-like function — well-understood.

4. **Multiple equilibria.** The posterior-function fixed point may have
   multiple solutions, just like the price-grid fixed point. The
   initialization μ⁰ = Λ(τu) (no-learning) seeds toward the PR branch.

---

## 15. RELATIONSHIP TO THE PRICE-GRID METHOD

The two methods solve the same economic problem. At convergence:
- The posterior-method μ*(u, p) implies a price function P*(u₁, u₂, u₃)
  via market clearing.
- The price-grid P*[i,j,l] implies a posterior function μ*(u, p) via
  contour integration.

They should give the same answer. The difference is computational:
which object to store and iterate on.

The posterior method can be viewed as "substituting out P" from the
system, just as the price-grid method "substitutes out μ." The
posterior method substitutes more aggressively — it doesn't store P
at all — and gains efficiency at the cost of an implicit price function.

---

## 16. ALGORITHM SUMMARY

```
Initialize:
    μ(u, p) = Λ(τu) for all (u, p) on the grid   [no-learning]

Iterate until convergence:
    For each grid point (u_i, p_j):

        1. DEMAND OF OWN AGENT
           R₁ = exp((logit(μ(u_i, p_j)) - logit(p_j)) / γ)
           x₁ = W(R₁ - 1) / ((1-p_j) + R₁·p_j)
           D₁ = -x₁

        2. SWEEP OTHER AGENTS
           For n = 1, ..., N_sweep:
               u₂ = sweep point (e.g. equally spaced or Gauss-Hermite)
               
               Compute x₂ = demand at (μ(u₂, p_j), p_j)
               [interpolate μ in u-direction if u₂ is off-grid]
               
               Target for agent 3: T₃ = D₁ - x₂
               
               Root-find u₃* such that x₃(μ(u₃*, p_j), p_j) = T₃
               [1D root-find; interpolate μ(·, p_j) as needed]
               
               Record: f_v(u₂) · f_v(u₃*) → add to A_v

        3. UPDATE POSTERIOR
           μ_new(u_i, p_j) = f₁(u_i)·A₁ / (f₀(u_i)·A₀ + f₁(u_i)·A₁)

    Damped update or Anderson acceleration on the μ vector.
```

---

## 17. TASKS

1. Implement in Python at small scale (G_u = G_p = 20, N_sweep = 20)
2. Verify against price-grid method at G=5
3. Run at G_u = G_p = 100 and compare speed
4. Measure 1-R² at converged μ
5. Compare convergence behavior (Picard + Anderson)
