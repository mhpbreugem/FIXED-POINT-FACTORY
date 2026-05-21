# MODEL AND FIXED-POINT SYSTEM

A condensed reference for the equations the solver actually computes.
For derivations and discretisation tradeoffs, read POSTERIOR_METHOD_V2.md.

---

## 1. PRIMITIVES

| Symbol | Meaning |
|--------|---------|
| v ∈ {0, 1}              | binary asset payoff, prior P(v = 1) = ½ |
| K = 3                   | number of agents |
| s_k = v + ε_k           | private signal; ε_k iid N(0, 1/τ) |
| u_k = s_k − ½           | centred signal: u_k \| v ~ N(v − ½, 1/τ) |
| τ                       | signal precision |
| γ                       | CRRA risk aversion (common to all K) |
| W = 1                   | wealth normalization |
| f_v(u)                  | signal density: √(τ/2π) exp(−τ/2 (u − v + ½)²) |
| Λ(x) = 1 / (1 + e^{−x}) | logistic function (sigmoid) |
| logit p = log(p/(1−p))  | inverse logistic |

There are NO noise traders, NO supply shocks, NO random endowments.

---

## 2. CRRA DEMAND

```
R       = exp((logit μ − logit p) / γ)
x(μ, p) = W · (R − 1) / ((1 − p) + R · p)
```

Properties used by the solver:
- monotone increasing in μ at fixed p
- monotone decreasing in p at fixed μ
- x(p, p) = 0 (no trade when posterior equals price)
- CARA limit (γ → ∞ in a renormalized sense): x → (logit μ − logit p) / γ_CARA

---

## 3. THE UNKNOWN

The REE posterior function on a 2-D grid:

    μ : {u₁, …, u_G} × {p_j(u_i)} → (0, 1)
    μ[i, j] = P(v = 1 | own signal u_i, observed price p_j(u_i))

Storage is per-row ragged: each signal u_i has its own price grid

    p_j(u_i) ∈ [p_lo(u_i), p_hi(u_i)]

with G_p points in logit-space spacing. At the seed (γ = 0.5, τ = 2,
G = 20, UMAX = 5) this is 400 unknowns.

---

## 4. THE FIXED POINT  μ* = Φ(μ*)

The map Φ does Bayesian updating along the market-clearing contour.
For each grid cell (u_i, p_j):

### Step A — column extraction at fixed p_j
For each row i' = 1, …, G:
    μ_col[i'] = interp(μ[i', ·], p_grid[i'], p_j)
If p_j ∉ [p_lo(u_{i'}), p_hi(u_{i'})] use μ_col[i'] = Λ(τ u_{i'}).

### Step B — demand column
    R[i'] = exp((logit μ_col[i'] − logit p_j) / γ)
    d[i'] = W · (R[i'] − 1) / ((1 − p_j) + R[i'] · p_j)

d is monotone in i' (because μ_col is monotone in u and demand is
monotone in μ).

### Step C — vectorised contour inversion
For own-signal index i:
    D_i        = − d[i]
    target[i'] = D_i − d[i']                        (for each sweep i')
    u₃*[i']    = interp_invert(d, u_grid, target[i'])
    valid[i']  = u_min ≤ u₃*[i'] ≤ u_max

`interp_invert` is `np.interp` with d and u_grid swapped (with sign
flip if d is decreasing). No root-finding.

### Step D — densities along the contour
    f₁_sweep[i'] = f₁(u_{i'})              under v = 1
    f₀_sweep[i'] = f₀(u_{i'})              under v = 0
    f₁_root [i'] = f₁(u₃*[i'])
    f₀_root [i'] = f₀(u₃*[i'])

### Step E — contour integrals (dot products over valid sweep)
    A₁(u_i, p_j) = Σ_{i' valid} f₁_sweep[i'] · f₁_root[i']
    A₀(u_i, p_j) = Σ_{i' valid} f₀_sweep[i'] · f₀_root[i']

### Step F — Bayes
    Φ(μ)[i, j] = f₁(u_i) · A₁(u_i, p_j)
                 ----------------------------------------
                 f₀(u_i) · A₀(u_i, p_j) + f₁(u_i) · A₁(u_i, p_j)

The fixed point is

    μ*[i, j] = Φ(μ*)[i, j]      for every active cell (i, j)

A square G × G_p system in G × G_p unknowns.

---

## 5. CONTOUR INTEGRATION = WHY μ* HAS THE FORM IT DOES

The set
    C(u_i, p_j) = { (u₂, u₃) : x(μ_col(u_i), p) + x(μ_col(u₂), p)
                                                   + x(μ_col(u₃), p) = 0 }
is the locus of other-agent signal pairs consistent with the agent
seeing (u_i, p_j). Bayes weights each crossing by the joint density
under the two states:

    A_v(u_i, p_j) = ∫ f_v(u₂) · f_v(u₃*(u₂)) du₂

Discretised on the u-grid, this is a dot product. The Jacobian
|du₃*/du₂| cancels in the ratio A₁/A₀ to leading order (see
POSTERIOR_METHOD_V2.md §9 for the precise statement).

---

## 6. MARKET-CLEARING RECONSTRUCTION

To verify the solution and to measure 1-R², reconstruct prices for
all G³ = 8000 triples (u₁, u₂, u₃):

    F(p) = x(μ*(u₁, p), p) + x(μ*(u₂, p), p) + x(μ*(u₃, p), p) = 0

This is a 1-D root-find in p (F is monotone decreasing). At each
trial p, μ*(u_k, p) comes from interpolating row u_k of the converged
μ* in the p-direction.

Sufficient statistic: T* = τ (u₁ + u₂ + u₃).

Under CARA: μ*(u, p) = p, so logit p = T*/K exactly → 1-R² = 0.
Under CRRA: μ* depends on both arguments, the regression of logit p
on T* leaves a residual → 1-R² > 0.

---

## 7. WEIGHTED 1-R²  (the metric reported in TASK_QUEUE.json)

```python
def signal_density(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau / (2*np.pi)) * np.exp(-tau/2 * (u - mean)**2)

# For each triple (i, j, l) on the G^3 grid:
w[i,j,l] = 0.5 * ( f0(u_i)*f0(u_j)*f0(u_l)
                 + f1(u_i)*f1(u_j)*f1(u_l) )

# Weighted regression:
slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(w))
pred             = slope * Tstar + intercept

# Weighted R^2 (use np.average with weights):
mean_lp = np.average(logit_p,                  weights=w)
var_tot = np.average((logit_p - mean_lp)**2,   weights=w)
var_res = np.average((logit_p - pred)**2,      weights=w)
R2      = 1.0 - var_res / var_tot
one_minus_R2 = var_res / var_tot      # this is what we report
```

Unweighted 1-R² is unreliable: it depends on UMAX because the tails
contribute outliers that carry near-zero ex-ante probability. Weighted
1-R² is invariant across (G = 15-20, UMAX = 4-5) at γ = 0.5, τ = 2.

Verified anchor (γ = 0.5, τ = 2):
| Grid          | unwtd 1-R² | wtd 1-R² | wtd slope |
|---------------|-----------:|---------:|----------:|
| G=15 UMAX=4   |      0.191 |    0.078 |     0.521 |
| G=18 UMAX=4   |      0.195 |    0.083 |     0.545 |
| G=20 UMAX=5   |      0.230 |    0.085 |     0.543 |

---

## 8. CARA SANITY CHECK

Run the same solver with CARA demand x = (logit μ − logit p) / γ.
At convergence μ*(u, p) ≡ p, every triple gives logit p = T*/K, and
the weighted regression yields 1-R² = 0 to numerical tolerance. Any
deviation indicates a bug in the solver, not in the model.
