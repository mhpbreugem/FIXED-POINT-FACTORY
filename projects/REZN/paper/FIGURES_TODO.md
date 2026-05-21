# SOLVER INSTRUCTIONS — READ THIS FILE, EXECUTE IN ORDER
# Updated 2026-05-02

## GLOBAL PARAMETERS
- Standard gammas: γ = 0.5, 1.0, 4.0
- Grid: G=20, u_grid from -5 to +5 (UMAX=5), spacing 0.526
- Precision: mp50 (50 decimal digits) for sweeps
- Tolerance: ||F||∞ < 1e-25 (do NOT converge to machine precision)
- Seed checkpoint: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json
  (G=20, UMAX=5, γ=0.5, τ=2, ||F||=7.4e-119)

---

## TASK 1: EXTRACT FIGURES FROM EXISTING SEED (no new convergence runs)

Load the seed file posterior_v3_G20_umax5_notrim_mp300.json.
This contains the converged μ*(u,p) at G=20, γ=0.5, τ=2.

### Task 1a: Fig 3B — CRRA contour lines
- Fix u₁ = closest grid point to 1.0
- Create 300×300 fine grid in (u₂, u₃) ∈ [-3.5, 3.5]²
- At each (u₂, u₃): interpolate μ* for all 3 agents, solve market clearing → price
- Apply Gaussian smoothing (σ=1.5 pixels) to the 300×300 price surface
- Extract contour lines at p ∈ {0.2, 0.3, 0.5, 0.7, 0.8} using matplotlib contour
- For each contour: arc-length resample to 50 points
- Save: results/full_ree/fig3B_G20_pgfplots.tex
- Format: `\addplot coordinates {(x1,y1)(x2,y2)...};` per price level

### Task 1b: Fig 5 — price vs T* (three curves)
- Choose 50 evenly-spaced T* values from -8 to +8
- For each T*, set u₁ = u₂ = u₃ = T*/(3τ) (symmetric triple)
- Compute three prices:
  - p_FR = Λ(T*/3) (analytical)
  - p_NL = solve Σ crra_demand(Λ(τuₖ), p) = 0 (no-learning, private priors)
  - p_REE = solve Σ crra_demand(μ*(uₖ, p), p) = 0 (using converged μ*)
- Save: results/full_ree/fig5_G20_pgfplots.tex
- Format: three \addplot coordinate lists (FR, NL, REE)

### Task 1c: Fig 6B — CRRA posteriors vs T*
- Choose 50 T* values from -3 to +4 (transition zone)
- For each T*, use triple (u₁=+1, u₂=-1, u₃=T*/τ - u₁ - u₂ + 1)
  OR simpler: sweep T* and pick representative (u₁, u₂, u₃) with that T*
- At each triple: solve market clearing with μ* → get price p_REE
- Compute: μ₁ = μ*(u₁, p_REE), μ₂ = μ*(u₂, p_REE)
- Save: results/full_ree/fig6B_G20_pgfplots.tex
- Format: three \addplot coordinate lists (μ₁, μ₂, price)

### Task 1d: Measure 1-R² from seed
- From the converged μ*, compute 1-R² and slope at G=20 γ=0.5 τ=2
- Method: for all G³ triples, compute T* and logit(p_REE), regress
- Save: results/full_ree/G20_umax5_R2.json
- Include: 1-R², slope, n_triples

---

## TASK 2: γ-SWEEP AT τ=2 (5 new convergence runs)

Run the posterior-function fixed-point solver at G=20, UMAX=5, mp50.
Warm-start each from the nearest converged γ.

### Execution order (warm-start chain):
```
Run 1: γ=1.0, τ=2  — warm from γ=0.5 seed (interpolate μ* onto new p-grid)
Run 2: γ=2.0, τ=2  — warm from γ=1.0 result
Run 3: γ=4.0, τ=2  — warm from γ=2.0 result
Run 4: γ=0.25, τ=2 — warm from γ=0.5 seed
Run 5: γ=0.1, τ=2  — warm from γ=0.25 result
```

For each converged run, save:
- Checkpoint: results/full_ree/posterior_v3_G20_umax5_gXX_mp50.json
- Measure: 1-R², slope
- Summary: results/full_ree/fig4B_G20_gamma_sweep.json

The final fig4B_G20_gamma_sweep.json should contain:
```json
{
  "figure": "fig4B",
  "params": {"G": 20, "tau": 2.0, "umax": 5},
  "REE": [
    {"gamma": 0.1, "1-R2": ..., "slope": ...},
    {"gamma": 0.25, "1-R2": ..., "slope": ...},
    {"gamma": 0.5, "1-R2": ..., "slope": ...},
    {"gamma": 1.0, "1-R2": ..., "slope": ...},
    {"gamma": 2.0, "1-R2": ..., "slope": ...},
    {"gamma": 4.0, "1-R2": ..., "slope": ...}
  ],
  "no_learning": [
    {"gamma": 0.1, "1-R2": ...},
    ...
  ]
}
```

Also output: results/full_ree/fig4B_G20_pgfplots.tex with:
```latex
% REE
\addplot coordinates {(0.1,...)  (0.25,...) (0.5,...) (1,...) (2,...) (4,...)};
% no-learning
\addplot coordinates {(0.1,...) (0.25,...) (0.5,...) (1,...) (2,...) (4,...)};
```

---

## TASK 3: τ-SWEEP AT 3 γ VALUES (33 new convergence runs)

After Task 2 completes, you have converged μ* at τ=2 for γ=0.5, 1.0, 4.0.
Now sweep τ for each γ.

### τ values: [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
### γ values: [0.5, 1.0, 4.0]

τ=2.0 is already done (from Task 2). So 11 new τ values per γ = 33 runs.

### Warm-start chain for each γ:
```
τ=2.0 (done) → τ=1.5 → τ=1.0 → τ=0.8 → τ=0.5 → τ=0.3   (walk down)
τ=2.0 (done) → τ=3.0 → τ=4.0 → τ=5.0 → τ=7.0 → τ=10.0 → τ=15.0  (walk up)
```

When changing τ: the u_grid stays the same, but μ*(u,p) changes.
Interpolate the previous τ's converged μ* as the initial guess.

For each converged run, measure 1-R².

Save: results/full_ree/fig4A_G20_tau_sweep.json
```json
{
  "figure": "fig4A",
  "params": {"G": 20, "umax": 5},
  "curves": [
    {"gamma": 0.5, "points": [{"tau": 0.3, "1-R2": ...}, {"tau": 0.5, "1-R2": ...}, ...]},
    {"gamma": 1.0, "points": [...]},
    {"gamma": 4.0, "points": [...]}
  ]
}
```

Also output: results/full_ree/fig4A_G20_pgfplots.tex

---

## TASK 4: NO-LEARNING FIGURES (cheap, no REE)

### Task 4a: Fig R1 — 1-R² vs K (number of agents)
- For K = 3, 5, 7, 10, 15, 20
- For each γ ∈ {0.5, 1.0, 4.0}
- Compute no-learning 1-R² at τ=2, G=20
- Save: results/full_ree/figR1_G20_pgfplots.tex

### Task 4b: Fig R2 — lognormal payoff variant
- Same as knife-edge but with lognormal payoff v ~ LogNormal
- For γ ∈ {0.5, 1.0, 4.0}, sweep τ
- Save: results/full_ree/figR2_G20_pgfplots.tex

### Task 4c: Fig 11 — mechanisms bar chart
- Compute no-learning 1-R² at 6 configurations:
  1. CRRA symmetric (γ=0.5, equal τ)
  2. Het γ = (0.5, 3, 10), equal τ=2
  3. Het τ = (1, 3, 10), equal γ=0.5
  4. Het γ + het τ aligned (low γ = high τ)
  5. Het γ + het τ opposed (low γ = low τ)
  6. CRRA γ=2 (weak effect)
- Save: results/full_ree/fig11_G20_pgfplots.tex

---

## EXECUTION PRIORITY
1. Task 1 (extractions from seed) — DO FIRST, no convergence needed
2. Task 2 (5 γ-sweep runs) — DO SECOND
3. Task 4 (no-learning, cheap) — DO ANYTIME
4. Task 3 (33 τ-sweep runs) — DO OVERNIGHT after Task 2

---

## NOTES
- All pgfplots output: one \addplot coordinates {...}; per curve
- Arc-length resample contours to 40-50 points
- Use linear interpolation in pgfplots (no smooth keyword needed if 40+ pts)
- Colors will be applied in the paper's LaTeX, not in the pgfplots output

---

## CRITICAL FIX: USE WEIGHTED 1-R² EVERYWHERE

**All 1-R² measurements MUST use ex-ante probability weights.**

Unweighted gives wrong values (0.19-0.23) that depend on umax.
Weighted gives stable values (0.078-0.085) that converge across grids.

**Weight formula:**
```python
w(i,j,l) = 0.5 * (f_0(u_i)*f_0(u_j)*f_0(u_l) + f_1(u_i)*f_1(u_j)*f_1(u_l))

def signal_density(u, v):
    mean = v - 0.5
    return sqrt(tau/(2*pi)) * exp(-tau/2 * (u - mean)^2)
```

**Weighted regression:**
```python
slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))
```

**Weighted R²:**
```python
pred = slope * Tstar + intercept
w_norm = weights / weights.sum()
mean_lp = np.average(logit_p, weights=weights)
var_tot = np.average((logit_p - mean_lp)**2, weights=weights)
var_res = np.average((logit_p - pred)**2, weights=weights)
R2 = 1 - var_res / var_tot
```

**Verified results (γ=0.5, τ=2):**
| Grid            | Unwtd 1-R² |  Wtd 1-R² | Wtd slope |
|-----------------|------------|-----------|-----------|
| G=15 umax=4     |    0.191   |   0.078   |   0.521   |
| G=18 umax=4     |    0.195   |   0.083   |   0.545   |
| G=20 umax=5     |    0.230   |   0.085   |   0.543   |

Apply this to ALL 1-R² measurements: Fig 4A τ-sweep, Fig 4B γ-sweep.

---

## URGENT: RE-MEASURE 1-R² WITH WEIGHTED FORMULA FOR γ=1, 2, 4

The 1-R² values you reported for Task 2 are UNWEIGHTED and therefore WRONG:
- γ=1.0: you report 0.155 — true weighted value is much lower
- γ=2.0: you report 0.100 — same issue
- γ=4.0: you report 0.070 — same issue

For reference: γ=0.5 gives unweighted=0.230 but weighted=0.085.

**Re-measure each converged checkpoint using this code:**

```python
import numpy as np

def signal_density(u, v, tau):
    mean = v - 0.5
    return np.sqrt(tau/(2*np.pi)) * np.exp(-tau/2 * (u - mean)**2)

# For each triple (i,j,l):
w = 0.5 * (signal_density(u1,0,tau)*signal_density(u2,0,tau)*signal_density(u3,0,tau)
          + signal_density(u1,1,tau)*signal_density(u2,1,tau)*signal_density(u3,1,tau))

# Weighted regression:
slope, intercept = np.polyfit(Tstar, logit_p, 1, w=np.sqrt(weights))

# Weighted R²:
pred = slope * Tstar + intercept
mean_lp = np.average(logit_p, weights=weights)
var_tot = np.average((logit_p - mean_lp)**2, weights=weights)
var_res = np.average((logit_p - pred)**2, weights=weights)
weighted_1mR2 = 1 - (1 - var_res / var_tot)  # i.e. var_res / var_tot
```

**Do this for ALL checkpoint files:**
- posterior_v3_G20_umax5_*_g1.0_*.json
- posterior_v3_G20_umax5_*_g2.0_*.json
- posterior_v3_G20_umax5_*_g4.0_*.json

**Save updated summary:**
results/full_ree/fig4B_G20_gamma_sweep_weighted.json

**Also: still need γ=0.25 and γ=0.1 convergence runs to complete Fig 4B.**

## ALSO: FIG 5 FIX

The symmetric-triple method gives FR ≡ NL (correct but uninformative).
For Fig 5, use ASYMMETRIC triples instead:
- Fix u₁ = 1.0 (closest grid point)
- Fix u₂ = -1.0 (closest grid point)
- Vary u₃ from -3 to +3 (50 points)
- T* = τ(u₁ + u₂ + u₃) varies from -6 to +10
- At each u₃: compute p_FR = Λ(T*/3), p_NL, p_REE

This gives three DIFFERENT curves that show the PR gap.

---

## TODO: FIG 6B (posteriors) NEEDS HIGHER G OR UMAX=4

Current Fig 6B has visible artifacts from G=20 umax=5:
- Flat regions at extremes (μ₁≈0.324 for T*<-1, μ₂≈0.676 for T*>1)
- Step/kink at T*≈-1.1 where μ₂ jumps
- These are grid boundary effects, not real features

Options to fix:
1. Use G=18 umax=4 mp300 seed instead (tighter grid, no wasted tails)
2. Or use G=25 when it converges
3. Or clip T* range to [-0.5, 1.5] (transition zone only, no flat regions)
4. Or interpolate μ* more finely before evaluating

Currently has gray background (black!8) to flag as pending.
