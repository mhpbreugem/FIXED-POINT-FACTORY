# THEORY — Noise Traders Are Not a Primitive
# Author: Matthijs Breugem, Nyenrode Business University
# Target: Econometrica
# This file is the single source of truth for the theoretical framework.
# Claude Code: read this before any computation. Write results to numerics.md in the same folder.

---

## 1. MODEL

### 1.1 Asset
- Binary asset v ∈ {0,1}
- Prior P(v=1) = π₀ = 1/2
- Payoff: v

### 1.2 Agents
- K ≥ 3 groups (baseline K=3)
- Agent k receives private signal: s_k = v + ε_k, ε_k ~ N(0, 1/τ_k)
- Centered signal: u_k = s_k − 1/2
  - Under v=1: u_k ~ N(+1/2, 1/τ)
  - Under v=0: u_k ~ N(−1/2, 1/τ)
- Signal density: f_v(u) = √(τ/(2π)) exp(−τ/2 (u − v + 1/2)²)
- Likelihood ratio: f_1(u)/f_0(u) = exp(τu)
- Private posterior (prior only): μ_prior = Λ(τu) where Λ(z) = 1/(1+e^{-z})

### 1.3 Preferences
- CRRA utility: U(W) = W^{1−γ}/(1−γ)
- Special cases:
  - γ = 1: log utility U = log(W)
  - γ → ∞: CARA (exponential utility)
- Initial wealth W_k (baseline: equal, W=1)
- Zero net supply of the asset

### 1.4 Demands

**CARA demand** (linear in log-odds):
```
x_k = [logit(μ_k) − logit(p)] / γ
```
No wealth effects. Aggregation space = log-odds.

**CRRA demand** (nonlinear):
```
x_k = W_k(R_k − 1) / ((1−p) + R_k·p)
where R_k = exp((logit(μ_k) − logit(p)) / γ)
```
Has wealth effects. Aggregation space = probability.

**Log demand** (γ=1, closed form):
```
x_k = W_k(μ_k − p) / (p(1−p))
```
Market clearing price: p = Σ w_k·μ_k where w_k ∝ W_k.

### 1.5 Market Clearing
```
Σ_{k=1}^{K} x_k(μ_k, p) = 0
```
No noise traders. Zero supply.

### 1.6 Notation
- Λ(z) = logistic function = 1/(1+e^{-z})
- Λ⁻¹(p) = logit(p) = ln(p/(1−p))
- T* = Σ τ_k u_k = sufficient statistic
- f_v(u) = signal density under state v

---

## 2. KEY RESULTS

### Proposition 1: CARA ⟹ Full Revelation (FR)
For ANY K≥1, ANY heterogeneous (γ_k, τ_k, W_k):
CARA market clearing aggregates in log-odds space:
```
logit(p) = (1/K) Σ logit(μ_k) = (1/K) Σ τ_k u_k = T*/K
```
Price is a function of T* only. 1−R² = 0.
Extends to entire exponential family.

### Proposition 2: Non-CARA ⟹ Partial Revelation (PR)
Log utility market clearing:
```
p = (1/K) Σ Λ(τu_k) ≠ Λ((1/K) Σ τu_k)
```
by Jensen's inequality (Λ is nonlinear). 1−R² > 0.

### Proposition 3: Jensen Gap
```
Δp = −(τ³/48K)[U₃ − U₁³/K²] + O(τ⁵)
```
where U_n = Σ u_k^n. The gap is third-order in τ.

### Proposition 4: PR Survives at Full REE
When agents learn from prices (contour method), β_signal > 0 at the fixed point.
Proved numerically. See Section 5 for the method.

### Proposition 5: Positive Welfare
PR equilibrium has positive trade volume E[|x_k|] > 0.
FR equilibrium has zero trade (no-trade theorem).

### Proposition 6: Value of Information
Under CARA: V(τ) = 0 for all τ (signals worthless, price reveals all).
Under CRRA: V(τ) > 0, increasing in τ (signals have value).

### Proposition 7: Grossman-Stiglitz Resolution
Under CARA: net value of acquiring signal = V(τ) − c = −c < 0. Nobody acquires. Paradox.
Under CRRA: net value = V(τ) − c > 0 for c < V(τ). Agents acquire. Paradox resolved.
No noise traders needed.

### Proposition 8: Smooth Transition
1−R² is continuous and monotonically increasing as γ decreases from ∞.
CARA is a knife-edge: the limit γ→∞, not a robust feature.

---

## 3. AGGREGATION SPACE — THE CORE CONCEPT

The aggregation space is the mathematical space in which equilibrium prices combine private signals.

**CARA**: Demands are linear in logit(μ) − logit(p). Market clearing averages in log-odds:
```
logit(p) = Σ w_k · logit(μ_k)
```
This matches the Bayesian sufficient statistic T* = Σ τu_k. Full revelation.

**Non-CARA**: Demands are nonlinear. Market clearing averages in a different space (probability space for log utility):
```
p = Σ w_k · Λ(τu_k)    [log utility]
```
This does NOT match T*. The mismatch is the Jensen gap: Σ Λ(τu)/K ≠ Λ(Σ τu/K).

**The insight**: Full revelation requires the aggregation space to match the sufficient statistic space. Only CARA achieves this. Every other utility function aggregates in the wrong space.

---

## 4. THREE MECHANISMS FOR PARTIAL REVELATION

With heterogeneous agents, three sources of PR stack:

| Configuration | 1−R² | Mechanism |
|---|---|---|
| Equal γ, equal τ (CRRA) | 0.011 | Pure Jensen gap |
| Het γ=(1,3,10), equal τ | 0.247 | + heterogeneous risk aversion |
| Equal γ, het τ=(1,3,10) | 0.082 | + heterogeneous precision |
| Het γ + het τ aligned (low-γ = high-τ) | 0.100 | Stabilizing |
| Het γ + het τ opposed (low-γ = low-τ) | 0.211 | Destabilizing |
| Extreme opposed | 0.604 | Endogenous noise trader |

The "endogenous noise trader": an agent with low γ (aggressive) and low τ (uninformed) trades aggressively on weak signals. Same effect as exogenous noise traders, but arising from preferences.

---

## 5. THE CONTOUR METHOD — FULL REE SOLUTION

### 5.1 The Unknown
P[i,j,l] for i,j,l = 1,...,G. One price array, G³ scalars.

### 5.2 The Map Φ
P is the ONLY unknown. Everything else is a function of P. The equilibrium is:
```
P = Φ(P)
```
where Φ maps a price array to a new price array via:

**Step 1**: For each (i,j,l), read price p = P[i,j,l].

**Step 2**: For each agent k, extract her 2D slice of P:
- Agent 1: P[i,:,:]  (fix own signal, vary others)
- Agent 2: P[:,j,:]
- Agent 3: P[:,:,l]

**Step 3**: Trace the contour P = p through the slice:
- Pass A: sweep first axis on grid, root-find second axis off grid
- Pass B: sweep second axis on grid, root-find first axis off grid
- Each crossing contributes f_v(u_a) · f_v(u_b) to A_v

**Step 4**: Average the two passes:
```
A_v = ½(S_v^A + S_v^B)
```

**Step 5**: Bayes' rule with own signal:
```
μ_k = f_1(u_own) · A_1 / (f_0(u_own) · A_0 + f_1(u_own) · A_1)
```

**Step 6**: Market clearing:
```
Σ x_k(μ_k, p_new) = 0  →  P_new[i,j,l] = p_new
```

### 5.3 System of Equations per Realization (i,j,l)

| # | Name | Equation | Unknown | Count |
|---|---|---|---|---|
| D1 | Root-find (pass A, 3 agents) | P(...,u_jc,u_lc,...) = p | u_lc | 3G |
| D2 | Root-find (pass B, 3 agents) | P(...,u_jc,u_lc,...) = p | u_jc | 3G |
| D3-D5 | Contour integrals | A_v = ½[Σ f_v·f_v + Σ f_v·f_v] | A_v^(k) | 6 |
| D6 | Bayes' rule | μ_k = ... | μ_k | 3 |
| E1 | Market clearing | Σ x_k = 0 | P[i,j,l] | 1 |

D1–D6 are substituted out inside Φ. Only E1 is the equilibrium condition.
Total: G³ equations, G³ unknowns.

### 5.4 Solvers

**Picard iteration** (mimics learning):
```
P^{n+1} = α·Φ(P^n) + (1−α)·P^n
```
Damping α = 0.15–0.3. Symmetrize over all 6 permutations. Converges slowly.

**Anderson acceleration** (quasi-Newton):
Same Φ evaluations. Builds approximation from past residuals. Window m=6–8.
Converges in ~6 steps vs 30+ for Picard. Stalls at interpolation floor.

**Full Newton**: Stack all 12G⁴ + 10G³ equations as one system F(x)=0.
No sub-iteration, no damping. Requires Jacobian (Broyden or finite difference).

### 5.5 Why CARA → Straight Contour → FR
Under CARA: logit(p) = (1/K) Σ τu_k.
Level set: τu₂ + τu₃ = const — a STRAIGHT LINE.
Every crossing has the same T*. Agent extracts T* perfectly. → FR.

### 5.6 Why CRRA → Curved Contour → PR
Under CRRA: p ≈ (1/K) Σ Λ(τu_k).
Level set: Λ(τu₂) + Λ(τu₃) = const — CURVED (Λ nonlinear).
Different crossings have different T*. Agent cannot extract T*. → PR.

### 5.7 Root-Finding Detail
For no-learning initialization: P is defined analytically via market clearing.
No grid interpolation needed. Solve Σ x_k(Λ(τu_k), p) = 0 for any (u1,u2,u3).

For subsequent iterations: P stored on grid. Use linear interpolation along one axis.
Linear extrapolation beyond grid edges.

---

## 6. NUMERICAL RESULTS — NO-LEARNING

### 6.1 Smooth Transition Table (exact, G=20)

| γ | τ=0.5 | τ=1.0 | τ=2.0 |
|---|---|---|---|
| 0.1 | 0.146 | 0.145 | 0.137 |
| 0.3 | 0.044 | 0.070 | 0.090 |
| 0.5 | 0.016 | 0.038 | 0.062 |
| 1.0 | 0.004 | 0.013 | 0.029 |
| 3.0 | 0.000 | 0.002 | 0.006 |
| 10.0 | 0.000 | 0.000 | 0.001 |
| 100 | 0.000 | 0.000 | 0.000 |

### 6.2 1−R² vs τ (knife-edge figure data, G=10)

γ=0.25, 20 log-spaced τ from 0.1 to 10:
```
(0.1000,0.005112)(0.1274,0.010448)(0.1624,0.019278)(0.2069,0.031679)
(0.2637,0.046413)(0.3360,0.061466)(0.4281,0.075049)(0.5456,0.085972)
(0.6952,0.093733)(0.8859,0.098771)(1.1288,0.101453)(1.4384,0.103644)
(1.8330,0.108978)(2.3357,0.114996)(2.9764,0.116032)(3.7927,0.110778)
(4.8329,0.103607)(6.1585,0.092867)(7.8476,0.087005)(10.0000,0.085672)
```

γ=1.0:
```
(0.1000,0.000019)(0.1274,0.000048)(0.1624,0.000118)(0.2069,0.000282)
(0.2637,0.000643)(0.3360,0.001375)(0.4281,0.002708)(0.5456,0.004825)
(0.6952,0.007737)(0.8859,0.011336)(1.1288,0.015633)(1.4384,0.020802)
(1.8330,0.026948)(2.3357,0.033804)(2.9764,0.040475)(3.7927,0.045788)
(4.8329,0.049329)(6.1585,0.048715)(7.8476,0.051070)(10.0000,0.057204)
```

γ=5.0 (NEEDS RECOMPUTATION AT γ=4.0):
```
(0.1000,0.000001)(0.1274,0.000002)(0.1624,0.000006)(0.2069,0.000013)
(0.2637,0.000028)(0.3360,0.000058)(0.4281,0.000117)(0.5456,0.000220)
(0.6952,0.000384)(0.8859,0.000624)(1.1288,0.000959)(1.4384,0.001423)
(1.8330,0.002063)(2.3357,0.002924)(2.9764,0.004025)(3.7927,0.005314)
(4.8329,0.006659)(6.1585,0.007431)(7.8476,0.010594)(10.0000,0.012000)
```

NOTE: G=10 has grid artifacts at high τ (especially γ=4 at τ>7).
NEED: Recompute at G=20 with 30 points. See TASKS below.

### 6.3 1−R² vs K (number of agents, G=5, τ=1)

γ=0.25:
```
(3,0.113220)(4,0.130804)(5,0.129275)(6,0.123143)(7,0.118312)(8,0.112108)(9,0.105452)(10,0.098957)
```

γ=1.0:
```
(3,0.012076)(4,0.010444)(5,0.009121)(6,0.008095)(7,0.007285)(8,0.006628)(9,0.006084)(10,0.005626)
```

γ=5.0 (NEEDS RECOMPUTATION AT γ=4.0):
```
(3,0.000714)(4,0.000624)(5,0.000546)(6,0.000484)(7,0.000433)(8,0.000392)(9,0.000357)(10,0.000329)
```

CARA: identically 0 for all K.

---

## 7. NUMERICAL RESULTS — FULL REE

### 7.1 Converged Fixed Point at (u₁,u₂,u₃) = (1,−1,1), G=5, τ=2

**CARA** (analytical: logit(p) = T*/K = 2/3, p = Λ(2/3) = 0.6608):
Converged: price = 0.8808 = Λ(T*) = Λ(2). All posteriors = 0.8808. FR.

**CRRA (γ=0.5)**:
Price = 0.9077. μ₁ = 0.9185, μ₂ = 0.8889, μ₃ = 0.9185. Agents disagree. PR.

| | Prior | CARA | CRRA |
|---|---|---|---|
| μ₁ (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| μ₂ (u=−1) | 0.1192 | 0.8808 | 0.8889 |
| μ₃ (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| price | 0.648 | 0.8808 | 0.9077 |

### 7.2 Anderson Convergence
- ||G||∞ drops from 0.33 to 0.002 in ~6 iterations
- Stalls at 0.002 = G=5 interpolation floor
- Values stable from iteration ~10

### 7.3 Picard Convergence (α=0.3, 30 iterations)
- Price: 0.648 → 0.909
- Posteriors barely change across iterations
- Price overshoots at iteration 16 (Δp flips sign)

---

## 8. HARA AND THE KNIFE-EDGE

HARA: U(W) = (1−γ)/γ · (aW/(1−γ) + b)^γ
Risk tolerance: T(W) = aW/(1−γ) + b (linear in W)

Special cases:
- CARA: a=0, T(W)=b (constant)
- CRRA: b=0, T(W)=W/γ
- Log: b=0, γ→0

With homogeneous agents and equal wealth, the parameters (a,b) cancel in market clearing.
The equilibrium price depends ONLY on γ. A 2D contour plot in (a,b) at fixed γ is flat.

The honest knife-edge plot is 1−R² vs τ with multiple γ curves. CARA (γ→∞) is the only
curve at zero. All finite γ are above zero. This is done — see Section 6.2.

---

## 9. PAPER STRUCTURE

1. Introduction (3pp)
2. Model (2pp)
3. No-Learning Benchmark: Props 1-3 (3pp) — Figures 1-2
4. Full REE: Contour method, Prop 4 (5pp) — Figures 3-5
5. Welfare & Grossman-Stiglitz: Props 5-7 (4pp) — Figures 7-9
6. Three Mechanisms (2pp) — Figure 6
7. Discussion (2pp)
8. Conclusion (1pp)

---

## 10. FIGURES

### Figure 1: Smooth Transition Table
- Heatmap or styled table of 1−R² by (γ, τ)
- Data: Section 6.1
- Status: DATA DONE. Need final LaTeX table.

### Figure 2: Knife-Edge (1−R² vs τ)
- x: τ (log scale, 0.1 to 10), y: 1−R²
- 4 curves: γ=0.25 green solid, γ=1 red dashed, γ=4 blue dotted, CARA black dashdotted
- BC20 pgfplots style (see Section 12)
- Status: DONE at G=10. NEED G=20 recomputation (30 log-spaced τ points).

### Figure 3: CARA vs CRRA Contour
- Side-by-side: straight (CARA) vs curved (CRRA) level sets
- Agent 1 at u₁=1, price at (1,−1,1)
- Color by T* at each crossing
- Status: NEED G≥100.

### Figure 4: Converged Posteriors
- Table: prior, CARA posterior, CRRA posterior for all 3 agents
- Data: Section 7.1
- Status: DONE at G=5. NEED larger G for paper.

### Figure 5: Convergence Paths
- ||P−Φ(P)||∞ vs iteration (log scale)
- Picard vs Anderson
- Status: DONE at G=5.

### Figure 6: Three Mechanisms Table
- 1−R² by configuration
- Data: Section 4
- Status: DATA DONE.

### Figure 7: Trade Volume
- E[|x_k|] vs γ
- CARA: zero. CRRA: positive.
- Status: NEED computation at converged REE.

### Figure 8: Value of Information
- V(τ) vs τ for several γ
- Status: NEED computation.

### Figure 9: Grossman-Stiglitz Resolution
- Net value of acquiring vs cost c
- Status: NEED computation.

### Figure 10: 1−R² vs K (Number of Agents)
- x: K (3 to 10), y: 1−R²
- Same 4 curves
- Data: Section 6.3
- Status: DONE at G=5.

---

## 11. TASKS FOR CLAUDE CODE

### HIGH PRIORITY
1. **Recompute knife-edge data at G=20**: 30 log-spaced τ from 0.1 to 10, γ=0.25, 1.0, 4.0.
   Output as pgfplots coordinates. Save to Google Drive folder CLAUDE (ID: 19XlmnFAON2qufCeYUsRT6UBTjFUZqjo3).
   Algorithm: Section 5 steps, no-learning only (no iteration). See Section 6.2 for the code pattern.

2. **Converged REE at multiple (γ,τ)**: Run Anderson at G=20 for (γ,τ) ∈ {0.25,1.0,4.0} × {0.5,1.0,2.0}.
   Report: converged 1−R², ||F||∞, number of iterations.

3. **CARA vs CRRA contour figure**: G=100 or more. Extract level set at (1,−1,1).
   Report: crossing coordinates for both CARA and CRRA.

### MEDIUM PRIORITY
4. Trade volume E[|x_k|] at converged REE for γ = 0.1 to 100.
5. Value of information V(τ) for τ = 0 to 5, γ = 0.25, 1, 4.
6. K-agent sweep at G=10 (more precise than G=5).

### LOW PRIORITY
7. Three mechanisms table at G=20.
8. Full Newton (Broyden) solver comparison.

---

## 12. FIGURE STYLE — BC20

All pgfplots figures must match this style exactly:

```latex
\definecolor{red}{rgb}{0.7, 0.11, 0.11}
\definecolor{blue}{rgb}{0.0, 0.20, 0.42}
\definecolor{green}{rgb}{0.11, 0.35, 0.02}

\begin{axis}[
  legend style={draw=none, legend pos=north west, font=\footnotesize},
  ticklabel style={/pgf/number format/fixed, /pgf/number format/precision=5},
  scaled ticks=false,
  xticklabel style={/pgf/number format/1000 sep=, font=\scriptsize},
  width=8cm, height=8cm,
]
```

Line styles (in this order):
1. green, solid, very thick
2. red, dashed, very thick
3. blue, dotted, very thick
4. black, dashdotted, ultra thick (CARA baseline)

All curves use `smooth` for pgfplots interpolation.
ymin=-0.001 so CARA line is visible above x-axis.

---

## 13. PARAMETER DEFAULTS

- τ = 2.0 (baseline signal precision)
- γ = 0.5 (CRRA main case)
- W = 1.0 (equal wealth)
- K = 3 (number of agents)
- G = 5 (debug), 10 (quick), 20 (paper), 100+ (contour figure)
- u_grid = [−4, +4]
- Anderson window m = 6–8
- Picard damping α = 0.15–0.3

---

## 14. GITHUB REPO

Branch **REZN** of `mhpbreugem/REZN`.
```
git clone -b REZN https://github.com/mhpbreugem/REZN.git
```
Figures in `figures/`. Each `.tex` references a sibling `.pdf` via `\includegraphics`.
Include BOTH when copying figures into the paper.

---

## 15. GOOGLE DRIVE BRIDGE

Folder: Research/REZN/CLAUDE
- Folder ID: 19XlmnFAON2qufCeYUsRT6UBTjFUZqjo3
- Parent (REZN) ID: 11C_ewj3iyDkThWIBWFxW0ghqontj-uNx

Protocol:
- Claude Code writes `numerics.md` here with computation results
- This chat reads `numerics.md` and updates figures/paper
- This chat writes `theory.md` here with model updates
- Claude Code reads `theory.md` before computing

---

## 16. FOOTNOTES AND REMARKS FOR THE PAPER

### Footnote: Zero Net Supply
Add a footnote in Section 2 (Model) where zero supply is introduced. Text:

"Zero net supply is the strongest version of our result: partial revelation arises without any exogenous friction — no supply noise, no noise traders, no demand shocks. CRRA demand is self-bounding: as wealth approaches zero in either state, marginal utility diverges, preventing agents from taking unbounded positions. Market clearing at z-bar = 0 always admits a solution by the intermediate value theorem, with some agents long and others short. Introducing positive deterministic supply z-bar > 0 shifts the price level but does not affect the information content: under CARA, agents still extract T* perfectly from p (since z-bar is known); under CRRA, partial revelation persists. Our choice of z-bar = 0 thus isolates the pure preference channel."
