# FIGURES SPECIFICATION — Noise Traders Are Not a Primitive
# For Claude Code: read theory.md first, then follow these specs exactly.
# Save all outputs to figures/ in the REZN branch of mhpbreugem/REZN.

---

## STYLE — BC20 pgfplots (applies to ALL figures)

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

All curves use `smooth`. ymin=-0.001 so CARA line is visible above x-axis.

---

## SHARED ALGORITHM: No-Learning 1−R²

Used by Figures 1, 2, 6, 10. Given (γ, τ, G, K=3):

1. `u = linspace(-4, 4, G)`
2. `μ_i = Λ(τ · u_i)` for each grid point, where `Λ(z) = 1/(1+exp(-z))`
3. For each (i,j,l): solve `Σ x_k(μ_k, p) = 0` for p
   - CRRA demand: `x = (R-1)/((1-p)+R·p)`, `R = exp((logit(μ)-logit(p))/γ)`
   - Use brentq on [0.002, 0.998]
4. Store P[i,j,l] = p. Clip to [0.002, 0.998].
5. T* = τ(u_i + u_j + u_l) for each (i,j,l)
6. Weights: `w = 0.5·(Π_k f_1(u_k) + Π_k f_0(u_k))`
   - `f_v(u) = √(τ/(2π)) exp(-τ/2·(u - v + 1/2)²)`
7. Weighted regression of logit(P) on T* with weights w
8. Report 1−R². Filter: only 1e-4 < P < 1-1e-4.

---

## FIGURE 1: Smooth Transition Table

Section §3. LaTeX table, not a plot.

Parameters:
- G = 20
- γ = 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0
- τ = 0.5, 1.0, 2.0, 3.0

Output: LaTeX table, γ rows, τ columns, values to 3 decimal places.
γ=100 row should be all zeros (CARA verification).

Reference (G=20):

| γ | τ=0.5 | τ=1.0 | τ=2.0 |
|---|---|---|---|
| 0.1 | 0.146 | 0.145 | 0.137 |
| 0.5 | 0.016 | 0.038 | 0.062 |
| 1.0 | 0.004 | 0.013 | 0.029 |
| 10.0 | 0.000 | 0.000 | 0.001 |

---

## FIGURE 2: Knife-Edge (1−R² vs τ)

Section §3. The main visual result.

Parameters:
- G = 20
- γ = 0.25, 1.0, 4.0
- τ = 30 log-spaced points from 0.1 to 10: `np.logspace(log10(0.1), log10(10), 30)`

Output: pgfplots coordinates for each γ:
```
% gamma = 0.255
    (0.1000,0.005112)(0.1274,0.010448)...
```

Full .tex file:
```latex
\documentclass[border=2mm]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{red}{rgb}{0.7, 0.11, 0.11}
\definecolor{blue}{rgb}{0.0, 0.20, 0.42}
\definecolor{green}{rgb}{0.11, 0.35, 0.02}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
  legend style={draw=none,legend pos=north west,font=\footnotesize},
  ticklabel style={/pgf/number format/fixed,/pgf/number format/precision=5},
  scaled ticks=false,
  ymin=-0.001, ymax=0.13, ylabel={$1-R^2$},
  xmin=0.08, xmax=10, xmode=log,
  xtick={0.1, 0.2, 0.5, 1, 2, 5, 10},
  xticklabels={0.1, 0.2, 0.5, 1, 2, 5, 10},
  xticklabel style={/pgf/number format/1000 sep=,font=\scriptsize},
  width=8cm, height=8cm,
  title={Signal precision ($\tau$)}]

\addplot[very thick,color=green,smooth] coordinates {REPLACE_GAMMA_0.255};
\addplot[very thick,color=red,dashed,smooth] coordinates {REPLACE_GAMMA_1.0};
\addplot[very thick,color=blue,dotted,smooth] coordinates {REPLACE_GAMMA_4.0};
\addplot[ultra thick,color=black,dashdotted] coordinates {(0.08,0)(10,0)};
\legend{$\gamma = 0.25$, $\gamma = 1$, $\gamma = 4$, CARA}
\end{axis}
\end{tikzpicture}
\end{document}
```

Existing G=10 data has grid artifacts at high τ for γ=4. G=20 should fix this.

---

## FIGURE 3: CARA vs CRRA Contour

Section §4.1. Central theoretical figure. Two side-by-side panels.

Setup:
- Realization (u₁, u₂, u₃) = (1, -1, 1)
- Agent 1 fixes u₁ = 1, observes price p₀ = P(1, -1, 1)
- Contour: {(u₂, u₃) : P(1, u₂, u₃) = p₀}

Algorithm (no-learning prices, no iteration needed):
1. For each u₂ in a fine sweep (500 points, -4 to 4):
   - Root-find u₃ such that market clearing at (1, u₂, u₃) gives price = p₀
   - The market clearing is solved directly: `Σ x_k(Λ(τu_k), p₀) = 0` → solve for u₃
   - This is a 1D root-find in u₃, no grid interpolation
2. Record (u₂, u₃, T*) where T* = τ(1 + u₂ + u₃)

Do twice:
- Left panel: CARA demands. Contour = straight line. All T* identical.
- Right panel: CRRA γ=0.5. Contour = curved. T* varies.

Parameters:
- τ = 2.0
- γ = 0.5 for CRRA, any γ for CARA (result independent of γ)
- 500 crossing points
- No grid needed — everything is analytical via market clearing

Output: two CSV files or coordinate lists: (u₂, u₃, T*) per panel.

Colormap: blue (low T*) to red (high T*).
Mark actual realization (-1, 1) with a star.

---

## FIGURE 4: Converged Posteriors

Section §4.3. A LaTeX table.

Run full contour method (Anderson) to convergence.

Parameters:
- (u₁, u₂, u₃) = (1, -1, 1)
- τ = 2.0, γ = 0.5 (CRRA) and explicit CARA
- G = 20
- Anderson: m=8, run until ||G||∞ < 1e-6 or 50 iterations

The contour method (Φ map) — full algorithm in theory.md Section 5.2:
1. Init P from no-learning prices
2. For each (i,j,l): extract agent slices, 2-pass contour, posteriors, market clear → P_new
3. Symmetrize over 6 permutations
4. Anderson mixing
5. Repeat

Output table:

| | Prior | CARA posterior | CRRA posterior |
|---|---|---|---|
| μ₁ (u₁=+1) | | | |
| μ₂ (u₂=-1) | | | |
| μ₃ (u₃=+1) | | | |
| price | | | |
| ||F||∞ | | | |

Reference (G=5): CARA all = 0.8808, CRRA μ₁=0.9185, μ₂=0.8889, price=0.9077

---

## FIGURE 5: Convergence Paths

Section §4.3 or Appendix. Byproduct of Figure 4.

Run both Picard (α=0.3) and Anderson (m=8) for 50 iterations. Record ||P - Φ(P)||∞ each step.

Parameters: same as Figure 4.

Output: pgfplots with log y-axis:
```latex
\begin{axis}[ymode=log, xlabel={Iteration}, ylabel={$\|P - \Phi(P)\|_\infty$}, ...]
\addplot[very thick,color=green] coordinates {...};       % Picard
\addplot[very thick,color=red,dashed] coordinates {...};  % Anderson
\legend{Picard ($\alpha=0.3$), Anderson ($m=8$)}
\end{axis}
```

---

## FIGURE 6: Three Mechanisms Table

Section §6. LaTeX table.

Same algorithm as Figure 1 but with per-agent (γ_k, τ_k).

Configurations:
1. γ=(1,1,1), τ=(2,2,2): pure Jensen gap
2. γ=(1,3,10), τ=(2,2,2): het risk aversion
3. γ=(1,1,1), τ=(1,3,10): het precision
4. γ=(1,3,10), τ=(10,3,1): aligned (low-γ = high-τ)
5. γ=(1,3,10), τ=(1,3,10): opposed (low-γ = low-τ)
6. γ=(0.1,3,10), τ=(0.5,3,10): extreme opposed

Per-agent demands: agent k uses her own γ_k.
Per-agent signals: `μ_k = Λ(τ_k · u_k)`, `f_v(u) uses τ_k`.
T* = Σ τ_k · u_k.

G = 20. Output: LaTeX table with 1−R² per row.

---

## FIGURE 7: Trade Volume

Section §5. Requires converged REE.

At converged P, compute:
```
E[|x_k|] = Σ_{i,j,l} w(i,j,l) · |x_k(μ_k(i,j,l), P[i,j,l])|
```
where μ_k is the converged posterior and w is the signal density weight.

Parameters:
- τ = 2.0
- γ = 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0
- G = 15 (or 20)
- Anderson to convergence at each γ

Output: pgfplots coordinates (γ, E[|x|]).
```latex
\begin{axis}[xmode=log, xlabel={$\gamma$}, ylabel={$E[|x_k|]$}, ...]
\addplot[very thick,color=red] coordinates {...};
\end{axis}
```

CARA (γ=100) should give ≈ 0.

---

## FIGURE 8: Value of Information

Section §5.

V(τ) = E[U(informed)] - E[U(uninformed)]

Compute expected utility integrating over signals AND true state:
```
E[U] = Σ_{i,j,l} Σ_{v=0,1} 0.5 · f_v(u_i)·f_v(u_j)·f_v(u_l) · U(W + x_k·(v - P[i,j,l]))
```

Uninformed: x=0, so E[U_uninformed] = U(W) = U(1). For log: U(1) = 0.

Parameters:
- γ = 0.25, 1.0, 4.0
- τ = 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0
- G = 15

Output: three curves V(τ) vs τ, same BC20 style.

---

## FIGURE 9: Grossman-Stiglitz Resolution

Section §5. Derived from Figure 8.

For fixed τ = 2:
- x-axis: c (cost of signal), 0 to max(V(τ))
- y-axis: V(τ) - c
- Curves for γ = 0.25, 1.0, 4.0, 100.0 (CARA)

CARA: V=0, so line = -c (always negative).
CRRA: V>0, crosses zero at c* = V(τ).

Output: pgfplots with horizontal line at y=0.

---

## FIGURE 10: 1−R² vs K (Number of Agents)

Section §3. Uses symmetry trick for efficiency.

```python
from itertools import combinations_with_replacement
from math import factorial

for combo in combinations_with_replacement(range(G), K):
    counts = [combo.count(i) for i in range(G)]
    mult = factorial(K)
    for c in counts: mult //= factorial(c)
    # compute price, T*, weight for this combo
    # multiply everything by mult
```

Configs: C(G+K-1, K). At G=5, K=10: only 1001 configs.

For γ=1 (log): closed form p = mean(μ_k), no brentq.

Parameters:
- G = 5
- τ = 1.0
- γ = 0.25, 1.0, 4.0
- K = 3, 4, 5, 6, 7, 8, 9, 10

Output: pgfplots coordinates.
```latex
\begin{axis}[
  xmin=2.5, xmax=10.5, xtick={3,...,10},
  ymin=-0.001, ymax=0.15,
  xlabel={Number of agents ($K$)}, ylabel={$1-R^2$}, ...]
\addplot[very thick,color=green,smooth] coordinates {...};  % γ=0.25
\addplot[very thick,color=red,dashed,smooth] coordinates {...};  % γ=1
\addplot[very thick,color=blue,dotted,smooth] coordinates {...};  % γ=4
\addplot[ultra thick,color=black,dashdotted] coordinates {(2.5,0)(10.5,0)};
\legend{$\gamma = 0.25$, $\gamma = 1$, $\gamma = 4$, CARA}
\end{axis}
```

Reference (G=5):
- γ=0.25: (3,0.113)(4,0.131)(5,0.129)(6,0.123)(7,0.118)(8,0.112)(9,0.105)(10,0.099)
- γ=1.0: (3,0.012)(4,0.010)(5,0.009)(6,0.008)(7,0.007)(8,0.007)(9,0.006)(10,0.006)
- γ=4.0: (3,0.001)(4,0.001)(5,0.001)(6,0.000)(7,0.000)(8,0.000)(9,0.000)(10,0.000)

---

## PRIORITY ORDER

1. Figure 2 (knife-edge, G=20) — most important
2. Figure 3 (contour, G≥100) — central theoretical figure
3. Figure 1 (table, G=20) — quick
4. Figure 10 (K agents, G=5) — have data, verify
5. Figure 4 (posteriors, G=20) — needs contour iteration
6. Figure 5 (convergence, G=20) — byproduct of 4
7. Figure 7 (trade volume) — needs converged REE at many γ
8. Figure 8 (value of info) — careful integration
9. Figure 9 (GS resolution) — derived from 8
10. Figure 6 (three mechanisms) — heterogeneous agent code

Save all .tex and .pdf to figures/ in the REZN branch of mhpbreugem/REZN.
