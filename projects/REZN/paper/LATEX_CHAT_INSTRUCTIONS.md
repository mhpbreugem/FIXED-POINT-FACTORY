# SYSTEM INSTRUCTIONS — LaTeX Paper Chat
# Paste this at the start of a new Claude chat to write the paper.

---

## YOUR ROLE
You are writing an Econometrica paper titled "Noise Traders Are Not a Primitive" by Matthijs Breugem (Nyenrode Business University). You have full read/write access to the GitHub repo containing figures, data, and context files.

## GITHUB ACCESS
- Repo: mhpbreugem/REZN (may be private)
- Token: [PASTE_YOUR_TOKEN_HERE]
- Clone: `git clone https://[PASTE_YOUR_TOKEN_HERE]@github.com/mhpbreugem/REZN.git`
- You have full read+write access. Push to main.
- Configure git: `git config user.email "claude@anthropic.com" && git config user.name "Claude"`
- IMPORTANT: Never commit files containing the token. GitHub will block the push.

## FIRST THING TO DO
1. Clone the repo
2. Read CHAT_MEMORY.md — this is the living memory from the theory/content chat. It has all decisions, results, figure status, and open questions.
3. Read contour.md — explains the numerical method in plain language
4. Read project_summary.txt — the original project spec
5. Look at figures/ — all .tex and .pdf files. Some are done, some are placeholders (yellow background).

## THE PAPER

### Core Claim
CRRA preferences produce partial revelation (PR) of information through prices WITHOUT noise traders. CARA is a knife-edge: only exponential utility gives full revelation (FR). The mechanism is the aggregation space mismatch — CARA aggregates in log-odds (matching the Bayesian sufficient statistic), non-CARA aggregates in probability space (mismatching it).

### Model
- Binary asset v in {0,1}, prior P(v=1) = 1/2
- K >= 3 agent groups
- Signal: s_k = v + epsilon_k, epsilon_k ~ N(0, 1/tau_k)
- Centered signal: u_k = s_k - 1/2
- Signal density: f_v(u) = sqrt(tau/(2pi)) exp(-tau/2 (u - v + 1/2)^2)
- Private posterior: mu = Lambda(tau*u) where Lambda is the logistic function
- CRRA utility: U(W) = W^{1-gamma}/(1-gamma)
- CRRA demand: x_k = W(R-1)/((1-p)+R*p), R = exp((logit(mu)-logit(p))/gamma)
- CARA demand: x_k = (logit(mu)-logit(p))/gamma (linear in log-odds)
- Log demand (gamma=1): x_k = W(mu-p)/(p(1-p))
- Zero net supply, NO noise traders
- Notation: Lambda(z) = 1/(1+e^{-z}), logit(p) = ln(p/(1-p)), T* = sum tau_k u_k

### Propositions

**Proposition 1 (CARA = FR):** For any K >= 1, any heterogeneous (gamma_k, tau_k, W_k): CARA market clearing aggregates in log-odds. logit(p) = (1/K) sum logit(mu_k) = T*/K. Price is a function of T* only. 1-R^2 = 0. Extends to entire exponential family.

**Proposition 2 (Non-CARA = PR):** Log utility: p = (1/K) sum Lambda(tau*u_k) != Lambda((1/K) sum tau*u_k) by Jensen's inequality. 1-R^2 > 0.

**Proposition 3 (Jensen Gap):** Delta_p = -(tau^3/48K)[U_3 - U_1^3/K^2] + O(tau^5) where U_n = sum u_k^n.

**Proposition 4 (PR at REE):** When agents learn from prices (contour method), beta_signal > 0 at the fixed point. Proved numerically.

**Proposition 5 (Positive Welfare):** PR equilibrium has positive trade volume. FR has zero trade.

**Proposition 6 (Value of Info):** V(tau) = 0 under CARA. V(tau) > 0 under CRRA, increasing in tau.

**Proposition 7 (GS Resolution):** Under CARA: net value = -c < 0 (paradox). Under CRRA: net value = V(tau) - c > 0 for c < V(tau) (resolved).

**Proposition 8 (Smooth Transition):** 1-R^2 is continuous and monotonically increasing as gamma decreases from infinity. CARA is knife-edge.

### Paper Structure
1. Introduction (3pp) — motivation, contribution, literature
2. Model (2pp) — setup, demands, aggregation space concept
3. No-Learning Benchmark (3pp) — Props 1-3, Figures 1-2
4. Full REE (5pp) — contour method, Prop 4, Figures 3-5
5. Welfare & Grossman-Stiglitz (4pp) — Props 5-7, Figures 7-9
6. Three Mechanisms (2pp) — Figure 6
7. Discussion (2pp) — literature connections, extensions
8. Conclusion (1pp)

### Figures (all in figures/ directory)

**DONE (use these directly):**
- fig_knife_edge.tex/pdf — 1-R^2 vs tau, G=20, 35 log-spaced points. Main result.
- fig1_smooth_transition.tex/pdf — Heatmap of 1-R^2 by (gamma, tau)
- fig1_table.tex/pdf — LaTeX table version of same data
- fig3_contour.tex/pdf — CARA vs CRRA contour side-by-side. Central figure.
- fig5_convergence.tex/pdf — Picard vs Anderson convergence paths
- fig_knife_edge_K.tex/pdf — 1-R^2 vs number of agents K
- fig_knife_edge_lognormal.tex/pdf — Lognormal signal variant

**PLACEHOLDER (yellow background, being computed by Claude Code):**
- fig4_posteriors — converged REE posteriors table
- fig6_mechanisms — three mechanisms table
- fig7_volume — trade volume vs gamma
- fig8_value_info — value of information V(tau) vs tau
- fig9_GS — Grossman-Stiglitz resolution
- fig10_K_agents — K agents (alt version)

Include placeholders with a note like "Figure forthcoming" and \includegraphics pointing to the existing PDF. They'll be updated in place.

### Figure Style (BC20 pgfplots)
All figures use this exact style:
```latex
\definecolor{red}{rgb}{0.7, 0.11, 0.11}
\definecolor{blue}{rgb}{0.0, 0.20, 0.42}
\definecolor{green}{rgb}{0.11, 0.35, 0.02}
```
8cm x 8cm, legend draw=none footnotesize, smooth curves.
Line order: green solid, red dashed, blue dotted, black dashdotted.
Very thick CRRA, ultra thick CARA. ymin=-0.001.

### Key Numerical Results

**No-learning smooth transition (G=20):**

| gamma | tau=0.5 | tau=1.0 | tau=2.0 |
|-------|---------|---------|---------|
| 0.1   | 0.146   | 0.145   | 0.137   |
| 0.3   | 0.044   | 0.070   | 0.090   |
| 0.5   | 0.016   | 0.038   | 0.062   |
| 1.0   | 0.004   | 0.013   | 0.029   |
| 3.0   | 0.000   | 0.002   | 0.006   |
| 10.0  | 0.000   | 0.000   | 0.001   |
| 100   | 0.000   | 0.000   | 0.000   |

**Converged REE at (u1,u2,u3)=(1,-1,1), G=5, tau=2:**

| | Prior | CARA | CRRA(0.5) |
|---|---|---|---|
| mu1 (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| mu2 (u=-1) | 0.1192 | 0.8808 | 0.8889 |
| mu3 (u=+1) | 0.8808 | 0.8808 | 0.9185 |
| price | 0.648 | 0.8808 | 0.9077 |

CARA: all posteriors equal (FR). CRRA: agents disagree (PR).

### The Contour Method (Section 4)

The equilibrium is P = Phi(P) where P[i,j,l] is the price for signal realization (u_i, u_j, u_l). G^3 unknowns, G^3 equations.

Phi computes:
1. Each agent extracts her 2D slice of P
2. Traces the contour P = p through the slice (2-pass: sweep one axis, root-find the other)
3. Integrates signal densities along the contour -> A_v
4. Bayes' rule: mu_k = f1(own)*A1 / (f0(own)*A0 + f1(own)*A1)
5. Market clearing: sum x_k(mu_k, p_new) = 0

Under CARA: contour is a straight line (same T* everywhere) -> agent extracts T* -> FR.
Under CRRA: contour is curved (T* varies) -> agent cannot extract T* -> PR.

P is the ONLY unknown. D1-D6 are inside Phi, not separate equations.

### HARA Note
HARA: T(W) = aW/(1-gamma) + b. With homogeneous agents, (a,b) cancel in market clearing. The price depends only on gamma. A 2D (a,b) plot would be flat — not informative.

### Grossman-Stiglitz (Section 5)
Under CARA: FR -> zero trade -> signals worthless -> V(tau)=0 -> nobody acquires -> paradox.
Under CRRA: PR -> positive trade -> signals valuable -> V(tau)>0 -> agents acquire -> paradox resolved. No noise traders needed.

### Three Mechanisms (Section 6)

| Configuration | 1-R^2 | Mechanism |
|---|---|---|
| Equal gamma, equal tau | 0.011 | Pure Jensen gap |
| Het gamma=(1,3,10), equal tau | 0.247 | + het risk aversion |
| Equal gamma, het tau=(1,3,10) | 0.082 | + het precision |
| Het gamma + het tau aligned | 0.100 | Low-gamma = high-tau (stabilizing) |
| Het gamma + het tau opposed | 0.211 | Low-gamma = low-tau (destabilizing) |
| Extreme opposed | 0.604 | Endogenous noise trader |

### Key Literature
- DeMarzo-Skiadas 1998 JET: CARA-normal FR uniqueness
- Breon-Drish 2015 RFS: exponential family + CARA with noise
- Albagli-Hellwig-Tsyvinski 2024 JF: non-Gaussian REE
- Vives 2011 Ecta: private values route
- Grossman-Stiglitz 1980: the paradox
- Breugem-Buss: projection method for non-standard REE

### Writing Style
- Econometrica standard: formal, precise, no hand-waving
- Define everything before using it
- Proofs in appendix where possible, sketch in main text
- Figures referenced as Figure X, not "the figure above"
- Each section opens with a one-paragraph preview
- Notation section in Section 2

### LaTeX Setup
Use standard Econometrica class if available, otherwise article class with:
```latex
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{booktabs}
\usepackage{natbib}

\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{definition}{Definition}
```

Figures: use \includegraphics{figures/fig_name.pdf} pointing to the repo's figures/ directory.

### Workflow
1. Write sections as separate .tex files or one main file
2. Include figures from figures/ directory
3. Push to the repo when done
4. The author will review and iterate

---

## SUMMARY
Read CHAT_MEMORY.md from the repo first — it has everything. Write the paper. Push to the repo. Use the done figures directly. Mark placeholder figures as forthcoming. Ask the author if anything is unclear about the economics or the narrative.

---

## FOOTNOTES TO INCLUDE

### Zero Net Supply (Section 2, where z-bar=0 is stated)
Add a footnote explaining: "Zero net supply is the strongest version of our result: partial revelation arises without any exogenous friction. CRRA demand is self-bounding (U' -> inf as W -> 0 prevents unbounded positions). Market clearing at z-bar = 0 always admits a solution by IVT. Positive deterministic supply z-bar > 0 shifts the price level but not information content: under CARA, agents still extract T* (z-bar is known); under CRRA, PR persists. Our choice isolates the pure preference channel."
