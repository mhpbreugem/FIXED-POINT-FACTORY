# PAPER ADDITIONS — Content to insert into main.tex
# Ordered by insertion point.

---

## 1. Definition of 1−R² (insert in Section 3, before Proposition 1)

```latex
\begin{definition}[Revelation deficit]\label{def:deficit}
The revelation deficit of a price function $p(u_{1},\ldots,u_{K})$ relative
to the Bayesian sufficient statistic $\Tstar=\sum_{k}\tau_{k}u_{k}$ is
\begin{equation}
   1 - R^{2} \;\equiv\; 1 - \frac{\bigl[\Cov_{w}(\logit p,\,\Tstar)\bigr]^{2}}
   {\Var_{w}(\logit p)\,\cdot\,\Var_{w}(\Tstar)},
\label{eq:deficit}
\end{equation}
where the weighted moments use $w(u_{1},\ldots,u_{K})=\tfrac12\bigl[\prod_{k}f_{1}(u_{k})
+\prod_{k}f_{0}(u_{k})\bigr]$, the ex ante signal density averaged over the
two states. The deficit equals zero if and only if $\logit p$ is an affine
function of $\Tstar$ (full revelation), and is bounded above by one.
\end{definition}
```

---

## 2. Existence/uniqueness lemma (insert in Section 2.3, after eq:no-learning)

```latex
\begin{lemma}[Existence and uniqueness of the no-learning equilibrium]\label{lem:existence}
For any $\gamma\in(0,\infty]$, $K\geq 1$, $\tau_{k}>0$, $W_{k}>0$, and any
posterior vector $(\mu_{1},\ldots,\mu_{K})\in(0,1)^{K}$, the market-clearing
equation $\sum_{k}x_{k}(\mu_{k},p)=0$ has a unique solution $p^{\star}\in(0,1)$.
\end{lemma}

\begin{proof}
Define $Z(p)\equiv\sum_{k}x_{k}(\mu_{k},p)$. Under both CARA and CRRA,
$x_{k}$ is continuous in $p$ on $(0,1)$ and strictly decreasing: at any
price $p$, raising $p$ reduces $R_{k}$ and hence $x_{k}$. As $p\to 0^{+}$,
$R_{k}\to\infty$ for every $k$ with $\mu_{k}>0$, so
$Z(p)\to+\infty$; as $p\to 1^{-}$, $R_{k}\to 0$ and $Z(p)\to-\infty$.
By the intermediate value theorem $Z$ has at least one root; strict
monotonicity delivers uniqueness.
\end{proof}
```

---

## 3. HARA remark (insert in Section 7.1 Discussion, after "What CARA does and does not do")

```latex
\begin{remark}[HARA and the knife-edge]\label{rem:hara}
The CRRA family is the $b=0$ subfamily of the HARA class with risk
tolerance $T(W)=aW/(1-\gamma)+b$. With homogeneous agents and equal
wealth, the parameters $(a,b)$ enter the equilibrium only through the
overall demand scale: each HARA demand equals a $(a,b)$-dependent scalar
multiple of the CRRA demand at parameter $\gamma$. Market clearing
annihilates the scalar, so the equilibrium price depends solely on
$\gamma$, not on $(a,b)$. The knife-edge is therefore robust to the full
HARA generalisation: moving within HARA at fixed $\gamma$ does not change
the revelation deficit; only moving to $\gamma=\infty$ (CARA, equivalently
$a=0$ in the HARA parametrisation) recovers full revelation.
\end{remark}
```

---

## 4. Zero-supply footnote (insert at line 266, after "z-bar=0")

```latex
\footnote{Zero net supply is the strongest version of our result: partial
revelation arises without any exogenous friction. CRRA demand is
self-bounding --- as wealth approaches zero in either state, marginal
utility diverges --- so no agent takes an unbounded position or crosses
into negative wealth. Market clearing at $\bar{z}=0$ always admits a
solution by Lemma~\ref{lem:existence}. Introducing a positive
deterministic supply $\bar{z}>0$ shifts the price level (and under CARA
shifts the sufficient statistic by a known constant $\gamma\bar{z}$) but
does not affect the informational content of the aggregation: the
alignment between demand linearity and the Bayesian sufficient statistic
is a preference property, not a supply property. Our choice of
$\bar{z}=0$ isolates the pure preference channel.}
```

---

## 5. Multiple equilibria paragraph (insert in Section 4.4 after the convergence figure)

```latex
\paragraph{Equilibrium selection.}
The contour map $\Phi$ generically admits at least two fixed points. The
first is the fully-revealing price function $p=\Lam(\Tstar)$: under any
preferences, identical posteriors clear the market at zero trade. The
second is the partially-revealing branch characterised by
Proposition~\ref{prop:ree}. The fully-revealing fixed point is a
knife-edge in the sense that any perturbation of the price function away
from $\Lam(\Tstar)$ induces contour curvature and posterior disagreement,
pushing the map toward the partially-revealing branch. The vanishing-noise
selection of Proposition~\ref{prop:vanishing} provides the natural
selection criterion: at every finite $\gamma$, the limit of vanishing noise
selects the partially-revealing branch. Only at $\gamma=\infty$ does the
noise limit coincide with the fully-revealing fixed point.
```

---

## 6. Stronger conclusion (replace last two sentences of Section 8)

```latex
The remaining open question is whether the alignment principle carries
through to dynamic and intermediated environments, where the interaction
between preferences and learning unfolds over multiple rounds. The
contour-integration fixed point developed here extends in principle to
those settings, and we leave the dynamic analysis to future work. The
broader message is simple: what the literature has modelled as exogenous
noise in the price is, in significant part, the endogenous consequence of
preferences that do not align with the Bayesian sufficient statistic.
Noise traders are not a primitive of rational markets; they are an artefact
of CARA.
```

---

## 7. Lemma 3 proof — move to appendix

Currently the proof of Lemma 3 (CARA as CRRA limit) appears inline at lines 374-388. For Econometrica consistency, move it to Appendix A between the proofs of Lemma 2 and Proposition 1. Replace the inline proof with:

```latex
The proof is in Appendix~\ref{app:proofs}. The convergence is uniform
on compact subsets of $(0,1)^{2}$ and preserves the market-clearing
identity.
```

---

## FIGURE REORDERING SUGGESTION

Current order:
- Fig 1: Heatmap (smooth transition)
- Fig 2: Robustness (K agents + lognormal)
- Fig 3: CARA vs CRRA contour
- Fig 4: Posteriors (placeholder)
- Fig 5: Convergence
- Figs 6-10: Welfare, mechanisms, etc.

Suggested order:
- Fig 1: Knife-edge (1−R² vs τ) — THE visual punchline, should come first
- Fig 2: Heatmap/table (smooth transition) — supports Fig 1
- Fig 3: CARA vs CRRA contour — explains WHY
- Fig 4: Posteriors — confirms at REE
- Fig 5: Convergence — technical support
- Rest unchanged

The knife-edge figure is the single image that captures the paper's contribution. A referee glancing at Figure 1 should immediately understand the claim.
