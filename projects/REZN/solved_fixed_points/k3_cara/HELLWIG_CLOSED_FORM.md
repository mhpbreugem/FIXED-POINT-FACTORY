# Closed-form CARA equilibrium for our K=3 binary-asset model (Hellwig-style)

## Setup
- Asset value v ∈ {0,1}, flat prior.
- Signals u_i = vm_v + noise; f_v(u) = √(τ/2π) exp(−τ(u−vm_v)²/2) with vm_0 = −1/2, vm_1 = +1/2.
- K=3 agents.
- CARA + binary asset → linearized demand x_i = W_i(m_i − π)/a with m_i = logit μ_i, π = logit p.
- Market clearing Σ W_i x_i = 0 with W_i = 1 ⇒ **π = mean_i(m_i)** ⇒ **p = sigmoid(mean(logit μ_i))**.
- *No noise traders.*

## The sufficient statistic is τΣu
Log-likelihood ratio per signal:
$$\log\frac{f_1(u)}{f_0(u)} = -\frac{\tau}{2}\Big[(u-\tfrac12)^2-(u+\tfrac12)^2\Big] = \tau u.$$
Total log-LR given all signals = τ(u_1+u_2+u_3) = **τΣu**. With flat prior, **E[v=1 | u_1,…,u_K] = σ(τΣu) = lam(τΣu)**.

## FR is a CARA fixed point
Candidate equilibrium price function: **p*(u) = lam(τΣu)** ("full revelation").

Verification:
1. **Inference:** observing p*(u) reveals τΣu (since lam is monotone and the price is a function of the sufficient statistic). Combined with own signal u_i, the posterior is E[v=1 | u_i, τΣu] = E[v=1 | τΣu] = lam(τΣu) (u_i is conditionally redundant given the sufficient stat). So **μ_i = lam(τΣu) for all i**.
2. **Aggregation:** mean(logit μ_i) = logit(lam(τΣu)) = **τΣu** = logit(p*).
3. **Clearing:** sigmoid(mean(logit μ_i)) = lam(τΣu) = p*(u). ✓

So p* = lam(τΣu) is a self-consistent CARA equilibrium → **FR is a CARA fixed point** with revelation deficit identically zero (1−R² of logit(p*) on τΣu = 0).

## Uniqueness (Hellwig 1980 analog)
In Hellwig's CARA-normal model without noise traders, the linear-RE equilibrium is unique and fully revealing — Grossman's paradox. Our setting is the binary-asset analog with linear-log-odds clearing; the *continuum* counterpart of the Hellwig argument yields the same conclusion: **FR is the unique CARA equilibrium**, so the deficit should be zero.

## So why does the numerical CARA op give deficit > 0 (and growing with G)?
Three possibilities — discriminated by the FR test:

- **(i)** FR is a stable Picard FP of the discrete operator → Picard should converge to it; if it doesn't, the operator has a bug.
- **(ii) FR is a Picard SADDLE** of the discrete operator (ρ(Φ′(P_FR)) ≥ 1): naive iteration *escapes* FR and settles on a *non-FR* stable fixed point of the discrete map. Hellwig's existence/uniqueness holds in the continuum; what fails is Picard's *dynamical reachability* on the finite grid — exactly the same dynamic-vs-equilibrium tension we found for the K=3 CRRA PR.
- **(iii)** FR isn't a discrete fixed point at all → operator/model implementation issue.

The accompanying script `test_FR_is_fixed_point.py` runs both checks: residual ‖Φ_CARA(P_FR) − P_FR‖_∞ and the spectral radius ρ(Φ′_CARA at P_FR) via Arnoldi.

## What the result means for the paper

- If **(ii)**: the existence claim ("CRRA admits a noiseless PR equilibrium, CARA does not") is correct in the CONTINUUM as Hellwig predicts, but the numerical *dynamics* in both CRRA and CARA admit non-FR stable plateaus due to Picard instability. The contrast becomes: for CRRA, even Newton finds a non-FR fixed point (the genuine deterministic PR equilibrium with positive deficit ~0.26); for CARA, **Newton from an FR-region IC converges to FR (deficit = 0)**, recovering the Hellwig result, while Picard from NL converges to a non-FR Picard-attractor (artifact of dynamics, not equilibrium).
- If **(iii)**: we have an operator bug to fix.

The decisive distinction: nail CARA with **Newton from the FR seed** and check whether it converges to deficit ≈ 0. That's the right "continuum CARA equilibrium" check, not Picard from NL.
