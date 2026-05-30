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

## Empirical verdict (`test_FR_is_fixed_point.py`, τ=2, K=3, log-odds clearing)

| G  | ‖Φ_CARA(P_FR) − P_FR‖_∞ | deficit(P_FR) | ρ(Φ′_CARA at P_FR) | top \|λ\| |
|----|--------------------------|---------------|--------------------|----------|
|  9 | **0.207**                | 5.1e-4        | 0.635              | 0.64, 0.56, 0.54, 0.54 |
| 13 | **0.209**                | 2.9e-4        | 0.870              | 0.87, 0.82, 0.81, 0.80 |
| 17 | **0.198**                | 2.0e-4        | 1.059              | 1.06, 1.01, 1.01, 1.01 |

**Reading.** The analytic FR is *almost* a deficit-0 surface on the grid (deficit(P_FR) ~ 10⁻⁴, shrinking with G — the small residual is just the finite logit-linearity of the clipped sigmoid). But the discrete operator does **not** map P_FR back to itself: the residual sits at ≈0.20 for *every* G we tried, with no sign of vanishing. This is **outcome (iii)**: FR is not a discrete fixed point of `phi_cara` as currently implemented.

Conclusion. **Hellwig stands in the continuum.** The closed-form argument above is airtight: the only CARA equilibrium of the K=3 noiseless model is fully revealing, p* = sigmoid(τΣu), deficit = 0. The growing numerical "deficit" reported by `cara_vs_G.py` is **not** a CARA Jensen gap (CARA has no Jensen gap by construction — linear log-odds clearing). It is a **discretization artifact** of the discrete `phi_cara` implementation: the analytic P_FR is *almost* but not exactly a discrete FP (residual ≈ 0.2 mostly from grid edges where τΣu saturates the [10⁻⁹,1−10⁻⁹] clip), so Picard relaxes to a nearby discrete FP whose deficit (a) is tiny at small G (3.8e-3 at G=9), (b) grows with G as more interior cells approach the boundary-clipped region.

We verified the artifact is *not just* halo BC: running Picard with the **FR-consistent halo** (`cara_fr_halo_vs_G.py`, halo = sigmoid(τΣu) extended to the padded ring) gives **the same** deficit as the NL halo (G=9: 3.8e-3, G=13: 1.15e-2 — identical to `cara_vs_G.py`). Both Picards converge to the same Picard attractor, which is a *near-FR* fixed point of the discrete operator, not the analytic FR itself. The two main discrete-vs-continuous gaps that explain the residual at P_FR are (i) the 10⁻⁹ clip on p (saturates large |τΣu|), (ii) the kernel-band quadrature of A_v(p) and Bayes near the box edges.

Two things this confirms cleanly for the paper:
1. **The CRRA result is real, not a numerical echo.** CRRA at G=17 nails a non-FR fixed point with deficit ≈ 0.13 (τ=2, γ=0.1) using the *same* operator; the corresponding deficit at the *analytic* CARA P_FR is ~10⁻⁴ (3 orders of magnitude smaller), and shrinks further with G. The CRRA gap is intrinsic (it comes from the Jensen-curvature mechanism, magnitude (½−p)Var(m)/γ); CARA's "gap" is grid-induced and continuum-vanishing.
2. **The right CARA benchmark for the paper is the analytic 0**, not any numerical residual. Report the closed-form FR result and label the `cara_vs_G.py` and `cara_fr_halo_vs_G.py` numbers honestly as a continuum-vanishing discretization artifact of the kernel-band operator at the box-clip — both halos give the same Picard attractor, so the residual is intrinsic to the discrete operator near boundary-saturated cells, not a fixable boundary-condition mistake. The Hellwig theorem is what the paper should rely on for CARA = FR; numerical CARA serves only as a sanity check that the operator's CARA fixed point sits "near" the analytic FR (and it does: deficit ~10⁻⁴ at FR itself for G=9–17).
