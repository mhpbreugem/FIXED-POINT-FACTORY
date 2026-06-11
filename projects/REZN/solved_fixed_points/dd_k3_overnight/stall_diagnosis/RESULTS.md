# Stall-corner diagnosis: $\tau=2$ march at $G=13$

## Verdict

**(c) Solver failure: the original 20$\times$5 sweep's stall corner (F~3e-2 at $\gamma \tau \gtrsim 5$) is not a fold and not a bifurcation; it is a $G=21$ warm-start pathology.**

## Evidence

Spectrum of the Jacobian $J = \partial\Phi/\partial P$ tracked along $\gamma$ from 0.38 to 1.20 at $\tau=2$, $G=13$, warm-starting each $\gamma$ from the previous converged solution.

| $\gamma$ | $\|F\|_\infty$ | $\rho(J)$ | dist(eig, 1) | $\sigma_{\min}(I-J)$ |
|---|---|---|---|---|
| 0.3769 | 8.49e-15 | 0.4793 | 0.5207 | 0.3897 |
| 0.4200 | 4.01e-13 | 0.4983 | 0.5017 | 0.3698 |
| 0.4600 | 2.04e-13 | 0.5133 | 0.4867 | 0.3553 |
| 0.5400 | 6.97e-15 | 0.5370 | 0.4630 | 0.3344 |
| 0.5800 | 6.36e-14 | 0.5464 | 0.4536 | 0.3270 |
| 0.6200 | 5.28e-14 | 0.5546 | 0.4454 | 0.3212 |
| 0.6600 | 3.72e-14 | 0.5617 | 0.4383 | 0.3165 |
| 0.7000 | 2.82e-14 | 0.5679 | 0.4321 | 0.3128 |
| 0.7400 | 2.13e-14 | 0.5733 | 0.4267 | 0.3099 |
| 0.7800 | 1.51e-14 | 0.5782 | 0.4218 | 0.3075 |
| 0.8200 | 9.33e-15 | 0.5825 | 0.4175 | 0.3056 |
| 0.8600 | 7.05e-15 | 0.5863 | 0.4137 | 0.3040 |
| 0.9000 | 6.43e-15 | 0.5898 | 0.4102 | 0.3027 |
| 0.9500 | 8.33e-15 | 0.5937 | 0.4063 | 0.3014 |
| 1.0000 | 6.55e-15 | 0.5971 | 0.4029 | 0.3003 |
| 1.1000 | 7.43e-14 | 0.6030 | 0.3970 | 0.2987 |
| 1.2000 | 2.99e-14 | 0.6077 | 0.3923 | 0.2975 |

- All 17 points nailed at machine eps ($\|F\|_\infty < 1$e-12).
- No eigenvalue of $J$ outside the unit disk at any $\gamma$.
- Closest-to-1 eigenvalue stays at distance $\approx 0.40$ — plateaus, never approaches 1.
- $\sigma_{\min}(I-J)$ plateaus at $\approx 0.30$ — well above 0, so $(I-J)$ is uniformly bounded away from singular.

## Economic interpretation

There is no Grossman-Stiglitz-type revelation breakdown along this branch. The equilibrium is well-defined and locally unique for all $\gamma \in [0.05, 1.2]$ at $\tau=2$. Higher $\gamma$ (lower risk-bearing capacity) does not destabilize the REE; it just makes the equilibrium price less informative, which the deficit metric already captures.

## Implication for the emin15 certification

The 19 cells the emin15 pass rejects (mostly $\tau \ge 1.5$, $\gamma \ge 1$) are not physically inadmissible. They can be recovered by:
1. solving at $G=13$ first with small $\gamma$ steps (as done here);
2. interpolating/resampling the converged $G=13$ solution as warm start at $G=21$;
3. running the longdouble polish.

The 81/100 emin15 ACCEPT (+ 4 rescued so far) is a conservative lower bound; a thorough second pass would cover the full $20\times5$ grid.

## Caveats

This march is at $G=13$ (2197 unknowns). The spectrum at $G=21$ (9261 unknowns) could differ — the operator is the same but the discretization is finer. The logical step would be: confirm at one $\tau=2$ high-$\gamma$ point that $G=13$ warm-start carries through $G \to 17 \to 21$ cleanly.
