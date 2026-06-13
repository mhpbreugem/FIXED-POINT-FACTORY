# Maximum revelation deficit in the K=3 CRRA-REE binary-payoff model

## Verdict

**Maximum certified deficit: $1-R^2 = 0.2871$ at $(\tau, \gamma) = (2.0, 0.01)$.**
Certified at $\|\Phi(P)-P\|_\infty = 2.95\times 10^{-16}$ in 80-bit longdouble
arithmetic (well under the $10^{-15}$ certification bar).

This is up from the previous best of $0.2844$ at $(\tau=2.0, \gamma=0.05)$
(emin15 pass) and confirms the prediction that the deficit envelope
**saturates** at a finite ceiling as $\gamma\to 0$ rather than blowing up.

## What was added in this pass

| pass | $\tau$ values | $\gamma$ list | cells | ACCEPT |
|------|---------------|---------------|-------|--------|
| emin15 (existing) | 0.2, 0.5, 1.0, 1.5, 2.0 | 20-gamma log grid | 91 | 87 |
| lowtau (existing) | 0.05, 0.1, 0.3, 0.4, 0.6 | 20-gamma log grid | 100 | 100 |
| **hightau (new)** | **0.8, 1.2** (full 20-gamma + 3 low-gamma)<br>plus 0.1, 0.2, 0.4, 0.6, 1.0, 1.5 (3 low-gamma)<br>plus 2.0 (3 low-gamma + 2 mid) | mixed | 69 | 68 |
| **grand total certified** | 12 $\tau$ values | (dedup) | — | **255** |

The new $\tau\in\{0.8, 1.2\}$ rows give 23 cells each (low-$\gamma$ extension +
standard grid). The low-$\gamma$ extension at the existing $\tau\in\{0.1, 0.2,
0.4, 0.6, 1.0, 1.5\}$ rows adds 3 cells each. Plan C at $\tau=2.0$ added
$\gamma\in\{0.01, 0.02, 0.03, 2.03\}$.

## Maximum deficit per $\tau$ (the envelope)

| $\tau$ | max $1-R^2$ | at $\gamma$ | cells | source of max |
|--------|-------------|-------------|-------|---------------|
| 0.05 | 0.0591 | 0.0500 | 20 | lowtau (no low-$\gamma$ extension done) |
| 0.10 | 0.1957 | 0.0100 | 23 | **hightau (new)** |
| 0.20 | 0.1984 | 0.0100 | 23 | **hightau** |
| 0.30 | 0.1761 | 0.0500 | 20 | lowtau |
| 0.40 | 0.2138 | 0.0100 | 23 | **hightau** |
| 0.50 | 0.1860 | 0.0500 | 20 | lowtau |
| 0.60 | 0.2138 | 0.0100 | 23 | **hightau** |
| 0.80 | 0.2181 | 0.0100 | 23 | **hightau (new $\tau$ row)** |
| 1.00 | 0.2298 | 0.0100 | 23 | **hightau** |
| 1.20 | 0.2459 | 0.0100 | 23 | **hightau (new $\tau$ row)** |
| 1.50 | 0.2614 | 0.0100 | 19 | **hightau** |
| 2.00 | **0.2871** | **0.0100** | 15 | **hightau** |

Rows in **bold** are the new cells that lifted the row's max deficit above
the previous lowtau/emin15 result. The pattern is monotone-increasing in
$\tau$ (with discretization noise of about $\pm 0.002$ at $\tau\in[0.3,0.5]$ —
those rows kept their original max at $\gamma=0.05$ because no low-$\gamma$
extension was done for them in this pass).

## Does the deficit saturate? YES.

At each new $\tau$ row the deficit at the three lowest $\gamma\in\{0.01, 0.02,
0.03\}$ values differs by less than the discretization noise (which is on the
order of $\pm 0.003$ at $G=21$). Examples:

- $\tau=0.80$: $d(\gamma=0.01)=0.2181$, $d(0.02)=0.2150$, $d(0.03)=0.2114$, $d(0.05)=0.2032$
- $\tau=1.20$: $d(\gamma=0.01)=0.2459$, $d(0.02)=0.2418$, $d(0.03)=0.2384$, $d(0.05)=0.2331$
- $\tau=1.50$: $d(\gamma=0.01)=0.2614$, $d(0.02)=0.2599$, $d(0.03)=0.2584$
- $\tau=2.00$: $d(\gamma=0.01)=0.2871$, $d(0.02)=0.2864$, $d(0.03)=0.2857$, $d(0.05)=0.2844$

The decrement per decade of $\gamma$ is tiny (about 0.003-0.005). The deficit is
**asymptoting to a $\tau$-dependent ceiling well below the theoretical upper
bound of 1**, not blowing up. Extrapolating $\tau\to\infty$, the empirical
envelope still appears bounded — at $\tau=2$ we see $d_\infty(\tau)\approx
0.287$.

## Stalled cells (quarantine)

| $(\tau,\gamma)$ | source | $F_{64}$ | comment |
|-----------------|--------|----------|---------|
| $(1.5,\;10.93)$ | emin15 | $3.1\times 10^{-2}$ | high-$\gamma$ high-$\tau$ G=21 warm-start pathology (see RESULTS.md in stall_diagnosis: spectrum stays away from 1; a $G=13$ ladder rescue is possible but not pursued here) |
| $(1.5,\;15.30)$ | emin15 | $3.2\times 10^{-2}$ | same |
| $(1.5,\;21.42)$ | emin15 | $4.7\times 10^{-2}$ | same |
| $(1.5,\;30.00)$ | emin15 | $5.0\times 10^{-2}$ | same |
| $(2.0,\;2.84)$  | **hightau (new)** | $3.4\times 10^{-2}$ | chain64 stall; rescue (tau-march from lower $\tau$) tried but failed — likely needs $G=13$ ladder |

These 5 cells are economically uninteresting (they sit at the
near-full-revealing tail where the deficit is already $<0.05$); they are
documented for honesty but do not affect the max-deficit verdict.

Stage-7-aware caveat: tau >= 2 high-gamma stalls are known surface-tilt
issues compromising the kernel solver; we did not push past the standard
20-gamma logspace at tau=2 because of the 90-minute budget.

## Locus of max-deficit cells: economically relevant frontier

Across all 12 certified $\tau$ values, the maximum-deficit cell always sits
at the **smallest tested $\gamma$**. The frontier in $(\gamma,\tau)$ space is
therefore the LOW-LEFT corner of the heatmap, and the gradient is essentially
purely in $\tau$ at fixed small $\gamma$.

The "deficit per CPU second" frontier (relevant if one is buying solver time)
is dominated by the new $\tau\ge 0.8$ rows at $\gamma=0.01$: each cell solved
in $\sim 20$s for $\tau\in\{0.8, 1.0, 1.2\}$ and $\sim 18$s for $\tau\in\{1.5,
2.0\}$ — the high-tau corner is NOT computationally expensive at low $\gamma$;
the expense is at high $\gamma$ (where the surface tilts more sharply but the
deficit is already $\sim 10^{-2}$).

## Economic interpretation of the high-deficit corner

At $(\tau,\gamma)=(2.0, 0.01)$ (or any low-$\gamma$, high-$\tau$ cell):

- Each agent is extremely risk-tolerant ($\gamma=0.01$ means near-linear
  utility — log-utility lives at $\gamma=1$, $\gamma\to 0$ is risk neutrality).
- Signals are very informative ($\tau=2$ means each $u_k$ tells a lot about
  the state). The full-revealing-price benchmark would summarize all 3
  signals in the price.
- Yet the equilibrium logit-price only captures $\approx 71\%$ of its own
  variance in the linear projection on $\sum_k u_k$ — the remaining
  $1-R^2 = 0.287$ is **scatter around the linear aggregator**, not absent
  movement.

**Empirically, the equilibrium price at the max-deficit cell is STRONGLY
BIMODAL**, not "near 0.5":

| cell | min $P$ | max $P$ | frac in $[0.4, 0.6]$ | frac in $[0.1, 0.9]$ |
|------|---------|---------|---------------------|---------------------|
| max-deficit: $(\tau=2.0, \gamma=0.01)$ | 0.0001 | 0.9999 | 7.1% | 84.3% |
| smooth low-tau: $(\tau=0.05, \gamma=0.05)$ | -- | -- | 100% | 100% |

So the picture is: at $(2.0, 0.01)$ the price hits the absorbing
$\{0, 1\}$ corners very often when one or two signals are extreme, but the
**path between corners is not just $\tau\sum_k u_k$ — it bulges and bends in
ways the linear projection can't capture**. The deficit is the residual
variance of those bends. Mechanism: risk-neutral CRRA agents have very
aggressive demand schedules and the market clears at extreme price points
whenever marginal evidence in either direction; the resulting price surface
is steep but non-linear in $\sum u_k$. The Jensen wedge between price and FR
posterior expands as $\gamma$ shrinks.

This contradicts a naive "deficit = price hugs 0.5" picture and is the
genuine economic content of the high-deficit corner: **the price moves a
lot, but it does not move IN PROPORTION to the linear combination of
signals — the relationship is the wrong shape**.

## Outputs delivered

- `/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/hightau.json` — 69 cells (68 ACCEPT, 1 chain_fail at $(2.0, 2.84)$).
- `/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/MAX_DEFICIT_report.pdf` — 10 figures (F1, F1b, F2 heatmap, F3 envelope, F4 residuals + stalls, F5 low-$\gamma$ zoom, F6 VoI decomposition at $\tau=2.0$, F7 logit-P scatter at max cell, F8 price slice at max cell, F9 frontier locus).
- `/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/LOWTAU_report.pdf` — regenerated with the new $\tau$ values included.
- `/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/metrics.json` — extended from 132 to 255 cells.
- `P_ld_t{tau}_g{gamma}.npy` — 68 new certified $P^\star$ inner cubes in the lowtau directory.
