# Stage 7: strict h=0 tau-ladder — tilt screening, tau*, and the revealing-equilibrium discovery

**Hypothesis under test (user's):** the Stage 6b strict-h=0 ill-posedness (tilt saturation at
sqrt(2) on the one-signal-dominance band; CRRA price atoms near 1/3, 2/3) is monotone in tau;
below some tau*(gamma) strict h=0 should be well-posed and convergent, yielding the first table
of strict-h=0 deficits.

**Code:** `projects/REZN/solver_code/cmm_endogenous/cmm_stage7_tau_ladder.py`
(+ `cmm_stage7_plots.py`, `cmm_stage7_revealing_check.py`), reusing the Stage 6b graph solver
(`cmm_stage6b_solver.py`: heval/build_pad, Problem residual+FD Jacobian, build_initial_H,
deficit_from_H) verbatim.
**Artifacts:** `stage7_tilt_screen.json` (87 cells), `stage7_tau_ladder.json`,
`stage7_revealing_check.json`, `stage7_tilt_heatmap.png`, `stage7_ladder_plot.png`,
`stage7_H_t{tau}.npy` (gamma=0.098 rungs) and `stage7_H_t{tau}_g{gamma}.npy` (all rungs).

---

## HEADLINE (not the hypothesized outcome — something sharper)

**The strict h=0 operator admits the FULLY REVEALING equilibrium
`P(u) = sigma(tau * (u1+u2+u3))` as an EXACT fixed point at every (tau, gamma).**
Flat level surfaces `H_m(a,b) == logit(p_m) / (tau*sqrt(3))` give every agent the identical
posterior `mu_k = sigma(tau*(u1+u2+u3))` (the slice-curve evidence integral is Gaussian and
closes analytically: log-odds = tau*X + 2*tau*c = tau*sqrt(3)*t, independent of the own
signal X), and CRRA clearing with identical posteriors returns p = mu for any gamma.
**Verified numerically in the Stage 6b graph machinery: max|r| = 1.7e-15 .. 3.8e-15 at all 9
ladder cells** (each with its own M=8 quantile p-levels and quadrature), including the
"ill-posed" cell (tau=2, gamma=0.098). Its revelation deficit is identically 0 (reconstruction:
|1-R^2| < 4e-15 — exact, since logit p is linear in t).

Consequences:
- The "first table of strict h=0 deficits" exists and is **trivial: d = 0 at every (tau, gamma)**.
- The kernel/h-blur is **not merely a regularizer that makes the equations solvable — it is the
  selection device for the nontrivial, partially revealing branch.** The deep-ladder anchor
  d_inf = 0.268 +/- 0.010 at (tau=2, gamma=0.098) is the h->0 limit of the *kernel* branch; it is
  NOT the deficit of any strict-h=0 fixed point reachable by this program (Grossman–Stiglitz-type
  degeneracy: at exact h=0 the conditioning is on a measure-zero level set and full revelation
  becomes self-consistent).
- Stage 6b's plateau is re-interpreted: LM from the kernel warm start is trying to march from the
  kernel branch toward the revealing solution and **jams in the supercritical-tilt (atom) band**
  whenever that band is present.

## T1 — tilt-audit screening of all 87 certified kernel FPs (18 s)

For each cell: kernel FP -> Stage 6b `build_initial_H` (M=8 quantile surfaces, 15x15 H grid on
[-4.5,4.5]^2) -> |grad H_m| on a 61x61 sample (Stage 6b tilt-audit machinery). Classes on
max tilt: WELL-POSED < 1.0 <= MARGINAL < sqrt(2) <= ILL-POSED.

**Counts: 15 WELL-POSED / 1 MARGINAL / 71 ILL-POSED.**

The boundary on the (tau, gamma) grid (see `stage7_tilt_heatmap.png`):
- The **entire well-posed region is the tau=0.2 row at gamma >= 0.2692** (max tilt 0.64-0.85);
  gamma = 0.1922 is the single MARGINAL cell (1.19); gamma <= 0.1373 at tau=0.2 is ill-posed
  (1.56-2.18).
- **Every cell with tau >= 0.5 is ill-posed**, at all gamma up to 30 (max tilt 1.69 at
  (0.5, 30) rising to 5.42 at (2.0, 1.4493)). So the hypothesized monotonicity in tau is
  real (tilt falls as tau drops) but the well-posed window only opens at tau ~ 0.2, and only
  for moderate-to-large gamma.
- Posedness correlates with the kernel deficit, not with tau per se: **all 15 well-posed cells
  have kernel G=21 deficit <= 0.0153** (i.e. the kernel FP is already nearly revealing);
  every cell with kernel deficit >= 0.04 is marginal/ill-posed. This is exactly the
  revealing-equilibrium picture: the strict-h=0 problem is well-posed where the nontrivial
  kernel branch has (almost) merged with the revealing branch.

## T2 — tau-ladder at gamma = 0.098 (anchored column)

M=8 surfaces, 15x15 heights, 13x13 vertices (all density-active at these tau), graph-in-s
residual, masked LM (no Tikhonov anchor, step cap, |J|>100 phantom suppression, NO row
dropping — an unsatisfiable band must show up as a plateau). s-quadrature widened vs Stage 6b
(s_max = 6/sqrt(tau); Stage 6b's +/-4 truncates at low tau). tau=0.2/0.5/1.0/2.0 warm-started
from the kernel FP (own quantile levels); tau=0.1/0.05 by continuation down from tau=0.2
(no kernel FP exists there).

| tau | warm start | init max\|r\| | plateau max\|r\| | plateau med\|r\| | tilt(warm) | tilt(final H) | kernel d |
|-----|-----------|-----------|--------------|--------------|-----------|--------------|----------|
| 0.05| contin.   | 6.6e-2 | **6.1e-2** | 2.5e-2 | 19.6 (polluted) | 24.1 | – |
| 0.1 | contin.   | 8.2e-2 | **7.4e-2** | 1.5e-2 | 7.8 (polluted)  | 19.6 | – |
| 0.2 | kernel FP | 2.3e-1 | **2.2e-1** | 6.3e-2 | 1.75 | 7.8  | 0.1183 |
| 0.5 | kernel FP | 2.3e-1 | **2.1e-1** | 1.4e-1 | 3.07 | 6.2  | 0.1557 |
| 1.0 | kernel FP | 3.2e-1 | **3.2e-1** | 1.5e-1 | 2.85 | 9.6  | 0.2046 |
| 2.0 | kernel FP | 3.2e-1 | **3.2e-1** | 1.6e-1 | 2.51 | 13.3 | 0.2796 |

- **No tau in [0.05, 2] converges; tau*(0.098) does not exist** (the entire column is ill-posed,
  as the screen predicted: max tilt 1.75 at the warm start even at tau=0.2; the M=8 quantile
  levels at tau=0.2 already span [0.348, 0.652] — the gamma-driven 1/3–2/3 atoms are alive at
  ALL tau in this column; the worst surfaces are always the ones nearest p = 1/3, 2/3).
- The plateau does shrink as tau drops (0.32 -> 0.06 max) — the hypothesized tau-monotonicity
  is visible in the *size* of the unsatisfiable residual, but it never reaches solvability.
- LM steepens the surfaces while grinding (final-H tilt 6-24, sign-flip drift up to 3.5):
  the iteration is pulled toward {u_k = const} configurations — the Stage 6b mechanism.
- "deficit_strict_h0" entries in the json for these rungs are reconstructions of partial LM
  states, NOT equilibrium deficits (ladder crossings / far-field drift; e.g. 0.99 values are
  pure artifacts). The only certified strict-h=0 equilibrium deficit is 0 (revealing).
- The continuation rungs (0.1, 0.05) inherit the polluted tau=0.2 surfaces; their plateaus
  (7.4e-2, 6.1e-2) bound the strict-h=0 inconsistency of the kernel-branch shape from above.

## T3 — gamma = 1.035 row (small-wedge column; the actual tau* row)

| tau | warm start | init max\|r\| | final max\|r\| | behavior | tilt(warm) | kernel d |
|-----|-----------|-----------|------------|----------|-----------|----------|
| 0.2 | kernel FP | 2.2e-1 | 1.6e-1 (m0; mid-surfaces 9.4e-4) | **steady march toward revealing** (all 8 surfaces descend every iteration; time-capped, not stuck) | 0.68 WELL-POSED | 0.0003 |
| 0.5 | kernel FP | 2.5e-1 | 2.2e-1 | jams (15 its, outer surfaces frozen) | 1.94 ILL | 0.0066 |
| 1.0 | kernel FP | 2.6e-1 | 2.5e-1 | **frozen** (max\|r\| moves 1%) | 3.55 ILL | 0.0414 |

Distance to the revealing planes (core nodes |a|,|b|<=2.5, `stage7_revealing_check.json`):
g1.035 tau=0.2: 1.436 -> 0.928 (marching, ~35% of the way when time-capped at ~7 min);
tau=0.5: 1.308 -> 1.105 (slowing); tau=1.0: 1.058 -> 1.011 (jammed).
gamma=0.098 column: tau=0.2: 1.89 -> 1.47; tau=2.0: 1.387 -> 1.392 (completely jammed).

**tau*(1.035) lies in (0.2, 0.5]** — kernel-FP surfaces are subcritical everywhere at tau=0.2
(max tilt 0.68 vs sqrt(2)) and supercritical at tau=0.5 (1.94); the LM behavior flips from
monotone global descent to a frozen plateau across the same interval. Finer localization would
require kernel FPs at intermediate tau (not in the emin15 archive).

Note on convergence speed at the well-posed cell: the march is genuinely long (heights must
move O(2-4) because at tau=0.2 the price surface is nearly flat — dP/dt ~ 0.08/unit), and the
unanchored LM lets far-field (insensitive) nodes drift (final-H corner tilt/sign-flip artifacts
at this cell are LM null-space drift, not physics). Driving max|r| to 1e-6 from the kernel warm
start was abandoned as a vanity metric once the destination was identified analytically:
the exact solution it is marching to is the revealing equilibrium (residual 1.7e-15 directly).

## Verdict

1. **Does strict h=0 work for low tau?** Yes, but trivially: where it is well-posed
   (tau ~ 0.2, gamma >= ~0.27 — equivalently wherever the kernel deficit is already < ~0.02),
   the strict-h=0 equilibrium is the fully revealing one, deficit 0, and LM flows toward it.
   There is no tau regime with a *nontrivial* convergent strict-h=0 fixed point.
2. **Is the kernel necessary only at high tau, or always?** Always — but for a sharper reason
   than Stage 6b's: not just to regularize the singular band, but to *select the partially
   revealing branch at all*. At exact h=0 the only equilibrium this machinery can certify is
   full revelation (at every tau and gamma); the d_inf = 0.268 continuum anchor is a property
   of the h->0 LIMIT of the kernel family, attained along h>0, not at h=0.
3. **Consistent with the Stage 6b mechanism?** Yes, quantitatively: ill-posedness (tilt >=
   sqrt(2), CRRA atoms at 1/3, 2/3) weakens monotonically as tau drops (max-tilt heatmap) and
   as gamma rises (atom wedge shrinks), and the LM plateau shrinks with it (0.32 at tau=2 to
   0.06 at tau=0.05 on the gamma=0.098 column). But at gamma=0.098 the atoms never disappear
   for any tau in [0.05, 2] — the hypothesized tau*(0.098) does not exist; the well-posed
   window opens only where the atoms (and with them the kernel deficit) have already collapsed.
4. **Recommendation:** the h-continuation program (Stage 6b recommendation) is now mandatory
   rather than optional: the nontrivial branch must be followed in h, since at h=0 it merges
   into/disconnects from the revealing branch. The Stage 7 screen provides the map of where
   that continuation will be hard (supercritical band) vs easy.

## Run notes
- T1 screen: 87 cells in 18 s. Ladder legs: ~3-7 min/rung (numba, 4 cores), checkpointed
  per rung in `stage7_tau_ladder.json`.
- Quadrature: graph-in-s residual, s_max = 6/sqrt(tau), ds = 0.10/0.15/0.25 for
  tau >= 1 / >= 0.2 / < 0.2; revealing-solution residual is 1e-15 under the same quadrature
  (the integral closes exactly for flat surfaces, so quadrature error cancels in mu).
- The 4 emin15 REJECT cells (t1.5, gamma >= 10.93) were excluded.
