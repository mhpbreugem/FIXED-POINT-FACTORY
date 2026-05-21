# Status of "full-equilibrium" figures (Figs 4, 6, 7, 8, 9, 10)

## TL;DR
Six figures are flagged with a yellow background as placeholders. The
production PCHIP+contour kernel cannot reach the strong-PR REE branch
from any seed I have tried tonight. All solver paths collapse onto the
near-CARA / FR fixed point, contradicting the HANDOFF's reported
strong-PR values (1-R² ≈ 0.057 at γ=3, τ=3).

## Evidence collected tonight (G=11, UMAX=2)

### γ=0.5, τ=2:
- Cold no-learning seed: 1-R²=0.113, PR-gap=0.092 (NL itself NOT a fixed
  point — Finf=0.325)
- Picard α=0.3 maxiters=2000: collapses to 1-R²=4e-6, PR-gap=0.0002
- Anderson m=8 + Picard chain: same FR result
- Analytic Newton 15 iters: 1-R²=0.10, PR-gap=0.083, but Finf=0.086
  (transient on PR manifold, not converged)

### γ=3, τ=3 (the HANDOFF's documented PR regime):
- Picard α=0.3: 1-R²=1.5e-7, PR-gap=0 (FR)
- scipy newton_krylov from no-learning: p=0.769 (near-FR)
- NK from NL+0.1·rand: non-converged spurious (p=0.349)
- NK from NL+0.2·rand: non-converged (p=0.794)
- NK from NL + sign-tilt: p=0.767 (FR)
- *None* reached the documented PR target (p=0.632, μ=(0.645, 0.633, 0.645))

### Forward τ-homotopy at γ=3 (τ = 3.00 → 3.50, step 0.05, warm-starting):
| τ    | 1-R²    | PR-gap | p     |
|------|---------|--------|-------|
| 3.00 | 5.4e-7  | 0      | 0.770 |
| 3.05 | 2.4e-7  | 0      | 0.774 |
| 3.10 | 1.7e-7  | 0      | 0.777 |
| 3.15 | 1.5e-7  | 0      | 0.781 |
| 3.20 | 1.4e-7  | 0      | 0.784 |
| 3.25 | 1.3e-7  | 0      | 0.787 |
| 3.30 | 1.3e-7  | 0      | 0.791 |
| 3.35 | 1.3e-7  | 0      | 0.794 |
| 3.40 | 1.3e-7  | 0      | 0.797 |
| 3.45 | 1.3e-7  | 0      | 0.800 |
| 3.50 | 4.9e-3  | 0      | 0.804 (Finf=0.27, not converged) |

The basin jump documented in HANDOFF.md does not occur — Picard-warm-
started NK stays glued to FR all the way through τ=3.5. Above τ=3.5
there is simply no convergent fixed point in either basin.

## Why
The HANDOFF says the strong-PR branch was found by "Newton-Krylov from
warm starts that drift across the basin boundary at τ ≈ 3.39" — but
that requires a warm start with the *right initial perturbation* such
that NK lands in the PR basin instead of the FR basin. Tonight's seeds
either (a) snap back to FR via Picard pre-conditioning or (b) miss the
PR basin and land in the no-fixed-point divergent regime.

The historical PR results in `pchip_G11_backward_snap.csv` and
`pchip_G11_forward_snap.csv` came from runs whose specific seed
sequence is no longer reproducible — the cache.pkl currently in
the repo holds only FR-branch tensors.

## Effect on each placeholder

| Fig | Spec wants          | Currently shows       | Gap            |
|-----|---------------------|-----------------------|----------------|
| 4   | full REE @ γ=0.5    | full REE on FR branch | wrong fixed pt |
| 6   | full REE per config | no-learning           | needs PR REE   |
| 7   | full REE            | no-learning E[\|x\|]  | needs PR REE   |
| 8   | full REE            | no-learning V(τ)      | needs PR REE   |
| 9   | derived from Fig 8  | derived from no-learn | inherits #8    |
| 10  | full REE per K      | no-learning           | needs PR REE   |

## Files added during the hunt
- `python/probe_PR_seeds.py`        — analytic-Newton from 6 seeds
- `python/probe_picard_G11.py`      — Picard test at G=11
- `python/probe_picard_g3.py`       — Picard sweep around γ=3
- `python/probe_nk_pr.py`           — scipy newton_krylov from 4 seeds
- `python/forward_homotopy.py`      — τ-homotopy with warm-start chain
- `python/ree_solver.py`            — common Picard+Anderson helper

## Recommended unblock
1. Recover or recompute the historical PR-branch tensors (need to find
   the exact seed schedule that originally produced
   `pchip_G11_backward_snap.csv`).
2. OR: implement a globalised Newton-Krylov with perturbation-based
   homotopy that systematically scans for stable-but-not-Picard-
   reachable fixed points.

Without one of those, the production solver can only deliver the FR
branch — and these six figures will need to remain yellow placeholders
or be reframed as "no-learning benchmark" plots.
