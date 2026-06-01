# FIXED-POINT-FACTORY — Extended Session Summary & Handoff

> Purpose: a self-contained brain-dump so a **fresh chat in a fresh repo** can continue
> the research with zero loss of context. Read this top-to-bottom before starting.

---

## 0. TL;DR (read this first)

We study the **noiseless (h=0) K=3 CRRA Rational-Expectations Equilibrium (REE)**:
3 agents, each sees a private Gaussian signal of a binary asset value `v ∈ {0,1}`,
signal precision `τ=2`, CRRA risk aversion `γ`. The object solved for is the
equilibrium **price function** `P(u₁,u₂,u₃) ∈ [0,1]` on a 3-D cube.

**The single most important result of the whole session:**
> The strict-h=0 K=3 CRRA REE **has a genuine partially-revealing (PR) fixed point**
> at γ=0.1, τ=2. It was nailed to **machine precision ‖F‖∞ = 9.4×10⁻¹⁶** on a
> linear u-grid at G=9 (the `hfree_smooth` operator), and a **γ-sweep** from
> γ=0.000576 → 0.261 was produced, all points at ‖F‖ ≤ 1×10⁻¹¹.

**Three more load-bearing facts:**
1. **Initial condition matters as much as the operator.** The same h=0 operator has
   (≥) two basins: a **PR basin** (slope≈0.36, deficit≈0.17) reachable from a
   *kernel-warm-start + Newton*, and a **near-FR basin** (slope≈0.85–0.96) reachable
   from *no-learning Picard*. Months of apparent "the operator drifts to FR" were
   really "Picard from the wrong IC lands in the wrong basin."
2. **The residual floor is discretization, not arithmetic.** Verified twice:
   `mpmath dps=50` and `flint dps=100` both reproduce the float64 worst-cell residual
   to the same order (e.g. 5.08e-3 vs 4.67e-3). Double-double / arbitrary precision
   **cannot** lower the ‖F‖ floor or remove the fold — only larger G or a better
   operator can.
3. **There is a fold/bifurcation around γ ≈ 0.26–0.30** at G=9 where the PR branch
   destabilizes. From the CARA side (γ large, FR ansatz), Newton descends into a
   **second PR branch** (slope≈0.66, deficit≈0.10) instead of full FR — so the CARA
   limit at G=9 is itself PR, not FR. The two PR branches don't meet across the fold.

---

## 1. The economic model (precise statement)

- Binary asset value `v ∈ {0,1}`, symmetric prior P(v=1)=½.
- Agent k ∈ {1,2,3} observes `u_k = v + noise`, noise ~ N(0, 1/τ), τ=2.
  Signal density: `f_v(u) = sqrt(τ/2π) · exp(-½ τ (u - m_v)²)`, with means
  `m_0 = -½, m_1 = +½`.
- Each agent forms a posterior μ_k = P(v=1 | u_k, price) and submits a CRRA demand.
- Market clears → equilibrium price P ∈ [0,1].
- **Noiseless**: there is NO supply noise. Partial revelation (if any) comes from the
  Jensen gap of CRRA aggregation, NOT from supply shocks. (Contrast with the
  *standard* Hellwig model in §9 which DOES have supply noise and a closed form.)

**The fixed-point operator Φ** (Bayes + clear), one application:
```
for each cube cell (i,j,k) with price p = P[i,j,k]:
    for each agent: A_v(p) = co-area evidence integral over the other 2 signals
                    μ_agent = f_1·A_1 / (f_0·A_0 + f_1·A_1)        # Bayes
    P_new[i,j,k] = CRRA_clear(μ_1, μ_2, μ_3, γ)                    # market clearing
```
The **co-area evidence** is the contour integral
`A_v(p) = ∫_{P=p} f_v f_v / |∇P| dσ`, computed on a 2-D slice of the cube.

**CRRA clearing** (bisection on log-price, matches all our code):
```
given μ's and γ, find m solving  Σ_k (R_k - 1)/((1-m) + R_k·m) = 0,
where R_k = exp((logit(μ_k) - logit(m))/γ).  Return m.
```
As γ→∞ this → CARA log-odds average `sigmoid(mean(logit μ_k))`.

**Key diagnostics** (computed on the inner cube vs `T = τ·Σu_k`):
- `slope_T` = slope of regression logit(P) ~ T. FR (fully-revealing) ⇒ slope=1.
- `deficit = 1 - R²` of that regression. FR ⇒ 0. PR ⇒ >0.
- `d_FR` = RMS distance of P from the FR price `sigmoid(T)`.

---

## 2. Operator classes explored (the zoo)

| Operator | Smoothing | Where | Nails PR? |
|---|---|---|---|
| **Kernel co-area** | Gaussian bandwidth h≈0.32 (= C·√du, C=0.45) | `k3_coarea_sweep`, `k3_coarea_2dsweep` | YES, ‖F‖~5e-11, but **over-smoothed** (slope=0.18, deficit=0.28) |
| **hfree_smooth** ★ | NONE (h=0): cubic spline + Gauss-Legendre (decoupled) + partition-of-unity + smooth contour root-find | `k3_hfree_smooth/hfree_operator.py` | **YES, ‖F‖=9.4e-16** (slope=0.364, deficit=0.172) ← the winner |
| hfree_morse | h=0 + Morse-critical-robust quadrature | `k3_hfree_morse` | yes-ish, cures a 1e-3 floor |
| strict_h0_cdf | h=0 sub-level-set CDF-derivative (flint arb) | `k3_strict_h0_cdf` | diverged in original test |
| σ-δ V11 | h=0 cubic spline on (u₁,Σ,δ) atanh ξ-cube | `solver_code/sigma_delta/v11_strict_h0_hardwired.py` | YES via NK+FR-BC+warm-start, ‖F‖=8.4e-7 |
| u-grid linear scan | h=0 linear interp marching-squares | `contour_K3_halo.phi_K3_halo` | NO — oscillates near-FR |
| u-grid Hermite cubic | h=0 Hermite scan | `phi_K3_halo_cubic` | NO — oscillates near-FR |
| CDF-cube (this session) | h=0, ζ=F̄(u) axes | `solver_code/sigma_delta/cdf_cube_*.py` | soft only, ‖F‖~0.04 floor |

★ = the operator that works. Everything important traces back to `hfree_smooth`.

---

## 3. The three grid/coordinate choices

The cube of unknowns `P_ijk` can be laid out three ways (decouple grid from operator!):

1. **Linear u-grid**: `u_i` uniform in [-4, 4]. Original `hfree_smooth` rep. Wastes
   nodes in tails (signal density peaks at u=±½).
2. **σ-δ atanh ξ-cube**: coords (u₁, Σ=u₂+u₃, δ=u₂-u₃), each atanh-stretched
   `ξ_k = tanh(u_k/T_k)` onto bounded (-1,1)³. FR boundary at ξ=±1 (u=±∞ ⇒ P=0/1).
3. **CDF-cube** (user's idea this session): axes `ζ_k = F̄(u_k)` where
   `F̄(u) = ½Φ(√τ(u+½)) + ½Φ(√τ(u-½))` is the unconditional Gaussian-mixture signal
   CDF. Uniform in ζ ⇒ nodes concentrate where signal mass is. `u_k = F̄⁻¹(ζ_k)` via
   Brent. **Wider PR basin**: no-learning IC is already PR-like; first Picard step
   moves slope 0.13→0.45 (vs linear u-grid 0.19→0.73, σ-δ 0.25→0.79 — both escape PR).

**Cross-check result:** none of the u-grid PR FPs are CDF-cube FPs (all ferr≈0.5).
Each (grid × operator) defines a distinct discrete operator with its own PR FP.

---

## 4. The γ-sweep (the headline deliverable)

Adaptive-step continuation on `hfree_smooth` G=9, strict tol 1e-11, warm-start from
the machine-precision γ=0.1 anchor, halve step on rejection. **21 accepted points.**

**DOWN (γ < 0.1), all ‖F‖ ≤ 1e-12:**
```
γ        slope_T   deficit   d_FR
0.07     0.36261   0.17507   0.2583
0.049    0.36167   0.17650   0.2592
0.04165  0.36141   0.17687   0.2594
0.029155 0.36104   0.17735   0.2595
0.020409 0.36075   0.17763   0.2595
0.014286 0.36051   0.17781   0.2596
0.010000 0.36033   0.17792   0.2596
0.007000 0.36020   0.17799   0.2596
0.004900 0.36010   0.17804   0.2596
0.003430 0.36003   0.17808   0.2597
0.002401 0.35998   0.17810   0.2597
0.001681 0.35995   0.17812   0.2597
0.001177 0.35992   0.17813   0.2597
0.000824 0.35990   0.17814   0.2597
0.000576 0.35989   0.17814   0.2597
```
**γ→0 asymptote: slope ≈ 0.3599, deficit ≈ 0.1781, d_FR ≈ 0.2597.**

**Anchor + UP (γ > 0.1):**
```
0.1      0.36412   0.17249   0.2566   (anchor, ‖F‖=9.4e-16)
0.15     0.36681   0.16812   0.2535
0.225    0.37082   0.16283   0.2502
0.253125 0.37246   0.16157   0.2498
0.25886  0.37283   0.16109   0.2494
0.261247 0.37298   0.16088   0.2493   ← last accepted before fold
```
UP-sweep aborts at γ≈0.263 (adaptive step shrinks below 0.5% min). **Fold here.**

**CARA-side descent** (FR ansatz at γ=100, descend): all REJ at 1e-11 but lands in a
**second PR branch**: γ=100→3 all give slope≈0.66, deficit≈0.10, d_FR≈0.003,
‖F‖~2e-4 (G=9 discretization floor for that branch). So the fold separates a
"deep PR" branch (slope 0.36) from a "shallow PR" branch (slope 0.66).

---

## 5. The 7 repair attempts on the ‖F‖~0.04 CDF-cube floor (all documented)

Tried to push the CDF-cube G=13 below its ~0.04 NK plateau:
1. **cubic-spline-clip** (clip spline eval to [0,1]): best PR result, ‖F‖=0.038, slope=0.66.
2. **PCHIP** (monotone, no overshoot): WORSE, ‖F‖=0.46 — monotonicity kinks break NK.
3. **Morse partition-of-unity** (per-root w=dA²/(dA²+dB²)): ‖F‖=0.051, PR.
4. **sub-level-set CDF** (bilinear, no contour, no topology jumps): ‖F‖=0.46 — bilinear C⁰ seams break NK.
5. **smooth-A_v(p)** (exact line integral at 64 p-samples + spline-in-p fit): ‖F‖=0.073, PR, best "smooth" idea.
6. **pinned boundary** (pin ζ-faces to P=0/1): ‖F‖_inner=0.005 BUT forces near-FR (slope=0.96) — residual tight, equilibrium character lost. Don't use.
7. (Spectral diagnostic): sampled Jacobian σ_min(I-J)=0.50 → healthy; NK *should* converge.
   The blocker is **operator non-smoothness at Morse-critical prices** (contour topology
   changes as p crosses a critical value of P — root count jumps discretely). The
   kernel (h>0) smears this over O(h); h=0 has the jump. This is the deep "why h=0 is hard."

**Why IFT doesn't save us:** the implicit function theorem gives a locally smooth
contour ζ_b(ζ_a, p) *only where ∂P/∂ζ_b ≠ 0*. At Morse-critical points (saddles/extrema
of the slice surface) the gradient vanishes, two roots merge/are born, the *root list*
changes discretely, and Φ is only C⁰ (not C¹) in p there. That's the irreducible
non-smoothness of strict h=0.

---

## 6. Precision investigation (settled)

- `quick_gmpy2_test.py`: gmpy2 mpfr 50-digit `crra_clear` vs float64 → diff = 0.000e+00.
- `cdf_cube_mpmath.py` dps=50: Picard iter-1 ferr = 0.4938 = **identical** to float64 numba.
- `flint_dps100_one_cell.py` / `flint_dps100_off_center`: worst-cell |Φ-FR| at γ=10 G=9:
  float64 = 4.673e-3, flint dps=100 = 5.080e-3 (same order, 8.7% diff = discretization noise).

**Verdict: arithmetic precision is NOT the bottleneck anywhere we looked.** Float64 is
sufficient. The lever is **larger G** (discretization order) or a **different operator**.
flint speed note from earlier: flint arb ≈ 3× faster than gmpy2 for bulk ops at dps=100.

---

## 7. Solver lessons

- **Picard from no-learning IC → near-FR basin** (the trap). Adaptive damped Picard
  oscillates (ferr alternates ~0.07/0.50) around the PR FP but never nails it.
- **Anderson(m=8)** accelerates but plateaus at the same operator floor; good for
  warm-starting NK.
- **Newton-Krylov (scipy `newton_krylov`, method='lgmres')** is the only thing that
  nails to machine precision — BUT only from a PR-basin warm-start. From no-learning it
  diverges or lands near-FR.
- **The winning recipe:** kernel-co-area PR FP (or analytic warm-start) → interpolate →
  NK on the strict-h=0 operator → machine precision. Then **γ-continuation** with NK,
  small steps, warm-start each γ from the previous.
- σ-δ V11 needs **FR boundary conditions** at ξ=±1 (P=0/1); the "warm-halo" variant
  (no FR-BC) diverges to NaN.

---

## 8. Platform / tooling decisions (discussed, not yet acted on)

- **Python + numba is the right stack.** Measured on a 4-core box, our kernel pattern
  (cube loop × quadrature × bisection): pure Python 1367 ms → numba serial 2.9 ms (476×)
  → numba parallel 0.8 ms (1791×). numpy can't vectorize the data-dependent bisection.
- **Julia ≈ numba for float64** (both LLVM); Julia's only real win is precision-generic
  code (one `phi(P::Array{T})` runs Float64/Double64/BigFloat) + no per-process JIT
  warmup. NOT worth a port for speed. Reconsider only if precision-generic + big-G scaling
  becomes the wall. High-prec in Julia: DoubleFloats.jl (~32 dig), MultiFloats.jl (~64/128),
  Quadmath, BigFloat (MPFR), ArbNumerics/Nemo (= flint/arb native).
- **GPU is the WRONG tool** for this code: branchy, data-dependent per-cell work
  (bisection, variable root counts) → warp divergence. GitHub Codespaces is **CPU-only**
  anyway (GPU beta deprecated).
- **Compute platforms:**
  - **GitHub Codespaces** = cloud VM, connect from browser or local VS Code. Free tier
    120 core-hours/mo (=60 hr on 2-core, 7.5 hr on 16-core). Beyond: ~$0.09/core-hour
    ($0.18/hr 2-core … $2.88/hr 32-core) + $0.07/GB-mo storage. Aggressive auto-stop is key.
  - **Dev Containers** = same `devcontainer.json` run LOCALLY on your own PC via Docker
    Desktop — free, your cores. Best for day-to-day.
  - **GitHub Actions** = free unlimited minutes on **public** repos, 2-core runners,
    ~20 concurrent jobs, 6-hr job cap. Great for **matrix-fanout γ-sweeps** (one job per γ,
    commit results back). Caveat: warm-start continuation is sequential (do a 2-phase:
    cheap sequential coarse anchors → parallel independent refine). Mild TOS caution:
    keep it tied to repo artifacts, don't run a 20×6h farm continuously.
  - **JuliaHub** only if you commit to Julia.
  - **Recommendation: develop locally via Dev Containers (free) + burst to a 16-32 core
    Codespace or an Actions matrix for the heavy sweeps.**

---

## 9. The NEXT TASK (what we were mid-building): clean 5-step skeleton

User wants a **fresh, modular, bite-size rewrite** they can code themselves one piece at
a time, starting from the **standard Hellwig (1980) CARA model with supply noise** (which
has a **closed-form linear REE** to test against — unlike the noiseless CRRA model above).

### The classic Hellwig 5 steps
1. **Conjecture price** P(·) (initial guess)
2. **Bayesian learning** — given conjectured P, each trader's posterior
3. **Individual demand** — CARA: `x_i = (E_i - p)/(ρ·Var_i)`
4. **Aggregate demand** — integrate over traders
5. **Market clearing & verify conjecture** — clear, compare P_new to P

### Proposed file layout (each file tiny, pure functions, arrays in/out)
```
main.py              # the 5-step loop, ~30 lines, reads like the paper
config.py            # τ_θ, τ_ε, τ_u, ρ, grid params — one place (dataclass Params)
analytic.py          # closed-form linear REE coeffs (for tests)
grid.py              # build (θ, u) grid — swappable
signals.py           # Gaussian posterior algebra + price-coeff regression
step1_conjecture.py  # P_conjecture(grid) -> P array  (naive linear init, NOT the answer)
step2_learning.py    # learn(P, grid, params) -> beliefs (E_bar, v, tau_phi, a, b, d)
step3_demand.py      # individual_demand(E, v, P, params) -> x = (E-P)/(ρv)
step4_aggregate.py   # aggregate(x, ...) -> D  (continuum: avg signal = θ)
step5_clearing.py    # clear_and_verify(D, beliefs, grid, params, P) -> (P_new, resid)
solvers/picard.py    # update(P, P_new, damping)
tests/               # test_signals, test_grid, test_operator_fixed_point (analytic
                     #   self-consistency to machine eps), test_convergence
```

### The standard Hellwig CARA-Gaussian model (THE MATH, verified this session)
Primitives: θ~N(θ̄,1/τ_θ) payoff; s_i=θ+ε_i, ε~N(0,1/τ_ε); supply u~N(ū,1/τ_u);
CARA ρ; continuum of traders (avg signal = θ exactly).

**Closed-form linear REE** (price p = a + bθ − cu):
```
τ_φ = τ_u · (τ_ε/ρ)²            # price informativeness about θ — CLOSED FORM, no fixed point
τ_1 = τ_θ + τ_ε + τ_φ
a   = τ_θ·θ̄ / τ_1
b   = (τ_ε + τ_φ) / τ_1
c   = ρ·(τ_ε + τ_φ) / (τ_1·τ_ε)     # note b/c = τ_ε/ρ exactly
```
Clean default params: τ_θ=1, τ_ε=2, τ_u=1, ρ=2, θ̄=ū=0
⇒ τ_φ=1, τ_1=4, a=0, b=0.75, c=0.75.

**The numerical 5-step map on a (θ,u) grid** (converges to the above, verifiable):
```
step2: regress P on (θ,u) → (a, b, d=-c); τ_φ = τ_u·(b/d)²;
       price-signal φ = (P-a)/b;  E_bar = (τ_θθ̄ + τ_ε·θ + τ_φ·φ)/τ_1;  v = 1/τ_1
step3: x = (E_bar - P)/(ρ·v)
step4: D = x         (continuum averaging: avg signal = θ; documented identity for CARA)
step5: excess = D - U(supply);  P_new = P + excess·(ρv);  resid = max|P_new - P|
       (one-step exact for CARA: P_new = E_bar - ρ·v·u)
```
**Proven self-consistent:** applying this map to the analytic price returns it to
machine precision (the analytic REE is the operator's fixed point). That's the gold
test `test_operator_fixed_point`.

This CARA model is the **warm-up / scaffolding**. Once the 5-step structure is clean and
tested, swap `step3` (CARA→CRRA) and `step2` (Gaussian-closed-form → co-area integral)
to rebuild the hard noiseless-CRRA operator from §1–§7 inside the same clean architecture.

---

## 10. Repo geography (where the good stuff lives)

Current repo `mhpbreugem/fixed-point-factory`, branch
`claude/study-fixed-point-economics-y12PB`:

- `projects/REZN/solved_fixed_points/k3_hfree_smooth/`
  - `hfree_operator.py` — **THE working strict-h=0 operator** (cubic spline + GL + PoU)
  - `P_nailed_G9.npy` — PR FP at G=9 (slope 0.364)
  - `gamma_sweep/` — the 21-point γ-sweep .npy files
  - `gamma_sweep_adaptive.py`, `cara_side_descent.py`
- `projects/REZN/solver_code/sigma_delta/`
  - `hfree_G9_machine_prec.npy` — the ‖F‖=9.4e-16 anchor
  - `v11_strict_h0_hardwired.py` — σ-δ V11 operator
  - `cdf_cube_numba.py` (+ `_pchip`, `_morse`, `_sublevel`, `_smooth_p`, `_pinned`) — CDF-cube zoo
  - `nail_machine_prec.py`, `spectral_diag.py`, `quick_gmpy2_test.py`, `flint_dps100_*`
  - `check_ugrid_FPs_on_cdf_cube.py` — cross-check
- `projects/REZN/solved_fixed_points/k3_coarea_2dsweep/reznsrc/contour_K3_halo.py`
  — kernel co-area operators (`phi_K3_halo_smooth` = the h=0.32 one that finds PR)
- `projects/REZN/solved_fixed_points/methodology/` — LaTeX papers
  - `extended_paper.{tex,pdf}`, `cdf_cube_paper.{tex,pdf}` (9 sections, figures)
- `projects/REZN/solved_fixed_points/plots/` — all figures
- `hellwig/` — (just started) the clean 5-step skeleton; **incomplete, can be regenerated**

---

## 11. Pitfalls / gotchas (don't repeat these)

- Don't conclude "operator drifts to FR" from Picard alone — **try NK from a PR warm-start**.
- Don't reach for higher precision to fix a discretization floor — **it won't help** (proven).
- Don't pin boundaries to tighten ‖F‖ — it **silently forces near-FR** (lost the PR).
- Don't use PCHIP / bilinear for the slice interp — their kinks/seams **break NK**.
- Keep **grid and operator decoupled** — half the early confusion was tangling them.
- The kernel bandwidth h is NOT noise; it's a smoothing that regularizes the Morse-critical
  topology jumps. Removing it (h=0) is what makes the problem genuinely hard.
- numba `parallel=True` + `cache=True`; pay JIT warmup once per process (1–40 s).
- Background sweeps: write to a log file, `tail -F | grep`, auto-commit+push per accepted point.

---

## 12. Suggested first moves in the fresh repo

1. Make it a **public** GitHub repo (free unlimited Actions).
2. Add `.devcontainer/devcontainer.json` (Python 3.11 + numpy scipy matplotlib numba pytest).
3. Build the **CARA Hellwig 5-step skeleton** from §9 first — it has a closed-form
   answer so every step is unit-testable. Get `test_operator_fixed_point` green
   (analytic price is a fixed point to machine eps).
4. Then swap in CRRA demand (§1) and the co-area learning step, reusing the working
   `hfree_operator.py` math, inside the clean architecture.
5. For the γ-sweep at scale: GitHub Actions matrix (2-phase: coarse anchors → parallel refine).
6. To push past the γ≈0.26 fold or the ‖F‖ floor: **increase G** (9→13→17→21) — the only
   lever that moves discretization — and/or pseudo-arclength continuation across the fold.

---

*End of handoff. This document is saved at `HANDOFF_SUMMARY.md` in the repo root.*
