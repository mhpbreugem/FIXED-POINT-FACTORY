# MIZN — code architecture (tree structure)

Modular, bite-size rebuild of the K=3 CRRA REE solver.
**Each file < 200 lines, one clear purpose, arrays in / arrays out.**

## The 5-step Hellwig loop (the architectural skeleton)

```
P = step1.conjecture(grid)
for it in range(maxit):
    mu     = step2.learn(P, grid, params)       # Bayesian updating
    x      = step3.demand(mu, P, params)        # individual demand
    Z      = step4.aggregate(x, grid, params)   # aggregate demand
    P_new, resid = step5.clear_and_verify(Z, P, params)
    P = solver.update(P, P_new, params)         # picard / anderson / NK
    if resid < tol: break
```

Each step is a **pure function**. Swap CARA → CRRA by flipping one string in
`config.py`. Swap linear u-grid → CDF ζ-grid by flipping another string.
`main.py` itself never changes.

## The tree

```
MIZN/
│
├── README.md                              one paragraph: what + how to run
├── pyproject.toml                         deps: numpy scipy matplotlib numba pytest jinja2 pyyaml
├── HANDOFF_SUMMARY.md                     why this repo exists, prior findings
├── MIZN_DESIGN.md                         this file
├── EXEC_LOG_DESIGN.md                     per-run logging system
├── MIZN_STARTUP.md                        reading order for new chat
│
├── .devcontainer/
│   └── devcontainer.json                  Python 3.11 + stack; local & Codespaces
│
├── .github/workflows/
│   ├── tests.yml                          pytest on every push
│   └── gamma_sweep.yml                    matrix fan-out for γ-sweep
│
├── src/mizn/
│   │
│   ├── __init__.py
│   ├── config.py                          ★ @dataclass Params: tau_th, tau_eps, tau_u,
│   │                                        rho, gamma, theta_bar, u_bar, G,
│   │                                        model ('cara'|'crra'),
│   │                                        grid ('u'|'xi'|'zeta'),
│   │                                        solver ('picard'|'anderson'|'nk'),
│   │                                        tol, maxit
│   │
│   ├── main.py                            ★ THE 5-STEP LOOP, ~30 lines
│   │
│   ├── signals.py                         f_v(u) Gaussian; logit; sigmoid;
│   │                                        Gauss-Legendre nodes/weights
│   │
│   ├── analytic.py                        ★ closed-form CARA-Gaussian REE
│   │                                        linear_REE(params) -> (a, b, c, tau_phi)
│   │                                        price_at(theta, u, params) -> P
│   │
│   ├── grid/                              SWAPPABLE coordinate system
│   │   ├── __init__.py                    def build(params) → dispatches
│   │   ├── base.py                        @dataclass Grid: nodes, weights, jacobian, edges
│   │   ├── linear_u.py                    uniform u ∈ [-Umax, Umax]
│   │   ├── atanh_xi.py                    ξ = tanh(u/T), bounded (-1,1)
│   │   └── cdf_zeta.py                    ζ = F̄(u) (Gaussian-mixture CDF); F̄⁻¹ via Brent
│   │
│   ├── step1_conjecture.py                P_conjecture(grid, params) → P array
│   │
│   ├── step2_learning/                    BAYESIAN UPDATING — swappable variants
│   │   ├── __init__.py                    def learn(P, grid, params) → mu
│   │   ├── gaussian.py                    Hellwig closed-form (CARA + Gaussian + supply noise)
│   │   ├── coarea_kernel.py               h>0: Gaussian kernel smoothing
│   │   └── coarea_h0.py                   h=0: cubic spline + GL + partition-of-unity
│   │                                        + smooth contour root-find (hfree_smooth port)
│   │
│   ├── step3_demand/
│   │   ├── __init__.py                    def demand(mu, P, params) → x
│   │   ├── cara.py                        x = (E_bar - P)/(ρ·v)   closed-form
│   │   └── crra.py                        x = (mu - P)/(ρ·mu(1-mu))   binary CRRA
│   │
│   ├── step4_aggregate.py                 ∫ x_i di  (continuum: avg signal = θ)
│   │                                        or K=3 finite sum across agents
│   │
│   ├── step5_clearing/
│   │   ├── __init__.py                    def clear_and_verify(Z, P, params) → (P_new, resid)
│   │   ├── cara.py                        P_new = E_bar - ρ·v·u   one-shot exact
│   │   └── crra.py                        bisection on logit (matches v11 reference)
│   │
│   ├── solvers/
│   │   ├── __init__.py                    def update(P, P_new, params)
│   │   ├── picard.py                      damped, adaptive ω
│   │   ├── anderson.py                    m=8 history, lstsq, clip
│   │   └── newton_krylov.py               scipy newton_krylov wrapper
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── metrics.py                     slope_T, deficit=1-R², d_FR, ‖F‖∞, ‖F‖₂
│   │   └── plotting.py                    contours, slices, ferr trajectory,
│   │                                        γ-sweep summary plots
│   │
│   └── continuation/                      γ-sweep / pseudo-arclength
│       ├── __init__.py
│       ├── adaptive_step.py               halve step on rejection
│       └── arclength.py                   pseudo-arclength for past-fold
│
├── tests/                                 ONE PER MODULE — tiny fixtures, fast
│   ├── conftest.py                        shared Params() fixtures
│   ├── test_config.py
│   ├── test_grid.py
│   ├── test_signals.py
│   ├── test_analytic.py
│   ├── test_step1.py
│   ├── test_step2.py                      ★ applied to analytic P → returns same beliefs
│   ├── test_step3.py
│   ├── test_step4.py
│   ├── test_step5.py                      ★ at analytic FP, clearing returns same P
│   ├── test_solvers.py
│   └── test_full_loop.py                  ★ GOLD: CARA default → analytic REE to eps
│                                            in ≤10 iters
│
├── tools/                                 EXEC LOG SYSTEM (see EXEC_LOG_DESIGN.md)
│   ├── runner.py                          THE wrapper: dirs, manifest, latex, pdf
│   ├── manifest.py                        per-file git_blob + sha256
│   ├── report_template.tex.j2             Jinja2 LaTeX template
│   └── index_filter.py                    search runs/index.json
│
├── scripts/                               RUNNABLE FROM CLI / ACTIONS
│   ├── nail_anchor.py                     (model, γ, τ, G) → nail to 1e-11, save
│   ├── gamma_sweep_adaptive.py            warm-start continuation, auto-commit per γ
│   ├── cara_descent.py                    FR ansatz at γ=∞, descend
│   ├── compare_operators.py               cross-check FPs across grids/operators
│   └── make_plots.py                      regenerate paper figures from data/
│
├── runs/                                  ONE FOLDER PER EXECUTION (see EXEC_LOG_DESIGN.md)
│   ├── INDEX.md                           append-only one-line summary table
│   ├── index.json                         machine-readable index
│   └── 2026-06-01T14-23-18Z__cara_default__a4f7c2/
│       ├── config.yaml
│       ├── manifest.json                  per-file versions
│       ├── git_state.txt
│       ├── env.txt
│       ├── stdout.log
│       ├── results.json
│       ├── artifacts/
│       │   ├── P_final.npy
│       │   ├── ferr_history.npy
│       │   └── plots/*.png
│       ├── report.tex                     auto-generated
│       └── report.pdf                     compiled summary
│
├── data/                                  COMMITTED FIXED POINTS (small .npy)
│   ├── fixed_points/
│   │   ├── cara_default.npy
│   │   ├── hfree_G9_g0.1_machine_prec.npy ← prior session ‖F‖=9.4e-16 anchor
│   │   └── ...
│   ├── sweeps/
│   │   └── hfree_G9_gamma_sweep.json      21-point table from prior session
│   └── README.md                          provenance
│
├── notebooks/                             OPTIONAL — Colab-friendly exploration
│   ├── 01_hellwig_walkthrough.ipynb
│   └── 02_strict_h0_PR.ipynb
│
└── paper/                                 LaTeX writeup
    ├── main.tex
    ├── refs.bib
    └── figures/                           generated by scripts/make_plots.py
```

## Build order (each step ~1 PR, < 200 lines)

1. `pyproject.toml`, `.devcontainer/`, `.github/workflows/tests.yml`
2. `src/mizn/config.py` + `tests/test_config.py`
3. `signals.py` + `analytic.py` + tests
4. `grid/linear_u.py` + `grid/base.py` + `grid/__init__.py` + test
5. `step1_conjecture.py` (CARA analytic init) + test
6. `step2_learning/gaussian.py` (Hellwig closed-form) + test
7. `step3_demand/cara.py` + `step4_aggregate.py` + `step5_clearing/cara.py` + each test
8. `solvers/picard.py` + test
9. `main.py` + `tests/test_full_loop.py` ← **green = scaffolding works**
10. `tools/runner.py` + `tools/manifest.py` + `report_template.tex.j2`
11. Run `nail_anchor.py` on CARA default → first PDF report
12. Add `solvers/anderson.py`, `solvers/newton_krylov.py`
13. Add `grid/atanh_xi.py`, `grid/cdf_zeta.py` — confirm CARA test green on each
14. Add `step2_learning/coarea_kernel.py` (h>0, easier) — re-find kernel PR FP
15. Add `step2_learning/coarea_h0.py` (port hfree_smooth) — re-nail machine precision
16. Add `step3_demand/crra.py`, `step5_clearing/crra.py` — switch model='crra'
17. `scripts/gamma_sweep_adaptive.py` + Actions matrix → reproduce 21-point sweep

## Design rules (don't break these)

- **One file, one purpose.** If `step2_learning/coarea_h0.py` exceeds 250 lines,
  split into `coarea_h0_spline.py` + `coarea_h0_quadrature.py`.
- **Pure functions.** Arrays in, arrays out. No global state in `src/mizn/`.
- **Tests run in <30 seconds total.** Use G=5 fixtures for tests; G=9+ for actual runs.
- **Grid never knows the operator; operator never knows the grid.** Both talk
  through the `Grid` dataclass (nodes, weights, jacobian, edges).
- **The dispatcher pattern** in `step*/__init__.py` makes adding a new variant
  a one-file change (e.g. `coarea_dd.py` for double-double later).
- **Every script calls `tools/runner.run(...)`.** No direct experiments outside
  the logging system.

## The CARA Hellwig math (verified — needed for step2_learning/gaussian.py)

Primitives: θ~N(θ̄,1/τ_θ), s_i=θ+ε_i with ε~N(0,1/τ_ε), supply u~N(ū,1/τ_u),
CARA risk ρ, continuum of traders.

**Closed-form linear REE** (price P = a + bθ − cu):
```
τ_φ = τ_u · (τ_ε/ρ)²                # price informativeness
τ_1 = τ_θ + τ_ε + τ_φ
a   = τ_θ·θ̄ / τ_1
b   = (τ_ε + τ_φ) / τ_1
c   = ρ·(τ_ε + τ_φ) / (τ_1·τ_ε)
```
Default params τ_θ=1, τ_ε=2, τ_u=1, ρ=2, θ̄=ū=0  →  τ_φ=1, τ_1=4, a=0, b=0.75, c=0.75.

**The 5-step numerical operator** (converges to the above):
- step2 (gaussian): regress P on (θ,u) → (a,b,d=-c); τ_φ = τ_u(b/d)²;
  φ=(P-a)/b; E_bar = (τ_θθ̄ + τ_ε·θ + τ_φ·φ)/τ_1; v = 1/τ_1
- step3 (cara):     x = (E_bar - P)/(ρ·v)
- step4:            D = x   (continuum: avg signal = θ exactly)
- step5 (cara):     excess = D - u; P_new = P + excess·(ρv);
                    one-shot exact: P_new = E_bar - ρ·v·u

**Gold test:** apply this map to the analytic price → returns it to
machine eps. That's `tests/test_full_loop.py`.
