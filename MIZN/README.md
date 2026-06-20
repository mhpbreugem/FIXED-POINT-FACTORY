# MIZN

Modular spectral solver for the noiseless K=3 CRRA / CARA rational-expectations
equilibrium. Clean rebuild of the work from
[fixed-point-factory](https://github.com/mhpbreugem/fixed-point-factory);
see the top-level `MIZN_DESIGN.md`, `MIZN_EXEC_LOG_DESIGN.md`, and
`MIZN_STARTUP.md` for the architectural spec.

## Quick start

```bash
pip install -e .[test]
pytest             # full suite, < 30s
python -m mizn     # default CARA run, prints residual & metrics
```

Each `scripts/*.py` calls `tools.runner.run(...)` which writes a timestamped
folder under `runs/` with config, manifest, results, plots, and a compiled
LaTeX report.

## Layout

5-step Hellwig loop (`src/mizn/main.py`):
```
P  = step1.conjecture(grid)
for it in range(maxit):
    mu     = step2.learn(P, grid, params)
    x      = step3.demand(mu, P, params)
    Z      = step4.aggregate(x, grid, params)
    P_new, resid = step5.clear_and_verify(Z, P, params)
    P = solver.update(P, P_new, params)
```
Each step is a pure function. Swap models/grids/solvers by flipping a string
in `Params`.
