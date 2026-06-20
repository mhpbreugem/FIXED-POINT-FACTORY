# MIZN — fresh-start reading order

Welcome. This repo (`MIZN`) is a clean rebuild of the noiseless K=3 CRRA REE
solver work whose history lives in `mhpbreugem/fixed-point-factory`. The four
documents below are the **complete handoff**. Read them in this order:

1. **`HANDOFF_SUMMARY.md`** — what we learned in the previous repo
   (the model, the operator zoo, the machine-precision PR result at γ=0.1,
   the γ-sweep, the fold, what worked and what didn't). ~3000 words, 12 sections.

2. **`DESIGN.md`** — the tree structure for THIS repo: 5-step Hellwig
   skeleton (CARA closed-form warm-up → swap in CRRA noiseless), pure
   functions, swappable grids, swappable operators. Build order.

3. **`EXEC_LOG_DESIGN.md`** — per-run logging system: every execution
   produces a versioned subfolder with `manifest.json` (per-file git_blob +
   sha256), `config.yaml`, `results.json`, auto-generated LaTeX `report.tex`
   and compiled `report.pdf`. Top-level `runs/INDEX.md` for search.

## Quick rules of engagement for the fresh chat

- **Code by hand in bite-size pieces.** Each PR < 200 lines, ideally one
  file. Tests stay green at every step.
- **Decouple grid from operator** (the #1 confusion source last time).
- **Don't reach for higher precision to fix a discretization floor** —
  it doesn't help (proven with flint dps=100).
- **Always use a PR-basin warm-start before Newton-Krylov.** No-learning
  IC + Picard lands in the wrong (near-FR) basin.
- **Start with the CARA Hellwig model** — it has a closed-form linear REE
  that lets every step be unit-testable. Then swap in CRRA.

## Suggested first commits in this order

1. `pyproject.toml`, `.devcontainer/devcontainer.json`, `.github/workflows/tests.yml`
2. `src/mizn/config.py` + `tests/test_config.py`
3. `src/mizn/signals.py` + `src/mizn/analytic.py` + tests
4. `src/mizn/grid/linear_u.py` + test
5. `step1_conjecture.py` (CARA analytic init) + test
6. `step2_learning/gaussian.py` (Hellwig closed-form) + test
7. `step3_demand/cara.py`, `step4_aggregate.py`, `step5_clearing/cara.py` + tests
8. `solvers/picard.py` + test
9. `main.py` + `tests/test_full_loop.py` ← gold test: green = scaffolding works
10. Then add `tools/runner.py` + LaTeX report template (per EXEC_LOG_DESIGN.md)
11. Then add Anderson, NK, atanh-ξ grid, CDF-ζ grid, kernel co-area operator
12. Then port the strict-h=0 `hfree_smooth` operator (the one that nails PR)
13. Then CRRA demand+clearing → re-find the machine-precision PR FP at γ=0.1
14. Then `scripts/gamma_sweep_adaptive.py` + Actions matrix → reproduce the 21-point sweep

If you keep this discipline, you'll re-derive everything we have plus the
modularity to push past the fold.
