# Chebyshev h=0 prototype (in-progress)

This is a minimal proof-of-concept for the symmetric Chebyshev spectral
solver described in `design_notes/mizn_book/`. It is built fresh in this
session at the user's request: STRICT h=0, Chebyshev basis, pure (dense)
Newton.

## Status (in development)

Working:
- atanh ξ-coordinates, c=2
- Chebyshev-Lobatto nodes at N=6 (343 cube cells)
- 3D coefficient ↔ value transforms (chebfit/chebval per axis)
- Co-area integral via:
  - 2D slice extraction (Einstein-summed contraction)
  - 1D Chebyshev polynomial extraction along contour axis
  - companion-matrix root-find via `chebroots` (exact, all roots)
  - Gauss-Legendre quadrature in transverse axis
  - Chebyshev-derivative computation via `chebder`
- Bayes posterior μ from co-area evidence A_v
- CRRA bisection clearing
- Damped Picard preconditioning
- Pure (dense) Newton with forward-difference Jacobian + Armijo line search

Not yet:
- S_3 × Z_2 symmetry reduction (would shrink 343 → ~30 unknowns)
- Sigmoid asymptotic lift (would handle boundaries exactly)
- numba JIT (currently ~2s per Phi evaluation pure Python)
- Tuned c stretch parameter

## Files

- `cheby_h0_solver.py`: the core operator (Phi, evaluation, root-find).
- `cheby_solve.py`: Picard + Newton wrapper, generates `run.log`.

## First results (CRRA γ=1, τ=1, N=6, c=2)

Picard 5 iters from no-learning IC:
- ||F||: 0.50 → 0.11
- slope_T: 0.12 → 0.26 (climbing toward deep PR)
- deficit: 0.32 → 0.21

Newton iter 1: in progress (343 forward-diff Phi-evals = ~12 min wall).

This run is at `cheby_h0_prototype/`; final results will be reported as
the Newton iterations complete.
