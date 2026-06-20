# Newton-Krylov attempt on the (u_1, Σ̂, δ̂) cube — fails

## Setup

- (u_1, Σ̂, δ̂) cube, G=31 uniform, γ=0.1
- F(P) = Φ(P) − P with Fix A+B (tail BC + symmetry projection)
- Picard warmup: 5 iters (reaches ferr ≈ 1.5e-2, d_FR ≈ 4.6e-4 — near a fixed point)
- Newton via scipy.sparse.linalg.lgmres with FD-based Jacobian-vector products
- LGMRES: inner_m=15, maxiter=6 outer (capped to avoid runaway cost)

## Result

After 5 Picard warmup steps the state is well-conditioned:
- ‖F‖_∞ = 0.0146
- 1−R²(T*) = 2.5e-6 (essentially FR with small wedge content)
- d_FR = 4.6e-4

First Newton-Krylov step:
- LGMRES returns info=6 (maxiter reached without convergence)
- ‖δ‖_∞ = 2.7 × 10³ — HUGE relative to P ∈ [0, 1]

The step is in a direction where (∂Φ/∂P − I) is near-singular: GMRES
amplifies noise into the null-space-like direction. After clipping P
back into [ε, 1−ε], the state is destroyed:
- ‖F‖_∞ = 0.69 (50× worse than before Newton)
- 1−R² = 0.68 (260,000× worse)
- d_FR = 0.36 (800× worse)

A second Newton step worsens it further.

## Why this happens

1. **Ill-conditioned Jacobian.** ∂Φ/∂P − I has eigenvalues clustered near 0 
   (the model is near a marginal contractor at γ=0.1 — that's why Picard
   limit-cycles). GMRES on such a matrix gives unstable steps.

2. **FD Jacobian noise.** Each matvec is one extra Φ call with ε=1e-6
   perturbation. The discrete contour scan + bisection clearing has a
   noise floor ~1e-7 even without perturbation; the FD divides by ε so
   the noise floor in J·v is ~1e-1 / 1e-6 = 1e+5 amplified. Bad.

3. **Symmetry-projection in the matvec.** Applying Fix B inside the matvec
   means each matvec is itself a non-trivial operation; the effective J
   has small eigenvalues from the (3-dimensional) symmetric-subspace
   projector that GMRES sees as singular directions.

## What works instead

The earlier 5K-iter **adaptive-damping Picard** run (γ=0.05 case)
contracted to ferr = 4.5×10⁻⁵ once ω auto-tuned to ~0.05. Damped Picard
is the correct tool here:

- ω = 1 (pure Picard) → limit cycle at ferr ≈ 10⁻²
- ω = 0.05 (auto-damped) → contracting to ferr ≈ 10⁻⁵ over 5K iters

The reason: Picard with ω < 1 effectively places (1−ω)·I + ω·∂Φ inside
the contraction map, which has spectral radius < 1 when ∂Φ has spectral
radius < 2 (much weaker condition than Picard's ρ(∂Φ) < 1).

## Recommendation

- Use **damped Picard** for routine fixed-point finding.
- If Newton is needed: pair it with **trust-region clipping** (cap ‖δ‖
  per step) or use a **preconditioned** Newton-Krylov where M ≈ J^{-1}
  has been built analytically. Neither is implemented.

## Files

- `AB_newton.py` — Newton-Krylov attempt (run for reproducibility)
- `figures/AB_newton_trajectory.png` — final state if the run completes;
  currently it shows the divergence (Newton step destroys Picard's work)
