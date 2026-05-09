# Symmetric-K REE Solver

Solves the homogeneous-CRRA rational expectations equilibrium for K = 3..8
agents at G = 15, γ = 0.5, τ = 2.0 using permutation symmetry.

## Architecture

### Key insight

When all K agents share the same γ, τ, W, the price function P(u₁,...,uₖ) is
permutation-invariant. Instead of storing a G^K array, we store only the
C(G+K-1, K) sorted multiset cells. At G=15:

| K | Sorted cells | Full cells | Compression |
|---|-------------|------------|-------------|
| 3 | 680 | 3,375 | 5× |
| 4 | 3,060 | 50,625 | 17× |
| 5 | 11,628 | 759,375 | 65× |
| 6 | 38,760 | 11,390,625 | 294× |
| 7 | 116,280 | 170,859,375 | 1,469× |
| 8 | 319,770 | 2,562,890,625 | 8,016× |

### Files

- `contour_KN_sym.py` — symmetric Phi map and SymGrid indexing
- `solve.py` — dispatch hook: tasks with `"symmetric": true` run `_run_sym_task`

### Algorithm (`sym_phi`)

For each sorted cell s = (i₁ ≤ i₂ ≤ ... ≤ iₖ):
1. Representative agent 0 has signal u[i₁], observes price p = P[s].
2. Compute A₀, A₁ via contour integral over the (K-1)-D slice P[i₁, :,...,:]:
   scan all K-1 axes; at each sign-change crossing of P = p use linear
   interpolation to find the off-grid signal value; accumulate f₀/f₁ products.
   Average K-1 scan passes (mirrors `_agent_evidence_K3`).
3. Bayes update: μ = f₁(u[i₁])·A₁ / (f₀(u[i₁])·A₀ + f₁(u[i₁])·A₁).
4. Market clearing: since all K posteriors equal (by symmetry, same agent-
   type at same price), bisect over p for sum_k x_CRRA(μ, p, γ, W) = 0,
   which simplifies to x_CRRA(μ, p, γ, W) = 0, so p = μ.
5. Store result in new_P_sorted[s].

Iteration uses damped Picard: P ← (1-α)P + α·Φ(P) with α = 0.3.

### Validation

At K=3, G=15, γ=0.5, τ=2.0:
- `sym_phi` (no halo) converges to 1-R² ≈ 2.08e-5
- `phi_K3_halo(pad=0)` (same conditions) converges to 1-R² ≈ 2.12e-5
- Delta ≈ 3.8e-7 < 1e-6 ✓
- At G=5 pad=0: sym_phi == phi_K3_halo to machine precision (diff = 0.0) ✓

Note: the halo-padded K=3 reference (pad=4) converges to a different fixed
point (1-R² ≈ 0.0875) because halo boundary conditions alter the domain.

## Task queue

Tasks `symK3_g050_t0200_G15` through `symK8_g050_t0200_G15` form a warm-start
chain: K=3 runs first (cold start), each subsequent K warm-starts from K-1.
symK3 status: ready. symK4..K8 status: blocked (unblock after K-1 completes).
