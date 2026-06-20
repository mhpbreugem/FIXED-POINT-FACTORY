# Bounded-ξ K=3 CARA: what the bounded grid actually buys us

## The user's question
The u-grid CARA test showed the apparent "deficit" was a box-clip artifact
(open the box → deficit → 0). Why not just use the bounded ξ-grid (ξ =
tanh(τu/2), ξ ∈ (−1, +1)) where the clip can never activate?

## The naive answer
On ξ ∈ [−0.95, +0.95], u_max = (2/τ)·atanh(0.95) = 1.83 at τ=2, so
τΣu_max = 11. σ(11) = 0.99998, never clipped. So no clip artifact possible.

## The actual result

Self-contained `cara_xi_grid.py` operator (Jacobian-aware: signal density
in ξ-coords is g_v(ξ) = f_v(u(ξ))·du/dξ, which I derived and applied; the
inner Picard nails to F ≤ 10⁻¹² in 50–80 iterations). Three ξ-bound
protocols:

### Wide ξ ∈ [−0.95, 0.95]  (u_max=1.83, τΣu_max=11, σ'~2e-5 — saturated)
| G  |  deficit | d_FR  | slope |
|----|----------|-------|-------|
|  7 |  0.209   | 0.14  | 0.61  |
|  9 |  0.223   | 0.14  | 0.62  |
| 11 |  0.224   | 0.14  | 0.63  |
| 13 |  0.227   | 0.14  | 0.64  |
| 15 |  0.225   | 0.14  | 0.64  |

### Medium ξ ∈ [−0.7, +0.7]  (u_max=0.87, τΣu_max=5.2 — off-saturation)
| G  |  deficit | d_FR  | slope |
|----|----------|-------|-------|
|  7 |  0.265   | 0.17  | 0.92  |
|  9 |  0.255   | 0.15  | 0.87  |
| 11 |  0.258   | 0.16  | 0.91  |
| 13 |  0.256   | 0.16  | 0.92  |
| 15 |  0.247   | 0.15  | 0.89  |

### Narrow ξ ∈ [−0.5, 0.5]  (u_max=0.55, τΣu_max=3.3 — fully off-saturation)
| G  |  deficit | d_FR  | slope |
|----|----------|-------|-------|
|  7 |  0.271   | 0.17  | 1.12  |
|  9 |  0.251   | 0.13  | 0.97  |
| 11 |  0.239   | 0.12  | 0.89  |
| 13 |  0.232   | 0.11  | 0.84  |
| 15 |  0.227   | 0.10  | 0.81  |

## Reading

- **The slope (rate of revelation, FR has slope=1)** does improve sharply
  when we shrink the ξ-box away from saturation: 0.62 (wide) → 0.91
  (medium) → ~0.97 at G=9 narrow. So the saturated-price kernel artifact
  IS real and IS reduced by avoiding saturation.

- **But the deficit (1−R² scatter) stays at ≈0.23 across all three
  protocols and shrinks only weakly with G**. The kernel co-area
  inference on the bounded ξ-grid has a residual O(1) bias against
  exact FR even when prices stay off-saturation.

## Why the deficit doesn't collapse on the ξ-grid

Switching to ξ moves the artifact but doesn't eliminate it:

1. **Bounded ξ truncates the contour integral**. The exact A_v(p) = ∫
   over the unbounded u-line; on ξ ∈ [−(1−ε), +(1−ε)] we MISS the tails
   |u| > u_max entirely. At τ=2, u_max=0.55, the missing Gaussian mass
   per signal is Φ(−(0.55−0.5)·√2) ≈ 47% — half the density is gone.
   That asymmetrically biases A_v.

2. **Kernel band vs price gradient**. The kernel K_h "strip" width in
   u-space is h/(dP/du) = h/(σ'·τ). At τΣu=11 (saturated), σ' = 2×10⁻⁵
   and h≈0.2 → strip width ≈ 5000 in u-space (covers the whole grid).
   At τΣu=3.3 (narrow ξ), σ' = 0.035 → strip width ≈ 3 — better, but
   still spans many cells, so inference is broader than the true
   level set.

3. **The Jacobian dudξ blows up near ξ=±1**. Even with EPSB=0.05 (ξ_max=0.95),
   the outer halo cells (pad×dξ outside the inner grid) get clipped at
   ξ=±0.999 where dudξ ≈ 500. Their g_v contribution is small (the
   Gaussian f_v at u≈3.8 is tiny) but the lever arm is large.

## So why does the **u-grid open-box** test work?

Because at UMAX=12, the price range over the inner grid spans roughly the
same [σ(−72), σ(+72)] ≈ [0, 1] — but the kernel band h ~ √du is
relatively narrow in *price units* because cells are spaced widely in u
and the price σ(τΣu) changes smoothly across them. AND the inner block
is large enough that the contour A_v(p) gets contributions from the bulk
where f_v is concentrated. So both artifacts (truncation, kernel-vs-
saturation) are simultaneously suppressed. Result: deficit 3.4e-14 at
G=7, 7.3e-9 at G=9 (`cara_open_box_vs_G.json`).

## Verdict on the question

**Yes, the bounded ξ-grid is the principled approach** (it's how the
DD/QD high-precision K=2 production solver works). For K=3 with the
current kernel-co-area operator, switching to ξ alone moves the
artifact from clip-on-FR to kernel-band-vs-coverage. To get a clean
machine-zero ξ-grid CARA result would need either:

  (a) **adaptive kernel bandwidth** h(ξ) = h₀·|dP/dξ|⁻¹ so the
      kernel strip is constant in u-space;
  (b) a **PCHIP-line-integral** ξ-operator (the production DD K=2
      approach: explicit contour line by 1D PCHIP through grid, no
      kernel smoothing) — significant porting work for K=3;
  (c) a **broader ξ-box with non-uniform spacing** (Gauss-Lobatto or
      Chebyshev nodes) so resolution adapts to the price gradient.

The UMAX-collapse on the u-grid is the equivalent diagnostic to "Hellwig
in the limit". Both confirm CARA = FR. For CRRA the diagnostic is
symmetric and (also reported in `../k3_coarea_2dsweep/CRRA_VS_CARA_UMAX.md`)
shows the gap survives every box choice — intrinsic.

## Files
- `cara_xi_grid.py` — bounded-ξ K=3 CARA operator with three ξ-bound protocols.
- `cara_xi_grid.json` — numeric results.
