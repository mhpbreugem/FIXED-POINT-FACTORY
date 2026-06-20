"""PDF describing the discrete-price-grid proposal (user idea).

Pages:
 1. Title + proposal text
 2. Why GS degenerates at strict h=0 (visual: level sets)
 3. The discrete-price proposal: positive-measure partition cells (visual)
 4. Bayes comparison: measure-zero conditioning vs positive-measure
 5. Algorithm pseudocode (text)
 6. Why this should give a nontrivial FP (analytic argument)
 7. Initial guess: discretize the kernel solution into M bands
 8. Two solver paths: hard K-means vs soft-max relaxation
 9. Expected outcome and pass/fail criteria
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import build_grid, extract_surfaces_marching_cubes
from reznsrc.contour_K3_halo import init_no_learning_K3

BUILD = '/tmp/discrete_pdf'
os.makedirs(BUILD, exist_ok=True)
PAGES = []

def save(fig, name):
    fig.savefig(f"{BUILD}/{name}.png", dpi=140, bbox_inches='tight')
    plt.close(fig); PAGES.append(name)

# ---------- P1: Title + proposal ----------
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("Discrete-Price-Grid Strict-$h{=}0$ Equilibrium\n"
             "A proposal to break the Grossman-Stiglitz degeneracy",
             fontsize=14, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
THE PROBLEM (recap):
Under binary payoff, the strict-$h{=}0$ operator $\Phi_0$ admits the
fully revealing $P_{FR}(u) = \sigma(\tau \sum_k u_k)$ as its unique fixed
point.  Conditioning on the level set $\{P_{FR} = p\}$ (a measure-zero
hyperplane) gives every trader $\mu_k = p$ pointwise, so no one trades
and self-consistency holds trivially.  The partially-revealing
equilibrium found by the kernel-$h>0$ operator does not satisfy any
natural strict-$h{=}0$ fixed-point equation (verified empirically).

THE NEW PROPOSAL (user's idea):
Force the price function to take values on a FIXED DISCRETE GRID:
   $P : \mathbb{R}^3 \to \{p_1, p_2, \ldots, p_M\}$
with $p_1 < p_2 < \ldots < p_M$ prescribed in $(0, 1)$.

Then the conditioning event on observing $P = p_m$ is the
PARTITION CELL
   $\Pi_m = \{u \in \mathbb{R}^3 \,:\, P(u) = p_m\}$,
a positive-measure subset of $\mathbb{R}^3$, NOT a measure-zero
level surface.

CONSEQUENCES:
1. Bayes is classical: $\mu_k(u_k, \Pi_m) = \Pr(v=1 \mid u_k, u \in \Pi_m)$,
   well-defined as a conditional probability with positive Bayes denominator.
2. The GS sufficient-statistic identity that pinned $\mu_k = p$ at the
   FR level set NO LONGER HOLDS on the positive-measure region.  There
   is "room" for $\mu_k \ne p_m$ within each $\Pi_m$.
3. Equilibrium = a PARTITION $\{\Pi_1, \ldots, \Pi_M\}$ of $\mathbb{R}^3$
   such that, for every $u \in \Pi_m$, the CRRA clearing price (given the
   partition-induced posteriors) equals $p_m$.

THIS IS A GENUINELY NEW EQUILIBRIUM CONCEPT.  It is different from:
  -- the kernel-$h>0$ family (which smooths the conditioning event)
  -- the pointwise strict-$h{=}0$ equilibrium (which has only $P_{FR}$)

It corresponds to a real-world feature: prices are quoted on a tick
grid (discrete cents), so traders observe a tick-discretized price.

PASS/FAIL:
  PASS: the solver converges to a partition with $\Phi$-residual zero,
        positive deficit, distinct from $P_{FR}$.
  FAIL: the only partition that satisfies clearing collapses to the
        "P_FR-quantized" assignment (each $u$ goes to nearest $P_{FR}(u)$
        value on the grid), with degenerate posteriors.

In the FAIL case, even the partition formulation cannot escape
$P_{FR}$ -- a much stronger GS degeneracy than the level-set version.
"""
ax.text(0.04, 0.94, txt, fontsize=9.5, va='top', ha='left', family='serif')
save(fig, 'p01_proposal')

# ---------- P2: Smooth P vs discrete P (schematic 2D) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
# Smooth FR slice
x = np.linspace(-3, 3, 200)
ax = axes[0]
ax.plot(x, 1.0 / (1.0 + np.exp(-2.0*x)), 'b-', linewidth=2, label=r'$P_{FR}(u) = \sigma(\tau \cdot \sum u)$')
ax.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.7, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\sum_k u_k$', fontsize=13); ax.set_ylabel(r'$P$', fontsize=13)
ax.set_title('Smooth (continuous-value) FR price\n'
             'level sets are measure-zero hyperplanes',
             fontsize=11)
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
# Discrete P (step function)
p_levels = np.linspace(0.1, 0.9, 9)
discrete_P = np.array([p_levels[np.argmin(np.abs(p_levels - 1/(1+np.exp(-2.0*xi))))] for xi in x])
ax = axes[1]
ax.step(x, discrete_P, where='mid', color='C3', linewidth=2,
        label=r'$P_{\rm disc}(u) \in \{0.1, 0.2, \ldots, 0.9\}$')
ax.plot(x, 1.0 / (1.0 + np.exp(-2.0*x)), 'b--', alpha=0.5, label='smooth FR')
ax.set_xlabel(r'$\sum_k u_k$', fontsize=13); ax.set_ylabel(r'$P$', fontsize=13)
ax.set_title('Discrete-price proposal: P takes only finite values\n'
             'level sets are POSITIVE-MEASURE BANDS',
             fontsize=11)
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
fig.suptitle('Two representations of the equilibrium price',
             fontsize=12, weight='bold')
save(fig, 'p02_smooth_vs_discrete')

# ---------- P3: 3D conditioning event comparison ----------
# Use a kernel solution to show partition cells
Gi = 21
du, uf, lo, hi = build_grid(Gi)
tau, gamma = 2.0, 0.01
P_inner = np.load(f"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/P_ld_t{tau}_g{gamma}.npy")
P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
P_full[lo:hi, lo:hi, lo:hi] = P_inner
u_in = uf[lo:hi]

fig = plt.figure(figsize=(13, 6))
# Left: measure-zero level set
ax = fig.add_subplot(121, projection='3d')
surf = extract_surfaces_marching_cubes(P_full, uf, [0.5])[0]
if surf is not None:
    v, f = surf
    tri = v[f[::3]]
    col = Poly3DCollection(tri, alpha=0.5, color='C0', edgecolor='none')
    ax.add_collection3d(col)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$'); ax.set_zlabel('$u_3$')
ax.set_title('Strict-$h{=}0$: $\\{P = 0.5\\}$\n2D surface (MEASURE-ZERO)\n'
             'GS degeneracy: $\\mu_k = p$ on this set',
             fontsize=10)
ax.view_init(elev=18, azim=35)
# Right: positive-measure partition cell (price band)
ax = fig.add_subplot(122, projection='3d')
# Show the band [0.4, 0.6] by extracting outer and inner surfaces
for p_band in [0.4, 0.5, 0.6]:
    surf = extract_surfaces_marching_cubes(P_full, uf, [p_band])[0]
    if surf is None: continue
    v, f = surf
    tri = v[f[::3]]
    alpha_v = 0.4 if abs(p_band - 0.5) < 0.01 else 0.15
    col_v = 'C2' if abs(p_band - 0.5) < 0.01 else 'C2'
    col = Poly3DCollection(tri, alpha=alpha_v, color=col_v, edgecolor='none')
    ax.add_collection3d(col)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$'); ax.set_zlabel('$u_3$')
ax.set_title('Discrete-grid: $\\Pi_{0.5} = \\{P \\in [0.4, 0.6]\\}$\n'
             '3D BAND (POSITIVE-MEASURE)\nBayes classical, no GS degeneracy',
             fontsize=10)
ax.view_init(elev=18, azim=35)
fig.suptitle('The key difference: conditioning event geometry',
             fontsize=12, weight='bold')
save(fig, 'p03_conditioning')

# ---------- P4: Bayes comparison (math) ----------
fig = plt.figure(figsize=(8.5, 9))
fig.suptitle('Bayes posterior: measure-zero vs positive-measure conditioning',
             fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
STRICT $h{=}0$ (measure-zero level set):
   $\mu_k(u_k, p) \;=\; \dfrac{f_1(u_k) \cdot A_1(p, u_k)}{f_0(u_k) \cdot A_0(p, u_k) + f_1(u_k) \cdot A_1(p, u_k)}$

   where
   $A_v(p, u_k) \;=\; \int_{\{P = p\} \cap \{u_k\ {\rm fixed}\}} f_v(u_j) f_v(u_l) \, d\sigma$

   PROBLEM: at $P = P_{FR}$, the constraint $\{P = p\}$ pins $\sum_k u_k$
   exactly.  The Gaussian integrals close in closed form, every trader's
   posterior collapses to $\mu_k = p$, CRRA clearing returns $p$ trivially.
   $P_{FR}$ is an exact fixed point but the equilibrium is degenerate.

DISCRETE-PRICE GRID (positive-measure partition cell):
   $\mu_k(u_k, \Pi_m) \;=\; \dfrac{\int_{\Pi_m \cap \{u_k\ {\rm fixed}\}} f_1(u_j) f_1(u_l) \, du}
                                  {\int_{\Pi_m \cap \{u_k\ {\rm fixed}\}} f_0(u_j) f_0(u_l) \, du + (\textrm{symmetric})}$

   The conditioning set $\Pi_m \cap \{u_k\ {\rm fixed}\}$ is a 2D region
   (positive measure in $\mathbb{R}^2$), not a 1D curve.  Standard Bayes.
   The Gaussian-closed-form identity that pinned $\mu_k = p$ NO LONGER
   APPLIES: the integration is over a 2D region, not a 1D level set.

KEY OBSERVATION:
   As the price grid is refined ($M \to \infty$), the partition cells
   $\Pi_m$ shrink toward the level surfaces $\{P = p\}$.  The Bayes
   posteriors approach the strict-$h{=}0$ posteriors.
   But for any FINITE $M$, the equilibrium concept is genuinely
   different from the strict pointwise version.

   For the binary-payoff GS degeneracy, the issue is specifically the
   MEASURE-ZERO conditioning -- any positive-measure conditioning
   (whether by kernel smoothing OR by discrete-price quantization)
   should break the degeneracy.

ECONOMIC INTERPRETATION:
   A discrete price grid is what we observe in real markets: prices
   are quoted in ticks (e.g., \$0.01 increments).  An equilibrium under
   tick-discretized prices is the natural object; the continuous-price
   abstraction is the model assumption that creates the GS degeneracy
   under binary payoff.
"""
ax.text(0.03, 0.93, txt, fontsize=9.5, va='top', family='serif')
save(fig, 'p04_bayes_math')

# ---------- P5: Algorithm ----------
fig = plt.figure(figsize=(8.5, 9))
fig.suptitle('Solver algorithm', fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
STATE:
  Discrete-price grid $p_1 < p_2 < \ldots < p_M$ (fixed).
  Partition assignment $a : \{1, \ldots, G^3\} \to \{1, \ldots, M\}$
  (which cube cell goes to which price level).

EQUILIBRIUM CONDITIONS:
  For every cube cell $c$ with assignment $a(c) = m$ and signal vector
  $u^{(c)}$:
    1. Posteriors $\mu_k = \mu_k(u_k^{(c)}, \Pi_m)$ via Bayes on the
       partition.
    2. CRRA clearing: $p^{\rm clear}(\mu_1, \mu_2, \mu_3; \gamma) = p_{a(c)}$.

TWO SOLVER PATHS:

  Path A (hard K-means style):
    Iterate:
      i. For each cell $c$: compute current $\mu_k$, then $p^{\rm clear}$.
      ii. Reassign $a(c) := \arg\min_m |p_m - p^{\rm clear}(c)|$.
      iii. Stop when assignment stops changing.
    Fixed point: partition matches clearing prices.
    Discrete dynamics; may oscillate; easy to monitor.

  Path B (soft-max relaxation):
    Replace hard assignment by soft weights
      $w_m(c) = {\rm softmax}_m({\rm logit}(p_m) - {\rm logit}(p^{\rm clear}(c)) / T)$
    with temperature $T > 0$.  The price field is
      $P(c) = \sum_m p_m w_m(c)$.
    Solve $P(c) = p^{\rm clear}(c)$ as a continuous fixed-point problem
    via scipy.optimize.least_squares.  Anneal $T \to 0$.

INITIAL GUESS:
  Quantize the kernel-$h>0$ certified solution $P_h$:
    $a^{(0)}(c) := \arg\min_m |p_m - P_h^*(c)|$.
  This places the kernel solution on the discrete grid.

PASS CRITERION:
  Converged partition with $\max_c |p_{a(c)} - p^{\rm clear}(c)| < {\rm tol}$,
  where ${\rm tol}$ should be smaller than half the grid spacing
  $\min_m |p_m - p_{m+1}| / 2$.

FAIL CRITERION:
  Iteration converges to the "P_FR-quantized" partition with all
  $\mu_k = p_m$ exactly (degenerate posteriors, same as continuous P_FR).
"""
ax.text(0.03, 0.93, txt, fontsize=9.5, va='top', family='serif')
save(fig, 'p05_algorithm')

# ---------- P6: Why this might work (analytic argument) ----------
fig = plt.figure(figsize=(8.5, 9))
fig.suptitle('Why a non-trivial discrete-price fixed point should exist',
             fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
INFORMAL ARGUMENT:

1.  The kernel-$h>0$ family has a unique nontrivial fixed point $P_h^*$
    for every $h > 0$, with positive deficit.  This is established by
    the existence theorem of the main paper.

2.  The kernel smoothing replaces the indicator $\mathbf{1}\{P(u) = p\}$ by a
    Gaussian $K_h(P - p)$ supported on $[p - h, p + h]$.  This is
    equivalent to averaging over a band of width $\sim h$.

3.  The discrete-price quantization replaces $\mathbf{1}\{P(u) = p\}$ by
    $\mathbf{1}\{P(u) \in [p_m - \delta_m, p_m + \delta_m]\}$ where the bands
    are the partition cells $\Pi_m$.  This is a HARD-EDGED averaging
    over an explicit partition.

4.  Both modifications turn the measure-zero conditioning event into a
    positive-measure event.  By continuity:
       $P_h^*$ exists for all $h > 0$ (proven)
       $\Rightarrow$ discrete-grid fixed point should exist for any
         partition that "resembles" a kernel-induced one.

5.  The natural candidate:
       Discretize the kernel solution $P_h^*$ on a price grid with
       spacing $\sim h$.  The resulting partition has $\Pi_m$ similar
       in size to the kernel's effective conditioning band.

6.  If the soft-max relaxation converges, the limit $T \to 0$ gives a
    hard partition that approximately satisfies the discrete-grid
    fixed-point equation.

POTENTIAL OBSTRUCTION:
  The hard-K-means iteration may have NO fixed point.  This happens if
  the clearing price under any candidate partition always pulls cells
  AWAY from their assigned level.  In particular, if all clearing
  prices cluster near 0.5 (regardless of partition), then the
  "P_FR-quantized" partition is the unique attractor, even at the
  partition level.  This would be a STRONGER GS degeneracy.

EITHER OUTCOME IS INFORMATIVE:
  PASS: we have a strict-$h{=}0$ partition equilibrium with positive
    deficit, breaking the degeneracy that the continuous-P version had.
    The paper gains a new equilibrium concept and a real-world
    interpretation (tick-discretized prices).

  FAIL: the GS degeneracy is even stronger than previously thought
    -- even discrete-price conditioning collapses to FR.  This would be
    a striking impossibility result.
"""
ax.text(0.03, 0.93, txt, fontsize=9.5, va='top', family='serif')
save(fig, 'p06_why_should_work')

# ---------- P7: Initial guess visual ----------
fig, ax = plt.subplots(figsize=(9, 6))
# Show how the kernel solution gets quantized
P_slice = P_inner[10]  # middle slice at u_3=0
levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ax.contourf(u_in, u_in, P_slice, levels=levels, cmap='RdBu_r')
ax.contour(u_in, u_in, P_slice, levels=[0.2, 0.5, 0.8], colors='k', linewidths=0.5)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$')
ax.set_title('Initial guess: certified kernel $P_h^*(u_1, u_2, u_3=0)$ quantized to\n'
             'the discrete price grid $\\{0.05, 0.1, 0.2, 0.3, \\ldots, 0.95\\}$\n'
             '(each color band = one partition cell $\\Pi_m$)', fontsize=11)
fig.colorbar(ax.collections[0], ax=ax)
save(fig, 'p07_initial_guess')

# ---------- P8: Roadmap ----------
fig = plt.figure(figsize=(8.5, 9))
fig.suptitle('Implementation roadmap', fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
ROADMAP (this PDF will be followed by the actual implementation):

STAGE 1: hard K-means iteration at G=8
  - Compute kernel-certified P at G=8 (~5 seconds)
  - Quantize to discrete grid M = 8, 16
  - Iterate: clearing price -> reassign -> repeat
  - Monitor: assignment change rate, max |p_clear - p_assigned|
  - Expected wall: minutes per (G, M) combination

STAGE 2: soft-max relaxation at G=8, then G=11
  - scipy.optimize.least_squares on soft assignment weights
  - Temperature anneal T = 0.1 -> 0.01
  - Final hard assignment via argmax of soft weights
  - Verify hard partition satisfies the equilibrium

STAGE 3: refinement test
  - Refine M (8 -> 16 -> 32 levels) and check whether deficit
    converges to the kernel-h limit
  - Refine G (8 -> 11 -> 15) and check stability

STAGE 4: documentation
  - Build "discrete-price equilibrium" report with all results
  - Update main paper with new equilibrium concept
  - Compare with kernel-h limit empirically

PASS/FAIL DELIVERABLE:
  At end of Stage 1-2, definitive answer to "does a non-trivial
  discrete-price strict-h=0 equilibrium exist for this model?"
"""
ax.text(0.03, 0.93, txt, fontsize=9.5, va='top', family='serif')
save(fig, 'p08_roadmap')

# ---------- combine ----------
imgs = [Image.open(f"{BUILD}/{p}.png").convert('RGB') for p in PAGES]
W = max(im.width for im in imgs)
norm = []
for im in imgs:
    if im.width != W:
        h2 = int(im.height * W / im.width)
        im = im.resize((W, h2))
    norm.append(im)
dst = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cmm_endogenous/DISCRETE_PRICE_PROPOSAL.pdf'
norm[0].save(dst, save_all=True, append_images=norm[1:])
print(f"saved {dst} ({len(PAGES)} pages)")
