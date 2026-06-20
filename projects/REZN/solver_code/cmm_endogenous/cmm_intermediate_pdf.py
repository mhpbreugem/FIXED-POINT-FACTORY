"""CMM intermediate-results PDF: progression Stage 1c -> 3a -> 3b (and 3c when ready)."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cmm_endogenous'
BUILD = '/tmp/cmm_int_build'
os.makedirs(BUILD, exist_ok=True)

# Stage 1c
m1c = json.load(open(f"{ROOT}/stage1c_meta.json"))
Pn_kernel = np.load(f"{ROOT}/Pn_cmm_nbins128.npy")  # actually this IS Phi_CMM at n=128
# Stage 3b
m3b = json.load(open(f"{ROOT}/stage3b_jit.json"))
# Stage 3a sample
m3a = json.load(open(f"{ROOT}/stage3a_sample.json"))
# Stage 3c (may not exist yet)
have_3c = os.path.exists(f"{ROOT}/stage3c_newton.json")
m3c = json.load(open(f"{ROOT}/stage3c_newton.json")) if have_3c else None

# ============ PAGE 1: progression ============
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("Co-Area Mesh Method — strict $h{=}0$ progression",
              fontsize=14, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = [
    "Test case throughout: $(\\tau,\\gamma)=(2,0.098)$ at $G{=}21$",
    "(emin15-certified longdouble FP of the kernel operator).",
    "",
    "Stage 1c — cube-based strict-$h{=}0$ forward eval ($\\Phi_{\\rm CMM}$):",
    f"  cube cell residual (median): {6.230e-3:.3e}",
    f"  cube cell residual (max): {3.30e-1:.3e}",
    f"  wall time (full G=21): {813:.0f}s",
    "  -> Confirms strict h=0 matches kernel within O(h^2) bulk,",
    "     O(h) tails near critical-price regions.",
    "",
    "Stage 2 — iteration test of cube-based CMM from kernel FP:",
    "  iter 0 -> 1 -> 2: 0.33 -> 0.46 -> 0.47 (DIVERGES)",
    "  -> Cube state is the wrong representation: marching cubes",
    "     re-triangulates as P[i,j,l] updates, breaking smoothness.",
    "",
    "Stage 3a — moving-mesh prototype:",
    f"  Mesh: {m3a['n_surfaces']} surfaces, {m3a['total_verts']} vertices",
    f"  Vertex residual (500-sample): max {m3a['max']:.2e}, "
    f"median {m3a['median']:.2e}",
    "  -> Vertex-level residual is 30x larger than cube-level",
    "     residual: strict h=0 disequilibrium was hidden by coarse sampling.",
    "  -> Pure Python: 120 ms/vertex (~90 min full eval, too slow for Newton).",
    "",
    "Stage 3b — JIT'd residual (numba, per-triangle 2-point Gauss-Legendre):",
    f"  Mesh: {m3b['n_surfaces']} surfaces, {m3b['total_v']} verts",
    f"  Full-residual eval: {m3b['eval_s']:.1f}s (was 90 min in Python; "
    f"{int(90*60/m3b['eval_s'])}x speedup)",
    f"  max |r|: {m3b['max_r']:.3e}",
    f"  median |r|: {m3b['median_r']:.3e}",
    f"  p90 |r|:  {m3b['p90_r']:.3e}",
    "  -> Confirms 3a numbers; opens Newton at minute-scale cost.",
    "",
    "Stage 3c — Newton-Krylov on moving mesh:",
]
if m3c is not None:
    hist = m3c['F_history']
    txt += [
        f"  Iterations: {m3c['n_iter']}",
        f"  Final max |r|: {m3c['final_max_r']:.3e}",
        f"  Reduction: {m3c['reduction']:.1f}x",
        f"  Residual history: {' -> '.join(f'{x:.2e}' for x in hist)}",
    ]
else:
    txt += ["  (still running — see below)"]
ax.text(0.05, 0.94, "\n".join(txt), fontsize=10, va='top', ha='left', family='sans-serif')
plt.savefig(f"{BUILD}/p1.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 2: histograms ============
fig, axes = plt.subplots(2, 1, figsize=(8.5, 9))
fig.suptitle("Residual distributions: cube vs mesh", fontsize=14, weight='bold')
ax = axes[0]
# fake cube hist proxy (just illustrate values from m1c)
labels = ['Stage 1c (cube cells)', 'Stage 3b (mesh vertices)']
medians = [6.23e-3, m3b['median_r']]
maxes = [3.30e-1, m3b['max_r']]
xs = np.arange(2)
ax.bar(xs - 0.2, medians, width=0.4, label='median |r|', color='C0')
ax.bar(xs + 0.2, maxes, width=0.4, label='max |r|', color='C1')
ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_yscale('log'); ax.set_ylabel('|residual|')
ax.set_title("median is 30x higher at vertices: cube hid the strict-h=0 disequilibrium")
ax.legend(); ax.grid(alpha=0.3, axis='y')
# Newton history if available
ax = axes[1]
if m3c is not None:
    hist = m3c['F_history']
    ax.semilogy(hist, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Newton iteration'); ax.set_ylabel('max |r|')
    ax.set_title("Stage 3c Newton-Krylov contraction history")
else:
    ax.text(0.5, 0.5, '(Stage 3c history will appear when it completes.)',
             ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.axis('off')
ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f"{BUILD}/p2.png", dpi=150, bbox_inches='tight'); plt.close()

# ============ PAGE 3: takeaways ============
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("Key learnings", fontsize=14, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = [
    "1. The strict-$h{=}0$ forward operator EXISTS, is computable in"
    " seconds at $G{=}21$, and disagrees with the kernel operator at",
    "   the rate $O(h)$ in tails, $O(h^2)$ in bulk -- matching what the",
    "   deep G-ladder data already implied.",
    "",
    "2. Cube-based CMM cannot be solved by Newton:",
    "   re-triangulation jumps in $\\Phi_{\\rm CMM}$ as cube values update",
    "   destroy smoothness. This is the same defect that killed the",
    "   historical strict $\\varphi_{K3}^{\\rm halo}$ operator.",
    "",
    "3. Moving-mesh CMM (vertex positions = primary DOFs, topology",
    "   pinned) eliminates that defect by construction.",
    "",
    "4. JIT'd residual at $\\sim 0.6$s/eval enables Newton iteration at",
    "   minute-scale cost. The first Newton step's contraction will tell",
    "   us if the strict $h{=}0$ fixed point is reachable from the kernel FP.",
    "",
    "Pending (Stage 3c, in progress):",
    "  - Newton-Krylov on a 22734-vertex per-surface scalar displacement",
    "    parametrization (vertex moves along its surface normal).",
    "  - Convergence test: residual must contract to <1e-6 over <=8 iters.",
    "  - If converged: compute strict-h=0 deficit at $(\\tau,\\gamma)=(2,0.098)$",
    "    and compare with the deep-ladder continuum estimate",
    "    $d_\\infty = 0.268 \\pm 0.010$.",
    "",
    "Caveats:",
    "  - Mesh quality must be monitored (no triangle flips).",
    "  - The initial mesh comes from marching cubes of the kernel FP;",
    "    if the strict h=0 fixed point has different surface topology,",
    "    Newton will fail or land on a wrong solution.",
    "  - Stage 3c uses normal-only displacement (1 DOF/vertex). Tangential",
    "    rearrangement, if needed, requires the full 3 DOF/vertex.",
]
ax.text(0.05, 0.94, "\n".join(txt), fontsize=10, va='top', ha='left')
plt.savefig(f"{BUILD}/p3.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- combine ----------
imgs = [Image.open(f"{BUILD}/p{i}.png").convert("RGB") for i in (1, 2, 3)]
dst = f"{ROOT}/CMM_intermediate_results.pdf"
imgs[0].save(dst, save_all=True, append_images=imgs[1:])
print(f"saved {dst}")
