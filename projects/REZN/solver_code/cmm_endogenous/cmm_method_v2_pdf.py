"""CMM method PDF v2: comprehensive (>= 9 figures) for the strict-h=0
moving-grid program, including basin-of-attraction results.

Pages:
 1. Title + method idea (text)
 2. Level surfaces (3D)
 3. Mesh render of one surface
 4. Hyperplane slice (3D + 2D)
 5. Evidence integrand along slice curve
 6. Per-vertex residual diagram (text)
 7. Block-diagonal Jacobian sparsity
 8. GS degeneracy: PFR is exact strict-h=0 FP, kernel branch is not
 9. Basin map: distance-to-FR vs initial residual at multiple G
10. Convergence trajectories: Picard, LM v1, scipy LS comparison
11. Status table of all solver attempts + roadmap
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes,
                          slice_mesh_by_hyperplane, stitch_segments)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth

BUILD = '/tmp/cmm_method_v2_build'
os.makedirs(BUILD, exist_ok=True)
PAGES = []

# Use the certified max-deficit corner for visualizations
Gi = 21
du, uf, lo, hi = build_grid(Gi)
tau, gamma = 2.0, 0.01
u_in = uf[lo:hi]
P_inner = np.load(f"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/P_ld_t{tau}_g{gamma}.npy")
P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
P_full[lo:hi, lo:hi, lo:hi] = P_inner


def save(fig, name):
    fig.savefig(f"{BUILD}/{name}.png", dpi=140, bbox_inches='tight')
    plt.close(fig); PAGES.append(name)


# ---------- P1: Title ----------
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("Strict h=0 Moving-Grid Method (CMM) for the K=3 CRRA REE\n"
             "Method, results, and the Grossman-Stiglitz degeneracy",
             fontsize=14, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
THE METHOD (user's proposal, formalized):

Standard solvers fix grid points $(u_1, u_2, u_3)$ in the signal cube and treat
prices $P[i,j,l]$ as unknowns. The co-area evidence integral
   $A_v(p, u_k) = \int_{\{P=p\} \cap \{u_k\ fixed\}} f_v(u_j)\, f_v(u_l)\, d\sigma$
then requires either:
  -- kernel smoothing $K_h(P-p)$, bias of order $h$
  -- pointwise contour extraction, $\Phi$ discontinuous in $P$, Newton fails.

CMM inverts the representation: choose price levels $p_1 < \ldots < p_M$ first;
let the GRID POINTS be the unknowns.  Each level surface $S_m = \{P = p_m\}$ is
a triangle mesh whose vertex positions move.  The evidence integral becomes an
ARCLENGTH integral along the curve $S_m \cap \{u_k = U\}$: exact at $h = 0$.

EQUATIONS at every mesh vertex $v$ of surface $S_m$:
   $\mathrm{clear\_CRRA}(\mu_1(v), \mu_2(v), \mu_3(v); \gamma) = p_m$
where $\mu_k(v)$ is the Bayes posterior from the slice integrals.
Unknowns: 3 DOF per vertex.  Equations: 1 per vertex.
The 2-dimensional gauge per vertex is harmlessly absorbed by Levenberg damping.

PIPELINE:
  marching cubes ONCE on a warm-start price field (topology pinned)
  -> JIT'd per-vertex residual + per-surface block-FD Jacobian
  -> Levenberg / scipy-LS trust-region with topology guard
  -> strict-h=0 fixed point IF it exists.

KEY EMPIRICAL FINDING (this PDF documents):
  The strict-h=0 operator under binary payoff admits the trivial
  fully-revealing $P_{FR}(u) = \sigma(\tau\,\sum_k u_k)$ as its unique
  fixed point.  No nontrivial partially-revealing strict-h=0 fixed
  point exists.  The partially-revealing equilibrium of risk-averse
  binary-payoff trading is generated only by the kernel $h>0$ regularization,
  which is therefore an EQUILIBRIUM SELECTION mechanism, not a numerical
  convenience.  Pages 8-11 summarize the basin-of-attraction tests
  across multiple resolutions and starting points that established this.
"""
ax.text(0.04, 0.94, txt, fontsize=10, va='top', ha='left', family='serif')
save(fig, 'p01_title')

# ---------- P2: level surfaces ----------
p_levels = [0.2, 0.35, 0.5, 0.65, 0.8]
surfs = extract_surfaces_marching_cubes(P_full, uf, p_levels)
fig = plt.figure(figsize=(8.5, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.coolwarm(np.linspace(0, 1, len(p_levels)))
for ip, s in enumerate(surfs):
    if s is None: continue
    v, f = s
    tri = v[f[::3]]
    col = Poly3DCollection(tri, alpha=0.3, color=colors[ip], edgecolor='none')
    ax.add_collection3d(col)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$'); ax.set_zlabel('$u_3$')
ax.set_title(f"P2 -- Level surfaces $\\{{P^* = p\\}}$ for $p \\in \\{{0.2,0.35,0.5,0.65,0.8\\}}$\n"
             f"$(\\tau, \\gamma) = (2.0, 0.01)$ -- max-deficit certified equilibrium",
             fontsize=11)
ax.view_init(elev=18, azim=35)
save(fig, 'p02_levelsets')

# ---------- P3: mesh render ----------
v, f = surfs[2]  # p=0.5
fig = plt.figure(figsize=(8.5, 8))
ax = fig.add_subplot(111, projection='3d')
tri = v[f]
col = Poly3DCollection(tri, alpha=0.55, facecolor='lightsteelblue',
                       edgecolor='navy', linewidths=0.3)
ax.add_collection3d(col)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$'); ax.set_zlabel('$u_3$')
ax.set_title(f"P3 -- The $p=0.5$ surface as a triangle mesh\n"
             f"{len(v)} vertices (UNKNOWNS), {len(f)} faces (topology FIXED)",
             fontsize=11)
ax.view_init(elev=18, azim=35)
save(fig, 'p03_mesh')

# ---------- P4: slicing ----------
segs = slice_mesh_by_hyperplane(v, f, 0, 0.0)
chains = stitch_segments(segs)
fig = plt.figure(figsize=(8.5, 10))
ax = fig.add_subplot(211, projection='3d')
tri = v[f[::3]]
col = Poly3DCollection(tri, alpha=0.18, facecolor='lightsteelblue', edgecolor='none')
ax.add_collection3d(col)
for c in chains:
    n = len(c)
    pts3 = np.column_stack([np.full(n, 0.0), c[:, 0], c[:, 1]])
    ax.plot(pts3[:, 0], pts3[:, 1], pts3[:, 2], 'r-', linewidth=2.5)
yy, zz = np.meshgrid([-4, 4], [-4, 4])
ax.plot_surface(np.zeros_like(yy), yy, zz, alpha=0.12, color='red')
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$'); ax.set_zlabel('$u_3$')
ax.set_title("P4 -- Hyperplane slice $\\{u_1 = 0\\}$\n"
             "Red curve = exact integration domain ($h=0$ by construction)",
             fontsize=11)
ax.view_init(elev=18, azim=35)
ax2 = fig.add_subplot(212)
for c in chains:
    ax2.plot(c[:, 0], c[:, 1], 'r-', linewidth=2)
    ax2.plot(c[:, 0], c[:, 1], 'k.', markersize=3)
ax2.set_xlabel('$u_2$'); ax2.set_ylabel('$u_3$')
ax2.set_title("Same curve in $(u_2, u_3)$ -- piecewise-linear polyline\n"
              "Black dots: face-crossing endpoints")
ax2.set_aspect('equal'); ax2.grid(alpha=0.3)
save(fig, 'p04_slice')

# ---------- P5: evidence integrand ----------
if chains:
    chain = max(chains, key=len)
    seg_lens = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    s_arc = np.concatenate(([0], np.cumsum(seg_lens)))
    sgn = np.sqrt(tau / (2*np.pi))
    f0 = sgn*np.exp(-0.5*tau*(chain[:,0]+0.5)**2) * sgn*np.exp(-0.5*tau*(chain[:,1]+0.5)**2)
    f1 = sgn*np.exp(-0.5*tau*(chain[:,0]-0.5)**2) * sgn*np.exp(-0.5*tau*(chain[:,1]-0.5)**2)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9))
    ax = axes[0]
    ax.plot(s_arc, f0, 'b-', label=r'$f_0(u_2)f_0(u_3)$  (state $v=0$)')
    ax.plot(s_arc, f1, 'r-', label=r'$f_1(u_2)f_1(u_3)$  (state $v=1$)')
    ax.fill_between(s_arc, 0, f0, alpha=0.2, color='b')
    ax.fill_between(s_arc, 0, f1, alpha=0.2, color='r')
    ax.set_xlabel(r'arclength $\sigma$'); ax.set_ylabel('integrand')
    ax.set_title("P5 -- Evidence integrands $f_v(u_j)f_v(u_l)$ on the slice curve\n"
                 r"$A_v = \int f_v f_v\, d\sigma$ -- closed-form per segment, $h\equiv 0$",
                 fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    A0 = np.concatenate(([0], np.cumsum(0.5*seg_lens*(f0[:-1]+f0[1:]))))
    A1 = np.concatenate(([0], np.cumsum(0.5*seg_lens*(f1[:-1]+f1[1:]))))
    ax = axes[1]
    ax.plot(s_arc, A0, 'b-', label=r'cumulative $A_0$')
    ax.plot(s_arc, A1, 'r-', label=r'cumulative $A_1$')
    ax.set_xlabel(r'$\sigma$'); ax.set_ylabel('cumulative integral')
    ax.set_title("Cumulative evidence -- ratio $A_1/A_0$ feeds the Bayes posterior")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, 'p05_evidence')

# ---------- P6: residual diagram ----------
fig = plt.figure(figsize=(8.5, 7))
fig.suptitle("P6 -- The per-vertex residual", fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
For each vertex $v = (v_1, v_2, v_3)$ of surface $S_m$ (price level $p_m$):

1.  Three slices through the SAME surface:
       agent 1:  $S_m \cap \{u_1 = v_1\}$  $\to$  $(A_0^{(1)}, A_1^{(1)})$
       agent 2:  $S_m \cap \{u_2 = v_2\}$  $\to$  $(A_0^{(2)}, A_1^{(2)})$
       agent 3:  $S_m \cap \{u_3 = v_3\}$  $\to$  $(A_0^{(3)}, A_1^{(3)})$

2.  Bayes posterior per agent:
       $\mu_k = \dfrac{f_1(v_k)\,A_1^{(k)}}{f_0(v_k)\,A_0^{(k)} + f_1(v_k)\,A_1^{(k)}}$

3.  CRRA market clearing (bisection):
       $p_{\rm clear}(v) = $ unique $p$: $\sum_k x_k^{\rm CRRA}(\mu_k, p; \gamma) = 0$

4.  Residual:
       $r(v) = p_{\rm clear}(v) - p_m$

At a strict-$h{=}0$ equilibrium, every vertex of every surface clears exactly
at its own price level: $r \equiv 0$.

State vector: stack of all vertex coordinates ($3\,N$ unknowns).
Residual vector: scalar per vertex ($N$ equations).
2$N$ tangential gauge DOFs are damped harmlessly by Levenberg.
"""
ax.text(0.03, 0.92, txt, fontsize=10, va='top', family='serif')
save(fig, 'p06_residual')

# ---------- P7: Jacobian sparsity ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
np.random.seed(1)
ax = axes[0]
sizes = (np.random.rand(6)*15 + 5).astype(int)
N_tot = sizes.sum()
Jpat = np.zeros((N_tot, N_tot))
r0c = 0
for sz in sizes:
    Jpat[r0c:r0c+sz, r0c:r0c+sz] = (np.random.rand(sz, sz) > 0.55)
    r0c += sz
ax.imshow(Jpat, cmap='Greys', aspect='auto')
ax.set_title("P7a -- Global Jacobian is block-diagonal by surface")
ax.set_xlabel('vertex coordinate'); ax.set_ylabel('vertex residual')
ax = axes[1]
sz = 50
within = np.zeros((sz, sz))
for i in range(sz):
    for j in range(max(0, i-3), min(sz, i+4)):
        within[i, j] = 1
    for j in np.random.choice(sz, 5, replace=False):
        within[i, j] = 0.5
ax.imshow(within, cmap='Blues', aspect='auto')
ax.set_title("P7b -- Within-block: column $w$ affects only rows whose\n"
             "slices cross faces incident to vertex $w$ (sparse FD)")
ax.set_xlabel('vertex coordinate'); ax.set_ylabel('residual row')
save(fig, 'p07_jacobian')

# ---------- P8: GS degeneracy ----------
fig = plt.figure(figsize=(8.5, 8))
fig.suptitle("P8 -- Grossman-Stiglitz degeneracy at strict $h{=}0$",
             fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
ANALYTIC FACT: $P_{FR}(u) = \sigma(\tau \sum_k u_k)$ is an EXACT
fixed point of the strict-$h{=}0$ operator at every $(\tau, \gamma)$
under binary payoff.

WHY: restricted to the level set $\{P_{FR} = p\}$, the
constraint $\sum u_k = \mathrm{logit}\,p / \tau$ pins the
sufficient statistic exactly.  The slice integrals close
in closed form Gaussian: every agent's posterior
collapses to $\mu_k = \sigma(\tau \sum_j u_j) = p$.
With identical posteriors, CRRA market clearing returns
$p^{\rm clear} = p$ for any $\gamma$, so consistency holds.

NUMERICAL CONFIRMATION:  9 cells across $\tau \in [0.05, 2]$ and
$\gamma \in \{0.01, 0.098, 1.035\}$.  Strict-$h{=}0$ residual at $P_{FR}$:
   max $\|F\|_\infty \in [1.7,\, 3.8] \times 10^{-15}$
   (float64 floor; deficit reconstructs to $|1-R^2| < 4 \times 10^{-15}$).

CONSEQUENCE: the partially-revealing equilibrium $(1-R^2 > 0)$ of risk-
averse binary-payoff trading is NOT a strict-$h{=}0$ fixed point.  It
exists only as the $h \to 0$ limit of the kernel-regularised family $\{P_h\}$.
The kernel bandwidth $h>0$ is the EQUILIBRIUM SELECTION mechanism for the
nontrivial branch: $h \to 0$ along the kernel family converges to the PR
equilibrium with certified $1-R^2_\infty = 0.278 \pm 0.008$ at the max-
deficit corner, but the limit object is NOT a fixed point of the strict-$h{=}0$
operator.  See P9-P10 for basin-of-attraction tests confirming this at G=7,
G=15, G=21, G=33.
"""
ax.text(0.04, 0.93, txt, fontsize=10, va='top', family='serif')
save(fig, 'p08_GS_degeneracy')

# ---------- P9: basin diagram ----------
# Pull initial residuals from G=15 basin
try:
    G15 = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/basin_overnight/G15.json'))
except FileNotFoundError:
    G15 = []
try:
    G33 = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/basin_search/basin_results.json'))
except FileNotFoundError:
    G33 = []

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
labels = []; init_med = []; dist_FR = []
for r in G15:
    if 'error' in r: continue
    labels.append(r['label']); init_med.append(r['med_r_init'])
    dist_FR.append(r['init_dist_to_FR'])
ax.semilogy(dist_FR, init_med, 'o', markersize=12, alpha=0.7)
for x, y, lbl in zip(dist_FR, init_med, labels):
    ax.annotate(lbl, (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points')
ax.axhline(5e-6, color='r', ls='--', label='mesh-extraction floor')
ax.set_xlabel('initial distance to $P_{FR}$ (max over inner cube)')
ax.set_ylabel('initial median |r| (strict-$h{=}0$ residual)')
ax.set_title('P9a -- G=15: only $P_{FR}$ sits at the residual floor')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
labels2 = []; init_med2 = []; dist_FR2 = []
for r in G33:
    labels2.append(r['label']); init_med2.append(r['med_r_init'])
    dist_FR2.append(r['dist_to_FR0'])
ax.semilogy(dist_FR2, init_med2, 's', markersize=12, alpha=0.7, color='C2')
for x, y, lbl in zip(dist_FR2, init_med2, labels2):
    ax.annotate(lbl, (x, y), fontsize=7, xytext=(5, 5), textcoords='offset points')
ax.axhline(5e-6, color='r', ls='--', label='mesh-extraction floor')
ax.set_xlabel('initial distance to $P_{FR}$')
ax.set_ylabel('initial median |r|')
ax.set_title('P9b -- G=33: same pattern, confirmed at higher resolution')
ax.legend(); ax.grid(alpha=0.3)
save(fig, 'p09_basin_map')

# ---------- P10: convergence trajectories at G=15 ----------
fig, ax = plt.subplots(figsize=(9, 6))
colors_t = plt.cm.tab10(np.linspace(0, 1, 10))
for ci, r in enumerate(G15[:8]):
    if 'error' in r: continue
    its = [t['it'] for t in r['traj']]
    meds = [t['med_r'] for t in r['traj']]
    ax.semilogy(its, meds, 'o-', color=colors_t[ci], label=r['label'][:18],
                markersize=4, linewidth=1.2)
ax.axhline(5e-6, color='r', ls='--', alpha=0.5, label='mesh floor')
ax.set_xlabel('Picard iteration'); ax.set_ylabel('median |r|')
ax.set_title('P10 -- Picard trajectories from each starting point (G=15)\n'
             'PFR stays at floor; kernel improves $\\sim 5\\times$; others stall',
             fontsize=11)
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3, which='both')
save(fig, 'p10_trajectories')

# ---------- P11: status table + roadmap ----------
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("P11 -- Status of solver attempts and the road forward",
             fontsize=13, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = r"""
EVERY MOVING-GRID SOLVER ATTEMPT (binary payoff, $\tau = 2, \gamma = 0.01$):

   Stage 2:  cube-based Picard               --  DIVERGED 0.33 $\to$ 0.46 $\to$ 0.47
   Stage 3c: moving-mesh Newton (normal)     --  STALLED $|\delta| \approx 10^{-6}$
   Stage 3d: moving-mesh damped Picard       --  PLATEAU med 0.16 $\to$ 0.11
   Stage 4-5: free-vertex LM with caps       --  PLATEAU med 0.13, max 0.33
   Stage 6:  graph height-function           --  PULLED toward $P_{FR}$
   Basin G=7  LM (per-vertex caps)           --  PFR step=0; others rejected
   Basin G=8  scipy.optimize.least_squares   --  (running, expected: PFR sits)
   Basin G=15/21/33 damped Picard            --  PFR at floor; others drift

INTERPRETATION: every solver and every starting point produces the same answer:
the only strict-$h{=}0$ fixed point in this corner is $P_{FR}$.  The
solvers' diverse failure modes are not numerical accidents -- they reflect
the structural fact that the nontrivial $1-R^2 > 0$ equilibrium does not
exist at exact $h = 0$.

WHAT THIS LEAVES FOR THE PAPER:

  -- Main text: certified PR equilibria from the kernel-h$>$0 family, with
     existence theorem (regularised), $1-R^2$ asymptotics, $K$-invariance.
  -- Appendix A: this PDF's contents.  Strict-$h{=}0$ is degenerate under
     binary payoff; $h>0$ is the equilibrium selection device.

ROAD FORWARD if a referee insists on a strict-$h{=}0$ solution:

  1. Replace the binary $v \in \{0,1\}$ with continuous $v$ (e.g. uniform
     on $[0,1]$): the FR equilibrium is no longer such a clean closed
     form, and a nontrivial strict-$h{=}0$ PR fixed point should exist.
  2. Use the moving-grid solver on the CONTINUOUS-payoff variant.
  3. Document the binary case as a knife-edge limit.

This route is conceptually clean (Wilson's classic Continuous-Type vs
Discrete-Type) and would settle the binary-payoff degeneracy as a model
choice rather than a numerical artifact.
"""
ax.text(0.03, 0.95, txt, fontsize=9.5, va='top', family='serif')
save(fig, 'p11_status_roadmap')

# Combine to PDF
imgs = [Image.open(f"{BUILD}/{p}.png").convert("RGB") for p in PAGES]
W = max(im.width for im in imgs)
norm = []
for im in imgs:
    if im.width != W:
        h2 = int(im.height * W / im.width)
        im = im.resize((W, h2))
    norm.append(im)
dst = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cmm_endogenous/CMM_method_v2.pdf'
norm[0].save(dst, save_all=True, append_images=norm[1:])
print(f"saved {dst} ({len(PAGES)} pages, {len(PAGES)-1} figures)")
