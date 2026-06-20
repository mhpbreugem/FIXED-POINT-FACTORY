"""Generate ~12 figures for the Chebyshev + σ-δ + symmetry PDF."""
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = '/tmp/chebsym/figs'
os.makedirs(OUT, exist_ok=True)

# ===== Fig 1: K=3 cube with three agent labels =====
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
# Cube edges
r = [-1, 1]
for s, e in [((-1,-1,-1),(1,-1,-1)),((-1,-1,-1),(-1,1,-1)),((-1,-1,-1),(-1,-1,1)),
             ((1,1,1),(-1,1,1)),((1,1,1),(1,-1,1)),((1,1,1),(1,1,-1)),
             ((1,-1,-1),(1,1,-1)),((1,-1,-1),(1,-1,1)),
             ((-1,1,-1),(1,1,-1)),((-1,1,-1),(-1,1,1)),
             ((-1,-1,1),(1,-1,1)),((-1,-1,1),(-1,1,1))]:
    ax.plot(*zip(s,e), 'k-', alpha=0.4, lw=1)
# Three agents at characteristic corners
ax.scatter([0],[0],[0], c='black', s=80, label='center (u=0)')
ax.scatter([1.2],[0],[0], c='tab:red', s=120, marker='o')
ax.scatter([0],[1.2],[0], c='tab:green', s=120, marker='s')
ax.scatter([0],[0],[1.2], c='tab:blue', s=120, marker='^')
ax.text(1.4, 0.1, 0.05, 'Agent 1\nu₁ private', color='tab:red', fontsize=11, ha='left')
ax.text(0.1, 1.4, 0.05, 'Agent 2\nu₂ private', color='tab:green', fontsize=11, ha='left')
ax.text(0.1, 0.1, 1.4, 'Agent 3\nu₃ private', color='tab:blue', fontsize=11, ha='left')
ax.set_xlabel('u₁'); ax.set_ylabel('u₂'); ax.set_zlabel('u₃')
ax.set_title('The K=3 cube: P(u₁, u₂, u₃) lives on this cube\n'
              'Each agent k sees u_k as their own signal,\n'
              '(u_others) as the joint distribution to integrate over.')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.5, 1.5)
plt.tight_layout()
plt.savefig(f'{OUT}/01_cube.png', dpi=130, bbox_inches='tight')
plt.close()
print('01_cube.png')

# ===== Fig 2: per-agent σ-δ rotation (3 panels) =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, ax in enumerate(axes):
    title = ['Agent 1: rotate (u₂,u₃)→(Σ,δ)',
              'Agent 2: rotate (u₁,u₃)→(Σ,δ)',
              'Agent 3: rotate (u₁,u₂)→(Σ,δ)'][k]
    other = [(2,3), (1,3), (1,2)][k]
    # 2-D plane showing the rotation
    n = 11
    grid = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(grid, grid)
    # Show original axes
    ax.scatter(X.flatten(), Y.flatten(), s=20, c='lightblue', alpha=0.6, label=f'u_{other[0]}, u_{other[1]} grid')
    # Show rotated diamond
    Sig = (X + Y) / np.sqrt(2)
    Del = (X - Y) / np.sqrt(2)
    ax.scatter(Sig.flatten(), Del.flatten(), s=15, c='tab:orange', alpha=0.7, marker='x', label='Σ, δ rotation')
    # Mark agent's own axis (going into screen — show as central dot)
    ax.scatter([0], [0], s=300, c='black', marker='*', label=f'u_{k+1} (into page)', zorder=10)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'u_{other[0]}  /  Σ = (u_{other[0]}+u_{other[1]})/√2')
    ax.set_ylabel(f'u_{other[1]}  /  δ = (u_{other[0]}-u_{other[1]})/√2')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
plt.suptitle('Per-agent (Σ_k, δ_k) coordinates: each agent integrates over its own (Σ, δ) plane', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/02_per_agent_rotation.png', dpi=130, bbox_inches='tight')
plt.close()
print('02_per_agent_rotation.png')

# ===== Fig 3: σ-δ ξ-cube with atanh stretching =====
fig = plt.figure(figsize=(12, 5))
# Left: u-coordinates (unbounded)
ax1 = fig.add_subplot(121)
u = np.linspace(-3, 3, 100)
xi = np.tanh(u)
ax1.plot(u, xi, 'b-', lw=2)
ax1.axhline(1, color='r', linestyle=':', alpha=0.5, label='ξ=+1 ↔ u=+∞')
ax1.axhline(-1, color='r', linestyle=':', alpha=0.5, label='ξ=-1 ↔ u=-∞')
ax1.set_xlabel('u (unbounded)'); ax1.set_ylabel('ξ = tanh(u)')
ax1.set_title('atanh stretch: ξ ∈ [-1, 1] is the bounded coordinate')
ax1.legend(); ax1.grid(alpha=0.3)
# Right: ξ-cube in 3D
ax2 = fig.add_subplot(122, projection='3d')
for s, e in [((-1,-1,-1),(1,-1,-1)),((-1,-1,-1),(-1,1,-1)),((-1,-1,-1),(-1,-1,1)),
             ((1,1,1),(-1,1,1)),((1,1,1),(1,-1,1)),((1,1,1),(1,1,-1)),
             ((1,-1,-1),(1,1,-1)),((1,-1,-1),(1,-1,1)),
             ((-1,1,-1),(1,1,-1)),((-1,1,-1),(-1,1,1)),
             ((-1,-1,1),(1,-1,1)),((-1,-1,1),(-1,1,1))]:
    ax2.plot(*zip(s,e), 'k-', lw=2)
# Cheb-Lobatto nodes
N = 7
nodes = -np.cos(np.pi * np.arange(N+1) / N)
X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing='ij')
ax2.scatter(X, Y, Z, c='tab:blue', s=10, alpha=0.4)
ax2.set_xlabel('ξ_{u₁}'); ax2.set_ylabel('ξ_Σ'); ax2.set_zlabel('ξ_δ')
ax2.set_title(f'σ-δ ξ-cube with Chebyshev-Lobatto nodes (N={N})\n'
                'Boundaries ξ=±1 = perfect-information limits')
plt.tight_layout()
plt.savefig(f'{OUT}/03_sigma_delta_cube.png', dpi=130, bbox_inches='tight')
plt.close()
print('03_sigma_delta_cube.png')

# ===== Fig 4: 1D Chebyshev polynomials T_0...T_5 =====
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-1, 1, 400)
for n in range(6):
    Tn = np.cos(n * np.arccos(x))
    ax.plot(x, Tn, lw=1.5, label=f'T_{n}(ξ)')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('ξ'); ax.set_ylabel('T_n(ξ)')
ax.set_title('Chebyshev polynomials of the first kind: T_n(ξ) = cos(n·arccos(ξ))\n'
              'Parity: T_n(-ξ) = (-1)^n T_n(ξ) — key for Z₂ symmetry')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/04_chebyshev_basis.png', dpi=130, bbox_inches='tight')
plt.close()
print('04_chebyshev_basis.png')

# ===== Fig 5: Chebyshev-Lobatto nodes in 1D and 2D =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
N = 10
nodes = -np.cos(np.pi * np.arange(N+1) / N)
# 1D
ax = axes[0]
uniform = np.linspace(-1, 1, N+1)
ax.scatter(uniform, np.zeros_like(uniform)+0.1, s=80, c='tab:red', label=f'Uniform ({N+1} nodes)')
ax.scatter(nodes, np.zeros_like(nodes)-0.1, s=80, c='tab:blue', label=f'Chebyshev-Lobatto ({N+1} nodes)')
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.5, 0.5)
ax.set_yticks([0.1, -0.1]); ax.set_yticklabels(['uniform', 'Cheb-Lobatto'])
ax.set_xlabel('ξ')
ax.set_title(f'1D: Chebyshev nodes cluster at the edges\n'
              '(prevents Runge phenomenon, makes spectral convergence)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.3)
# 2D
ax = axes[1]
X, Y = np.meshgrid(nodes, nodes, indexing='ij')
ax.scatter(X.flatten(), Y.flatten(), s=30, c='tab:blue')
# show density gradient with circle
from matplotlib.patches import Circle
ax.add_patch(Circle((0,0), 1, fill=False, color='gray'))
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_xlabel('ξ_a'); ax.set_ylabel('ξ_b')
ax.set_title(f'2D tensor: ({N+1})² = {(N+1)**2} nodes\n'
              'Edge-clustered → spectral accuracy')
plt.tight_layout()
plt.savefig(f'{OUT}/05_lobatto_nodes.png', dpi=130, bbox_inches='tight')
plt.close()
print('05_lobatto_nodes.png')

# ===== Fig 6: Spectral convergence demo =====
fig, ax = plt.subplots(figsize=(10, 5))
# Smooth function: sigmoid
def f(x): return 1/(1+np.exp(-3*x))
Ns = [4, 8, 12, 16, 20, 24, 28]
errs_sigmoid = []
for N in Ns:
    nodes = -np.cos(np.pi*np.arange(N+1)/N)
    vals = f(nodes)
    # FFT to get Chebyshev coeffs
    # Use DCT (type I) for Cheb-Lobatto
    from scipy.fft import dct
    a = dct(vals, type=1)/N
    a[0] /= 2; a[-1] /= 2
    # Reconstruction error on a finer grid
    xtest = np.linspace(-1, 1, 200)
    # Naive eval: sum a_n T_n(x)
    rec = np.zeros_like(xtest)
    for n in range(N+1):
        rec += a[n] * np.cos(n * np.arccos(xtest))
    errs_sigmoid.append(np.max(np.abs(rec - f(xtest))))

ax.semilogy(Ns, errs_sigmoid, 'o-', lw=2, markersize=10, color='tab:blue', label='Chebyshev (sigmoid f(x)=σ(3x))')
# Uniform polynomial fit for comparison (Runge / bad)
errs_uniform = []
for N in Ns:
    nodes_u = np.linspace(-1, 1, N+1)
    vals_u = f(nodes_u)
    # Lagrange interp at uniform nodes
    coeffs = np.polynomial.polynomial.polyfit(nodes_u, vals_u, N)
    xtest = np.linspace(-1, 1, 200)
    rec_u = np.polynomial.polynomial.polyval(xtest, coeffs)
    errs_uniform.append(np.max(np.abs(rec_u - f(xtest))))
ax.semilogy(Ns, errs_uniform, 's-', lw=2, markersize=10, color='tab:red', label='Uniform polynomial (Runge)')
ax.set_xlabel('N (polynomial degree)')
ax.set_ylabel('max |f(ξ) - p_N(ξ)|')
ax.set_title('Spectral convergence of Chebyshev vs Runge instability of uniform polynomial\n'
              '(target: smooth sigmoid f)')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUT}/06_spectral_convergence.png', dpi=130, bbox_inches='tight')
plt.close()
print('06_spectral_convergence.png')

# ===== Fig 7: 3D tensor structure visualization =====
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
N = 6
nodes = -np.cos(np.pi * np.arange(N+1) / N)
X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing='ij')
sc = ax1.scatter(X, Y, Z, c=X+Y+Z, cmap='viridis', s=30)
ax1.set_xlabel('ξ_{u₁}'); ax1.set_ylabel('ξ_Σ'); ax1.set_zlabel('ξ_δ')
ax1.set_title(f'Cheb-Lobatto tensor on σ-δ ξ-cube\n({N+1}³ = {(N+1)**3} nodes)')

# Right: coefficients
ax2 = fig.add_subplot(122)
# Indices (i,j,k) with i+j+k <= N
N_modes = 8
ijk = []
for i in range(N_modes+1):
    for j in range(N_modes+1):
        for k in range(N_modes+1):
            ijk.append((i,j,k))
# Color by total order
ijk = np.array(ijk)
sums = ijk.sum(axis=1)
sc = ax2.scatter(ijk[:,0]+0.1*ijk[:,2], ijk[:,1]+0.1*ijk[:,2],
                   c=sums, cmap='plasma', s=40, alpha=0.7)
ax2.set_xlabel('i + 0.1·k')
ax2.set_ylabel('j + 0.1·k')
ax2.set_title(f'(N+1)³ = {(N_modes+1)**3} coefficients a_{{ijk}}\n'
                '(naive 3D tensor)')
plt.colorbar(sc, ax=ax2, label='i+j+k (total order)')
plt.tight_layout()
plt.savefig(f'{OUT}/07_3d_tensor.png', dpi=130, bbox_inches='tight')
plt.close()
print('07_3d_tensor.png')

# ===== Fig 8: per-agent evaluation via symmetry =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Show 3 evaluations of "same" P at permuted points
for k, ax in enumerate(axes):
    triples = [(0.3, 0.5, -0.2), (0.5, 0.3, -0.2), (-0.2, 0.5, 0.3)][k]
    permname = ['canonical (u₁,u₂,u₃)', 'agent 2: input as (u₂,u₁,u₃)', 'agent 3: input as (u₃,u₂,u₁)'][k]
    note = ['stored coefficients',
             'permute (u₁↔u₂): same coefficients give same value (S₃ sym)',
             'permute (u₁↔u₃): same coefficients give same value (S₃ sym)'][k]
    ax.bar(['u₁', 'u₂', 'u₃'], triples, color=['tab:red','tab:green','tab:blue'])
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylim(-0.5, 0.7)
    ax.set_title(f'{permname}\nP(input) = {triples}\n{note}', fontsize=9)
    ax.set_ylabel('input value')
plt.suptitle('Per-agent evaluation under S₃ symmetry:\n'
              'store ONE coefficient set; evaluate at PERMUTED inputs for each agent.\n'
              'No re-fitting, no separate stores, no interpolation noise.', fontsize=11, y=1.05)
plt.tight_layout()
plt.savefig(f'{OUT}/08_per_agent_eval.png', dpi=130, bbox_inches='tight')
plt.close()
print('08_per_agent_eval.png')

# ===== Fig 9: S_3 orbits of multi-indices =====
fig, ax = plt.subplots(figsize=(10, 8))
N = 5
# Generate all (i,j,k) with 0 <= i,j,k <= N, color by orbit
orbits = {}
for i in range(N+1):
    for j in range(N+1):
        for k in range(N+1):
            key = tuple(sorted([i,j,k]))
            orbits.setdefault(key, []).append((i,j,k))

cmap = plt.cm.tab20
orbit_colors = {key: cmap(idx % 20) for idx, key in enumerate(sorted(orbits.keys()))}

for orbit_id, (key, members) in enumerate(orbits.items()):
    xs = [m[0]+0.15*m[2] for m in members]
    ys = [m[1]+0.15*m[2] for m in members]
    color = orbit_colors[key]
    ax.scatter(xs, ys, c=[color], s=140, edgecolors='black', linewidths=1)
    # annotate canonical (sorted) member
    if len(members) == 1:
        ax.annotate(f'{key}', (xs[0], ys[0]), fontsize=7, ha='center', va='center')

ax.set_xlabel('i (with offset for k)'); ax.set_ylabel('j (with offset for k)')
ax.set_title(f'S₃ orbits of multi-indices (i,j,k) at N={N}\n'
              f'{(N+1)**3} indices → {len(orbits)} orbits (= independent coefficients under S₃)\n'
              'Same color = same orbit = forced equal coefficients')
plt.tight_layout()
plt.savefig(f'{OUT}/09_S3_orbits.png', dpi=130, bbox_inches='tight')
plt.close()
print('09_S3_orbits.png')

# ===== Fig 10: Z_2 reflection symmetry =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
# Show P and 1-P_flipped
xi = np.linspace(-1, 1, 100)
# Some asymmetric P that should satisfy P(-x) = 1-P(x)
def P(x): return 1/(1+np.exp(-3*x))
ax.plot(xi, P(xi), 'b-', lw=2, label='P(ξ)')
ax.plot(xi, 1-P(-xi), 'r--', lw=2, label='1 - P(-ξ)')
ax.axhline(0.5, color='gray', lw=0.5, alpha=0.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('ξ'); ax.set_ylabel('P')
ax.set_title('Z₂ symmetry P(-ξ) = 1 - P(ξ)\n(binary asset value v ↔ 1-v)')
ax.legend()
ax.grid(alpha=0.3)
# right: which coefficients survive
ax = axes[1]
N = 8
allowed = []
killed = []
for i in range(N+1):
    for j in range(N+1):
        for k in range(N+1):
            if i+j+k == 0:
                ax.scatter([i+0.15*k],[j+0.15*k], c='black', marker='*', s=200, label='fixed a_{000}=1/2' if i+j+k==0 else None)
            elif (i+j+k) % 2 == 0:
                killed.append((i, j, k))
            else:
                allowed.append((i, j, k))
xs = [m[0]+0.15*m[2] for m in killed]
ys = [m[1]+0.15*m[2] for m in killed]
ax.scatter(xs, ys, c='red', marker='x', s=80, label=f'even sum → 0 ({len(killed)})')
xs = [m[0]+0.15*m[2] for m in allowed]
ys = [m[1]+0.15*m[2] for m in allowed]
ax.scatter(xs, ys, c='blue', marker='o', s=80, label=f'odd sum → free ({len(allowed)})')
ax.set_xlabel('i + 0.15·k'); ax.set_ylabel('j + 0.15·k')
ax.set_title(f'Z₂ on Chebyshev coefficients (N={N}):\n'
              f'kills half — only i+j+k odd survive\n'
              f'({(N+1)**3 - len(allowed) - 1} killed of {(N+1)**3} total)')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/10_Z2_parity.png', dpi=130, bbox_inches='tight')
plt.close()
print('10_Z2_parity.png')

# ===== Fig 11: combined S_3 x Z_2 reduction =====
fig, ax = plt.subplots(figsize=(10, 7))
Ns = list(range(2, 25))
naive = [(N+1)**3 for N in Ns]
# S_3 only: multisets
def n_S3(N):
    n = 0
    for i in range(N+1):
        for j in range(i, N+1):
            for k in range(j, N+1):
                n += 1
    return n
S3 = [n_S3(N) for N in Ns]
# combined: multisets with i<=j<=k and i+j+k odd (plus a_000=1/2 fixed)
def n_combined(N):
    n = 0
    for i in range(N+1):
        for j in range(i, N+1):
            for k in range(j, N+1):
                if (i+j+k) % 2 == 1:
                    n += 1
    return n
both = [n_combined(N) for N in Ns]

ax.semilogy(Ns, naive, 'r-', lw=2, marker='s', markersize=8, label='Naive (N+1)³')
ax.semilogy(Ns, S3, 'g-', lw=2, marker='^', markersize=8, label='S₃ only (multisets)')
ax.semilogy(Ns, both, 'b-', lw=2, marker='o', markersize=8, label='S₃ × Z₂ (multisets, odd-sum)')
# Annotate ratios at N=24
N_show = 24
ratio = naive[N_show-2]/both[N_show-2]
ax.annotate(f'{ratio:.1f}× reduction', xy=(N_show, both[N_show-2]), xytext=(N_show-2, naive[N_show-2]),
              arrowprops=dict(arrowstyle='->', color='black'), fontsize=11)
ax.set_xlabel('N (max Chebyshev degree per axis)')
ax.set_ylabel('number of independent coefficients (log)')
ax.set_title('Coefficient count: full tensor vs S₃ vs S₃ × Z₂\n'
              '(asymptotic factor 6 from S₃, additional 2 from Z₂ = 12 total)')
ax.legend(fontsize=11); ax.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(f'{OUT}/11_coefficient_reduction.png', dpi=130, bbox_inches='tight')
plt.close()
print('11_coefficient_reduction.png')

# ===== Fig 12: sigmoid lift decomposition =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
T = np.linspace(-6, 6, 200)
# (1) FR limit
ax = axes[0]
P_FR = 1/(1+np.exp(-T))
ax.plot(T, P_FR, 'b-', lw=2, label='P_FR = σ(T) (CARA limit)')
ax.set_xlabel('T = τ·Σu'); ax.set_ylabel('P_FR')
ax.set_title('FR limit (γ→∞ CARA):\nP_FR(u) = σ(T)\nThis is the leading-order ansatz')
ax.grid(alpha=0.3); ax.legend(fontsize=10)

# (2) PR with slope alpha < 1
ax = axes[1]
alpha = 0.36
P_PR = 1/(1+np.exp(-alpha*T))
ax.plot(T, P_FR, 'b-', lw=1, alpha=0.5, label='FR (slope=1)')
ax.plot(T, P_PR, 'r-', lw=2, label=f'PR ansatz σ(α·T), α={alpha}')
ax.set_xlabel('T = τ·Σu'); ax.set_ylabel('P')
ax.set_title('PR scalar α captures the slope.\n'
              'In our problem α≈0.36 was the verified γ=0.1 value.\n'
              'This is ONE coefficient.')
ax.grid(alpha=0.3); ax.legend(fontsize=10)

# (3) Full: P = σ(α·T + h(ξ))
ax = axes[2]
ax.plot(T, P_PR, 'r-', lw=1.5, alpha=0.6, label='σ(α·T)')
h_correction = 0.4 * np.sin(2*T) * np.exp(-T**2/8)
P_full = 1/(1+np.exp(-(alpha*T + h_correction)))
ax.plot(T, P_full, 'k-', lw=2, label='σ(α·T + h(ξ))  [example]')
ax.set_xlabel('T = τ·Σu'); ax.set_ylabel('P')
ax.set_title('Sigmoid lift: P = σ(α·T + h(ξ))\n'
              'h is a SMALL correction in Chebyshev modes;\n'
              'sigmoid handles boundaries P→0/1 exactly.')
ax.grid(alpha=0.3); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT}/12_sigmoid_lift.png', dpi=130, bbox_inches='tight')
plt.close()
print('12_sigmoid_lift.png')

# ===== Fig 13: companion-matrix root-finding visualization =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Left: a 1D slice of P with multiple roots P=p
ax = axes[0]
xi = np.linspace(-1, 1, 200)
# Chebyshev polynomial example
P_slice = 0.5 + 0.3*np.cos(3*np.arccos(xi)) + 0.1*np.cos(5*np.arccos(xi))
p_target = 0.5
ax.plot(xi, P_slice, 'b-', lw=2, label='P(ξ) = Σ a_n T_n(ξ)')
ax.axhline(p_target, color='r', linestyle='--', label=f'p_target = {p_target}')
# find roots numerically
from scipy.optimize import brentq
roots = []
for i in range(len(xi)-1):
    if (P_slice[i]-p_target)*(P_slice[i+1]-p_target) <= 0:
        try:
            r = brentq(lambda x: 0.5 + 0.3*np.cos(3*np.arccos(x)) + 0.1*np.cos(5*np.arccos(x)) - p_target, xi[i], xi[i+1])
            roots.append(r)
        except Exception:
            pass
for r in roots:
    ax.scatter([r], [p_target], s=120, c='red', zorder=5)
ax.set_xlabel('ξ'); ax.set_ylabel('P')
ax.set_title(f'1D Chebyshev slice. Find all roots of P(ξ) = p_target.\n'
              f'{len(roots)} roots: companion-matrix eigenvalues give them all in one shot.')
ax.legend(); ax.grid(alpha=0.3)

# Right: companion matrix sketch
ax = axes[1]
ax.text(0.5, 0.9, 'Companion-matrix root-find', ha='center', fontsize=13, weight='bold', transform=ax.transAxes)
ax.text(0.5, 0.78, 'Polynomial p(ξ) = Σ aₙTₙ(ξ) - p_target', ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.66, 'Convert to monomial: p(ξ) = c₀ + c₁ξ + ... + cₙξⁿ', ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.50, 'Build companion matrix C:', ha='center', fontsize=11, transform=ax.transAxes)
cm_text = (
    '⎡ 0   0   ...   -c₀/cₙ ⎤\n'
    '⎢ 1   0   ...   -c₁/cₙ ⎥\n'
    '⎢ 0   1   ...   -c₂/cₙ ⎥\n'
    '⎢ ... ... ... ......  ⎥\n'
    '⎣ 0   0   ... -cₙ₋₁/cₙ ⎦'
)
ax.text(0.5, 0.32, cm_text, ha='center', fontsize=10, family='monospace', transform=ax.transAxes)
ax.text(0.5, 0.10, 'Eigenvalues of C = all roots of p (in closed form).\n'
                      'No bisection, no missed roots, no topology-jump artifacts.',
        ha='center', fontsize=11, transform=ax.transAxes, style='italic')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT}/13_companion_root_find.png', dpi=130, bbox_inches='tight')
plt.close()
print('13_companion_root_find.png')

print('\nALL FIGURES GENERATED')
