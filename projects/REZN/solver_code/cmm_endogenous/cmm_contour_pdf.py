"""Contour-plot 5-page PDF for the CMM stage 1c results.

Pages:
1. Title page: setup + headline numbers (timing per n_bins, residuals).
2. Anchor fixed point P* slices: contour plots of P*(u1, u2, u3=u3_fixed)
   for u3 in {-1.5, 0, +1.5}, plus the diagonal slice P*(u, u, u).
3. CMM operator output: same slices for Pn_cmm (n_bins=128) vs P*; overlay.
4. CMM error map: |Phi_cmm - Phi_kernel| heatmap in the same slices; show
   where the strict-h=0 vs kernel discrepancy is large (the O(h) signature).
5. Level surfaces: extracted S_p contour curves in the (u2, u3) plane at
   fixed u1=0 for p in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} — these are
   exactly the curves CMM integrates over (illustrates h=0).
"""
import os, sys, json, subprocess, shutil
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cmm_stage1 import (build_grid, extract_surfaces_marching_cubes, EMIN15, OUT, C, UMAX, pad)
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth

BUILD = '/tmp/cmm_pdf_build'
os.makedirs(BUILD, exist_ok=True)

Gi = 21
du, uf, lo, hi = build_grid(Gi)
tau, gamma = 2.0, 0.0980
h = C * np.sqrt(du)
u_in = uf[lo:hi]
P_inner = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy")
P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
P_full[lo:hi, lo:hi, lo:hi] = P_inner
Pn_kernel = phi_K3_halo_smooth(P_full, uf, lo, hi, np.full(3, tau),
                                np.full(3, gamma), np.full(3, 1.0), h)
Pn_kernel_in = Pn_kernel[lo:hi, lo:hi, lo:hi]
Pn_cmm = None
for nb in (128, 64, 32):
    p = f"{OUT}/Pn_cmm_nbins{nb}.npy"
    if os.path.exists(p):
        Pn_cmm = np.load(p); used_nb = nb; break
assert Pn_cmm is not None, "no CMM cube saved yet"
meta = json.load(open(f"{OUT}/stage1c_meta.json"))

# ---------- Page 1: title + numbers ----------
fig = plt.figure(figsize=(8.5, 11))
fig.suptitle("CMM Stage 1c — strict h=0 co-area mesh forward operator",
              fontsize=14, weight='bold')
ax = fig.add_subplot(111); ax.axis('off')
txt = [
    rf"Test cell: $(\tau,\gamma) = ({tau}, {gamma})$, $G={Gi}^3$, "
    rf"$h_{{\rm kernel}} = 0.45\sqrt{{\Delta u}} = {h:.3f}$",
    rf"Fixed point $P^*$: emin15-certified longdouble FP "
    rf"($\|\Phi_{{\rm kernel}}(P^*)-P^*\|_\infty = {meta['F_kernel']:.2e}$)",
    "",
    "CMM forward operator: surfaces extracted by marching cubes at "
    "$N_{\\rm bins}$ price levels, sliced by hyperplanes $u_k=U$ "
    "(arc-length integration; $h\\equiv 0$).",
    "",
    f"CMM (n_bins={used_nb}) cube available.",
    f"$\\|\\Phi_{{\\rm CMM}} - \\Phi_{{\\rm kernel}}\\|_\\infty$ on whole cube: "
    f"{float(np.max(np.abs(Pn_cmm - Pn_kernel_in))):.3e}",
    f"median: {float(np.median(np.abs(Pn_cmm - Pn_kernel_in))):.3e}",
    f"p90: {float(np.percentile(np.abs(Pn_cmm - Pn_kernel_in), 90)):.3e}",
    "",
    "Interpretation: the cell-wise discrepancy IS the O(h) error of the "
    "kernel operator, varied locally (small in flat regions, large near "
    "Morse-critical price levels). Median is O(h^2), tails reach O(h).",
    "",
    "Pages 2-5 show: $P^*$ slices; $\\Phi_{\\rm CMM}$ output slices; "
    "$|\\Phi_{\\rm CMM}-\\Phi_{\\rm kernel}|$ error map; "
    "and the extracted level curves $\\{P^*=p\\}$ in 2D --- those are "
    "exactly the curves CMM integrates over.",
]
ax.text(0.05, 0.92, "\n".join(txt), fontsize=10, va='top', ha='left',
        wrap=True)
plt.savefig(f"{BUILD}/page1.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- Page 2: P* slices ----------
fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
fig.suptitle(r"Page 2 — Fixed point $P^*(u_1,u_2,u_3)$ slices",
              fontsize=14, weight='bold')
slices = [(-1.5, 0, 0), (0.0, 0, 1), (1.5, 0, 2)]
for (uval, _, axj) in slices:
    ax = axes.flat[axj]
    k = int(np.argmin(np.abs(u_in - uval)))
    cs = ax.contourf(u_in, u_in, P_inner[k], levels=np.linspace(0, 1, 21),
                      cmap='RdBu_r')
    ax.contour(u_in, u_in, P_inner[k], levels=np.linspace(0.1, 0.9, 9),
               colors='k', linewidths=0.6, alpha=0.5)
    ax.set_xlabel(r'$u_2$'); ax.set_ylabel(r'$u_3$')
    ax.set_title(rf"$P^*(u_1{{=}}{uval}, u_2, u_3)$")
    ax.set_aspect('equal')
plt.colorbar(cs, ax=axes[:, -1].tolist(), label='$P$', fraction=0.04, pad=0.04)
ax = axes[1, 1]
diag = np.array([P_inner[i,i,i] for i in range(Gi)])
ax.plot(u_in, diag, '-o', markersize=4)
ax.set_xlabel(r'$u$'); ax.set_ylabel(r'$P^*(u,u,u)$')
ax.set_title('Diagonal slice')
ax.grid(alpha=0.3)
plt.savefig(f"{BUILD}/page2.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- Page 3: CMM vs P* overlay ----------
fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
fig.suptitle(r"Page 3 — CMM operator $\Phi_{\rm CMM}(P^*)$ vs $P^*$",
              fontsize=14, weight='bold')
for row, uval in enumerate([-1.5, 0.0, 1.5]):
    k = int(np.argmin(np.abs(u_in - uval)))
    ax = axes[row, 0]
    cs1 = ax.contourf(u_in, u_in, P_inner[k], levels=np.linspace(0,1,21), cmap='RdBu_r')
    ax.set_title(rf"$P^*(u_1{{=}}{uval})$")
    ax.set_xlabel(r'$u_2$'); ax.set_ylabel(r'$u_3$'); ax.set_aspect('equal')
    ax = axes[row, 1]
    cs2 = ax.contourf(u_in, u_in, Pn_cmm[k], levels=np.linspace(0,1,21), cmap='RdBu_r')
    ax.set_title(rf"$\Phi_{{\rm CMM}}(P^*)(u_1{{=}}{uval})$")
    ax.set_xlabel(r'$u_2$'); ax.set_ylabel(r'$u_3$'); ax.set_aspect('equal')
plt.colorbar(cs1, ax=axes.ravel().tolist(), label='$P$', fraction=0.04, pad=0.04)
plt.savefig(f"{BUILD}/page3.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- Page 4: error map ----------
fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
fig.suptitle(r"Page 4 — $|\Phi_{\rm CMM} - \Phi_{\rm kernel}|$: where O(h) error lives",
              fontsize=14, weight='bold')
err = np.abs(Pn_cmm - Pn_kernel_in)
vmax = float(np.percentile(err, 99))
for row, uval in enumerate([-1.5, 0.0, 1.5]):
    if row >= 2: break
    ax = axes.flat[row]
    k = int(np.argmin(np.abs(u_in - uval)))
    cs = ax.imshow(err[k], origin='lower',
                    extent=[u_in[0], u_in[-1], u_in[0], u_in[-1]],
                    cmap='hot', vmin=0, vmax=vmax)
    ax.contour(u_in, u_in, P_inner[k], levels=np.linspace(0.1, 0.9, 9),
                colors='cyan', linewidths=0.5, alpha=0.6)
    ax.set_xlabel(r'$u_2$'); ax.set_ylabel(r'$u_3$')
    ax.set_title(rf"$|\Phi_{{\rm CMM}}-\Phi_{{\rm kernel}}|$ at $u_1{{=}}{uval}$")
    plt.colorbar(cs, ax=ax, fraction=0.04, pad=0.04)
# histogram of errors
ax = axes[1, 1]
ax.hist(err.ravel(), bins=60, color='C1', edgecolor='k')
ax.set_xlabel(r'$|\Phi_{\rm CMM}-\Phi_{\rm kernel}|$'); ax.set_ylabel('cells')
ax.set_title(f"Distribution (median {np.median(err):.2e}, p99 {vmax:.2e}, h={h:.2f})")
ax.set_yscale('log')
plt.savefig(f"{BUILD}/page4.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- Page 5: level curves ----------
fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
fig.suptitle(r"Page 5 — Level curves CMM integrates over (h=0 by construction)",
              fontsize=14, weight='bold')
plevels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
colors = plt.cm.viridis(np.linspace(0, 1, len(plevels)))
surfs = extract_surfaces_marching_cubes(P_full, uf, plevels)
for row, u1val in enumerate([-1.5, 0.0]):
    if row >= 2: break
    ax = axes[row, 0]
    k_glob = int(np.argmin(np.abs(uf - u1val)))
    for ip, p in enumerate(plevels):
        s = surfs[ip]
        if s is None: continue
        verts, faces = s
        # slice mesh at u1 = uf[k_glob] using cmm_stage1 routines
        from cmm_stage1 import slice_mesh_by_hyperplane, stitch_segments
        segs = slice_mesh_by_hyperplane(verts, faces, 0, uf[k_glob])
        chains = stitch_segments(segs)
        for c in chains:
            ax.plot(c[:,0], c[:,1], '-', color=colors[ip], linewidth=1.5,
                    label=f"p={p}" if c is chains[0] else None)
    ax.set_xlabel(r'$u_2$'); ax.set_ylabel(r'$u_3$')
    ax.set_title(rf"Level curves $\{{P^*=p\}}$ at $u_1{{=}}{u1val}$ (slice)")
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    if row == 0: ax.legend(fontsize=7, loc='lower left')
# 3D surface preview (matplotlib 3D)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
for ip, p in enumerate([0.3, 0.5, 0.7]):
    ip_full = plevels.index(p)
    s = surfs[ip_full]
    if s is None: continue
    verts, faces = s
    tri = verts[faces]
    col = Poly3DCollection(tri, alpha=0.3, color=colors[ip_full],
                            edgecolor='none')
    ax3.add_collection3d(col)
ax3.set_xlim(u_in[0], u_in[-1]); ax3.set_ylim(u_in[0], u_in[-1])
ax3.set_zlim(u_in[0], u_in[-1])
ax3.set_xlabel(r'$u_1$'); ax3.set_ylabel(r'$u_2$'); ax3.set_zlabel(r'$u_3$')
ax3.set_title('Selected level surfaces $\\{P=0.3, 0.5, 0.7\\}$')
ax = axes[1, 1]
# integrated arclength per (price, own_signal) — heat map
arc_grid = np.zeros((len(plevels), Gi))
for ip, p in enumerate(plevels):
    s = surfs[ip]
    if s is None: continue
    verts, faces = s
    for ki in range(Gi):
        segs = slice_mesh_by_hyperplane(verts, faces, 0, u_in[ki])
        chains = stitch_segments(segs)
        arc_grid[ip, ki] = sum(float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
                                for c in chains)
im = ax.imshow(arc_grid, aspect='auto', origin='lower',
                extent=[u_in[0], u_in[-1], plevels[0], plevels[-1]],
                cmap='magma')
ax.set_xlabel(r'own $u_1$'); ax.set_ylabel(r'price level $p$')
ax.set_title('Arc length of $\\{P=p\\}\\cap\\{u_1=U\\}$')
plt.colorbar(im, ax=ax, label='arclen')
plt.savefig(f"{BUILD}/page5.png", dpi=150, bbox_inches='tight'); plt.close()

# ---------- Combine ----------
from PIL import Image
imgs = [Image.open(f"{BUILD}/page{p}.png").convert("RGB") for p in range(1, 6)]
dst = f"{OUT}/CMM_stage1c_contours.pdf"
imgs[0].save(dst, save_all=True, append_images=imgs[1:])
print(f"saved {dst}")
