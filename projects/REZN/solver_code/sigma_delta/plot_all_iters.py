"""Plot all 50 iters of the gmpy2 Picard run."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

snaps = np.load('/tmp/gmpy2_snapshots.npy')  # shape (51, G, G, G), iters 0..50
with open('/tmp/gmpy2_ferr.json') as f:
    info = json.load(f)
ferr = info['ferr']
G = snaps.shape[1]
mid = G // 2
TAU = 2.0; UMAX = 3.0
u_full = np.linspace(-UMAX, UMAX, G)
U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')
P_FR = 1.0/(1.0+np.exp(-TAU*(U1+U2+U3)))

FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'

# All 51 P slices at u_1=middle (5×11 grid + 1 trailing or 6×9 + ...)
NCOLS = 11
NROWS = int(np.ceil(51 / NCOLS))
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(2*NCOLS, 2*NROWS), dpi=130)
axes = axes.ravel()
extent = [u_full[0], u_full[-1], u_full[0], u_full[-1]]
for it in range(51):
    ax = axes[it]
    sl = snaps[it, mid, :, :]
    im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                   extent=extent, aspect='auto')
    title = 'iter 0\n(FR IC)' if it == 0 else f'iter {it}\nferr={ferr[it]:.1e}'
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
# blank remaining
for j in range(51, len(axes)):
    axes[j].axis('off')
plt.suptitle(f'gmpy2 200-dec Picard, all 51 P snapshots at u_1=0 slice, γ={info["GAMMA"]}',
              fontsize=12, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/gmpy2_all_P.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/gmpy2_all_P.png')

# All 51 deviation slices
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(2*NCOLS, 2*NROWS), dpi=130)
axes = axes.ravel()
for it in range(51):
    ax = axes[it]
    dev = snaps[it, mid, :, :] - P_FR[mid, :, :]
    vmax = max(abs(dev).max(), 1e-200)
    im = ax.imshow(dev.T, origin='lower', cmap='PiYG', vmin=-vmax, vmax=vmax,
                   extent=extent, aspect='auto')
    title = f'iter {it}\nmax|dev|={vmax:.1e}'
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
for j in range(51, len(axes)):
    axes[j].axis('off')
plt.suptitle(f'gmpy2 200-dec Picard, all 51 deviation snapshots P − P^FR at u_1=0',
              fontsize=12, weight='bold', y=1.0)
plt.tight_layout()
plt.savefig(f'{FIG}/gmpy2_all_dev.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/gmpy2_all_dev.png')

# Full trajectory diagnostics
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
ax = axes[0]
iters = np.arange(0, len(ferr))
ax.semilogy(iters, np.maximum(ferr, 1e-200), 'o-', lw=1.5, ms=5)
ax.set_xlabel('iter'); ax.set_ylabel('ferr')
ax.set_title('ferr over 50 iters (no damping, ω=1.0 throughout)')
ax.grid(True, ls=':', alpha=0.5)
ax = axes[1]
maxdev = [np.max(np.abs(snaps[i] - P_FR)) for i in range(51)]
ax.semilogy(iters, np.maximum(maxdev, 1e-200), 's-', lw=1.5, ms=5, color='C2')
ax.set_xlabel('iter'); ax.set_ylabel('max|P − P^FR|')
ax.set_title('max deviation from FR per iter')
ax.grid(True, ls=':', alpha=0.5)
plt.suptitle(f'gmpy2 200-dec Picard, full trajectory γ={info["GAMMA"]}',
              fontsize=11.5, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/gmpy2_all_trajectory.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'wrote {FIG}/gmpy2_all_trajectory.png')
