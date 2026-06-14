"""Generate contour-slice and regression plots for representative
discrete-price equilibria, and bundle into INTEGRATED_RESULTS.pdf.

Picks G=21, M in {16, 32, 64} as the highest-G finished cells from
the base sweep.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel, solve_kmeans, compute_deficit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

TAU = float(os.environ.get('TAU', '2.0'))
GAMMA = float(os.environ.get('GAMMA', '0.01'))
OUT = os.environ.get('OUT',
    '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_sweep')
os.makedirs(OUT, exist_ok=True)


def slice_plot(ax, p_field, uf, lo, hi, k_slice, title):
    """Plot a 2D slice of the discrete-price field at u3 = uf[lo+k_slice]."""
    Gi = hi - lo
    u_in = uf[lo:hi]
    Sl = p_field[:, :, k_slice]
    im = ax.imshow(Sl.T, origin='lower', extent=[u_in[0], u_in[-1], u_in[0], u_in[-1]],
                    cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(title, fontsize=10)
    return im


def regression_plot(ax, p_field, uf, lo, hi, tau, title):
    Gi = hi - lo
    u_in = uf[lo:hi]
    sum_u = np.add.outer(np.add.outer(u_in, u_in), u_in).ravel()
    pc = np.clip(p_field.ravel(), 1e-12, 1-1e-12)
    y = np.log(pc/(1-pc))
    a = np.polyfit(sum_u, y, 1)
    pr = a[0]*sum_u + a[1]
    R2 = 1.0 - float(np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30))
    deficit = 1.0 - R2
    ax.scatter(sum_u, y, s=4, alpha=0.3, color='steelblue')
    xs = np.linspace(sum_u.min(), sum_u.max(), 100)
    ax.plot(xs, a[0]*xs + a[1], 'r--', lw=1.4,
            label=f'slope={a[0]:.3f},  $R^2$={R2:.3f}')
    # also FR baseline
    ax.plot(xs, tau*xs, 'g:', lw=1.2, alpha=0.7,
            label=fr'FR: slope $=\tau=${tau:.1f}')
    ax.set_xlabel(r'$\sum_k u_k$'); ax.set_ylabel(r'$\log(p/(1-p))$')
    ax.set_title(f'{title}  (deficit = {deficit:.3f})', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return R2


def main():
    Gi = 21
    print(f"\nSolving kernel at G={Gi}...", flush=True)
    t0 = time.time()
    P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    M_list = [16, 32, 64]
    fields = {}
    for M in M_list:
        p_levels = np.linspace(0.05, 0.95, M)
        print(f"Solving G={Gi} M={M}...", flush=True)
        t0 = time.time()
        assignment, hist, period = solve_kmeans(
            P_kernel, p_levels, uf, lo, hi, TAU, GAMMA, max_iter=60)
        # Use final assignment's discretized field
        p_field = p_levels[assignment]
        fields[M] = (p_field, p_levels, period)
        print(f"  done in {time.time()-t0:.1f}s  period={period}", flush=True)

    # Also keep the kernel-h>0 smooth field for comparison
    fields['kernel'] = (P_kernel, None, None)

    # === Figure 1: 3x3 grid of slices at u3 = -2, 0, +2 for M=16, 32, 64 ===
    fig, axes = plt.subplots(3, 4, figsize=(15, 11),
                              gridspec_kw=dict(width_ratios=[1, 1, 1, 0.06]))
    u_in = uf[lo:hi]
    slice_indices = [int(np.argmin(np.abs(u_in - v))) for v in [-2, 0, 2]]
    row_labels = ['M=16', 'M=32', 'M=64']
    for r, M in enumerate(M_list):
        p_field, _, _ = fields[M]
        for c, k_sl in enumerate(slice_indices):
            ax = axes[r, c]
            im = slice_plot(ax, p_field, uf, lo, hi, k_sl,
                             f'{row_labels[r]}, $u_3={u_in[k_sl]:+.2f}$')
            if c > 0:
                ax.set_ylabel('')
        # Colorbar at end of row
        cax = axes[r, 3]
        fig.colorbar(im, cax=cax, label='$P(u)$')
    plt.suptitle(rf'Discrete-price equilibrium $P(u_1, u_2 \mid u_3)$  --  $G={Gi}$',
                 y=0.995)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slices_G{Gi}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved slices_G{Gi}.png", flush=True)

    # === Figure 2: regression scatters, 1x4 (kernel + 3 discrete) ===
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    pk = fields['kernel'][0]
    R2_ker = regression_plot(axes[0], pk, uf, lo, hi, TAU,
                              f'Kernel-$h{{>}}0$  (G={Gi})')
    for i, M in enumerate(M_list):
        p_field, _, _ = fields[M]
        regression_plot(axes[i+1], p_field, uf, lo, hi, TAU,
                         f'Discrete  G={Gi}  M={M}')
    plt.suptitle(r'Log-odds regression: $\log(p/(1-p))$ vs $\sum_k u_k$ '
                 r'(red dashed = OLS, green dotted = FR slope $\tau$)',
                 y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/regression_G{Gi}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved regression_G{Gi}.png", flush=True)

    # === Figure 3: kernel-h>0 slices for comparison ===
    fig, axes = plt.subplots(1, 4, figsize=(15, 4),
                              gridspec_kw=dict(width_ratios=[1, 1, 1, 0.06]))
    for c, k_sl in enumerate(slice_indices):
        ax = axes[c]
        im = slice_plot(ax, pk, uf, lo, hi, k_sl,
                         f'kernel-$h{{>}}0$, $u_3={u_in[k_sl]:+.2f}$')
    fig.colorbar(im, cax=axes[3], label='$P(u)$')
    plt.suptitle(rf'Kernel-$h{{>}}0$ certified solution  --  $G={Gi}$', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slices_kernel_G{Gi}.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved slices_kernel_G{Gi}.png", flush=True)


if __name__ == "__main__":
    main()
