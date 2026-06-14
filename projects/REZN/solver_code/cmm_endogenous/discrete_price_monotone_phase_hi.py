"""High-gamma / high-tau extension of the monotone phase diagram.

Sweep: gamma in {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0}
       tau   in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0}
       G=21, M=32  (same as the original (tau, gamma) phase run)

Companion to discrete_price_monotone_phase.py -- the previous sweep
showed the monotone "good zone" expanding with gamma and contracting
with tau. This script extends both axes to map the upper-right corner.
Cells whose solver fails (e.g. tau=3 with weak coupling) are flagged
'solver_failed' but the sweep continues.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel, compute_deficit
from discrete_price_soft_monotone import (
    solve_kmeans_soft_monotone, max_mono_violation,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_monotone_phase_hi'
os.makedirs(OUT, exist_ok=True)


def main():
    Gi = 21
    M = 32
    gammas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    taus   = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"=== Monotone phase diagram (HIGH gamma/tau) ===", flush=True)
    print(f"G={Gi}, M={M}, gammas={gammas}, taus={taus}\n", flush=True)
    rows = []
    for ig, gamma in enumerate(gammas):
        for it, tau in enumerate(taus):
            print(f"[gamma={gamma}, tau={tau}] ", end='', flush=True)
            t0 = time.time()
            try:
                P_kernel, uf, lo, hi = solve_kernel(Gi, tau, gamma)
                kernel_viol = max_mono_violation(P_kernel)
                kernel_def = compute_deficit(P_kernel, uf, lo, hi, tau)
                p_levels = np.linspace(0.05, 0.95, M)
                p_field, hist, status, cycle = solve_kmeans_soft_monotone(
                    P_kernel, p_levels, uf, lo, hi, tau, gamma,
                    max_iter=40, h_p_factor=1.5)
                final = hist[-1]
                tail_def = float(np.mean([h['deficit'] for h in hist[-min(10, len(hist)):]]))
                tail_viol = float(np.mean([h['max_viol_raw'] for h in hist[-min(10, len(hist)):]]))
                wall = time.time() - t0
                print(f"status={status:>22} viol_raw={final['max_viol_raw']:.2e} "
                      f"deficit={tail_def:.3f}  wall={wall:.1f}s", flush=True)
                rows.append(dict(
                    gamma=gamma, tau=tau, status=status, cycle=int(cycle),
                    kernel_max_viol=float(kernel_viol),
                    kernel_deficit=float(kernel_def),
                    deficit_final=float(final['deficit']),
                    deficit_tail=tail_def,
                    max_viol_raw_final=float(final['max_viol_raw']),
                    max_viol_raw_tail=tail_viol,
                    max_res=float(final['max_res']),
                    wall=float(wall)))
            except Exception as e:
                wall = time.time() - t0
                print(f"FAILED ({e}) wall={wall:.1f}s", flush=True)
                rows.append(dict(gamma=gamma, tau=tau, status='solver_failed',
                                  error=str(e), wall=float(wall)))
            json.dump(dict(rows=rows, G=Gi, M=M,
                            gammas=gammas, taus=taus),
                      open(f"{OUT}/phase.json", 'w'), indent=2, default=str)
    # Build status matrix
    code = {'monotone_converged': 0, 'fixed_but_nonmonotone': 1,
            'cycle': 2, 'not_converged': 3, 'solver_failed': 4}
    status_mat = np.full((len(gammas), len(taus)), -1, dtype=int)
    deficit_mat = np.full((len(gammas), len(taus)), np.nan)
    viol_mat = np.full((len(gammas), len(taus)), np.nan)
    for r in rows:
        ig = gammas.index(r['gamma']); it = taus.index(r['tau'])
        status_mat[ig, it] = code.get(r['status'], -1)
        if 'deficit_tail' in r:
            deficit_mat[ig, it] = r['deficit_tail']
            viol_mat[ig, it] = r['max_viol_raw_tail']
    # Plot using categorical indices (works for non-uniform grids)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    cmap = ListedColormap(['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#7f8c8d'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    im0 = axes[0].imshow(status_mat, origin='lower', aspect='auto',
                          cmap=cmap, norm=norm)
    axes[0].set_xticks(range(len(taus)));
    axes[0].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
    axes[0].set_yticks(range(len(gammas)));
    axes[0].set_yticklabels([f'{g:g}' for g in gammas])
    axes[0].set_xlabel(r'$\tau$'); axes[0].set_ylabel(r'$\gamma$')
    axes[0].set_title('Status: green=monotone, orange=non-mono, red=cycle,\n'
                       'purple=no-conv, grey=solver-failed')
    cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0,1,2,3,4])
    cbar0.ax.set_yticklabels(['mono-conv', 'non-mono', 'cycle', 'no-conv', 'fail'])
    for ig in range(len(gammas)):
        for it in range(len(taus)):
            axes[0].text(it, ig, str(status_mat[ig, it]),
                          ha='center', va='center', fontsize=8, color='black')
    im1 = axes[1].imshow(deficit_mat, origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_xticks(range(len(taus)));
    axes[1].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
    axes[1].set_yticks(range(len(gammas)));
    axes[1].set_yticklabels([f'{g:g}' for g in gammas])
    axes[1].set_xlabel(r'$\tau$'); axes[1].set_ylabel(r'$\gamma$')
    axes[1].set_title('Tail-mean deficit  $1-R^2$')
    fig.colorbar(im1, ax=axes[1])
    for ig in range(len(gammas)):
        for it in range(len(taus)):
            v = deficit_mat[ig, it]
            if np.isfinite(v):
                axes[1].text(it, ig, f'{v:.2f}',
                              ha='center', va='center', fontsize=7, color='white')
    im2 = axes[2].imshow(np.log10(np.maximum(viol_mat, 1e-16)), origin='lower',
                          aspect='auto', cmap='magma')
    axes[2].set_xticks(range(len(taus)));
    axes[2].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
    axes[2].set_yticks(range(len(gammas)));
    axes[2].set_yticklabels([f'{g:g}' for g in gammas])
    axes[2].set_xlabel(r'$\tau$'); axes[2].set_ylabel(r'$\gamma$')
    axes[2].set_title(r'$\log_{10}$ RAW max non-monotone step')
    fig.colorbar(im2, ax=axes[2])
    dp = 0.9 / (M-1)
    thresh = 0.5 * dp
    cs = axes[2].contour(range(len(taus)), range(len(gammas)), viol_mat,
                          levels=[thresh], colors=['white'],
                          linestyles=['--'], linewidths=2)
    axes[2].clabel(cs, fmt=lambda v: f'  Δp/2 = {v:.3e}  ', fontsize=8)
    plt.suptitle(rf'Monotone-discrete-price phase diagram '
                 rf'(HIGH range), $G={Gi}$, $M={M}$', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/phase.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {OUT}/phase.png", flush=True)


if __name__ == "__main__":
    main()
