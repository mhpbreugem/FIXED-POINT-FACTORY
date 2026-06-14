"""Side-by-side SOFT slices at tau=2.0 vs tau=0.1, G=21, u3=0.

Tests the visual hypothesis: at low tau the SOFT partition is cleanly
monotone (no diagonal fragmentation, level sets ~ perpendicular to (1,1)).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel
from discrete_price_soft import solve_kmeans_soft, slice_panel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA = 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_soft_compare'
os.makedirs(OUT, exist_ok=True)


def main():
    Gi = 21
    M_list = [16, 32, 64]
    fields = {}  # (tau, M) -> field
    for tau in [2.0, 0.1]:
        print(f"\nSolving kernel at G={Gi}, tau={tau}...", flush=True)
        t0 = time.time()
        P_kernel, uf, lo, hi = solve_kernel(Gi, tau, GAMMA)
        print(f"  done in {time.time()-t0:.1f}s", flush=True)
        for M in M_list:
            p_levels = np.linspace(0.05, 0.95, M)
            print(f"  SOFT M={M}...", end=' ', flush=True)
            t0 = time.time()
            p_field, hist, per = solve_kmeans_soft(
                P_kernel, p_levels, uf, lo, hi, tau, GAMMA,
                max_iter=60, h_p_factor=1.5)
            print(f"period={per}  deficit={hist[-1]['deficit']:.4f}  "
                  f"wall={time.time()-t0:.1f}s", flush=True)
            fields[(tau, M)] = (p_field, uf, lo, hi)

    # Figure: 2 rows (tau=2, tau=0.1) x 3 cols (M=16, 32, 64); slice u3=0
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4),
                              gridspec_kw=dict(width_ratios=[1, 1, 1, 0.06]))
    for r, tau in enumerate([2.0, 0.1]):
        p_field0, uf0, lo0, hi0 = fields[(tau, M_list[0])]
        u_in = uf0[lo0:hi0]
        k0 = int(np.argmin(np.abs(u_in)))
        for c, M in enumerate(M_list):
            p_field, uf, lo, hi = fields[(tau, M)]
            im = slice_panel(axes[r, c], p_field, uf, lo, hi, k0,
                              rf'$\tau={tau}$,  SOFT  $M={M}$')
        fig.colorbar(im, cax=axes[r, 3], label='$P(u)$')
    plt.suptitle(r'SOFT lookup, $u_3=0$ slice: $\tau=2.0$ (top) vs $\tau=0.1$ (bottom), $G=21$',
                 y=1.0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/soft_slices_tau_compare.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {OUT}/soft_slices_tau_compare.png", flush=True)


if __name__ == "__main__":
    main()
