"""High-G extension of discrete_price_sweep.py.

Sweeps G in {27, 33, 41} x M in {16, 32, 64} at (tau, gamma) = (2.0, 0.01).

Vectorized partition-Bayes (einsum) keeps each K-means iter at O(M*G^3),
so even G=41, M=64 should finish in a few seconds per cell.
The dominant cost is the kernel solve (Newton-Krylov on G^3 unknowns).
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import (
    solve_kernel, compute_clearing_vectorized, compute_deficit, solve_kmeans,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAU, GAMMA = 2.0, 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_sweep_HIGH'
os.makedirs(OUT, exist_ok=True)


def main():
    print("=== Discrete-price-grid sweep HIGH-G ===\n", flush=True)
    print(f"(tau, gamma) = ({TAU}, {GAMMA})\n", flush=True)
    G_list = [27, 33, 41]
    M_list = [16, 32, 64]
    results = {}
    for Gi in G_list:
        print(f"\n========== G = {Gi} ==========", flush=True)
        t0 = time.time()
        P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
        t_ker = time.time() - t0
        kernel_deficit = compute_deficit(P_kernel, uf, lo, hi, TAU)
        print(f"  kernel solved in {t_ker:.1f}s, deficit = {kernel_deficit:.4e}",
              flush=True)
        for M in M_list:
            p_levels = np.linspace(0.05, 0.95, M)
            print(f"\n  --- M={M} (spacing={p_levels[1]-p_levels[0]:.3f}) ---",
                  flush=True)
            t0 = time.time()
            assignment, history, period = solve_kmeans(
                P_kernel, p_levels, uf, lo, hi, TAU, GAMMA, max_iter=60)
            wall = time.time() - t0
            final = history[-1]
            tail = history[-min(10, len(history)):]
            mean_def_tail = float(np.mean([h['deficit'] for h in tail]))
            print(f"    period={period} (1=converged), max_res={final['max_res']:.3e}, "
                  f"med_res={final['med_res']:.3e}, deficit_final={final['deficit']:.4e}, "
                  f"deficit_tail_mean={mean_def_tail:.4e}, wall={wall:.1f}s",
                  flush=True)
            results[f"G{Gi}_M{M}"] = dict(
                G=Gi, M=M,
                period=int(period),
                converged=(period == 1),
                max_res=final['max_res'], med_res=final['med_res'],
                deficit_final=final['deficit'],
                deficit_tail_mean=mean_def_tail,
                kernel_deficit=float(kernel_deficit),
                kernel_wall=float(t_ker),
                wall=float(wall), spacing=float(p_levels[1]-p_levels[0]),
                history=history)
            json.dump(results, open(f"{OUT}/sweep.json", 'w'), indent=2, default=str)
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for Gi in G_list:
        ms = [r['M'] for r in results.values() if r['G'] == Gi]
        defs = [r['deficit_tail_mean'] for r in results.values() if r['G'] == Gi]
        max_r = [r['max_res'] for r in results.values() if r['G'] == Gi]
        axes[0].plot(ms, defs, 'o-', label=f'G={Gi}')
        axes[1].plot(ms, max_r, 'o-', label=f'G={Gi}')
    for Gi in G_list:
        kd = next(r['kernel_deficit'] for r in results.values() if r['G'] == Gi)
        axes[0].axhline(kd, ls=':', alpha=0.4)
    axes[0].set_xlabel('M (price levels)'); axes[0].set_ylabel('deficit (tail mean)')
    axes[0].set_xscale('log', base=2); axes[0].set_title('Discrete-price deficit vs M (high G)')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('M'); axes[1].set_ylabel('max residual')
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
    axes[1].set_title('Max residual vs M')
    axes[1].legend(); axes[1].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/sweep.png", dpi=140); plt.close()
    print(f"\nSaved {OUT}/sweep.json, {OUT}/sweep.png", flush=True)


if __name__ == "__main__":
    main()
