"""Soft-bin discrete-price solver.

Same setup as discrete_price_sweep.py: P takes values on a fixed
discrete grid {p_1, ..., p_M}.  But the partition-Bayes lookup table
A_v(k, m, idx) is built with a SOFT Gaussian kernel weight, not a hard
mask:
    w_cell(m) = K_h(p_field[cell] - p_levels[m]),  sum_m w_cell(m) = 1
    A_v(k, m, idx) = sum_{cells} w_cell(m) * f_v(other signals)
Each cell contributes to multiple price levels with a smooth weight, so
A_v varies smoothly in m and the resulting posterior mu(u) is smooth in
u even though P(u) is still on the discrete grid.

Two passes per (tau, G, M):
  HARD:  the original argmin assignment / hard mask (for comparison)
  SOFT:  Gaussian kernel of bandwidth h_p = c * Delta p  (fixed)

Iteration projects p_clear back to the grid after each step:
    p_field_new[cell] = p_levels[argmin_m |p_clear[cell] - p_m|]
so the equilibrium price is always on the discrete grid; only the
lookup is smoothed.

Outputs: comparison table + before/after regression scatter at one cell.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel, solve_kmeans, compute_deficit
from reznsrc.demand import clear_crra
from reznsrc.signals import f_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAU = float(os.environ.get('TAU', '2.0'))
GAMMA = float(os.environ.get('GAMMA', '0.01'))
OUT = os.environ.get('OUT',
    '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_soft')
os.makedirs(OUT, exist_ok=True)


def compute_clearing_soft(p_field, p_levels, uf, lo, hi, tau, gamma, h_p):
    """Soft-kernel partition-Bayes.  A_v indexed by (k, m, idx) with
    smooth Gaussian weighting over m.  Returns the per-cell clearing
    price p_clear."""
    Gi = hi - lo
    M = len(p_levels)
    f0 = np.array([f_signal(uf[lo + i], 0, tau) for i in range(Gi)])
    f1 = np.array([f_signal(uf[lo + i], 1, tau) for i in range(Gi)])
    F0 = np.outer(f0, f0); F1 = np.outer(f1, f1)
    # Soft cell weights over m
    diffs = p_field[..., None] - p_levels[None, None, None, :]  # (G,G,G,M)
    W = np.exp(-0.5 * (diffs / h_p)**2)
    W /= np.maximum(W.sum(axis=-1, keepdims=True), 1e-300)
    # Build A_v(k, m, idx)
    A0 = np.zeros((3, M, Gi)); A1 = np.zeros((3, M, Gi))
    for m in range(M):
        w = W[..., m]
        A0[0, m, :] = np.einsum('ijl,jl->i', w, F0)
        A1[0, m, :] = np.einsum('ijl,jl->i', w, F1)
        A0[1, m, :] = np.einsum('ijl,il->j', w, F0)
        A1[1, m, :] = np.einsum('ijl,il->j', w, F1)
        A0[2, m, :] = np.einsum('ijl,ij->l', w, F0)
        A1[2, m, :] = np.einsum('ijl,ij->l', w, F1)
    # Per-cell mu: smooth weighted over m by the same cell weight
    p_clear = np.zeros((Gi, Gi, Gi))
    gam = np.full(3, gamma); Wv = np.full(3, 1.0)
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                w_c = W[i, j, l, :]  # (M,)
                mu = np.empty(3)
                for k_own, idx in [(0, i), (1, j), (2, l)]:
                    A0_v = A0[k_own, :, idx]; A1_v = A1[k_own, :, idx]
                    f0o = f0[idx]; f1o = f1[idx]
                    num = f1o * A1_v
                    den = f0o * A0_v + num
                    mu_m = num / np.maximum(den, 1e-300)
                    mu_cell = float(np.sum(w_c * mu_m))
                    mu[k_own] = max(1e-12, min(1-1e-12, mu_cell))
                p_clear[i, j, l] = clear_crra(mu, gam, Wv)
    return p_clear


def solve_kmeans_soft(P_kernel, p_levels, uf, lo, hi, tau, gamma,
                       max_iter=60, h_p_factor=1.5):
    """Soft-bin K-means.  Field is projected to grid each step; lookup
    is kernel-smoothed with bandwidth h_p = h_p_factor * Delta p."""
    dp = float(p_levels[1] - p_levels[0])
    h_p = h_p_factor * dp
    p_field = p_levels[np.argmin(
        np.abs(P_kernel[..., None] - p_levels[None, None, None, :]), axis=-1)]
    history = []
    p_field_hist = []
    for it in range(max_iter):
        p_clear = compute_clearing_soft(p_field, p_levels, uf, lo, hi, tau, gamma, h_p)
        a_new = np.argmin(np.abs(p_clear[..., None] - p_levels[None, None, None, :]), axis=-1)
        p_field_new = p_levels[a_new]
        n_changed = int(np.sum(p_field_new != p_field))
        max_res = float(np.max(np.abs(p_clear - p_field)))
        med_res = float(np.median(np.abs(p_clear - p_field)))
        deficit = compute_deficit(p_field, uf, lo, hi, tau)
        history.append(dict(it=it, h_p=h_p, n_changed=n_changed,
                             max_res=max_res, med_res=med_res, deficit=deficit))
        # Cycle detection
        p_field_hist.append(p_field.copy())
        cycle = 0
        for k in range(1, min(len(p_field_hist), 5) + 1):
            if k < len(p_field_hist) and np.array_equal(p_field, p_field_hist[-k-1]):
                cycle = k
                break
        if n_changed == 0:
            return p_field, history, 1
        if cycle > 0 and it > 5:
            return p_field, history, cycle
        p_field = p_field_new
    return p_field, history, 0


def regression_panel(ax, p_field, uf, lo, hi, tau, title):
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
    ax.plot(xs, tau*xs, 'g:', lw=1.2, alpha=0.7,
            label=fr'FR: $\tau$={tau}')
    ax.set_xlabel(r'$\sum_k u_k$'); ax.set_ylabel(r'$\log(p/(1-p))$')
    ax.set_title(f'{title}  (deficit = {deficit:.3f})', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return deficit


def slice_panel(ax, p_field, uf, lo, hi, k_slice, title):
    u_in = uf[lo:hi]
    Sl = p_field[:, :, k_slice]
    im = ax.imshow(Sl.T, origin='lower', extent=[u_in[0], u_in[-1], u_in[0], u_in[-1]],
                    cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel(r'$u_1$'); ax.set_ylabel(r'$u_2$')
    ax.set_title(title, fontsize=10)
    return im


def main():
    print(f"=== Soft-bin discrete-price solver  (tau={TAU}, gamma={GAMMA}) ===\n",
          flush=True)
    Gi = 21
    print(f"Solving kernel at G={Gi}...", flush=True)
    t0 = time.time()
    P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Sweep over M with HARD vs SOFT
    M_list = [16, 32, 64]
    results = {}
    for M in M_list:
        p_levels = np.linspace(0.05, 0.95, M)
        print(f"\n--- M={M} ---", flush=True)
        # Hard
        t0 = time.time()
        a_hard, h_hard, per_h = solve_kmeans(
            P_kernel, p_levels, uf, lo, hi, TAU, GAMMA, max_iter=60)
        p_hard = p_levels[a_hard]
        d_hard_final = h_hard[-1]['deficit']
        d_hard_tail = float(np.mean([h['deficit'] for h in h_hard[-10:]]))
        r_hard = h_hard[-1]['max_res']
        wall_h = time.time() - t0
        print(f"  HARD: period={per_h}  deficit_tail={d_hard_tail:.4f}  "
              f"max_res={r_hard:.3e}  wall={wall_h:.1f}s", flush=True)
        # Soft
        t0 = time.time()
        p_soft, h_soft, per_s = solve_kmeans_soft(
            P_kernel, p_levels, uf, lo, hi, TAU, GAMMA,
            max_iter=60, h_p_factor=1.5)
        d_soft_final = h_soft[-1]['deficit']
        d_soft_tail = float(np.mean([h['deficit'] for h in h_soft[-10:]]))
        r_soft = h_soft[-1]['max_res']
        wall_s = time.time() - t0
        print(f"  SOFT: period={per_s}  deficit_tail={d_soft_tail:.4f}  "
              f"max_res={r_soft:.3e}  wall={wall_s:.1f}s", flush=True)
        results[M] = dict(
            hard=dict(period=per_h, deficit_final=d_hard_final,
                       deficit_tail=d_hard_tail, max_res=r_hard, wall=wall_h),
            soft=dict(period=per_s, deficit_final=d_soft_final,
                       deficit_tail=d_soft_tail, max_res=r_soft, wall=wall_s,
                       h_p_factor=1.5, h_p=h_soft[-1]['h_p']),
            p_hard=p_hard, p_soft=p_soft, p_levels=p_levels,
        )

    # Save summary JSON
    summary = {f"M{M}": {kk: {k:v for k,v in vv.items() if not isinstance(v, np.ndarray)}
                          for kk, vv in r.items() if kk in ('hard','soft')}
               for M, r in results.items()}
    json.dump(summary, open(f"{OUT}/summary.json", 'w'), indent=2, default=str)

    # Figure: regression scatter, 2 rows x 3 cols (HARD top, SOFT bottom)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for c, M in enumerate(M_list):
        regression_panel(axes[0, c], results[M]['p_hard'], uf, lo, hi, TAU,
                          f'HARD  M={M}')
        regression_panel(axes[1, c], results[M]['p_soft'], uf, lo, hi, TAU,
                          f'SOFT  M={M}')
    plt.suptitle(rf'Log-odds regression --  HARD vs SOFT lookup --  $G={Gi}$, $\tau={TAU}$',
                 y=1.0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/regression_hard_vs_soft.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved regression_hard_vs_soft.png", flush=True)

    # Figure: slice at u3=0, 2 rows x 3 cols
    u_in = uf[lo:hi]
    k0 = int(np.argmin(np.abs(u_in)))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8),
                              gridspec_kw=dict(width_ratios=[1, 1, 1, 0.06]))
    for c, M in enumerate(M_list):
        im = slice_panel(axes[0, c], results[M]['p_hard'], uf, lo, hi, k0,
                          f'HARD  M={M},  $u_3=0$')
        im = slice_panel(axes[1, c], results[M]['p_soft'], uf, lo, hi, k0,
                          f'SOFT  M={M},  $u_3=0$')
    fig.colorbar(im, cax=axes[0, 3], label='$P(u)$')
    fig.colorbar(im, cax=axes[1, 3], label='$P(u)$')
    plt.suptitle(rf'$u_3=0$ slice  --  HARD (top) vs SOFT (bottom)  --  $G={Gi}$, $\tau={TAU}$',
                 y=1.0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slices_hard_vs_soft.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved slices_hard_vs_soft.png", flush=True)

    # Print final summary
    print(f"\n=== Summary (tau={TAU}) ===", flush=True)
    print(f"{'M':>4} {'mode':>5} {'period':>7} {'deficit':>9} {'max_res':>9}", flush=True)
    for M in M_list:
        for mode in ('hard', 'soft'):
            r = results[M][mode]
            print(f"{M:>4} {mode:>5} {r['period']:>7} {r['deficit_tail']:>9.4f} "
                  f"{r['max_res']:>9.3e}", flush=True)


if __name__ == "__main__":
    main()
