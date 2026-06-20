"""Soft-bin discrete-price solver with HARD monotonicity enforcement.

After each soft-lookup K-means step, project p_clear onto the cone of
componentwise-monotone functions P(u_1, u_2, u_3) increasing in each
argument, via iterated coordinate-wise PAV (pool-adjacent-violators).

Diagnostic on the RAW (unprojected) clearing field:
    max_violation = max over axes a of max(-(diff_a(p_clear))_negative)
A `monotone-converged' iterate has (n_changed == 0) AND max_violation
below MONO_TOL (default 0.5 * Delta p, i.e. the discretization floor).
Otherwise the iterate is flagged either "fixed but non-monotone'' (no
cell-flip but the raw field is non-monotone), or "not converged''
(cycle/max-iter).

Run on a tau ladder: tau in {0.1, 0.2, ..., 1.0}, G=21, M=32.
Produces JSON summary + deficit-vs-tau and violation-vs-tau plots.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from discrete_price_sweep import solve_kernel, compute_deficit
from discrete_price_soft import compute_clearing_soft, slice_panel
try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SK = True
except ImportError:
    HAS_SK = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA = 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_monotone_ladder'
os.makedirs(OUT, exist_ok=True)


def _pav_1d(y):
    """Pool-adjacent-violators (increasing) on a 1D array."""
    if HAS_SK:
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        return ir.fit_transform(np.arange(len(y)), y)
    # Fallback: simple O(n) PAV implementation
    y = y.astype(float).copy()
    n = len(y)
    w = np.ones(n)
    # Stack of (start, end, sum, weight)
    starts = [0]; ends = [0]; sums = [y[0]]; weights = [1.0]
    for i in range(1, n):
        starts.append(i); ends.append(i); sums.append(y[i]); weights.append(1.0)
        while len(starts) >= 2 and sums[-2]/weights[-2] > sums[-1]/weights[-1]:
            s2 = sums.pop(); w2 = weights.pop(); st2 = starts.pop(); en2 = ends.pop()
            sums[-1] += s2; weights[-1] += w2; ends[-1] = en2
    out = np.empty(n)
    for s, e, sm, wt in zip(starts, ends, sums, weights):
        out[s:e+1] = sm / wt
    return out


def isotonic_3d(P, max_sweeps=10, tol=1e-12):
    """Project P onto the cone of componentwise-monotone (increasing) 3D
    fields via iterated coordinate-wise PAV.  Converges in few sweeps."""
    P = P.astype(float).copy()
    n0, n1, n2 = P.shape
    for sweep in range(max_sweeps):
        P_prev = P.copy()
        # axis 0
        for j in range(n1):
            for l in range(n2):
                P[:, j, l] = _pav_1d(P[:, j, l])
        # axis 1
        for i in range(n0):
            for l in range(n2):
                P[i, :, l] = _pav_1d(P[i, :, l])
        # axis 2
        for i in range(n0):
            for j in range(n1):
                P[i, j, :] = _pav_1d(P[i, j, :])
        if np.max(np.abs(P - P_prev)) < tol:
            break
    return P


def max_mono_violation(P):
    """Largest negative step along any axis."""
    d0 = np.diff(P, axis=0); d1 = np.diff(P, axis=1); d2 = np.diff(P, axis=2)
    return float(max(-d0.min(initial=0.0),
                      -d1.min(initial=0.0),
                      -d2.min(initial=0.0)))


def project_to_grid(P, p_levels):
    a = np.argmin(np.abs(P[..., None] - p_levels[None, None, None, :]), axis=-1)
    return p_levels[a]


def solve_kmeans_soft_monotone(P_kernel, p_levels, uf, lo, hi, tau, gamma,
                                 max_iter=60, h_p_factor=1.5, mono_tol=None):
    """Soft K-means with monotone projection.  Tracks RAW violations."""
    dp = float(p_levels[1] - p_levels[0])
    h_p = h_p_factor * dp
    if mono_tol is None:
        mono_tol = 0.5 * dp
    p_field = project_to_grid(P_kernel, p_levels)
    history = []
    p_hist = []
    for it in range(max_iter):
        p_clear_raw = compute_clearing_soft(
            p_field, p_levels, uf, lo, hi, tau, gamma, h_p)
        max_viol = max_mono_violation(p_clear_raw)
        n_viol = int(np.sum(np.diff(p_clear_raw, axis=0) < -1e-12)
                     + np.sum(np.diff(p_clear_raw, axis=1) < -1e-12)
                     + np.sum(np.diff(p_clear_raw, axis=2) < -1e-12))
        p_clear_mono = isotonic_3d(p_clear_raw)
        p_field_new = project_to_grid(p_clear_mono, p_levels)
        n_changed = int(np.sum(p_field_new != p_field))
        max_res = float(np.max(np.abs(p_clear_mono - p_field)))
        med_res = float(np.median(np.abs(p_clear_mono - p_field)))
        deficit = compute_deficit(p_field, uf, lo, hi, tau)
        history.append(dict(it=it, h_p=h_p, n_changed=n_changed,
                             max_res=max_res, med_res=med_res, deficit=deficit,
                             max_viol_raw=max_viol, n_viol_raw=n_viol))
        p_hist.append(p_field.copy())
        cycle = 0
        for k in range(1, min(len(p_hist), 5) + 1):
            if k < len(p_hist) and np.array_equal(p_field, p_hist[-k-1]):
                cycle = k
                break
        if n_changed == 0:
            status = 'monotone_converged' if max_viol < mono_tol \
                     else 'fixed_but_nonmonotone'
            return p_field, history, status, cycle
        if cycle > 0 and it > 5:
            return p_field, history, 'cycle', cycle
        p_field = p_field_new
    return p_field, history, 'not_converged', 0


def main():
    Gi = 21
    M = 32
    taus = [round(0.1 + 0.1 * k, 2) for k in range(10)]  # 0.1 .. 1.0
    print(f"=== Monotone soft-bin discrete-price ladder ===", flush=True)
    print(f"G={Gi}, M={M}, tau in {taus}\n", flush=True)
    rows = []
    fields = {}
    for tau in taus:
        print(f"--- tau = {tau} ---", flush=True)
        t0 = time.time()
        P_kernel, uf, lo, hi = solve_kernel(Gi, tau, GAMMA)
        t_k = time.time() - t0
        p_levels = np.linspace(0.05, 0.95, M)
        kernel_def = compute_deficit(P_kernel, uf, lo, hi, tau)
        # Also: raw monotonicity of the kernel
        kernel_viol = max_mono_violation(P_kernel)
        t0 = time.time()
        p_field, hist, status, cycle = solve_kmeans_soft_monotone(
            P_kernel, p_levels, uf, lo, hi, tau, GAMMA,
            max_iter=40, h_p_factor=1.5)
        t_iter = time.time() - t0
        final = hist[-1]
        tail_def = float(np.mean([h['deficit'] for h in hist[-min(10, len(hist)):]]))
        tail_viol = float(np.mean([h['max_viol_raw'] for h in hist[-min(10, len(hist)):]]))
        print(f"  status={status} (cycle={cycle}), kernel_wall={t_k:.1f}s, "
              f"iter_wall={t_iter:.1f}s", flush=True)
        print(f"  kernel deficit={kernel_def:.4f}, kernel max_viol={kernel_viol:.3e}",
              flush=True)
        print(f"  final deficit={final['deficit']:.4f}, tail mean={tail_def:.4f}",
              flush=True)
        print(f"  RAW max violation (final iter)={final['max_viol_raw']:.3e}, "
              f"tail mean={tail_viol:.3e}", flush=True)
        rows.append(dict(
            tau=tau, status=status, cycle=int(cycle),
            kernel_deficit=float(kernel_def),
            kernel_max_viol=float(kernel_viol),
            deficit_final=float(final['deficit']),
            deficit_tail=tail_def,
            max_viol_raw_final=float(final['max_viol_raw']),
            max_viol_raw_tail=tail_viol,
            n_viol_raw_final=int(final['n_viol_raw']),
            max_res=float(final['max_res']),
            kernel_wall=float(t_k), iter_wall=float(t_iter),
        ))
        fields[tau] = (p_field, uf, lo, hi)
        json.dump(dict(rows=rows, G=Gi, M=M, gamma=GAMMA),
                  open(f"{OUT}/ladder.json", 'w'), indent=2, default=str)
    # Plots
    taus_arr = np.array([r['tau'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # 1) deficit
    axes[0].plot(taus_arr, [r['deficit_tail'] for r in rows], 'o-', label='discrete (tail)')
    axes[0].plot(taus_arr, [r['kernel_deficit'] for r in rows], 's--', label='kernel-h>0', alpha=0.6)
    axes[0].set_xlabel(r'$\tau$'); axes[0].set_ylabel('deficit $1-R^2$')
    axes[0].set_title(rf'Deficit vs $\tau$  (G={Gi}, M={M})')
    axes[0].grid(alpha=0.3); axes[0].legend()
    # 2) RAW monotonicity violation
    axes[1].plot(taus_arr, [r['max_viol_raw_tail'] for r in rows], 'o-', label='discrete RAW (tail)')
    axes[1].plot(taus_arr, [r['kernel_max_viol'] for r in rows], 's--', label='kernel', alpha=0.6)
    axes[1].axhline(0.5*(0.95-0.05)/(M-1), ls=':', color='gray',
                     label=r'$\Delta p / 2$ (grid floor)')
    axes[1].set_xlabel(r'$\tau$'); axes[1].set_ylabel('max RAW non-monotone step')
    axes[1].set_yscale('log'); axes[1].set_title('RAW monotonicity violation')
    axes[1].grid(alpha=0.3, which='both'); axes[1].legend(fontsize=8)
    # 3) status per tau
    status_colors = {'monotone_converged': 'tab:green',
                     'fixed_but_nonmonotone': 'tab:orange',
                     'cycle': 'tab:red', 'not_converged': 'tab:purple'}
    for r in rows:
        axes[2].scatter([r['tau']], [0], c=status_colors.get(r['status'], 'gray'),
                         s=200, edgecolor='black', linewidth=0.5)
        axes[2].annotate(r['status'].replace('_', '\n'),
                          (r['tau'], 0), ha='center', va='bottom', fontsize=7,
                          rotation=45)
    axes[2].set_xlabel(r'$\tau$'); axes[2].set_yticks([])
    axes[2].set_ylim(-0.2, 1.5)
    axes[2].set_title('Convergence status per $\\tau$')
    axes[2].grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f"{OUT}/ladder.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {OUT}/ladder.png", flush=True)
    # Slice gallery
    n_tau = len(taus)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    for k, tau in enumerate(taus):
        p_field, uf, lo, hi = fields[tau]
        u_in = uf[lo:hi]
        k0 = int(np.argmin(np.abs(u_in)))
        r, c = divmod(k, 5)
        slice_panel(axes[r, c], p_field, uf, lo, hi, k0, rf'$\tau={tau}$')
    plt.suptitle(rf'Monotone-projected SOFT slice $P(u_1, u_2 \mid u_3=0)$, $G={Gi}$, $M={M}$',
                 y=1.0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/slices.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUT}/slices.png", flush=True)


if __name__ == "__main__":
    main()
