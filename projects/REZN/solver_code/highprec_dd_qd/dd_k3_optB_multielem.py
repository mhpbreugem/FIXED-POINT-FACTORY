"""Multi-element Option B+: use critical p_c values as forced PCHIP knots.

Hypothesis: the existing OptB+ floors at max-error ~0.4 around the cusps
because its PCHIP knots = sorted cube-cell P values miss the EXACT critical
p_c locations. Inserting the critical p_c values (computed from the
critical-point detector) as additional knots aligns the PCHIP's natural
kink-handling with the true cusp locations.
"""
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
from lin_cdf_pchip import make_cdf_uniform_grid

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/optB_multielem"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def gauss_w(u_grid, mean, tau):
    """f_v(u) = N(u; v-0.5, tau^-1). v determines mean."""
    sd = 1.0 / np.sqrt(tau)
    return norm.pdf(u_grid, loc=mean, scale=sd)


def trap_w(u_grid):
    """Trapezoidal weights on a 1D grid."""
    n = u_grid.size
    w = np.zeros(n)
    if n < 2: return w + 1.0
    for i in range(n):
        if i == 0: w[i] = 0.5 * (u_grid[1] - u_grid[0])
        elif i == n - 1: w[i] = 0.5 * (u_grid[-1] - u_grid[-2])
        else: w[i] = 0.5 * (u_grid[i+1] - u_grid[i-1])
    return w


def make_p_grid(Gp, eps=1e-3):
    lo, hi = np.log(eps / (1 - eps)), np.log((1 - eps) / eps)
    return 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, Gp)))


def build_mu_global_pchip(P_vals, u_grid, p_grid, tau):
    """Plain OptB+ baseline: PCHIP through sorted cube-cell P values only."""
    G = u_grid.size; G_p = p_grid.size
    w = trap_w(u_grid)
    f0 = gauss_w(u_grid, -0.5, tau)
    f1 = gauss_w(u_grid, +0.5, tau)
    mu = np.empty((G_p, G))
    for k_node in range(G):
        P_arr, w_arr, f0_arr, f1_arr = [], [], [], []
        for i in range(G):
            for j in range(G):
                P_arr.append(P_vals[k_node, i, j])
                w_arr.append(w[i] * w[j])
                f0_arr.append(f0[i] * f0[j])
                f1_arr.append(f1[i] * f1[j])
        P_arr = np.array(P_arr); w_arr = np.array(w_arr)
        f0_arr = np.array(f0_arr); f1_arr = np.array(f1_arr)
        order = np.argsort(P_arr)
        P_sorted = P_arr[order]
        m0_sorted = (w_arr * f0_arr)[order]
        m1_sorted = (w_arr * f1_arr)[order]
        cum0 = np.cumsum(m0_sorted); cum1 = np.cumsum(m1_sorted)
        # Dedup ties
        Pu, F0u, F1u = [P_sorted[0]], [cum0[0]], [cum1[0]]
        for s in range(1, len(P_sorted)):
            if P_sorted[s] > Pu[-1] + 1e-15:
                Pu.append(P_sorted[s]); F0u.append(cum0[s]); F1u.append(cum1[s])
            else:
                F0u[-1] = cum0[s]; F1u[-1] = cum1[s]
        if len(Pu) < 4:
            mu[:, k_node] = 0.5
            continue
        Pu = np.array(Pu); F0u = np.array(F0u); F1u = np.array(F1u)
        # PCHIP density (derivative of cumulative mass)
        try:
            pchip0 = PchipInterpolator(Pu, F0u, extrapolate=True)
            pchip1 = PchipInterpolator(Pu, F1u, extrapolate=True)
            d0 = pchip0.derivative()(p_grid)
            d1 = pchip1.derivative()(p_grid)
            f0k, f1k = f0[k_node], f1[k_node]
            den = f0k * np.clip(d0, 0, None) + f1k * np.clip(d1, 0, None)
            m_vals = np.where(den > 1e-300, f1k * np.clip(d1, 0, None) / den, 0.5)
            mu[:, k_node] = np.clip(m_vals, 1e-9, 1 - 1e-9)
        except Exception:
            mu[:, k_node] = 0.5
    return mu


def compute_cube_cdf_at_pcs(P_vals, w_arr_2d, f_arr_2d, p_cs, k_node):
    """Cumulative cube-cell mass at p = p_c, slice u_k = k_node.
    Each cube cell contributes its full mass if P_cell <= p_c.
    P_cell taken as the vertex value P_vals[k_node, i, j] (no per-cell
    interpolation). The augmented knot set should produce a PCHIP that's
    pinned at the cusp p_c values.
    """
    cums = np.zeros_like(p_cs)
    P_slice = P_vals[k_node, :, :].ravel()
    masses = w_arr_2d.ravel() * f_arr_2d.ravel()
    for i, pc in enumerate(p_cs):
        mask = P_slice <= pc
        cums[i] = float(np.sum(masses[mask]))
    return cums


def build_mu_multielem_pchip(P_vals, u_grid, p_grid, tau, p_cs):
    """Multi-element OptB+: insert critical p_c values as forced PCHIP knots."""
    G = u_grid.size; G_p = p_grid.size
    w = trap_w(u_grid)
    f0 = gauss_w(u_grid, -0.5, tau)
    f1 = gauss_w(u_grid, +0.5, tau)
    w_2d = np.outer(w, w)
    f0_2d = np.outer(f0, f0)
    f1_2d = np.outer(f1, f1)
    mu = np.empty((G_p, G))
    p_cs_in = np.sort(p_cs[(p_cs > 0) & (p_cs < 1)])
    for k_node in range(G):
        P_arr = P_vals[k_node, :, :].ravel()
        m0_arr = (w_2d * f0_2d).ravel()
        m1_arr = (w_2d * f1_2d).ravel()
        order = np.argsort(P_arr)
        P_sorted = P_arr[order]
        cum0 = np.cumsum(m0_arr[order])
        cum1 = np.cumsum(m1_arr[order])
        # Dedup
        Pu = [P_sorted[0]]; F0u = [cum0[0]]; F1u = [cum1[0]]
        for s in range(1, len(P_sorted)):
            if P_sorted[s] > Pu[-1] + 1e-15:
                Pu.append(P_sorted[s]); F0u.append(cum0[s]); F1u.append(cum1[s])
            else:
                F0u[-1] = cum0[s]; F1u[-1] = cum1[s]
        # Insert critical p_c knots: for each p_c, compute cube CDF at that point
        F0_at_pc = compute_cube_cdf_at_pcs(P_vals, w_2d, f0_2d, p_cs_in, k_node)
        F1_at_pc = compute_cube_cdf_at_pcs(P_vals, w_2d, f1_2d, p_cs_in, k_node)
        # Merge with existing knots
        all_P = np.concatenate([Pu, p_cs_in])
        all_F0 = np.concatenate([F0u, F0_at_pc])
        all_F1 = np.concatenate([F1u, F1_at_pc])
        order2 = np.argsort(all_P)
        all_P = all_P[order2]; all_F0 = all_F0[order2]; all_F1 = all_F1[order2]
        # Dedup again
        Pn = [all_P[0]]; F0n = [all_F0[0]]; F1n = [all_F1[0]]
        for s in range(1, len(all_P)):
            if all_P[s] > Pn[-1] + 1e-12:
                Pn.append(all_P[s]); F0n.append(all_F0[s]); F1n.append(all_F1[s])
            else:
                # Take maximum (CDF should be non-decreasing)
                F0n[-1] = max(F0n[-1], all_F0[s])
                F1n[-1] = max(F1n[-1], all_F1[s])
        Pn = np.array(Pn); F0n = np.array(F0n); F1n = np.array(F1n)
        if len(Pn) < 4:
            mu[:, k_node] = 0.5; continue
        # PCHIP density (derivative)
        try:
            pchip0 = PchipInterpolator(Pn, F0n, extrapolate=True)
            pchip1 = PchipInterpolator(Pn, F1n, extrapolate=True)
            d0 = pchip0.derivative()(p_grid)
            d1 = pchip1.derivative()(p_grid)
            f0k, f1k = f0[k_node], f1[k_node]
            den = f0k * np.clip(d0, 0, None) + f1k * np.clip(d1, 0, None)
            m_vals = np.where(den > 1e-300, f1k * np.clip(d1, 0, None) / den, 0.5)
            mu[:, k_node] = np.clip(m_vals, 1e-9, 1 - 1e-9)
        except Exception:
            mu[:, k_node] = 0.5
    return mu


def test_cell(gamma, tau):
    print(f"\n=== gamma={gamma}, tau={tau} ===", flush=True)
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    P = fp["P_strict"].astype(np.float64)
    mu_strict = fp["mu_strict"].astype(np.float64)
    G = P.shape[0]; G_p = mu_strict.shape[0]
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)

    # Critical p_c
    cp = json.load(open(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/critpts/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.json"))
    p_cs = np.array(sorted(cp["p_c_values"]))
    print(f"  {len(cp['points'])} critical points, {len(p_cs)} unique p_c values")

    # Build mu with global PCHIP (baseline)
    t0 = time.time()
    mu_global = build_mu_global_pchip(P, u_grid, p_grid, tau)
    t_global = time.time() - t0
    err_global = np.abs(mu_global - mu_strict)
    print(f"  global PCHIP:    max={np.max(err_global):.3e}, "
          f"median={np.median(err_global):.3e}, RMS={np.sqrt(np.mean(err_global**2)):.3e}, "
          f"({t_global:.1f}s)")

    # Build mu with multi-element (forced knots at p_c)
    t0 = time.time()
    mu_multi = build_mu_multielem_pchip(P, u_grid, p_grid, tau, p_cs)
    t_multi = time.time() - t0
    err_multi = np.abs(mu_multi - mu_strict)
    print(f"  multi-elem PCHIP: max={np.max(err_multi):.3e}, "
          f"median={np.median(err_multi):.3e}, RMS={np.sqrt(np.mean(err_multi**2)):.3e}, "
          f"({t_multi:.1f}s)")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # max-err per p
    em_g = np.max(np.abs(mu_global - mu_strict), axis=1)
    em_m = np.max(np.abs(mu_multi - mu_strict), axis=1)
    axes[0, 0].semilogy(p_grid, em_g, 'b-', label='global PCHIP', alpha=0.7)
    axes[0, 0].semilogy(p_grid, em_m, 'r-', label='multi-element PCHIP', alpha=0.7)
    for pc in p_cs:
        axes[0, 0].axvline(pc, color='gray', alpha=0.2, lw=0.5)
    axes[0, 0].set_xlabel('p'); axes[0, 0].set_ylabel('max error vs strict')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3, which='both')
    axes[0, 0].set_title(f'$\\gamma={gamma}, \\tau={tau}$: error vs p')

    # mu(p, u_k=0)
    k0 = G // 2
    axes[0, 1].plot(p_grid, mu_strict[:, k0], 'k-', label='strict', lw=2)
    axes[0, 1].plot(p_grid, mu_global[:, k0], 'b--', label='global', alpha=0.7)
    axes[0, 1].plot(p_grid, mu_multi[:, k0], 'r:', label='multi-elem', alpha=0.7)
    axes[0, 1].set_xlabel('p'); axes[0, 1].set_ylabel(f'$\\mu(p, u_k=0)$')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_title(f'$\\mu(p, u_k=0)$')

    # Histogram of error reduction
    axes[1, 0].hist(np.log10(np.abs(mu_global.ravel() - mu_strict.ravel()) + 1e-20),
                     bins=50, alpha=0.6, label='global', color='blue')
    axes[1, 0].hist(np.log10(np.abs(mu_multi.ravel() - mu_strict.ravel()) + 1e-20),
                     bins=50, alpha=0.6, label='multi-elem', color='red')
    axes[1, 0].set_xlabel('$\\log_{10}$ |error|')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('error distribution')

    # error heatmap difference (where multi-elem helps)
    diff = np.abs(mu_global - mu_strict) - np.abs(mu_multi - mu_strict)
    im = axes[1, 1].imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                              extent=[u_grid[0], u_grid[-1], p_grid[0], p_grid[-1]])
    axes[1, 1].set_xlabel('$u_k$'); axes[1, 1].set_ylabel('p')
    axes[1, 1].set_title('improvement (red = multi-elem better)')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/g{gamma}_t{tau:.1f}.png", dpi=120)
    plt.close()

    return dict(
        gamma=gamma, tau=tau,
        n_critpts=len(cp["points"]),
        n_pc_unique=len(p_cs),
        global_max=float(np.max(err_global)),
        global_median=float(np.median(err_global)),
        global_rms=float(np.sqrt(np.mean(err_global**2))),
        multi_max=float(np.max(err_multi)),
        multi_median=float(np.median(err_multi)),
        multi_rms=float(np.sqrt(np.mean(err_multi**2))),
        improvement_max=float(np.max(err_global) / max(np.max(err_multi), 1e-30)),
        improvement_rms=float(np.sqrt(np.mean(err_global**2)) / max(np.sqrt(np.mean(err_multi**2)), 1e-30)),
    )


def main():
    cells = [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]
    results = []
    for g, t in cells:
        try:
            r = test_cell(g, t)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL: {e}", flush=True)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2, default=str)
    # Summary table
    print("\n" + "="*100)
    print(f"{'cell':>15} {'n_pc':>5} {'global_max':>12} {'multi_max':>12} {'imp_max':>8} {'global_rms':>12} {'multi_rms':>12} {'imp_rms':>8}")
    for r in results:
        print(f"g{r['gamma']:>4g}_t{r['tau']:.2f}".rjust(15) +
              f" {r['n_pc_unique']:>5d} {r['global_max']:>12.3e} {r['multi_max']:>12.3e} "
              f"{r['improvement_max']:>8.2f}x {r['global_rms']:>12.3e} {r['multi_rms']:>12.3e} "
              f"{r['improvement_rms']:>8.2f}x")
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
