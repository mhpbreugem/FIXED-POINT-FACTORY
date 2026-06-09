"""V2 of the (L_priv, L_pub) test. Three improvements:

1. Optimal 2-way ANOVA additive decomposition (minimizes ||delta||_2):
       L(p, u_k) = mu + a(u_k) + b(p) + delta(p, u_k)
       a, b sum-to-zero constraints.
   This gives the BEST additive separation, not the textbook-Gaussian one.

2. Optionally use a richer L_priv: fit a(u_k) to a Cheby polynomial in u_k.
   In the linear-Gaussian limit a(u_k) = tau * u_k; at higher tau it's
   nonlinear.

3. Decay profile uses the SUM of squared coefs above order n (cumulative
   tail norm) — a cleaner monotone measure than the per-anti-diagonal max.

Plus: test whether the residual delta has 'fast-decaying' or 'slow-decaying'
Cheby coefficients — that's the actual hypothesis to verify.
"""
import os, sys, time, json
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.stats import norm
import matplotlib.pyplot as plt

REPO = "/home/user/FIXED-POINT-FACTORY"
DATA = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight"
OUT = f"{DATA}/lpub_lpriv_v2"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/figs", exist_ok=True)


def make_cdf_uniform_grid(G, eps_q=0.01):
    qs = np.linspace(eps_q, 1 - eps_q, G)
    return norm.ppf(qs)


def make_p_grid(Gp, eps=1e-3):
    lo, hi = np.log(eps/(1-eps)), np.log((1-eps)/eps)
    return 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, Gp)))


def logit(p):
    return np.log(p / (1 - p))


def anova_2way(L):
    """Optimal additive decomposition L[i,j] = mu + a[i] + b[j] + delta[i,j]
    with sum(a)=sum(b)=0.  Returns (mu, a, b, delta).
    Here L has shape (N_row, N_col); we use rows=p, cols=u_k."""
    mu = float(L.mean())
    a = L.mean(axis=1) - mu      # row effect (p effect = L_pub)
    b = L.mean(axis=0) - mu      # col effect (u_k effect = L_priv)
    delta = L - mu - a[:, None] - b[None, :]
    return mu, a, b, delta


def fit_cheby_2d_lsq(F, xi_row, xi_col, deg_row, deg_col):
    Vr = cheb.chebvander(xi_row, deg_row)
    Vc = cheb.chebvander(xi_col, deg_col)
    coefs = np.linalg.pinv(Vr) @ F @ np.linalg.pinv(Vc).T
    return coefs


def tail_norm_profile(coefs):
    """Return cumulative tail-norm: tail[n] = sqrt(sum_{i+j >= n} coefs[i,j]^2).
    Monotone decreasing in n.  Faster decay = better representation."""
    nx, ny = coefs.shape
    abs2 = coefs**2
    tail = np.zeros(nx + ny)
    for n in range(nx + ny):
        s = 0.0
        for i in range(nx):
            for j in range(ny):
                if i + j >= n: s += abs2[i, j]
        tail[n] = np.sqrt(s)
    return tail


def fit_cheby_h_priv(L, p_grid, u_grid, deg_h=6):
    """Find the best Cheby polynomial h(u_k) such that
       L(p, u_k) ~ h(u_k) + b_h(p)  with smallest residual.
    Solve by alternating: for fixed h, b_h(p) = mean_uk(L - h);
    for fixed b_h, h(u_k) = (LSQ Cheby fit) of mean_p(L - b_h) onto Cheby basis.
    """
    G = u_grid.size
    Gp = p_grid.size
    U_MAX = abs(u_grid).max()
    xi_u = u_grid / U_MAX
    Vu = cheb.chebvander(xi_u, deg_h)  # (G, deg_h+1)
    # Init: h(u_k) = tau * u_k (caller controls; here we use 0 to start)
    h = np.zeros(G)
    b_h = np.zeros(Gp)
    for it in range(30):
        b_h = (L - h[None, :]).mean(axis=1)
        target = (L - b_h[:, None]).mean(axis=0)  # shape (G,)
        coefs = np.linalg.pinv(Vu) @ target
        h_new = Vu @ coefs
        d = float(np.max(np.abs(h_new - h)))
        h = h_new
        if d < 1e-12: break
    delta = L - h[None, :] - b_h[:, None]
    return h, b_h, delta, coefs


def test_cell(gamma, tau, deg_p=12, deg_u=8):
    print(f"\n=== gamma={gamma}, tau={tau} ===", flush=True)
    fp_file = f"{DATA}/dd_k3_strict_fp_g{gamma:g}_t{tau:.4f}.npz"
    if not os.path.exists(fp_file):
        print(f"  no FP file at {fp_file}", flush=True); return None
    d = np.load(fp_file)
    mu = d["mu_strict"].astype(np.float64)
    Gp_act, G_act = mu.shape
    u_grid = make_cdf_uniform_grid(G_act)
    p_grid = make_p_grid(Gp_act)
    eps = 1e-9
    L = logit(np.clip(mu, eps, 1 - eps))
    L_max = float(np.max(np.abs(L)))
    L_rms = float(np.sqrt(np.mean(L**2)))
    print(f"  |L|_inf = {L_max:.3f}, |L|_rms = {L_rms:.3f}", flush=True)

    # --- ANOVA decomposition ---
    mu_L, a_p_anova, b_uk_anova, delta_anova = anova_2way(L)
    print(f"  ANOVA: |a_p|_inf={np.max(np.abs(a_p_anova)):.3f}, "
          f"|b_uk|_inf={np.max(np.abs(b_uk_anova)):.3f}, "
          f"|delta|_inf={np.max(np.abs(delta_anova)):.3f}, "
          f"|delta|_rms={np.sqrt(np.mean(delta_anova**2)):.3f}",
          flush=True)

    # --- Cheby-h(u_k) decomposition (smoother L_priv) ---
    h_priv, b_pub, delta_cheby_h, h_coefs = fit_cheby_h_priv(L, p_grid, u_grid, deg_h=8)
    print(f"  Cheby-h: |h|_inf={np.max(np.abs(h_priv)):.3f}, "
          f"|b_pub|_inf={np.max(np.abs(b_pub)):.3f}, "
          f"|delta|_inf={np.max(np.abs(delta_cheby_h)):.3f}, "
          f"|delta|_rms={np.sqrt(np.mean(delta_cheby_h**2)):.3f}",
          flush=True)

    # --- Cheby decomposition: 2D LSQ tail norms ---
    U_MAX = abs(u_grid).max()
    xi_u = u_grid / U_MAX
    Lp = logit(np.clip(p_grid, eps, 1 - eps))
    LP_MAX = abs(Lp).max()
    xi_p = Lp / LP_MAX

    coefs_L = fit_cheby_2d_lsq(L, xi_p, xi_u, deg_p, deg_u)
    coefs_delta_anova = fit_cheby_2d_lsq(delta_anova, xi_p, xi_u, deg_p, deg_u)
    coefs_delta_cheby_h = fit_cheby_2d_lsq(delta_cheby_h, xi_p, xi_u, deg_p, deg_u)

    tail_L = tail_norm_profile(coefs_L)
    tail_d_anova = tail_norm_profile(coefs_delta_anova)
    tail_d_chebyh = tail_norm_profile(coefs_delta_cheby_h)

    # Reconstruction error at given truncation level
    def trunc_err(coefs, F, max_order):
        nx, ny = coefs.shape
        mask = np.zeros_like(coefs)
        for i in range(nx):
            for j in range(ny):
                if i + j <= max_order: mask[i, j] = 1
        coefs_trunc = coefs * mask
        Vr = cheb.chebvander(xi_p, deg_p)
        Vc = cheb.chebvander(xi_u, deg_u)
        return float(np.max(np.abs(Vr @ coefs_trunc @ Vc.T - F)))

    err_table = []
    for max_order in [2, 4, 6, 8, 10, 12]:
        e_L = trunc_err(coefs_L, L, max_order)
        e_d = trunc_err(coefs_delta_anova, delta_anova, max_order)
        # For delta_anova, reconstruction of L needs adding back the additive parts
        # err on L = ||L - (mu_L + a_p + b_uk + delta_trunc)||
        Vr = cheb.chebvander(xi_p, deg_p)
        Vc = cheb.chebvander(xi_u, deg_u)
        mask = np.zeros_like(coefs_delta_anova)
        for i in range(deg_p+1):
            for j in range(deg_u+1):
                if i + j <= max_order: mask[i, j] = 1
        delta_trunc = Vr @ (coefs_delta_anova * mask) @ Vc.T
        L_rec_anova = mu_L + a_p_anova[:, None] + b_uk_anova[None, :] + delta_trunc
        e_L_via_anova = float(np.max(np.abs(L_rec_anova - L)))
        err_table.append((max_order, e_L, e_L_via_anova))
        print(f"  trunc max_order={max_order}: "
              f"|L - L_cheby|={e_L:.2e},  |L - (anova+delta_cheby)|={e_L_via_anova:.2e}",
              flush=True)

    return dict(
        gamma=gamma, tau=tau, G=G_act, G_p=Gp_act,
        L_max=L_max, L_rms=L_rms,
        delta_anova_max=float(np.max(np.abs(delta_anova))),
        delta_anova_rms=float(np.sqrt(np.mean(delta_anova**2))),
        delta_cheby_h_max=float(np.max(np.abs(delta_cheby_h))),
        delta_cheby_h_rms=float(np.sqrt(np.mean(delta_cheby_h**2))),
        a_p_max=float(np.max(np.abs(a_p_anova))),
        b_uk_max=float(np.max(np.abs(b_uk_anova))),
        h_max=float(np.max(np.abs(h_priv))),
        tail_L=tail_L.tolist(),
        tail_d_anova=tail_d_anova.tolist(),
        tail_d_chebyh=tail_d_chebyh.tolist(),
        err_table=err_table,
        L=L, delta_anova=delta_anova, delta_cheby_h=delta_cheby_h,
        a_p_anova=a_p_anova, b_uk_anova=b_uk_anova,
        h_priv=h_priv, b_pub=b_pub, p_grid=p_grid, u_grid=u_grid,
        h_coefs=h_coefs,
    )


def plot_results(results):
    keys = list(results.keys())
    nk = len(keys)

    # Fig 1: cumulative tail norm decay
    fig, axes = plt.subplots(1, nk, figsize=(5*nk, 4))
    if nk == 1: axes = [axes]
    for ax, key in zip(axes, keys):
        r = results[key]
        ax.semilogy(r["tail_L"], 'b-o', label=r"L direct", ms=3)
        ax.semilogy(r["tail_d_anova"], 'r-s', label=r"$\delta$ (ANOVA-additive)", ms=3)
        ax.semilogy(r["tail_d_chebyh"], 'g-^', label=r"$\delta$ (Cheby-h)", ms=3)
        ax.set_xlabel("min order N")
        ax.set_ylabel("$\\|coefs\\|_2$ tail beyond order N")
        ax.set_title(f"$\\gamma={r['gamma']}, \\tau={r['tau']}$")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/fig1_tail_decay.png", dpi=120)
    plt.close()

    # Fig 2: heatmaps L vs delta_anova
    fig, axes = plt.subplots(nk, 3, figsize=(15, 4*nk))
    if nk == 1: axes = axes[None, :]
    for row, key in enumerate(keys):
        r = results[key]
        for ax, F, title in zip(
            axes[row, :],
            [r["L"], r["delta_anova"], r["delta_cheby_h"]],
            [f"L  (max={r['L_max']:.2f})",
             f"$\\delta_{{ANOVA}}$  (max={r['delta_anova_max']:.2f})",
             f"$\\delta_{{Cheby-h}}$  (max={r['delta_cheby_h_max']:.2f})"]):
            v = max(abs(F.min()), abs(F.max()))
            im = ax.imshow(F, aspect="auto", origin="lower",
                            extent=[r["u_grid"][0], r["u_grid"][-1],
                                    r["p_grid"][0], r["p_grid"][-1]],
                            cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"$\\gamma={r['gamma']}, \\tau={r['tau']}$: {title}")
            ax.set_xlabel("$u_k$"); ax.set_ylabel("p")
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/fig2_heat.png", dpi=120)
    plt.close()

    # Fig 3: 1D L_pub-like and L_priv-like functions
    fig, axes = plt.subplots(nk, 2, figsize=(10, 4*nk))
    if nk == 1: axes = axes[None, :]
    for row, key in enumerate(keys):
        r = results[key]
        axes[row, 0].plot(r["p_grid"], r["a_p_anova"], 'r-', label="$a(p)$ ANOVA")
        axes[row, 0].plot(r["p_grid"], r["b_pub"], 'g--', label="$b_{pub}(p)$ Cheby-h")
        axes[row, 0].axhline(0, color='k', lw=0.5)
        axes[row, 0].set_xlabel("p"); axes[row, 0].set_ylabel("p-effect")
        axes[row, 0].set_title(f"$\\gamma={r['gamma']}, \\tau={r['tau']}$: row-effect")
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(alpha=0.3)

        axes[row, 1].plot(r["u_grid"], r["b_uk_anova"], 'r-', label="$b(u_k)$ ANOVA")
        axes[row, 1].plot(r["u_grid"], r["h_priv"], 'g--', label="$h(u_k)$ Cheby-h (deg 8)")
        axes[row, 1].plot(r["u_grid"], r["tau"]*r["u_grid"], 'k:',
                            label=f"$\\tau u_k$ (closed form, $\\tau={r['tau']}$)")
        axes[row, 1].axhline(0, color='k', lw=0.5)
        axes[row, 1].set_xlabel("$u_k$"); axes[row, 1].set_ylabel("$u_k$-effect")
        axes[row, 1].set_title("col-effect")
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/fig3_effects.png", dpi=120)
    plt.close()


def main():
    cells = [(100, 0.2), (100, 1.0), (10, 1.0), (1000, 1.0)]
    results = {}; summary = {}
    for g, t in cells:
        key = f"g{g}_t{t:.4f}"
        try:
            r = test_cell(g, t)
            if r is None: continue
            results[key] = r
            summary[key] = {k: v for k, v in r.items()
                              if not isinstance(v, np.ndarray)}
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAIL: {e}", flush=True)
    plot_results(results)
    json.dump(summary, open(f"{OUT}/results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
