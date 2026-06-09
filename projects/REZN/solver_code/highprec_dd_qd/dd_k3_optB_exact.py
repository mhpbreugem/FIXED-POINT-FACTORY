"""Option B+ EXACT cube-cell mass integration for the cumulative CDF.

The baseline OptB+ computes F_v(p) by summing Gaussian mass of cube cells whose
*vertex* P value is <= p. This treats each cell's mass as a single point at the
vertex P, producing a step CDF that PCHIP smooths globally. The within-cell
sensitivity is lost.

This module replaces vertex-mass by EXACT per-cell mass: for each (u_a, u_b)
cell with bilinear interpolant P_hat(s,t), we densely sample (s,t) on an
NQK x NQK Gauss-Legendre tensor grid inside the cell, and treat each GL node
as a "micro-cell" contributing
    w_GL * |du_a/ds| * |du_b/dt| * f_v(u_a(s)) * f_v(u_b(t))
of Gaussian mass with bilinear-interpolated P value P_hat(s,t). The empirical
CDF then has  (G-1)^2 * NQK^2  knots per slice rather than  G^2.

For NQK=8 we get 64 micro-cells per cube cell -> 64*(G-1)**2 = 6400 knots for
G=11, vs 121 knots for vertex-only. Within-cell variations of P are resolved
to GL accuracy. The resulting empirical CDF, fed through PCHIP-derivative as
usual, gives a much smoother and more accurate density a_v(p, u_k).

Two interfaces are provided:
  build_mu_optB_exact: numba builder used by phi_optB_exact (parallel)
  phi_optB_exact     : full Phi(P) under exact-mass mu-table.

Test loop (under __main__) loads 4 strict FPs (g in {100,1000} x tau in
{0.2,1.0}), builds mu-tables via both step (OB.build_mu_optB) and exact
methods, and computes max/median/RMS errors vs mu_strict.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from numba import njit, prange
from cheby_numba import f_signal_jit, crra_clear_jit
from dd_k3_optB_numba import _pchip_slopes, _pchip_deriv_eval, _interp_mu


# ----------------------- Per-cell GL helper builders -----------------------
def make_gl_unit(NQK):
    """Gauss-Legendre nodes / weights on the unit interval [0,1]."""
    n_xi, w_xi = np.polynomial.legendre.leggauss(NQK)
    s = 0.5 * (n_xi + 1.0)          # in [0,1]
    w = 0.5 * w_xi                  # sum to 1
    return s.astype(np.float64), w.astype(np.float64)


# ----------------------- Exact-mass mu-table builder -----------------------
@njit(cache=True)
def build_mu_optB_exact(P_vals, u_grid, p_grid, tau,
                            gl_s, gl_w, f0_u, f1_u, mu_table):
    """Exact-cell-mass version of build_mu_optB.

    For each k_node (slice), iterate over (G-1)^2 cells. For each cell, evaluate
    bilinear P_hat at NQK x NQK GL nodes (s,t) in [0,1]^2. The micro-cell mass
    contribution is
        w_s * w_t * du_a_cell * du_b_cell * f_v(u_a(s)) * f_v(u_b(t))
    where du_a_cell = u_grid[ia+1]-u_grid[ia].

    All micro-cell (P_hat, mass) pairs across cells are then sorted by P_hat
    to build the cumulative empirical CDF F_v(p), PCHIP-smoothed, and
    differentiated for density a_v(p, u_k).
    """
    G = u_grid.size; G_p = p_grid.size; NQK = gl_s.size
    n_cells = (G - 1) * (G - 1)
    NPC = NQK * NQK                  # micro-cells per cube cell
    NTOT = n_cells * NPC

    # Pre-evaluate f_v at all GL nodes inside each interval, keyed by
    # (cell_ia, q_s) and (cell_ib, q_t)
    # f0a_node[ia, qs] = f_signal(u_grid[ia] + s_q * (u_grid[ia+1]-u_grid[ia]), 0)
    n_int = G - 1
    f0a_node = np.empty((n_int, NQK))
    f1a_node = np.empty((n_int, NQK))
    du_cell = np.empty(n_int)
    for ia in range(n_int):
        du = u_grid[ia + 1] - u_grid[ia]
        du_cell[ia] = du
        for qs in range(NQK):
            ua = u_grid[ia] + gl_s[qs] * du
            f0a_node[ia, qs] = f_signal_jit(ua, 0, tau)
            f1a_node[ia, qs] = f_signal_jit(ua, 1, tau)

    # Buffers for micro-cells
    P_arr = np.empty(NTOT)
    f0_arr = np.empty(NTOT)
    f1_arr = np.empty(NTOT)
    Pu = np.empty(NTOT + 2)
    F0u = np.empty(NTOT + 2)
    F1u = np.empty(NTOT + 2)
    m0 = np.empty(NTOT + 2)
    m1 = np.empty(NTOT + 2)

    for k_node in range(G):
        s = 0
        for ia in range(n_int):
            dua = du_cell[ia]
            for ib in range(n_int):
                dub = du_cell[ib]
                P_00 = P_vals[k_node, ia,     ib    ]
                P_10 = P_vals[k_node, ia + 1, ib    ]
                P_01 = P_vals[k_node, ia,     ib + 1]
                P_11 = P_vals[k_node, ia + 1, ib + 1]
                area = dua * dub
                for qs in range(NQK):
                    s_q = gl_s[qs]; ws_q = gl_w[qs]
                    f0a = f0a_node[ia, qs]; f1a = f1a_node[ia, qs]
                    for qt in range(NQK):
                        t_q = gl_s[qt]; wt_q = gl_w[qt]
                        f0b = f0a_node[ib, qt]; f1b = f1a_node[ib, qt]
                        # Bilinear P_hat(s,t)
                        Pst = ((1.0 - s_q) * (1.0 - t_q) * P_00
                                + s_q       * (1.0 - t_q) * P_10
                                + (1.0 - s_q) * t_q       * P_01
                                + s_q       * t_q       * P_11)
                        w = ws_q * wt_q * area
                        P_arr[s]  = Pst
                        f0_arr[s] = w * f0a * f0b
                        f1_arr[s] = w * f1a * f1b
                        s += 1
        # Sort micro-cells by P.
        # We DO NOT pad with explicit BCs at P=0,1 -- the empirical CDF lives on
        # [min(P_arr), max(P_arr)] and outside that range the PCHIP derivative
        # eval returns 0 (a_v=0), which falls back to mu=0.5. This matches the
        # OptB step-summation builder so the comparison is apples-to-apples on
        # the in-support region.
        order = np.argsort(P_arr)
        n_u = 0; cum0 = 0.0; cum1 = 0.0
        prev_P = P_arr[order[0]] - 1.0
        for s in range(NTOT):
            ii = order[s]
            cum0 += f0_arr[ii]
            cum1 += f1_arr[ii]
            P_cur = P_arr[ii]
            if P_cur > prev_P + 1e-15:
                Pu[n_u] = P_cur; F0u[n_u] = cum0; F1u[n_u] = cum1
                n_u += 1
                prev_P = P_cur
            else:
                F0u[n_u - 1] = cum0; F1u[n_u - 1] = cum1

        if n_u < 4:
            for ip in range(G_p): mu_table[ip, k_node] = 0.5
            continue
        _pchip_slopes(Pu, F0u, n_u, m0)
        _pchip_slopes(Pu, F1u, n_u, m1)
        f0k = f0_u[k_node]; f1k = f1_u[k_node]
        for ip in range(G_p):
            p = p_grid[ip]
            a0 = _pchip_deriv_eval(Pu, F0u, m0, n_u, p)
            a1 = _pchip_deriv_eval(Pu, F1u, m1, n_u, p)
            den = f0k * a0 + f1k * a1
            if den > 1e-300:
                m = f1k * a1 / den
                if m < 1e-9: m = 1e-9
                elif m > 1.0 - 1e-9: m = 1.0 - 1e-9
                mu_table[ip, k_node] = m
            else:
                mu_table[ip, k_node] = 0.5


# ----------------------- Full Phi (parallel) -----------------------
@njit(cache=True, parallel=True)
def phi_optB_exact_jit(P_vals, u_grid, p_grid, tau, gamma,
                            gl_s, gl_w, f0_u, f1_u, P_new):
    G = u_grid.size; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    build_mu_optB_exact(P_vals, u_grid, p_grid, tau, gl_s, gl_w,
                            f0_u, f1_u, mu_table)
    eps = 1e-9
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                pc = P_vals[i, j, k]
                if pc < eps: pc = eps
                elif pc > 1.0 - eps: pc = 1.0 - eps
                mu0 = _interp_mu(mu_table, pc, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, pc, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, pc, p_grid, k, G_p)
                if mu0 < eps: mu0 = eps
                elif mu0 > 1.0 - eps: mu0 = 1.0 - eps
                if mu1 < eps: mu1 = eps
                elif mu1 > 1.0 - eps: mu1 = 1.0 - eps
                if mu2 < eps: mu2 = eps
                elif mu2 > 1.0 - eps: mu2 = 1.0 - eps
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)


def make_helpers(u_grid, tau, NQK=8):
    gl_s, gl_w = make_gl_unit(NQK)
    f0_u = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
    f1_u = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
    return gl_s, gl_w, f0_u, f1_u


def phi(P, u_grid, p_grid, tau, gamma, NQK=8):
    gl_s, gl_w, f0_u, f1_u = make_helpers(u_grid, tau, NQK)
    P_new = np.empty_like(P)
    phi_optB_exact_jit(P, u_grid, p_grid, float(tau), float(gamma),
                            gl_s, gl_w, f0_u, f1_u, P_new)
    return P_new


# =========================================================================
#                       Test loop & comparison report
# =========================================================================
if __name__ == "__main__":
    import dd_k3_optB_numba as OB

    G = 11; G_p = 121
    NQK = 8
    FP_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
    OUT = os.path.join(FP_DIR, "optB_exact")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)

    from scipy.stats import norm
    def make_cdf_uniform_grid(G, eps_q=0.01):
        qs = np.linspace(eps_q, 1 - eps_q, G)
        return norm.ppf(qs)
    def make_p_grid(G_p=121, L=8.0):
        return 1.0 / (1.0 + np.exp(-np.linspace(-L, L, G_p)))

    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_s, gl_w = make_gl_unit(NQK)

    # JIT warmup
    print("JIT warmup...", flush=True); t0 = time.time()
    P0 = np.full((G, G, G), 0.5)
    phi(P0, u_grid, p_grid, 1.0, 1.0, NQK=NQK)
    OB.phi(P0, u_grid, p_grid, 1.0, 1.0)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    cells = [(g, t) for g in [100.0, 1000.0] for t in [0.2, 1.0]]
    results = {}
    for gamma, tau in cells:
        key = f"g{int(gamma)}_t{tau:.4f}"
        fp_path = os.path.join(FP_DIR, f"dd_k3_strict_fp_{key}.npz")
        if not os.path.exists(fp_path):
            print(f"\nSKIP {key}: missing {fp_path}", flush=True); continue
        d = np.load(fp_path)
        P_strict = d["P_strict"].astype(np.float64)
        mu_strict = d["mu_strict"].astype(np.float64)
        print(f"\n=== {key} (g={gamma}, tau={tau}) ===", flush=True)

        # Helpers
        f0_u = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
        f1_u = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
        w_trap, f0_ub, f1_ub = OB.make_helpers(u_grid, tau)

        # (a) baseline: vertex step-summation
        mu_step = np.empty((G_p, G))
        t0 = time.time()
        OB.build_mu_optB(P_strict, u_grid, p_grid, float(tau), w_trap,
                              f0_ub, f1_ub, mu_step)
        t_step = time.time() - t0

        # (b) exact-mass via GL
        mu_exact = np.empty((G_p, G))
        t0 = time.time()
        build_mu_optB_exact(P_strict, u_grid, p_grid, float(tau),
                                 gl_s, gl_w, f0_u, f1_u, mu_exact)
        t_exact = time.time() - t0

        # Comparison statistics (full table + in-support only)
        d_step  = np.abs(mu_step  - mu_strict)
        d_exact = np.abs(mu_exact - mu_strict)
        # "in support" = cells where strict actually has data (mu != default 0.5)
        sup = np.abs(mu_strict - 0.5) > 1e-9
        n_sup = int(sup.sum()); n_tot = int(d_step.size)
        st = dict(
            n_total=n_tot, n_in_support=n_sup,
            step_max=float(d_step.max()), step_med=float(np.median(d_step)),
            step_rms=float(np.sqrt(np.mean(d_step**2))), step_wall=t_step,
            exact_max=float(d_exact.max()), exact_med=float(np.median(d_exact)),
            exact_rms=float(np.sqrt(np.mean(d_exact**2))), exact_wall=t_exact,
            step_max_sup=float(d_step[sup].max()) if n_sup else 0.0,
            step_med_sup=float(np.median(d_step[sup])) if n_sup else 0.0,
            step_rms_sup=float(np.sqrt(np.mean(d_step[sup]**2))) if n_sup else 0.0,
            exact_max_sup=float(d_exact[sup].max()) if n_sup else 0.0,
            exact_med_sup=float(np.median(d_exact[sup])) if n_sup else 0.0,
            exact_rms_sup=float(np.sqrt(np.mean(d_exact[sup]**2))) if n_sup else 0.0,
        )
        st["max_ratio"] = st["step_max"] / max(st["exact_max"], 1e-300)
        st["rms_ratio"] = st["step_rms"] / max(st["exact_rms"], 1e-300)
        st["max_ratio_sup"] = st["step_max_sup"] / max(st["exact_max_sup"], 1e-300)
        st["med_ratio_sup"] = st["step_med_sup"] / max(st["exact_med_sup"], 1e-300)
        st["rms_ratio_sup"] = st["step_rms_sup"] / max(st["exact_rms_sup"], 1e-300)
        print(f"  full ({n_tot} cells):", flush=True)
        print(f"    step : max={st['step_max']:.3e} med={st['step_med']:.3e} "
                f"rms={st['step_rms']:.3e} ({t_step*1000:.1f}ms)", flush=True)
        print(f"    exact: max={st['exact_max']:.3e} med={st['exact_med']:.3e} "
                f"rms={st['exact_rms']:.3e} ({t_exact*1000:.1f}ms)", flush=True)
        print(f"  in-support ({n_sup} cells):", flush=True)
        print(f"    step : max={st['step_max_sup']:.3e} med={st['step_med_sup']:.3e} "
                f"rms={st['step_rms_sup']:.3e}", flush=True)
        print(f"    exact: max={st['exact_max_sup']:.3e} med={st['exact_med_sup']:.3e} "
                f"rms={st['exact_rms_sup']:.3e}", flush=True)
        print(f"    ratio: max={st['max_ratio_sup']:.2f}x med={st['med_ratio_sup']:.2f}x "
                f"rms={st['rms_ratio_sup']:.2f}x", flush=True)
        results[key] = dict(gamma=gamma, tau=tau, **st)

        # Save mu arrays for figures
        np.savez(os.path.join(OUT, f"mu_{key}.npz"),
                    mu_strict=mu_strict, mu_step=mu_step, mu_exact=mu_exact,
                    p_grid=p_grid, u_grid=u_grid)

    # Save results JSON
    with open(os.path.join(OUT, "exact_results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults -> {OUT}/exact_results.json", flush=True)

    # Build per-cell comparison figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key, st in results.items():
            d = np.load(os.path.join(OUT, f"mu_{key}.npz"))
            mu_strict = d["mu_strict"]; mu_step = d["mu_step"]
            mu_exact = d["mu_exact"]; pg = d["p_grid"]; ug = d["u_grid"]
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            # Top-left: mu vs p at u_k middle, all three
            k_mid = G // 2
            ax = axes[0, 0]
            ax.plot(pg, mu_strict[:, k_mid], "k-",  lw=1.4, label="mu_strict")
            ax.plot(pg, mu_step[:, k_mid],   "b--", lw=1.0, label="mu_step")
            ax.plot(pg, mu_exact[:, k_mid],  "r-.", lw=1.0, label="mu_exact")
            ax.set_xlabel("p"); ax.set_ylabel(f"mu(p, u_k={ug[k_mid]:.2f})")
            ax.legend(); ax.set_title(f"mu slice, {key}")
            # Top-right: pointwise error vs p
            ax = axes[0, 1]
            ax.semilogy(pg, np.abs(mu_step  - mu_strict).mean(axis=1),
                            "b-", label="step (mean over u_k)")
            ax.semilogy(pg, np.abs(mu_exact - mu_strict).mean(axis=1),
                            "r-", label="exact (mean over u_k)")
            ax.set_xlabel("p"); ax.set_ylabel("|mu - mu_strict|")
            ax.legend(); ax.set_title("u_k-mean error per p")
            # Bottom-left: heatmap of step error
            ax = axes[1, 0]
            im = ax.imshow(np.log10(np.abs(mu_step - mu_strict) + 1e-16),
                              origin="lower", aspect="auto",
                              extent=[ug[0], ug[-1], pg[0], pg[-1]],
                              cmap="viridis", vmin=-6, vmax=0)
            ax.set_xlabel("u_k"); ax.set_ylabel("p")
            ax.set_title("log10|mu_step - mu_strict|")
            plt.colorbar(im, ax=ax)
            # Bottom-right: heatmap of exact error
            ax = axes[1, 1]
            im = ax.imshow(np.log10(np.abs(mu_exact - mu_strict) + 1e-16),
                              origin="lower", aspect="auto",
                              extent=[ug[0], ug[-1], pg[0], pg[-1]],
                              cmap="viridis", vmin=-6, vmax=0)
            ax.set_xlabel("u_k"); ax.set_ylabel("p")
            ax.set_title("log10|mu_exact - mu_strict|")
            plt.colorbar(im, ax=ax)
            fig.suptitle(f"OptB+ exact-mass vs step-summation, {key}", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "figs", f"compare_{key}.png"), dpi=100)
            plt.close(fig)
            print(f"  fig: figs/compare_{key}.png", flush=True)
    except Exception as e:
        print(f"  figs FAILED: {e}", flush=True)

    print("\nDONE.", flush=True)
