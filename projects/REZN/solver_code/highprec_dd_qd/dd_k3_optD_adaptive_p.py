"""Option d (delta): equidistributed p-grid for the K=3 lookup table.

For each (gamma, tau) FP P, pick the p-grid endogenously so each bin
contains equal "monitor mass". Tested monitors:
  - alpha:  m(p) = a_v(p)             (equal lookup mass per bin)
  - beta:   m(p) = |d mu_avg / dp|    (equal mu-variation per bin)
  - gamma:  m(p) = sqrt(1 + |dmu|^2) (arc-length)
  - uniform: m(p) = 1                  (baseline = logit-uniform)

At each cell, we measure the lookup-table-vs-reference accuracy of each
monitor's p-grid (G_p = 121 points), compared to a dense reference
(G_p = 1001 logit-uniform).
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from scipy.interpolate import PchipInterpolator
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)


G = 11
G_p_REF = 1001     # reference fine grid
G_p_TEST = 121     # the grid size we want to compare


def equidistribute_p(monitor, p_dense, G_p_new):
    """Equidistribution: given monitor m(p) on dense p_dense, return G_p_new
    new p values such that each new bin contains equal cumulative monitor mass.
    """
    m_safe = np.maximum(monitor, 1e-30)
    M = np.zeros_like(p_dense)
    M[1:] = np.cumsum(0.5 * (m_safe[1:] + m_safe[:-1]) * np.diff(p_dense))
    if M[-1] <= 0: return np.linspace(p_dense[0], p_dense[-1], G_p_new)
    # quantile points
    qs = np.linspace(0.0, M[-1], G_p_new)
    new_p = np.interp(qs, M, p_dense)
    # ensure strictly increasing
    eps = 1e-12
    for i in range(1, len(new_p)):
        if new_p[i] <= new_p[i-1] + eps:
            new_p[i] = new_p[i-1] + eps
    return new_p


def interp_lookup(p_eval, p_known, mu_known):
    """Interpolate mu_known(p_known) onto p_eval (linear)."""
    out = np.empty((len(p_eval), mu_known.shape[1]))
    for k in range(mu_known.shape[1]):
        out[:, k] = np.interp(p_eval, p_known, mu_known[:, k])
    return out


def run():
    u_grid = make_cdf_uniform_grid(G)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], 16)
    p_dense = make_p_grid(G_p_REF)
    p_logit_test = make_p_grid(G_p_TEST)   # baseline (uniform)
    print(f"--- Adaptive p-grid (Option d) test ---", flush=True)
    print(f"reference G_p={G_p_REF}, test G_p={G_p_TEST}", flush=True)

    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    results = {}
    print("JIT...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    build_mu_table_lin_strict(P0, u_grid, p_dense, gl_u, gl_du, 1.0, G, 16)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    for fp in fp_files[:16]:
        d = np.load(fp)
        if d["mu_hi"].shape[0] != G: continue
        P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
        gamma = float(d["gamma"]); tau = float(d["tau"])
        key = os.path.basename(fp).replace(".npz", "")
        print(f"\n=== {key} ===", flush=True)
        # 1) Reference lookup at fine logit-uniform p-grid
        mu_ref = build_mu_table_lin_strict(P, u_grid, p_dense, gl_u, gl_du,
                                                    tau, G, 16)
        # 2) Average density across u_k as the "alpha" monitor
        a_avg = np.mean(np.gradient(mu_ref, p_dense, axis=0), axis=1)
        a_avg = np.abs(a_avg)
        # alpha: a_v approx (signed average density)
        m_alpha = a_avg.copy()
        # beta: |d mu_avg / dp|
        mu_avg = mu_ref.mean(axis=1)
        m_beta = np.abs(np.gradient(mu_avg, p_dense))
        # gamma: arc length
        m_gamma = np.sqrt(1.0 + m_beta**2)
        # uniform
        m_uniform = np.ones_like(p_dense)
        # Build each adaptive p-grid
        p_grids = {
            "uniform_logit": p_logit_test,
            "alpha_a_v":      equidistribute_p(m_alpha, p_dense, G_p_TEST),
            "beta_dmu":       equidistribute_p(m_beta,  p_dense, G_p_TEST),
            "gamma_arc":      equidistribute_p(m_gamma, p_dense, G_p_TEST),
        }
        cell = dict(gamma=gamma, tau=tau, mu_ref_range=[float(mu_ref.min()),
                                                                float(mu_ref.max())])
        for name, pg in p_grids.items():
            # Compute the lookup on this p-grid via strict-h=0
            mu_at_pg = build_mu_table_lin_strict(P, u_grid, pg, gl_u, gl_du,
                                                          tau, G, 16)
            # Linear interpolate back to the dense reference
            mu_interp = interp_lookup(p_dense, pg, mu_at_pg)
            d_max = float(np.max(np.abs(mu_interp - mu_ref)))
            d_med = float(np.median(np.abs(mu_interp - mu_ref)))
            cell[name] = dict(max=d_max, median=d_med,
                                 p_first=float(pg[0]), p_last=float(pg[-1]),
                                 p_density_at_0p5=float(np.sum(np.abs(pg-0.5)<0.05)))
            print(f"  {name:18s}  max|d|={d_max:.3e}  med|d|={d_med:.3e}  "
                  f"#pts near p=0.5: {cell[name]['p_density_at_0p5']}",
                  flush=True)
        results[key] = cell
        json.dump(results, open("/tmp/dd_k3_optD_results.json", "w"), indent=2,
                    default=str)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_optD_results.json",
          flush=True)


if __name__ == "__main__":
    run()
