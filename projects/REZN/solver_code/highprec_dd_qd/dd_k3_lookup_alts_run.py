"""Run all alternative lookup-table builders on the saved G=11 R4 FPs,
compare against strict-h=0 ground truth, and to each other.

Outputs to /tmp/dd_k3_alts_results.json + figures + report.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import dd_k3_lookup_alts as ALT
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)


G = 11
G_p = 121
NQ_STRICT = 16


def evaluate_cell(P_full, u_grid, p_grid, gamma, tau, key):
    """Run all five options + strict, return dict of diagnostics."""
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    P64 = P_full.astype(np.float64)
    print(f"\n=== {key} ===", flush=True)

    # Ground truth: strict-h=0
    t0 = time.time()
    mu_strict = build_mu_table_lin_strict(P64, u_grid, p_grid, gl_u_s, gl_du_s,
                                                  float(tau), G, NQ_STRICT)
    t_strict = time.time() - t0
    print(f"  strict-h=0:      {t_strict*1000:.0f}ms  range[{mu_strict.min():.4f},{mu_strict.max():.4f}]",
          flush=True)

    out = dict(t_strict=t_strict,
                 strict_min=float(mu_strict.min()), strict_max=float(mu_strict.max()))

    # Option A: cube-node KB at several h
    out["A"] = {}
    for h in [0.5, 0.3, 0.2, 0.1, 0.05]:
        t0 = time.time()
        mu_A = ALT.cube_node_mu(P64, u_grid, p_grid, float(tau), h)
        tA = time.time() - t0
        dmax = float(np.max(np.abs(mu_A - mu_strict)))
        dmed = float(np.median(np.abs(mu_A - mu_strict)))
        out["A"][f"h={h}"] = dict(max=dmax, med=dmed, wall=tA)
        print(f"  A (cube KB h={h}): {tA*1000:.0f}ms  max|d|={dmax:.3e}  med|d|={dmed:.3e}",
              flush=True)

    # Option B: empirical CDF + density
    t0 = time.time()
    try:
        mu_B = ALT.empirical_cdf_density(P64, u_grid, p_grid, float(tau))
        tB = time.time() - t0
        dmax_B = float(np.max(np.abs(mu_B - mu_strict)))
        dmed_B = float(np.median(np.abs(mu_B - mu_strict)))
        out["B"] = dict(max=dmax_B, med=dmed_B, wall=tB,
                          mu_min=float(mu_B.min()), mu_max=float(mu_B.max()))
        print(f"  B (CDF density): {tB*1000:.0f}ms  max|d|={dmax_B:.3e}  med|d|={dmed_B:.3e}",
              flush=True)
    except Exception as e:
        out["B"] = dict(error=str(e))
        print(f"  B FAILED: {e}", flush=True)

    # Option C: MC Bayes (small N for speed)
    t0 = time.time()
    try:
        mu_C = ALT.mc_bayes_mu(P64, u_grid, p_grid, float(tau),
                                       N_samples=80_000, kde_h_p=0.03, kde_h_u=0.5,
                                       seed=42)
        tC = time.time() - t0
        # Restrict comparison to mid p range (KDE has edge bias)
        midp = (p_grid > 0.05) & (p_grid < 0.95)
        dmax_C = float(np.max(np.abs(mu_C[midp] - mu_strict[midp])))
        dmed_C = float(np.median(np.abs(mu_C[midp] - mu_strict[midp])))
        out["C"] = dict(max=dmax_C, med=dmed_C, wall=tC, N=80_000)
        print(f"  C (MC Bayes):    {tC:.1f}s   max|d|={dmax_C:.3e}  med|d|={dmed_C:.3e}",
              flush=True)
    except Exception as e:
        out["C"] = dict(error=str(e))
        print(f"  C FAILED: {e}", flush=True)

    # Option D: level-set MC (smaller N)
    t0 = time.time()
    try:
        mu_D = ALT.levelset_mc_mu(P64, u_grid, p_grid, float(tau),
                                           N_samples=300_000, band_delta=0.02, seed=7)
        tD = time.time() - t0
        midp = (p_grid > 0.05) & (p_grid < 0.95)
        dmax_D = float(np.max(np.abs(mu_D[midp] - mu_strict[midp])))
        dmed_D = float(np.median(np.abs(mu_D[midp] - mu_strict[midp])))
        out["D"] = dict(max=dmax_D, med=dmed_D, wall=tD, N=300_000)
        print(f"  D (level-set MC):{tD:.1f}s   max|d|={dmax_D:.3e}  med|d|={dmed_D:.3e}",
              flush=True)
    except Exception as e:
        out["D"] = dict(error=str(e))
        print(f"  D FAILED: {e}", flush=True)

    # Option E: distribution diagnostic on strict-h=0 mu_table itself
    out["E"] = ALT.mu_distribution_diagnostic(mu_strict, p_grid)

    # Save mu tables for plotting
    np.savez(f"/tmp/dd_k3_alts_{key}.npz", mu_strict=mu_strict, P=P64,
               mu_A_h0p2=ALT.cube_node_mu(P64, u_grid, p_grid, float(tau), 0.2),
               mu_B=(mu_B if "B" in out and "error" not in out["B"] else np.full_like(mu_strict, np.nan)),
               mu_C=(mu_C if "C" in out and "error" not in out["C"] else np.full_like(mu_strict, np.nan)),
               mu_D=(mu_D if "D" in out and "error" not in out["D"] else np.full_like(mu_strict, np.nan)))
    return out


def main():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    # JIT warmup
    print("warmup...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    ALT.cube_node_mu(P0, u_grid, p_grid, 1.0, 0.3)
    build_mu_table_lin_strict(P0, u_grid, p_grid,
                                       *make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT),
                                       1.0, G, NQ_STRICT)
    print(f"  warm done {time.time()-t0:.1f}s", flush=True)

    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    # restrict to G=11
    cells = []
    for f in fp_files:
        d = np.load(f)
        if d["mu_hi"].shape[0] != G: continue
        cells.append((float(d["gamma"]), float(d["tau"]),
                        d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"]),
                        os.path.basename(f).replace(".npz", "")))
    cells.sort()
    print(f"\nFound {len(cells)} G={G} FP cells", flush=True)
    results = {}
    for gamma, tau, P, key in cells:
        results[key] = evaluate_cell(P, u_grid, p_grid, gamma, tau, key)
        results[key]["gamma"] = gamma; results[key]["tau"] = tau
        json.dump(results, open("/tmp/dd_k3_alts_results.json", "w"), indent=2,
                    default=str)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_alts_results.json", flush=True)


if __name__ == "__main__":
    main()
