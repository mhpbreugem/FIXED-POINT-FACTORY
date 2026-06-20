"""Cross-validate kernel-band Richardson against direct strict-h=0 co-area
(root-based, with partition-of-unity). At the saved G=11 FP for each (gamma, tau):

  1. Compute mu_table via STRICT-h=0 (no kernel; co-area + POU).
     This is the "ground truth" lookup function from a genuinely independent
     algorithm.
  2. Compute mu_table via kernel-band Richardson at R2, R3, R4, R5.
  3. Compare |mu_R - mu_strict|_inf for each R.

The R-order that minimizes the discrepancy is the right one.

Also: for each R, compute |Phi_R(P_FP) - P_FP| (how close P_FP is to FP under
each operator's Phi). This is a separate consistency check.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from lin_cdf_strict import build_mu_table_lin_strict
from lin_cdf_kern_tab import (build_mu_table_lin_kern, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)
from lin_cdf_richardson import richardson_weights


G = 11
G_p = 121
NQK = 16
NQ_STRICT = 16     # for the strict-h=0 GL inner quadrature
HS_R2 = (0.5, 0.2)
HS_R3 = (0.5, 0.35, 0.2)
HS_R4 = (0.5, 0.4, 0.3, 0.2)
HS_R5 = (0.5, 0.42, 0.35, 0.27, 0.2)


def build_mu_richardson(P, u_grid, p_grid, gl_u, gl_du, tau, hs, NQK):
    """Float64 mu-table via kernel-band Richardson with weights hs."""
    G = u_grid.size; G_p = p_grid.size
    w = richardson_weights(hs)
    mu = np.zeros((G_p, G))
    for hi, h in enumerate(hs):
        mu_h = build_mu_table_lin_kern(P, u_grid, p_grid, gl_u, gl_du, tau,
                                            G, NQK, float(h))
        mu += float(w[hi]) * mu_h
    return mu


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    # The strict-h=0 builder uses its own GL nodes -- same construction
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    print(f"--- Strict-h=0 cross-validation, G={G}, G_p={G_p}, NQK={NQK} ---",
          flush=True)
    # JIT warm
    print("warmup...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    build_mu_table_lin_strict(P0, u_grid, p_grid, gl_u_s, gl_du_s, 1.0, G, NQ_STRICT)
    build_mu_table_lin_kern(P0, u_grid, p_grid, gl_u, gl_du, 1.0, G, NQK, 0.3)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    target_cells = [(g, t) for g in [100.0, 10.0, 1.0, 1000.0] for t in [1.0, 0.2]]
    results = {}
    for gamma, tau in target_cells:
        key = f"g{gamma:.4g}_t{tau:.4f}"
        f = f"/tmp/dd_k3_sweep_fps/{key}.npz"
        if not os.path.exists(f):
            print(f"\nSKIP {key} (no FP)", flush=True); continue
        d = np.load(f)
        P_full = d["P"] if "P" in d.files else (d["mu_hi"] + d["mu_lo"])
        if P_full.shape[0] != G:
            print(f"\nSKIP {key} (G={P_full.shape[0]}!={G})", flush=True); continue
        print(f"\n=== FP {key} ===", flush=True)
        # strict-h=0 reference
        t0 = time.time()
        mu_strict = build_mu_table_lin_strict(P_full.astype(np.float64), u_grid,
                                                    p_grid, gl_u_s, gl_du_s,
                                                    float(tau), G, NQ_STRICT)
        t_strict = time.time() - t0
        print(f"  strict-h=0: {t_strict:.1f}s, mu range [{mu_strict.min():.4f}, "
              f"{mu_strict.max():.4f}]", flush=True)
        # Kernel-band Richardson at multiple orders
        cell = dict(strict_t=t_strict, strict_mu_min=float(mu_strict.min()),
                      strict_mu_max=float(mu_strict.max()))
        for label, hs in [("R2", HS_R2), ("R3", HS_R3),
                              ("R4", HS_R4), ("R5", HS_R5)]:
            t0 = time.time()
            mu_R = build_mu_richardson(P_full.astype(np.float64), u_grid, p_grid,
                                              gl_u, gl_du, float(tau), hs, NQK)
            tR = time.time() - t0
            dmax = float(np.max(np.abs(mu_R - mu_strict)))
            dmed = float(np.median(np.abs(mu_R - mu_strict)))
            print(f"  {label}: hs={[f'{h:.2g}' for h in hs]}  "
                  f"max|mu-strict|={dmax:.3e}  med={dmed:.3e}  ({tR:.1f}s)",
                  flush=True)
            cell[label] = dict(hs=list(hs), max_diff_vs_strict=dmax,
                                  median_diff_vs_strict=dmed, wall=tR)
        results[key] = cell
        json.dump(results, open("/tmp/dd_k3_strict_xv.json", "w"), indent=2,
                    default=str)
    print(f"\nDone. {len(results)} FPs. -> /tmp/dd_k3_strict_xv.json", flush=True)


if __name__ == "__main__":
    run()
