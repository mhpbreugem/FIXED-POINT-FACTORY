"""Does Richardson R4 converge to strict-h=0 as h shrinks?

At each saved G=11 R4 FP, compute mu_table via R4 Richardson at four progressively
smaller h-scales, each with matched NQK (so the kernel band is well-resolved):

  scale 0 (current): h ~ [0.2, 0.5],  NQK=16    -> coarse
  scale 1:            h ~ [0.1, 0.25], NQK=24
  scale 2:            h ~ [0.04, 0.10], NQK=32
  scale 3:            h ~ [0.02, 0.05], NQK=64    -> very fine

Compare each to strict-h=0 at the same P. If R4 -> strict as h decreases,
the Richardson approach is sound and we just need a smaller h. If R4 remains
biased even at h ~ 0.02, the operator has a fundamental issue.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)
from lin_cdf_kern_tab import build_mu_table_lin_kern
from lin_cdf_richardson import richardson_weights


G = 11
G_p = 121
NQ_STRICT = 16


def build_mu_R4(P, u_grid, p_grid, gl_u, gl_du, tau, hs, NQK):
    G = u_grid.size; G_p = p_grid.size
    w = richardson_weights(hs)
    mu = np.zeros((G_p, G))
    for hi, h in enumerate(hs):
        mu_h = build_mu_table_lin_kern(P, u_grid, p_grid, gl_u, gl_du, tau,
                                            G, NQK, float(h))
        mu += float(w[hi]) * mu_h
    return mu


SCALES = [
    dict(label="scale0_coarse", hs=(0.5, 0.4, 0.3, 0.2),       NQK=16),
    dict(label="scale1",          hs=(0.25, 0.20, 0.15, 0.10),  NQK=24),
    dict(label="scale2",          hs=(0.10, 0.08, 0.06, 0.04),  NQK=32),
    dict(label="scale3_fine",   hs=(0.05, 0.04, 0.03, 0.02),  NQK=64),
]


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], NQ_STRICT)
    # JIT warmup
    print("warmup jit...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    build_mu_table_lin_strict(P0, u_grid, p_grid, gl_u_s, gl_du_s, 1.0, G, NQ_STRICT)
    for sc in SCALES:
        gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], sc["NQK"])
        build_mu_table_lin_kern(P0, u_grid, p_grid, gl_u, gl_du, 1.0, G,
                                    sc["NQK"], sc["hs"][-1])
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    target_cells = [(g, t) for g in [100.0, 10.0] for t in [1.0, 0.2]]
    results = {}
    for gamma, tau in target_cells:
        key = f"g{gamma:.4g}_t{tau:.4f}"
        f = f"/tmp/dd_k3_sweep_fps/{key}.npz"
        if not os.path.exists(f):
            print(f"\nSKIP {key}", flush=True); continue
        d = np.load(f)
        P = (d["P"] if "P" in d.files else (d["mu_hi"] + d["mu_lo"])).astype(np.float64)
        if P.shape[0] != G: continue
        print(f"\n=== FP {key} ===", flush=True)
        t0 = time.time()
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       float(tau), G, NQ_STRICT)
        print(f"  strict-h=0  ({time.time()-t0:.1f}s): "
              f"mu range [{mu_strict.min():.4f}, {mu_strict.max():.4f}]",
              flush=True)
        cell = dict(strict_min=float(mu_strict.min()),
                      strict_max=float(mu_strict.max()))
        for sc in SCALES:
            gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], sc["NQK"])
            t0 = time.time()
            mu_R = build_mu_R4(P, u_grid, p_grid, gl_u, gl_du,
                                   float(tau), sc["hs"], sc["NQK"])
            tR = time.time() - t0
            dmax = float(np.max(np.abs(mu_R - mu_strict)))
            dmed = float(np.median(np.abs(mu_R - mu_strict)))
            print(f"  {sc['label']:15s} hs={[f'{h:.3g}' for h in sc['hs']]} "
                  f"NQK={sc['NQK']}  max|mu-strict|={dmax:.3e}  med={dmed:.3e}  "
                  f"({tR:.1f}s)", flush=True)
            cell[sc["label"]] = dict(hs=list(sc["hs"]), NQK=sc["NQK"],
                                        max_diff=dmax, median_diff=dmed,
                                        wall=tR)
        results[key] = cell
        json.dump(results, open("/tmp/dd_k3_small_h.json", "w"), indent=2,
                    default=str)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_small_h.json", flush=True)


if __name__ == "__main__":
    run()
