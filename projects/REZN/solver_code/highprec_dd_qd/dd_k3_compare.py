"""Compare lookup-table variants at saved G=11 FPs.

For each (gamma, tau) FP:
  - Build mu-table under each variant (V1 baseline ... V7).
  - Build "truth" with V8: PCHIP-p + Neville-R5 + NQK=24 + G_p=241.
  - Record per-variant:
      max |mu_var - mu_truth|     # on a common p-eval grid via PCHIP eval
      Phi(P_FP) under the variant; |F_var|_inf = max|Phi - P|
      slope, deficit unchanged (FP-properties)
      monotone-bracket diagnostic on mu_var
      wall time

Saves to /tmp/dd_k3_compare.json.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
import dd_ops as DO
import dd_k3_ops as DK
import dd_k3_variants as DV
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


FPS_DIR = "/tmp/dd_k3_sweep_fps"
OUT = "/tmp/dd_k3_compare.json"

G = 11
HS_R4 = (0.5, 0.4, 0.3, 0.2)
HS_R5_GEOM = tuple(0.6 * 0.7**k for k in range(5))     # geometric ratio 0.7

VARIANTS = {
    "V1_base":      dict(G_p=121, NQK=16, interp="linear", richardson="linear", hs=HS_R4),
    "V2_pchip":     dict(G_p=121, NQK=16, interp="pchip",  richardson="linear", hs=HS_R4),
    "V3_neville":   dict(G_p=121, NQK=16, interp="linear", richardson="neville", hs=HS_R5_GEOM),
    "V4_combined":  dict(G_p=121, NQK=16, interp="pchip",  richardson="neville", hs=HS_R5_GEOM),
    "V6_hi_NQK":    dict(G_p=121, NQK=24, interp="linear", richardson="linear", hs=HS_R4),
    "V7_hi_Gp":     dict(G_p=241, NQK=16, interp="linear", richardson="linear", hs=HS_R4),
    "V8_truth":     dict(G_p=241, NQK=24, interp="pchip",  richardson="neville", hs=HS_R5_GEOM),
}


def project_P_to_pgrid(P_H, P_L, u_grid_src, u_grid_dst):
    """Trilinear interp from u_grid_src to u_grid_dst (G=11 -> G=11 is identity)."""
    if u_grid_src.size == u_grid_dst.size and np.allclose(u_grid_src, u_grid_dst):
        return P_H, P_L
    from scipy.interpolate import RegularGridInterpolator
    pts = (u_grid_src,)*3
    intH = RegularGridInterpolator(pts, P_H, bounds_error=False, fill_value=None)
    intL = RegularGridInterpolator(pts, P_L, bounds_error=False, fill_value=None)
    U1, U2, U3 = np.meshgrid(u_grid_dst, u_grid_dst, u_grid_dst, indexing="ij")
    pts_eval = np.stack([U1.ravel(), U2.ravel(), U3.ravel()], axis=1)
    return (intH(pts_eval).reshape(U1.shape),
            intL(pts_eval).reshape(U1.shape))


def variant_phi(P_H, P_L, u_grid, gamma, tau, cfg):
    """Run one variant at fixed (P, gamma, tau). Returns dict with mu_table,
    P_new, F_inf, wall_seconds, NQK, G_p."""
    G_p = cfg["G_p"]; NQK = cfg["NQK"]
    interp = cfg["interp"]; rich = cfg["richardson"]
    hs = np.array(cfg["hs"], dtype=float)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    if rich == "linear":
        w = DK.richardson_weights(tuple(hs))
    else:
        w = DV.richardson_weights_neville(hs)
    th, tl = split(mp.mpf(repr(tau)))
    gh, gl_ = split(mp.mpf(repr(gamma)))
    t0 = time.time()
    PnH, PnL, muH, muL = DV.phi_dd_variant(P_H, P_L, u_grid, p_grid, gl_u, gl_du,
                                                  th, tl, gh, gl_, hs, w,
                                                  interp_mode=interp)
    wall = time.time() - t0
    # F_inf in DD
    diffH = PnH + PnL - (P_H + P_L)
    F_inf = float(np.max(np.abs(diffH)))
    # bracket diagnostic
    mu_full = muH + muL
    bracket = DV.monotone_bracket(mu_full)
    return dict(muH=muH, muL=muL, mu=mu_full, PnH=PnH, PnL=PnL,
                  F_inf=F_inf, wall=wall, G_p=G_p, NQK=NQK,
                  interp=interp, richardson=rich, n_hs=len(hs),
                  bracket=bracket, p_grid=p_grid)


def mu_diff_inf(mu_a, p_a, mu_b, p_b, k_eval=None):
    """Compare two mu-tables (possibly on different p-grids) by evaluating both
    on a fine common p-grid via linear interp."""
    p_eval = np.linspace(min(p_a[0], p_b[0])+1e-6,
                            max(p_a[-1], p_b[-1])-1e-6, 1001)
    G = mu_a.shape[1]
    if k_eval is None: k_eval = range(G)
    err = 0.0
    for k in k_eval:
        ya = np.interp(p_eval, p_a, mu_a[:, k])
        yb = np.interp(p_eval, p_b, mu_b[:, k])
        e = float(np.max(np.abs(ya - yb)))
        if e > err: err = e
    return err


def run():
    # JIT warmup
    print("JIT warmup...", flush=True); t0 = time.time()
    u_grid = make_cdf_uniform_grid(G)
    P_warm = np.zeros((G, G, G)) + 0.5; PL_warm = np.zeros_like(P_warm)
    variant_phi(P_warm, PL_warm, u_grid, 1.0, 1.0, VARIANTS["V1_base"])
    variant_phi(P_warm, PL_warm, u_grid, 1.0, 1.0, VARIANTS["V2_pchip"])
    print(f"  warm done {time.time()-t0:.1f}s", flush=True)

    # Choose FPs
    fp_files = sorted(glob.glob(f"{FPS_DIR}/*.npz"))
    print(f"Found {len(fp_files)} FP files in {FPS_DIR}", flush=True)
    results = {}
    for fp_file in fp_files:
        d = np.load(fp_file)
        gamma = float(d["gamma"]); tau = float(d["tau"])
        P_H = d["mu_hi"]; P_L = d["mu_lo"]
        if P_H.shape[0] != G:
            print(f"  skip {os.path.basename(fp_file)}: G={P_H.shape[0]} != {G}",
                  flush=True); continue
        print(f"\n=== FP: gamma={gamma}, tau={tau} ===", flush=True)
        cell = {}
        # Run truth first
        out_truth = variant_phi(P_H, P_L, u_grid, gamma, tau, VARIANTS["V8_truth"])
        print(f"  V8 truth: NQK=24 G_p=241 Neville-R5 PCHIP | F={out_truth['F_inf']:.2e} | "
              f"wall={out_truth['wall']:.1f}s", flush=True)
        cell["V8_truth"] = dict(F_inf=out_truth["F_inf"], wall=out_truth["wall"],
                                  bracket=out_truth["bracket"], G_p=out_truth["G_p"],
                                  NQK=out_truth["NQK"])
        # Run the rest
        for name in ["V1_base", "V2_pchip", "V3_neville", "V4_combined",
                       "V6_hi_NQK", "V7_hi_Gp"]:
            out = variant_phi(P_H, P_L, u_grid, gamma, tau, VARIANTS[name])
            dmu = mu_diff_inf(out["mu"], out["p_grid"],
                                  out_truth["mu"], out_truth["p_grid"])
            print(f"  {name:14s}: F={out['F_inf']:.2e} dmu_vs_truth={dmu:.2e} "
                  f"bracket_max={out['bracket']['max_bracket_p']:.3f} "
                  f"wall={out['wall']:.1f}s", flush=True)
            cell[name] = dict(F_inf=out["F_inf"], dmu_vs_truth=dmu,
                                wall=out["wall"], bracket=out["bracket"],
                                G_p=out["G_p"], NQK=out["NQK"],
                                interp=out["interp"], rich=out["richardson"])
        # Also: dmu of each variant vs baseline V1 (gives "what does each variant change")
        v1 = variant_phi(P_H, P_L, u_grid, gamma, tau, VARIANTS["V1_base"])
        for name in ["V2_pchip", "V3_neville", "V4_combined", "V6_hi_NQK", "V7_hi_Gp"]:
            out = variant_phi(P_H, P_L, u_grid, gamma, tau, VARIANTS[name])
            cell[name]["dmu_vs_V1"] = mu_diff_inf(out["mu"], out["p_grid"],
                                                       v1["mu"], v1["p_grid"])
        results[f"g{gamma:.4g}_t{tau:.4f}"] = cell
        json.dump(results, open(OUT, "w"), indent=2, default=str)

    print(f"\n=== DONE === {len(results)} FPs processed. Output: {OUT}",
          flush=True)


if __name__ == "__main__":
    run()
