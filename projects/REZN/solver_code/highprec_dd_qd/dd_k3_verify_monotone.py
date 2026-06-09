"""Verify that pipeline v4 OUTPUT itself is monotone, even where mu_strict has artifacts."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_optB_pieceCheby import detect_critical_points_from_mu
from dd_k3_zero_h_pipeline_v4 import build_full_v4, evaluate_full_v4


REPO = "/home/user/FIXED-POINT-FACTORY"


for gamma, tau in [(100, 0.2), (100, 1.0), (1000, 0.2), (1000, 1.0)]:
    fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
    mu_strict = fp["mu_strict"].astype(np.float64)
    G_p, G = mu_strict.shape
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)
    p_cs = detect_critical_points_from_mu(mu_strict, p_grid)

    # Evaluate pipeline on a DENSE grid (not just p_grid)
    p_dense = np.linspace(1e-6, 1 - 1e-6, 1001)
    n_mono_dense = 0
    n_mono_strict = 0
    min_diff_pipe_global = float("inf")
    min_diff_strict_global = float("inf")
    for k in range(G):
        knots, segs = build_full_v4(p_grid, mu_strict[:, k], p_cs, deg=6)
        if knots is None: continue
        mu_pipe_dense = np.array([evaluate_full_v4(p, knots, segs) for p in p_dense])
        min_d_pipe = float(np.min(np.diff(mu_pipe_dense)))
        min_diff_pipe_global = min(min_diff_pipe_global, min_d_pipe)
        if np.all(np.diff(mu_pipe_dense) >= -1e-9): n_mono_dense += 1
        min_d_strict = float(np.min(np.diff(mu_strict[:, k])))
        min_diff_strict_global = min(min_diff_strict_global, min_d_strict)
        if np.all(np.diff(mu_strict[:, k]) >= -1e-9): n_mono_strict += 1
    print(f"=== g={gamma}, t={tau} ===")
    print(f"  pipeline v4 monotone (dense grid): {n_mono_dense}/{G}, "
          f"global min diff = {min_diff_pipe_global:.3e}")
    print(f"  mu_strict monotone (p_grid):       {n_mono_strict}/{G}, "
          f"global min diff = {min_diff_strict_global:.3e}")
