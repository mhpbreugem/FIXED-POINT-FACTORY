"""Helper for saving a converged fixed point with full metadata.

Usage:
    from save_fp import save_fixed_point
    save_fixed_point(mu_vals, xi_grid, p_grid,
                     gamma=100.0, tau=2.0, A_logit=5.5,
                     F_final=5.4e-13, newton_iters=27,
                     tol=1e-12, solver_interp="logit_logit_pchip",
                     wall_seconds=33, script="tau_sweep_a55.py",
                     note="canonical anchor")
"""
import os, datetime, subprocess
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=THIS_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def save_fixed_point(mu_vals, xi_grid, p_grid, *,
                      gamma, tau, A_logit, F_final, newton_iters,
                      tol=1e-12, solver_interp="logit_logit_pchip",
                      wall_seconds=None, script="", note="",
                      filename=None):
    G, Gp = mu_vals.shape
    if filename is None:
        filename = f"g{gamma:.1f}_t{tau:.4f}_A{A_logit:.2f}_G{G}_Gp{Gp}.npz"
    out_path = os.path.join(THIS_DIR, filename)
    np.savez(
        out_path,
        mu_vals=mu_vals.astype(np.float64),
        xi_grid=xi_grid.astype(np.float64),
        p_grid=p_grid.astype(np.float64),
        gamma=np.float64(gamma),
        tau=np.float64(tau),
        G=np.int32(G), Gp=np.int32(Gp),
        A_logit=np.float64(A_logit),
        F_final=np.float64(F_final),
        newton_iters=np.int32(newton_iters),
        tol=np.float64(tol),
        solver_interp=np.str_(solver_interp),
        wall_seconds=np.float64(wall_seconds or -1.0),
        timestamp_utc=np.str_(datetime.datetime.utcnow().isoformat() + "Z"),
        commit_hash=np.str_(_git_commit_hash()),
        script=np.str_(script),
        note=np.str_(note),
    )
    return out_path
