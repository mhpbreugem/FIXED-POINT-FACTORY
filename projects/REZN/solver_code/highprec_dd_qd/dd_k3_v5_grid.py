"""Apply pipeline v5 (fix_fallback + per-segment Cheby) to all 36 ridge cells.

The previous grid sweep agent used the original pipeline (no fallback
fix), so 20/36 cells stalled because saved mu_strict has fallback
artifacts at low tau. Pipeline v5 should fix this by cleaning mu_strict
first.
"""
import os, sys, json, time, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_strict import build_mu_table_lin_strict, make_gl_for_u
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from dd_k3_optB_pieceCheby import detect_critical_points_from_mu
from dd_k3_zero_h_pipeline_v4 import build_full_v4, evaluate_full_v4
from dd_k3_strict_fixed_fallback import fix_fallback


REPO = "/home/user/FIXED-POINT-FACTORY"
RIDGE = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/strict_ridge_v5"
os.makedirs(f"{OUT}/figs", exist_ok=True)


def process_cell(gamma, tau, deg=6, G=11, G_p=121, nq=16):
    fp_file = f"{RIDGE}/strict_ridge_g{gamma}_t{tau:.4f}.npz"
    if not os.path.exists(fp_file):
        # Try alternate naming
        alt = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz"
        if os.path.exists(alt): fp_file = alt
        else: return None
    d = np.load(fp_file)
    P_strict = d["P_strict"].astype(np.float64) if "P_strict" in d.files else None
    if P_strict is None: return None
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(G_p)
    # Get mu_strict either from file or rebuild
    if "mu_strict" in d.files:
        mu_raw = d["mu_strict"].astype(np.float64)
    else:
        gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)
        mu_raw = build_mu_table_lin_strict(P_strict, u_grid, p_grid, gl_u, gl_du, tau, G, nq)

    # Step 1: fix fallback
    mu_clean = fix_fallback(mu_raw, p_grid)
    # Step 2: detect cusps on cleaned
    p_cs = detect_critical_points_from_mu(mu_clean, p_grid)
    # Step 3: pipeline v4
    mu_pipe = np.full((G_p, G), 0.5)
    for k in range(G):
        knots, segs = build_full_v4(p_grid, mu_clean[:, k], p_cs, deg=deg)
        if knots is None:
            mu_pipe[:, k] = mu_clean[:, k]; continue
        for ip in range(G_p):
            mu_pipe[ip, k] = evaluate_full_v4(p_grid[ip], knots, segs)
    err = np.abs(mu_pipe - mu_clean)
    # Monotonicity on dense
    p_dense = np.linspace(1e-6, 1 - 1e-6, 501)
    n_mono = 0; min_diff = float("inf")
    for k in range(G):
        knots, segs = build_full_v4(p_grid, mu_clean[:, k], p_cs, deg=deg)
        if knots is None: continue
        md = np.array([evaluate_full_v4(p, knots, segs) for p in p_dense])
        m = float(np.min(np.diff(md)))
        min_diff = min(min_diff, m)
        if np.all(np.diff(md) >= -1e-9): n_mono += 1
    np.savez(f"{OUT}/zero_h_v5_g{gamma}_t{tau:.4f}.npz",
             P_strict=P_strict, mu_raw=mu_raw, mu_clean=mu_clean, mu_pipe=mu_pipe,
             p_grid=p_grid, u_grid=u_grid, p_cs=p_cs)
    return dict(gamma=gamma, tau=tau, deg=deg, n_cusps=len(p_cs),
                max_err=float(err.max()), median_err=float(np.median(err)),
                rms_err=float(np.sqrt(np.mean(err**2))),
                n_mono=n_mono, min_diff=float(min_diff))


def main():
    GAMMAS = [1, 10, 30, 100, 300, 1000]
    TAUS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []
    t0 = time.time()
    for g in GAMMAS:
        for tau in TAUS:
            t1 = time.time()
            r = process_cell(g, tau)
            if r is None:
                print(f"  g={g}, t={tau}: SKIP (no FP file)", flush=True)
                continue
            wall = time.time() - t1
            print(f"  g={g:>5g}, t={tau:.1f}: cusps={r['n_cusps']:>3d}, "
                  f"max={r['max_err']:.2e}, med={r['median_err']:.2e}, "
                  f"mono={r['n_mono']}/11  ({wall:.1f}s)", flush=True)
            results.append(r)
    print(f"\nTotal wall: {time.time()-t0:.0f}s")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

    # Heatmap
    gam_idx = {g: i for i, g in enumerate(GAMMAS)}
    tau_idx = {t: i for i, t in enumerate(TAUS)}
    max_grid = np.full((len(GAMMAS), len(TAUS)), np.nan)
    med_grid = np.full((len(GAMMAS), len(TAUS)), np.nan)
    mono_grid = np.full((len(GAMMAS), len(TAUS)), np.nan)
    for r in results:
        max_grid[gam_idx[r['gamma']], tau_idx[r['tau']]] = np.log10(r['max_err'] + 1e-20)
        med_grid[gam_idx[r['gamma']], tau_idx[r['tau']]] = np.log10(r['median_err'] + 1e-20)
        mono_grid[gam_idx[r['gamma']], tau_idx[r['tau']]] = r['n_mono']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in zip(axes, [max_grid, med_grid, mono_grid],
                                  ['log10 max_err', 'log10 median_err', '# slices monotone']):
        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r' if 'err' in title else 'RdYlGn',
                        origin='lower')
        ax.set_xticks(range(len(TAUS))); ax.set_xticklabels([f'{t:.1f}' for t in TAUS])
        ax.set_yticks(range(len(GAMMAS))); ax.set_yticklabels([str(g) for g in GAMMAS])
        ax.set_xlabel('tau'); ax.set_ylabel('gamma')
        ax.set_title(title); plt.colorbar(im, ax=ax)
        # Annotate
        for i in range(len(GAMMAS)):
            for j in range(len(TAUS)):
                if not np.isnan(data[i, j]):
                    if 'err' in title: ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=8)
                    else: ax.text(j, i, f'{int(data[i,j])}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/heatmap_v5.png", dpi=120)
    plt.close()
    print(f"saved to {OUT}/")


if __name__ == "__main__":
    main()
