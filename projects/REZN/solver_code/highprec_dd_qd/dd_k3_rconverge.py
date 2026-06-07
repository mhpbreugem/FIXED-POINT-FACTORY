"""Richardson-order convergence study at a fixed FP.

For each Richardson order R2..R7 with both LINEAR h (arithmetic) and GEOMETRIC h
(ratio 0.7), compute Phi at the saved G=11 baseline FP and record:
  - F_inf (how far this operator says the baseline FP is from its own FP)
  - L1 weight norm (noise amplification factor)
  - mu_table max, bracket width

This will tell us if R5+ over-extrapolates or genuinely converges.
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
import dd_k3_ops as DK
import dd_k3_variants as DV
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


# Build Richardson families
def hs_linear(n, hmin=0.2, hmax=0.5):
    return tuple(np.linspace(hmin, hmax, n)[::-1])    # decreasing

def hs_geom(n, h0=0.5, r=0.75):
    return tuple(h0 * r**k for k in range(n))


G = 11
NQK = 16
G_p = 121


def study(P_H, P_L, gamma, tau, u_grid):
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    th, tl = split(mp.mpf(repr(tau)))
    gh, gl = split(mp.mpf(repr(gamma)))
    out = {}
    for family, hs_fn in [("linear", hs_linear), ("geom", hs_geom)]:
        for n in [2, 3, 4, 5, 6, 7]:
            hs = hs_fn(n)
            w_nev = DV.richardson_weights_neville(hs)
            w_van = DK.richardson_weights(hs)
            L1_nev = float(np.sum(np.abs(w_nev)))
            L1_van = float(np.sum(np.abs(w_van)))
            t0 = time.time()
            # use Vandermonde weights (same final result as Neville for matching hs)
            PnH, PnL, muH, muL = DV.phi_dd_variant(P_H, P_L, u_grid, p_grid,
                                                          gl_u, gl_du, th, tl, gh, gl,
                                                          np.array(hs, dtype=float),
                                                          w_van, interp_mode="linear")
            wall = time.time() - t0
            diff = PnH + PnL - (P_H + P_L)
            F_inf = float(np.max(np.abs(diff)))
            mu = muH + muL
            br = DV.monotone_bracket(mu)
            key = f"{family}_R{n}"
            out[key] = dict(family=family, n=n, hs=list(hs),
                              L1_norm=L1_nev, F_inf=F_inf,
                              max_bracket=br["max_bracket_p"],
                              min_diff=br["min_diff_p"], wall=wall)
            print(f"  {key:12s} hs[-1]={hs[-1]:.3g} hmin={min(hs):.3g} L1={L1_nev:6.2f} "
                  f"F={F_inf:.2e} br={br['max_bracket_p']:.3f}",
                  flush=True)
    return out


def main():
    u_grid = make_cdf_uniform_grid(G)
    # JIT warmup
    print("warmup...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5); P0L = np.zeros_like(P0)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    hs = (0.5, 0.4, 0.3, 0.2)
    DV.phi_dd_variant(P0, P0L, u_grid, p_grid, gl_u, gl_du, 1.0, 0.0, 1.0, 0.0,
                          np.array(hs, dtype=float),
                          DK.richardson_weights(hs), interp_mode="linear")
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    # Probe several FPs
    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    target_cells = [(g, t) for g in [100.0, 10.0, 1.0] for t in [1.0, 0.2]]
    results = {}
    for gamma, tau in target_cells:
        key = f"g{gamma:.4g}_t{tau:.4f}"
        f = f"/tmp/dd_k3_sweep_fps/{key}.npz"
        if not os.path.exists(f):
            print(f"\nSKIP {key} (FP not yet saved)", flush=True); continue
        d = np.load(f)
        if d["mu_hi"].shape[0] != G:
            print(f"\nSKIP {key} (wrong G)", flush=True); continue
        print(f"\n=== FP {key} ===", flush=True)
        results[key] = study(d["mu_hi"], d["mu_lo"], gamma, tau, u_grid)
        json.dump(results, open("/tmp/dd_k3_rconverge.json", "w"), indent=2,
                    default=str)
    print(f"\nDone. {len(results)} FPs studied. -> /tmp/dd_k3_rconverge.json",
          flush=True)


if __name__ == "__main__":
    main()
