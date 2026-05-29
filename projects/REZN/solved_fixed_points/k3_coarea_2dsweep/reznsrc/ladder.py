"""Gamma-ladder continuation sweep for K=4.

Walk gamma from large (CARA-like) down to small. Use the converged price
array P at gamma_n as the initial seed for gamma_{n+1}; this homotopy
keeps Anderson on the partially-revealing branch and dramatically
shortens convergence.

Usage (pinned to CPU 0):

    taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        python -m code.ladder --G 10 --tau 2.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import Config, SolverConfig
from .contour_K4 import init_no_learning, phi_K4
from .f128 import revelation_deficit_f128, symmetrize_f128
from .metrics import revelation_deficit
from .solver import solve


# Default ladder: dense near CARA so we can see the deficit lift off.
DEFAULT_GAMMAS = [
    100.0, 50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0,
    0.7, 0.5, 0.35, 0.25,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gamma-ladder K=4 continuation.")
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--W", type=float, default=1.0)
    p.add_argument("--u-min", type=float, default=-4.0)
    p.add_argument("--u-max", type=float, default=+4.0)
    p.add_argument("--gammas", type=str, default="",
                   help="comma-separated gamma list; default = built-in ladder")
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--tol", type=float, default=1.0e-6)
    p.add_argument("--anderson-m", type=int, default=8)
    p.add_argument("--seed", choices=("continuation", "no-learning"),
                   default="no-learning",
                   help="continuation: seed gamma_n+1 from converged P at "
                        "gamma_n (selects FR branch when starting from "
                        "large gamma). no-learning: seed each gamma from "
                        "its no-learning equilibrium (selects PR branch).")
    p.add_argument("--f128-symmetrize", action="store_true",
                   help="symmetrise the iterate in float128 each step")
    p.add_argument("--output-dir", type=Path, default=Path("output/ladder"))
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gammas:
        gammas = [float(g) for g in args.gammas.split(",") if g.strip()]
    else:
        gammas = DEFAULT_GAMMAS

    tag = f"_{args.tag}" if args.tag else ""
    base = f"K4_G{args.G}_tau{args.tau}{tag}"
    summary_path = args.output_dir / f"{base}_ladder.json"
    table_path = args.output_dir / f"{base}_ladder.txt"

    # Build the universal grid + tau / W vectors (gamma is per-step).
    cfg_template = Config(K=4, G=args.G, gamma=gammas[0], tau=args.tau,
                          W=args.W, u_min=args.u_min, u_max=args.u_max,
                          cara=False)
    u_grid = cfg_template.u_grid()
    tau_vec = cfg_template.tau_vec()
    W_vec = cfg_template.W_vec()

    print(f"[ladder] K=4 G={args.G} tau={args.tau} ladder of {len(gammas)} gammas")
    print(f"[ladder] seed={args.seed}  f128-sym={args.f128_symmetrize}")
    print(f"[ladder] gammas = {gammas}")

    # Continuation start: no-learning at the first gamma; we'll reseed
    # explicitly in the loop if seed=='no-learning'.
    P = init_no_learning(u_grid, tau_vec,
                         np.full(4, gammas[0], dtype=np.float64),
                         W_vec, cara=False)

    rows = []  # one dict per gamma
    t_total = time.perf_counter()

    for k, g in enumerate(gammas):
        gamma_vec = np.full(4, g, dtype=np.float64)
        scfg = SolverConfig(method="anderson", max_iters=args.max_iters,
                            tol=args.tol, anderson_m=args.anderson_m,
                            symmetrize=True, verbose=False, checkpoint_every=0)

        # Branch selection: reseed from no-learning at this gamma if asked.
        if args.seed == "no-learning":
            P = init_no_learning(u_grid, tau_vec, gamma_vec, W_vec,
                                 cara=False)

        deficit_seed = revelation_deficit_f128(P, u_grid, tau_vec, 4)

        def phi_fn(P_arr: np.ndarray, _g=gamma_vec) -> np.ndarray:
            P_phi = phi_K4(P_arr, u_grid, tau_vec, _g, W_vec, cara=False)
            if args.f128_symmetrize:
                # Override the f64 symmetrise: solver will symmetrise again
                # in f64, but our acc-in-f128 step removes one layer of noise.
                P_phi = symmetrize_f128(P_phi)
            return P_phi

        t0 = time.perf_counter()
        P_new, hist = solve(phi_fn, P, scfg)
        dt = time.perf_counter() - t0

        deficit_64 = revelation_deficit(P_new, u_grid, tau_vec, 4)
        deficit_128 = revelation_deficit_f128(P_new, u_grid, tau_vec, 4)
        row = {
            "gamma": g,
            "iters": len(hist),
            "residual": float(hist[-1]),
            "deficit_seed_f128": float(deficit_seed),
            "deficit_f64": float(deficit_64),
            "deficit_f128": float(deficit_128),
            "time_s": dt,
            "p_min": float(P_new.min()),
            "p_max": float(P_new.max()),
            "p_mean": float(P_new.mean()),
        }
        rows.append(row)
        print(f"[ladder] gamma={g:6.2f}  iters={len(hist):3d}  "
              f"res={hist[-1]:.2e}  seed={deficit_seed:.4e} -> "
              f"f64={deficit_64:.4e}  f128={deficit_128:.4e}  ({dt:.1f}s)")

        np.savez_compressed(args.output_dir / f"{base}_g{g:g}.npz",
                            P=P_new, history=hist, gamma=g, tau=args.tau,
                            G=args.G, deficit_f64=deficit_64,
                            deficit_f128=deficit_128)

        P = P_new  # if seed=='continuation' this is the next seed

    print(f"[ladder] total wall time: "
          f"{time.perf_counter() - t_total:.1f}s")

    # JSON summary
    with summary_path.open("w") as f:
        json.dump({"K": 4, "G": args.G, "tau": args.tau, "rows": rows},
                  f, indent=2)
    print(f"[ladder] wrote {summary_path}")

    # Plain-text table for quick eyeballing
    with table_path.open("w") as f:
        f.write(f"# K=4 G={args.G} tau={args.tau} seed={args.seed} "
                f"f128_sym={args.f128_symmetrize}\n")
        f.write(f"# {'gamma':>8s} {'iters':>5s} {'residual':>12s} "
                f"{'seed_f128':>14s} {'1-R^2_f64':>14s} "
                f"{'1-R^2_f128':>14s} {'time_s':>8s}\n")
        for r in rows:
            f.write(f"  {r['gamma']:8.4f} {r['iters']:5d} "
                    f"{r['residual']:12.4e} "
                    f"{r['deficit_seed_f128']:14.6e} "
                    f"{r['deficit_f64']:14.6e} {r['deficit_f128']:14.6e} "
                    f"{r['time_s']:8.2f}\n")
    print(f"[ladder] wrote {table_path}")

    # Monotonicity check (informational, not assertive: numerics may wobble)
    deficits = [r["deficit_f128"] for r in rows]
    monotone = all(deficits[i] <= deficits[i + 1] + 1e-6
                   for i in range(len(deficits) - 1))
    print(f"[ladder] deficit (f128) monotone-increasing as gamma falls: "
          f"{monotone}")


if __name__ == "__main__":
    main()
