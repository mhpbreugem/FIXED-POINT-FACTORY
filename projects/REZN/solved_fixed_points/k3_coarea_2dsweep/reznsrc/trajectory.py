"""Trajectory ladder for K=4: 1-R^2 along the Picard iterate at each gamma.

Records the deficit (in float128) every `record_every` iterations from
the no-learning seed. The point is to expose the convergence trajectory
of Phi to the paper's "PR vs FR branch" question rather than just
reporting a (possibly mid-flight) final number.

Usage:

    taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        python -m code.trajectory --G 8 --tau 2.0 \
            --max-iters 300 --damping 0.05
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .contour_K4 import init_no_learning, phi_K4, residual_inf
from .f128 import revelation_deficit_f128, symmetrize_f128


DEFAULT_GAMMAS = [
    100.0, 50.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0,
    0.7, 0.5, 0.35, 0.25,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K=4 trajectory ladder.")
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--gammas", type=str, default="")
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--damping", type=float, default=0.05)
    p.add_argument("--record-every", type=int, default=10)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/trajectory"))
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
    base = f"K4_G{args.G}_tau{args.tau}{tag}_traj"

    cfg_template = Config(K=4, G=args.G, gamma=gammas[0], tau=args.tau,
                          cara=False)
    u_grid = cfg_template.u_grid()
    tau_vec = cfg_template.tau_vec()
    W_vec = cfg_template.W_vec()

    print(f"[traj] K=4 G={args.G} tau={args.tau} damping={args.damping} "
          f"max_iters={args.max_iters}")
    print(f"[traj] {len(gammas)} gammas: {gammas}")

    all_results = []
    t_total = time.perf_counter()

    for g in gammas:
        gamma_vec = np.full(4, g, dtype=np.float64)
        P = init_no_learning(u_grid, tau_vec, gamma_vec, W_vec, cara=False)

        seed_def = revelation_deficit_f128(P, u_grid, tau_vec, 4)
        iters_record = [0]
        residuals = [float("nan")]
        deficits = [seed_def]

        t0 = time.perf_counter()
        last_residual = float("inf")
        for n in range(1, args.max_iters + 1):
            P_phi = symmetrize_f128(phi_K4(P, u_grid, tau_vec, gamma_vec,
                                           W_vec, cara=False))
            r = residual_inf(P, P_phi)
            P = (1.0 - args.damping) * P + args.damping * P_phi
            np.clip(P, 1.0e-12, 1.0 - 1.0e-12, out=P)
            last_residual = r
            if n % args.record_every == 0 or n == args.max_iters:
                d = revelation_deficit_f128(P, u_grid, tau_vec, 4)
                iters_record.append(n)
                residuals.append(r)
                deficits.append(d)
        dt = time.perf_counter() - t0

        final_def = deficits[-1]
        # crude detection of monotone decay (PR drifting to FR)
        decay_ratio = final_def / seed_def if seed_def > 0 else float("nan")
        print(f"[traj] gamma={g:7.3f}  "
              f"seed={seed_def:.4e} -> final={final_def:.4e}  "
              f"({decay_ratio:.3f}x of seed)  "
              f"residual {last_residual:.3e}  ({dt:.1f}s)")

        all_results.append({
            "gamma": g,
            "iters": iters_record,
            "residuals": residuals,
            "deficits_f128": deficits,
            "seed_deficit": seed_def,
            "final_deficit": final_def,
            "decay_ratio": decay_ratio,
            "time_s": dt,
        })

    print(f"[traj] total wall time {time.perf_counter() - t_total:.1f}s")

    json_path = args.output_dir / f"{base}.json"
    with json_path.open("w") as f:
        json.dump({"K": 4, "G": args.G, "tau": args.tau,
                   "damping": args.damping, "max_iters": args.max_iters,
                   "rows": all_results}, f, indent=2)
    print(f"[traj] wrote {json_path}")

    # Compact text table
    txt_path = args.output_dir / f"{base}.txt"
    with txt_path.open("w") as f:
        f.write(f"# K=4 G={args.G} tau={args.tau} damping={args.damping} "
                f"iters={args.max_iters}\n")
        f.write(f"# {'gamma':>8s} {'seed':>13s} {'final':>13s} "
                f"{'final/seed':>10s} {'residual':>12s}\n")
        for r in all_results:
            f.write(f"  {r['gamma']:8.3f} {r['seed_deficit']:13.6e} "
                    f"{r['final_deficit']:13.6e} {r['decay_ratio']:10.4f} "
                    f"{r['residuals'][-1]:12.4e}\n")
    print(f"[traj] wrote {txt_path}")


if __name__ == "__main__":
    main()
