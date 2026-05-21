"""Entry point for K=4 contour-method experiments.

Usage (run pinned to one core):

    taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        python -m code.run --G 10 --gamma 0.5 --tau 2.0

Outputs an .npz checkpoint and a small text summary in --output-dir.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from .config import DTYPE, Config, SolverConfig
from .contour_K4 import init_no_learning, phi_K4
from .metrics import revelation_deficit, summary
from .solver import solve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K=4 contour-method REE solver.")
    p.add_argument("--G", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--W", type=float, default=1.0)
    p.add_argument("--u-min", type=float, default=-4.0)
    p.add_argument("--u-max", type=float, default=+4.0)
    p.add_argument("--cara", action="store_true",
                   help="use CARA demand (alpha = gamma)")
    p.add_argument("--solver", choices=("picard", "anderson"),
                   default="anderson")
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--tol", type=float, default=1.0e-7)
    p.add_argument("--damping", type=float, default=0.3)
    p.add_argument("--anderson-m", type=int, default=8)
    p.add_argument("--no-symmetrize", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--tag", type=str, default="",
                   help="optional suffix for the output filename")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(K=4, G=args.G, gamma=args.gamma, tau=args.tau, W=args.W,
                 u_min=args.u_min, u_max=args.u_max, cara=args.cara)
    scfg = SolverConfig(method=args.solver, max_iters=args.max_iters,
                        tol=args.tol, damping=args.damping,
                        anderson_m=args.anderson_m,
                        symmetrize=not args.no_symmetrize,
                        checkpoint_every=args.checkpoint_every,
                        verbose=not args.quiet)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    label = ("CARA" if cfg.cara else f"CRRA-g{cfg.gamma}")
    base = f"K4_G{cfg.G}_{label}_t{cfg.tau}{tag}"
    npz_path = args.output_dir / f"{base}.npz"
    log_path = args.output_dir / f"{base}.log"

    u_grid = cfg.u_grid()
    tau_vec = cfg.tau_vec()
    gamma_vec = cfg.gamma_vec()
    W_vec = cfg.W_vec()

    print(f"[run] K={cfg.K} G={cfg.G} gamma={cfg.gamma} tau={cfg.tau} "
          f"cara={cfg.cara} solver={scfg.method} tol={scfg.tol}")
    print(f"[run] grid points: {cfg.G ** cfg.K} (= G^K)")

    t0 = time.perf_counter()
    P0 = init_no_learning(u_grid, tau_vec, gamma_vec, W_vec, cfg.cara)
    t_init = time.perf_counter() - t0
    print(f"[run] no-learning init done in {t_init:.2f}s")

    deficit0 = revelation_deficit(P0, u_grid, tau_vec, cfg.K)
    print(f"[run] no-learning 1-R^2 = {deficit0:.6e}")

    def phi_fn(P: np.ndarray) -> np.ndarray:
        return phi_K4(P, u_grid, tau_vec, gamma_vec, W_vec, cfg.cara)

    def checkpoint(n: int, P: np.ndarray) -> None:
        ckpt = args.output_dir / f"{base}.ckpt.npz"
        np.savez_compressed(ckpt, P=P, iter=n, gamma=cfg.gamma, tau=cfg.tau,
                            G=cfg.G, K=cfg.K, cara=cfg.cara)

    t0 = time.perf_counter()
    P, hist = solve(phi_fn, P0, scfg, checkpoint=checkpoint)
    t_solve = time.perf_counter() - t0
    print(f"[run] solver finished in {t_solve:.2f}s, "
          f"{len(hist)} iterations, final residual {hist[-1]:.3e}")

    deficit, pmin, pmax, pmean = summary(P, u_grid, tau_vec, cfg.K)
    print(f"[run] converged 1-R^2 = {deficit:.6e}")
    print(f"[run] price range  [{pmin:.4f}, {pmax:.4f}]  mean = {pmean:.4f}")

    np.savez_compressed(npz_path, P=P, P0=P0, history=hist,
                        u_grid=u_grid, tau_vec=tau_vec, gamma_vec=gamma_vec,
                        W_vec=W_vec, cara=cfg.cara, K=cfg.K, G=cfg.G,
                        gamma=cfg.gamma, tau=cfg.tau,
                        deficit=deficit, deficit_init=deficit0,
                        time_init=t_init, time_solve=t_solve)
    with log_path.open("w") as f:
        f.write(f"K={cfg.K} G={cfg.G} gamma={cfg.gamma} tau={cfg.tau} "
                f"cara={cfg.cara}\n")
        f.write(f"solver={scfg.method} max_iters={scfg.max_iters} "
                f"tol={scfg.tol}\n")
        f.write(f"iters_used={len(hist)}\n")
        f.write(f"residual_final={hist[-1]:.6e}\n")
        f.write(f"deficit_init={deficit0:.6e}\n")
        f.write(f"deficit_final={deficit:.6e}\n")
        f.write(f"price_min={pmin:.6e}\n")
        f.write(f"price_max={pmax:.6e}\n")
        f.write(f"price_mean={pmean:.6e}\n")
        f.write(f"time_init_s={t_init:.3f}\n")
        f.write(f"time_solve_s={t_solve:.3f}\n")
        f.write("\nresidual history:\n")
        for n, r in enumerate(hist, 1):
            f.write(f"  {n:3d}  {r:.6e}\n")
    print(f"[run] wrote {npz_path}")
    print(f"[run] wrote {log_path}")


if __name__ == "__main__":
    main()
