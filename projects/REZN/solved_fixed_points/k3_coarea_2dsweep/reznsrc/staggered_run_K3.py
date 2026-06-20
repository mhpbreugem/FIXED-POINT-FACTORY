"""Staggered-halo Newton-Krylov runner for K=3 het gamma + het tau.

K=3 is the paper's main case. Every Phi evaluation is roughly two
orders of magnitude cheaper than K=4, so we can run on a finer inner
grid (G_inner up to ~16) at modest wall-clock cost.

Default extreme-opposed configuration mirrors the paper's strongest
no-learning PR mechanism: low-gamma agent paired with low-tau (the
endogenous noise trader).

Usage:

    taskset -c 2,3 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
        python -m code.staggered_run_K3 --G-inner 12 --pad 4 \
            --gammas 0.25,1,10 --taus 0.25,1,10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .config import DTYPE
from .contour_K3_halo import (init_no_learning_K3, phi_K3_halo,
                              phi_K3_halo_cubic, phi_K3_halo_smooth)
from .f128 import revelation_deficit_f128
from .halo import extract_inner, replace_inner
from .staggered import staggered_solve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staggered halo K=3 het.")
    p.add_argument("--G-inner", type=int, default=12)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--u-inner-max", type=float, default=3.0)
    p.add_argument("--u-outer-max", type=float, default=6.0)
    p.add_argument("--gammas", type=str, default="0.25,1,10")
    p.add_argument("--taus", type=str, default="0.25,1,10")
    p.add_argument("--Ws", type=str, default="1,1,1")
    p.add_argument("--max-stages", type=int, default=6)
    p.add_argument("--stage-tol", type=float, default=1.0e-3)
    p.add_argument("--inner-tol", type=float, default=1.0e-7)
    p.add_argument("--inner-max-iter", type=int, default=30)
    p.add_argument("--inner-method", choices=("lgmres", "gmres", "bicgstab"),
                   default="lgmres")
    p.add_argument("--inner-maxiter", type=int, default=80)
    p.add_argument("--outer-k", type=int, default=40)
    p.add_argument("--rdiff", type=float, default=1.0e-4)
    p.add_argument("--presmooth-steps", type=int, default=15)
    p.add_argument("--presmooth-alpha", type=float, default=0.05)
    p.add_argument("--halo-update", choices=("no_learning", "extrapolate"),
                   default="no_learning")
    p.add_argument("--kernel", choices=("scan", "cubic", "smooth"),
                   default="smooth",
                   help="contour-evidence kernel: scan (linear interp), "
                        "cubic (Hermite interp), smooth (Gaussian K_h)")
    p.add_argument("--kernel-h", type=float, default=0.0,
                   help="Gaussian bandwidth for --kernel=smooth; "
                        "0 means auto = du / (2*sqrt(2)) where du is "
                        "the inner-grid spacing")
    p.add_argument("--heartbeat-s", type=float, default=60.0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/staggered_K3"))
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def parse_vec(s: str) -> np.ndarray:
    parts = [float(t) for t in s.split(",") if t.strip()]
    if len(parts) != 3:
        raise SystemExit(f"K=3 expects 3 entries, got {len(parts)}: {s}")
    return np.array(parts, dtype=DTYPE)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gamma_vec = parse_vec(args.gammas)
    tau_vec = parse_vec(args.taus)
    W_vec = parse_vec(args.Ws)

    G_inner = args.G_inner
    pad = args.pad
    G_full = G_inner + 2 * pad

    du = (2.0 * args.u_inner_max) / (G_inner - 1) if G_inner > 1 else 1.0
    u_full = np.empty(G_full, dtype=DTYPE)
    inner_lo = pad
    inner_hi = pad + G_inner
    for q in range(G_full):
        u_full[q] = -args.u_inner_max + (q - pad) * du
    u_grid_inner = u_full[inner_lo:inner_hi].copy()

    config_str = (f"Gi{G_inner}_pad{pad}"
                  f"_g{'-'.join(f'{g:g}' for g in gamma_vec)}"
                  f"_t{'-'.join(f'{t:g}' for t in tau_vec)}"
                  f"_{args.kernel}")
    tag = f"_{args.tag}" if args.tag else ""
    base = f"K3_staggered_{config_str}{tag}"

    print("=" * 78)
    print("Staggered halo K=3 het")
    print(f"  G_inner = {G_inner}  inner cells per axis")
    print(f"  pad     = {pad}     halo cells per side per axis")
    print(f"  G_full  = {G_full}    total cells per axis")
    print(f"  inner u in [{u_full[inner_lo]:.3f}, {u_full[inner_hi-1]:.3f}]"
          f"   (du={du:.3f})")
    print(f"  full  u in [{u_full[0]:.3f}, {u_full[-1]:.3f}]")
    print(f"  inner cells = {G_inner**3}, full cells = {G_full**3}")
    print(f"  gammas = {list(gamma_vec)}")
    print(f"  taus   = {list(tau_vec)}")
    print(f"  Ws     = {list(W_vec)}")
    print(f"  Stages: max={args.max_stages}  stage_tol={args.stage_tol}")
    print(f"  Inner: {args.inner_method}  inner_max_iter={args.inner_max_iter}"
          f"  inner_tol={args.inner_tol}")
    print(f"  Inner Krylov: maxiter={args.inner_maxiter}  outer_k={args.outer_k}"
          f"  rdiff={args.rdiff}")
    print(f"  Presmooth: {args.presmooth_steps} steps alpha={args.presmooth_alpha}")
    print(f"  Halo update: {args.halo_update}")
    print(f"  Kernel: {args.kernel}", end="")
    if args.kernel == "smooth":
        print(f"  (kernel_h = {args.kernel_h or 'auto'})", end="")
    print()
    print(f"  Heartbeat every {args.heartbeat_s}s")
    print("=" * 78)

    print("[seed] building no-learning halo over full grid ...", flush=True)
    t0 = time.perf_counter()
    halo = init_no_learning_K3(u_full, tau_vec, gamma_vec, W_vec)
    print(f"[seed] halo built in {time.perf_counter() - t0:.2f}s", flush=True)

    P_inner_seed = extract_inner(halo, inner_lo, inner_hi)

    # Bandwidth for smooth kernel.
    # Heuristic: the kernel should localise on a width comparable to one
    # grid cell along the steepest price gradient. Typical price slope on
    # a binary asset with K agents is |dP/du| ~ 0.1 per u-unit, so the
    # price difference between adjacent cells is roughly 0.1 * du. We use
    # h ~ a fraction of that so that 2-3 cells contribute on each side
    # of the level set, giving a smooth but localised kernel.
    kernel_h = args.kernel_h
    if args.kernel == "smooth" and kernel_h <= 0.0:
        kernel_h = 0.05 * du
        if kernel_h < 0.005:
            kernel_h = 0.005
    print(f"[seed] kernel={args.kernel}"
          + (f" kernel_h={kernel_h:.4f}" if args.kernel == "smooth" else ""),
          flush=True)

    if args.kernel == "scan":
        def phi_full_fn(P_full: np.ndarray) -> np.ndarray:
            return phi_K3_halo(P_full, u_full, inner_lo, inner_hi,
                               tau_vec, gamma_vec, W_vec)
    elif args.kernel == "cubic":
        def phi_full_fn(P_full: np.ndarray) -> np.ndarray:
            return phi_K3_halo_cubic(P_full, u_full, inner_lo, inner_hi,
                                     tau_vec, gamma_vec, W_vec)
    elif args.kernel == "smooth":
        def phi_full_fn(P_full: np.ndarray) -> np.ndarray:
            return phi_K3_halo_smooth(P_full, u_full, inner_lo, inner_hi,
                                      tau_vec, gamma_vec, W_vec, kernel_h)
    else:
        raise SystemExit(f"unknown kernel {args.kernel}")

    print("[seed] timing one Phi evaluation on padded grid ...", flush=True)
    P_full = replace_inner(halo, P_inner_seed, inner_lo, inner_hi)
    t0 = time.perf_counter()
    _ = phi_full_fn(P_full)
    print(f"[seed] one Phi on padded grid: "
          f"{time.perf_counter() - t0:.2f}s", flush=True)

    print("[run] starting staggered solve ...", flush=True)
    t0 = time.perf_counter()
    P_inner_final, history = staggered_solve(
        phi_full_fn, u_full, inner_lo, inner_hi,
        u_grid_inner=u_grid_inner, tau_vec=tau_vec, K=3,
        halo_initial=halo, inner_initial=P_inner_seed,
        max_stages=args.max_stages, stage_tol=args.stage_tol,
        inner_method=args.inner_method,
        inner_max_iter=args.inner_max_iter, inner_tol=args.inner_tol,
        inner_outer_k=args.outer_k,
        inner_inner_maxiter=args.inner_maxiter,
        inner_rdiff=args.rdiff,
        presmooth_steps=args.presmooth_steps,
        presmooth_alpha=args.presmooth_alpha,
        halo_update=args.halo_update,
        heartbeat_s=args.heartbeat_s,
    )
    t_solve = time.perf_counter() - t0

    P_full_final = replace_inner(halo, P_inner_final, inner_lo, inner_hi)
    F_full = phi_full_fn(P_full_final) - P_full_final
    F_inner = extract_inner(F_full, inner_lo, inner_hi)
    F_inf = float(np.max(np.abs(F_inner)))
    d_final = revelation_deficit_f128(P_inner_final, u_grid_inner, tau_vec, 3)

    print("=" * 78)
    print(f"[done] wall time {t_solve:.1f}s  stages={len(history.stages) - 1}")
    print(f"[done] inner ||F||inf = {F_inf:.4e}")
    print(f"[done] 1-R^2_f128 = {d_final:.6e}")
    print("=" * 78)

    npz_path = args.output_dir / f"{base}.npz"
    np.savez_compressed(
        npz_path,
        P_inner=P_inner_final, halo=halo, P_full=P_full_final,
        u_full=u_full, u_grid_inner=u_grid_inner,
        gamma_vec=gamma_vec, tau_vec=tau_vec, W_vec=W_vec,
        K=3, G_inner=G_inner, pad=pad, G_full=G_full,
        deficit_final=d_final, F_inner_inf_final=F_inf,
        elapsed_s=np.asarray([r.elapsed_s for r in history.stages]),
        stage_F_inf=np.asarray([r.F_inner_inf for r in history.stages]),
        stage_deficit=np.asarray([r.deficit_f128 for r in history.stages]),
        stage_drift=np.asarray([r.inner_drift_inf for r in history.stages]),
        stage_phi_calls=np.asarray([r.phi_calls for r in history.stages]),
    )
    print(f"[done] wrote {npz_path}")

    log_path = args.output_dir / f"{base}.log"
    with log_path.open("w") as f:
        f.write(f"# Staggered halo K=3 het\n")
        f.write(f"# gammas = {list(gamma_vec)}\n")
        f.write(f"# taus   = {list(tau_vec)}\n")
        f.write(f"# G_inner={G_inner} pad={pad} G_full={G_full}\n")
        f.write(f"# inner u in [{u_full[inner_lo]:.3f}, "
                f"{u_full[inner_hi-1]:.3f}]\n")
        f.write(f"# full  u in [{u_full[0]:.3f}, {u_full[-1]:.3f}]\n")
        f.write(f"# halo_update = {args.halo_update}\n")
        f.write(f"# wall time = {t_solve:.1f}s\n")
        f.write(f"# final 1-R^2_f128 = {d_final:.6e}\n")
        f.write(f"# final ||F_inner||inf = {F_inf:.6e}\n\n")
        f.write(f"# stage  elapsed_s   F_inner_inf  deficit_f128  drift_inf  "
                f"phi_calls\n")
        for r in history.stages:
            f.write(f"  {r.stage:5d}  {r.elapsed_s:9.2f}  "
                    f"{r.F_inner_inf:11.4e}  {r.deficit_f128:12.6e}  "
                    f"{r.inner_drift_inf:9.4e}  {r.phi_calls:9d}\n")
    print(f"[done] wrote {log_path}")


if __name__ == "__main__":
    main()
