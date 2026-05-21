"""Newton-Krylov from a heterogeneous-(gamma,tau) no-learning seed.

Picks the no-learning equilibrium for the user-specified (gamma_k, tau_k)
quartet as the initial point — this is the most natural seed for finding
a partial-revelation fixed point of Phi, because it is itself
partially revealing by Jensen's inequality and sits as far from the FR
no-trade fixed point Lambda(T*) as the no-learning manifold permits.

Usage (default = the K=4 endogenous-noise-trader configuration):

    taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        python -m code.newton_run --G 8 \
            --gammas 0.25,1,3,10 --taus 0.25,1,3,10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import DTYPE
from .contour_K4 import init_no_learning, phi_K4
from .contour_K4_het import init_no_learning_het, phi_K4_het
from .f128 import revelation_deficit_f128
from .newton import newton_krylov_solve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Newton-Krylov K=4 het-gamma het-tau.")
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--gammas", type=str, default="0.25,1,3,10",
                   help="comma-separated gamma_k for k=0..3")
    p.add_argument("--taus", type=str, default="0.25,1,3,10",
                   help="comma-separated tau_k for k=0..3")
    p.add_argument("--Ws", type=str, default="1,1,1,1",
                   help="comma-separated wealth W_k")
    p.add_argument("--u-min", type=float, default=-4.0)
    p.add_argument("--u-max", type=float, default=+4.0)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1.0e-7)
    p.add_argument("--inner-method", choices=("lgmres", "gmres", "bicgstab"),
                   default="lgmres")
    p.add_argument("--inner-maxiter", type=int, default=60)
    p.add_argument("--outer-k", type=int, default=30,
                   help="LGMRES outer-vector memory depth")
    p.add_argument("--rdiff", type=float, default=1.0e-4,
                   help="FD eps for Jacobian-vector products; bigger absorbs "
                        "contour-kink noise")
    p.add_argument("--presmooth-steps", type=int, default=10,
                   help="Picard pre-smoothing steps (set 0 to skip)")
    p.add_argument("--presmooth-alpha", type=float, default=0.05)
    p.add_argument("--heartbeat-s", type=float, default=15.0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("output/newton"))
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def parse_vec(s: str) -> np.ndarray:
    parts = [float(t) for t in s.split(",") if t.strip()]
    if len(parts) != 4:
        raise SystemExit(f"expected 4 entries, got {len(parts)}: {s}")
    return np.array(parts, dtype=DTYPE)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gamma_vec = parse_vec(args.gammas)
    tau_vec = parse_vec(args.taus)
    W_vec = parse_vec(args.Ws)

    u_grid = np.linspace(args.u_min, args.u_max, args.G, dtype=DTYPE)

    config_str = (f"G{args.G}_g{'-'.join(f'{g:g}' for g in gamma_vec)}"
                  f"_t{'-'.join(f'{t:g}' for t in tau_vec)}")
    tag = f"_{args.tag}" if args.tag else ""
    base = f"K4_{config_str}_newton{tag}"

    print("="*72)
    print("Newton-Krylov K=4 heterogeneous gamma + tau")
    print(f"  G       = {args.G}   (grid points = {args.G**4})")
    print(f"  gammas  = {list(gamma_vec)}")
    print(f"  taus    = {list(tau_vec)}")
    print(f"  Ws      = {list(W_vec)}")
    print(f"  u in   [{args.u_min}, {args.u_max}]")
    print(f"  Newton: max_iter={args.max_iter} tol={args.tol}")
    print(f"  Inner : method={args.inner_method} maxiter={args.inner_maxiter} "
          f"outer_k={args.outer_k}")
    print(f"  rdiff   = {args.rdiff}")
    print(f"  Picard pre-smooth: {args.presmooth_steps} steps "
          f"alpha={args.presmooth_alpha}")
    print(f"  heartbeat every {args.heartbeat_s}s")
    print("="*72)

    # Seed: no-learning equilibrium (partial revelation under non-CARA).
    print("[seed] computing no-learning equilibrium ...", flush=True)
    t0 = time.perf_counter()
    P0 = init_no_learning_het(u_grid, tau_vec, gamma_vec, W_vec)
    t_init = time.perf_counter() - t0
    d0 = revelation_deficit_f128(P0, u_grid, tau_vec, 4)
    print(f"[seed] no-learning P built in {t_init:.2f}s; "
          f"1-R^2_f128 = {d0:.6e}", flush=True)

    # Time one Phi evaluation for cost estimation.
    print("[seed] timing one Phi evaluation ...", flush=True)
    t0 = time.perf_counter()
    _ = phi_K4_het(P0, u_grid, tau_vec, gamma_vec, W_vec)
    t_phi = time.perf_counter() - t0
    print(f"[seed] Phi evaluation: {t_phi:.2f}s", flush=True)

    def phi_fn(P: np.ndarray) -> np.ndarray:
        # No symmetrisation: heterogeneous agents break S_4 symmetry.
        return phi_K4_het(P, u_grid, tau_vec, gamma_vec, W_vec)

    print("[newton] starting Newton-Krylov ...", flush=True)
    t0 = time.perf_counter()
    P, hist = newton_krylov_solve(
        phi_fn, P0, u_grid, tau_vec, K=4,
        max_iter=args.max_iter, tol=args.tol,
        method=args.inner_method,
        inner_maxiter=args.inner_maxiter, outer_k=args.outer_k,
        rdiff=args.rdiff,
        presmooth_steps=args.presmooth_steps,
        presmooth_alpha=args.presmooth_alpha,
        heartbeat_s=args.heartbeat_s,
    )
    t_solve = time.perf_counter() - t0

    d_final = revelation_deficit_f128(P, u_grid, tau_vec, 4)
    F_final = float(np.max(np.abs(phi_fn(P) - P)))
    print("="*72)
    print(f"[done] wall time {t_solve:.1f}s  phi_calls={hist.phi_calls[-1]}")
    print(f"[done] ||F||inf = {F_final:.4e}")
    print(f"[done] 1-R^2_f128 = {d_final:.6e}  "
          f"(seed was {d0:.6e}, ratio {d_final/d0:.4f}x)")
    print("="*72)

    # Save artifacts
    npz_path = args.output_dir / f"{base}.npz"
    np.savez_compressed(npz_path, P=P, P0=P0,
                        gamma_vec=gamma_vec, tau_vec=tau_vec, W_vec=W_vec,
                        u_grid=u_grid, K=4, G=args.G,
                        deficit_init=d0, deficit_final=d_final,
                        F_inf_final=F_final,
                        elapsed_s=np.asarray(hist.elapsed_s),
                        F_inf_history=np.asarray(hist.F_inf),
                        deficit_history=np.asarray(hist.deficit_f128),
                        phi_calls=np.asarray(hist.phi_calls))
    print(f"[done] wrote {npz_path}")

    log_path = args.output_dir / f"{base}.log"
    with log_path.open("w") as f:
        f.write(f"# Newton-Krylov K=4 het-gamma het-tau\n")
        f.write(f"# gammas = {list(gamma_vec)}\n")
        f.write(f"# taus   = {list(tau_vec)}\n")
        f.write(f"# G={args.G}  u in [{args.u_min}, {args.u_max}]\n")
        f.write(f"# seed deficit = {d0:.6e}\n")
        f.write(f"# final deficit = {d_final:.6e}\n")
        f.write(f"# final ||F||inf = {F_final:.6e}\n")
        f.write(f"# wall time = {t_solve:.1f}s\n\n")
        f.write(f"# {'label':>14s} {'elapsed_s':>10s} {'F_inf':>14s} "
                f"{'deficit_f128':>14s} {'phi_calls':>10s}\n")
        for i in range(len(hist.label)):
            f.write(f"  {hist.label[i]:>14s} "
                    f"{hist.elapsed_s[i]:10.2f} "
                    f"{hist.F_inf[i]:14.6e} "
                    f"{hist.deficit_f128[i]:14.6e} "
                    f"{hist.phi_calls[i]:10d}\n")
    print(f"[done] wrote {log_path}")


if __name__ == "__main__":
    main()
