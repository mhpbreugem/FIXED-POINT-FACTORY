"""Staggered (level-k continuation) solver for the K=4 halo problem.

Stage 0: P_full = no-learning everywhere. P_inner = no-learning.
Stage 1: halo = no-learning (frozen). Solve inner = Phi_halo(inner).
Stage s>=2: halo = update_rule(P_inner_{s-1}). Solve inner = Phi_halo(inner).

At each stage, the inner solve is a fixed-point problem with a fixed
boundary condition. We use scipy.optimize.newton_krylov as the inner
solver, optionally preceded by a few Picard pre-smooth steps.

Because the halo at stage s is informed only by stage s-1, the FR
no-trade fixed point Lambda(T*) is NOT a fixed point at any stage with
a non-FR halo. The equilibrium-selection burden is carried by the
halo's mismatch with FR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from .config import DTYPE
from .demand import EPS_PRICE
from .f128 import revelation_deficit_f128
from .halo import (extract_inner, extrapolated_halo, no_learning_halo,
                   replace_inner)


PhiFullFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class StageRecord:
    stage: int
    phi_calls: int
    F_inner_inf: float
    deficit_f128: float
    inner_drift_inf: float       # ||P_inner_{s} - P_inner_{s-1}||_inf
    elapsed_s: float


@dataclass
class StaggeredHistory:
    stages: List[StageRecord] = field(default_factory=list)
    # within-stage trace (for debugging)
    inner_trace: List[dict] = field(default_factory=list)


def _safe_clip(P: np.ndarray) -> np.ndarray:
    return np.clip(P, EPS_PRICE, 1.0 - EPS_PRICE)


def _picard_inner(phi_full: PhiFullFn, halo: np.ndarray,
                  P_inner: np.ndarray, inner_lo: int, inner_hi: int,
                  alpha: float, n_steps: int,
                  hb: dict, heartbeat_s: float,
                  history: StaggeredHistory,
                  u_grid: np.ndarray, tau_vec: np.ndarray,
                  K: int) -> np.ndarray:
    for n in range(1, n_steps + 1):
        P_full = replace_inner(halo, P_inner, inner_lo, inner_hi)
        P_full_new = phi_full(P_full)
        hb["phi_calls"] += 1
        P_inner_new = extract_inner(P_full_new, inner_lo, inner_hi)
        F_inner = P_inner - P_inner_new
        F_inf = float(np.max(np.abs(F_inner)))
        P_inner = _safe_clip((1.0 - alpha) * P_inner + alpha * P_inner_new)

        now = time.perf_counter()
        if now - hb["last_hb"] >= heartbeat_s or n == n_steps:
            d = revelation_deficit_f128(P_inner, u_grid, tau_vec, K)
            elapsed = now - hb["t0"]
            print(f"[t={elapsed:7.0f}s] stage={hb['stage']:2d} "
                  f"presmooth iter={n:3d}  ||F_inner||inf={F_inf:.4e}  "
                  f"1-R^2_f128={d:.6e}  phi_calls={hb['phi_calls']}",
                  flush=True)
            history.inner_trace.append({
                "stage": hb["stage"],
                "kind": "picard",
                "iter": n,
                "F_inf": F_inf,
                "deficit_f128": d,
                "elapsed_s": elapsed,
                "phi_calls": hb["phi_calls"],
            })
            hb["last_hb"] = now
    return P_inner


def staggered_solve(
        phi_full: PhiFullFn,
        u_full: np.ndarray, inner_lo: int, inner_hi: int,
        u_grid_inner: np.ndarray,            # for deficit regression
        tau_vec: np.ndarray,
        K: int,
        halo_initial: np.ndarray,
        inner_initial: np.ndarray,
        *,
        max_stages: int = 6,
        stage_tol: float = 1.0e-3,
        # inner-solver controls
        inner_method: str = "lgmres",
        inner_max_iter: int = 30,
        inner_tol: float = 1.0e-7,
        inner_outer_k: int = 30,
        inner_inner_maxiter: int = 60,
        inner_rdiff: float = 1.0e-4,
        presmooth_steps: int = 10,
        presmooth_alpha: float = 0.05,
        halo_update: str = "no_learning",   # "no_learning" or "extrapolate"
        heartbeat_s: float = 60.0,
        ) -> tuple[np.ndarray, StaggeredHistory]:
    """Run the staggered scheme.

    Returns (P_inner_final, history).
    """
    G_full = u_full.size
    G_inner = inner_hi - inner_lo
    n_inner = G_inner ** K

    # Halo for stage 1 = the seed halo (caller computed no-learning halo)
    halo = halo_initial.astype(DTYPE, copy=True)
    P_inner = inner_initial.astype(DTYPE, copy=True)

    history = StaggeredHistory()
    hb = {
        "t0": time.perf_counter(),
        "last_hb": time.perf_counter(),
        "phi_calls": 0,
        "stage": 0,
        "newton_iter": 0,
    }

    deficit0 = revelation_deficit_f128(P_inner, u_grid_inner, tau_vec, K)
    print(f"[t=      0s] stage 0 (seed): "
          f"1-R^2_f128={deficit0:.6e}  phi_calls=0", flush=True)
    history.stages.append(StageRecord(
        stage=0, phi_calls=0, F_inner_inf=float("nan"),
        deficit_f128=deficit0, inner_drift_inf=float("nan"),
        elapsed_s=0.0,
    ))

    for s in range(1, max_stages + 1):
        hb["stage"] = s
        P_inner_prev = P_inner.copy()

        # ------- update halo for this stage -------
        if s == 1:
            # halo unchanged, already no-learning
            pass
        elif halo_update == "no_learning":
            # static halo across all stages
            pass
        elif halo_update == "extrapolate":
            # build extrapolated halo from the previous full P
            P_full_prev = replace_inner(halo, P_inner_prev, inner_lo, inner_hi)
            halo = extrapolated_halo(P_full_prev, inner_lo, inner_hi)
        else:
            raise ValueError(f"unknown halo_update={halo_update}")

        # ------- inner solve -------
        elapsed = time.perf_counter() - hb["t0"]
        print(f"[t={elapsed:7.0f}s] === STAGE {s} === "
              f"halo_update={halo_update}  presmooth={presmooth_steps}  "
              f"newton_max={inner_max_iter}", flush=True)
        hb["last_hb"] = time.perf_counter() - heartbeat_s   # force a heartbeat

        # Picard pre-smoother
        if presmooth_steps > 0:
            P_inner = _picard_inner(phi_full, halo, P_inner,
                                    inner_lo, inner_hi,
                                    presmooth_alpha, presmooth_steps,
                                    hb, heartbeat_s, history,
                                    u_grid_inner, tau_vec, K)

        # Newton-Krylov on the inner residual
        def F_inner_flat(P_inner_flat: np.ndarray) -> np.ndarray:
            P_inner_arr = _safe_clip(P_inner_flat.reshape((G_inner,) * K))
            P_full = replace_inner(halo, P_inner_arr, inner_lo, inner_hi)
            P_full_new = phi_full(P_full)
            hb["phi_calls"] += 1
            F = (P_inner_arr - extract_inner(P_full_new, inner_lo, inner_hi))
            now = time.perf_counter()
            if now - hb["last_hb"] >= heartbeat_s:
                F_inf = float(np.max(np.abs(F)))
                d = revelation_deficit_f128(P_inner_arr, u_grid_inner,
                                            tau_vec, K)
                elapsed = now - hb["t0"]
                print(f"[t={elapsed:7.0f}s] stage={hb['stage']:2d} "
                      f"GMRES jvp newton_iter~{hb['newton_iter']:2d}  "
                      f"||F_inner||inf={F_inf:.4e}  "
                      f"1-R^2_f128={d:.6e}  phi_calls={hb['phi_calls']}",
                      flush=True)
                history.inner_trace.append({
                    "stage": hb["stage"],
                    "kind": "newton-jvp",
                    "iter": hb["newton_iter"],
                    "F_inf": F_inf,
                    "deficit_f128": d,
                    "elapsed_s": elapsed,
                    "phi_calls": hb["phi_calls"],
                })
                hb["last_hb"] = now
            return F.ravel()

        def newton_callback(x: np.ndarray, F: np.ndarray) -> None:
            hb["newton_iter"] += 1
            F_inf = float(np.max(np.abs(F)))
            P_inner_arr = x.reshape((G_inner,) * K)
            d = revelation_deficit_f128(P_inner_arr, u_grid_inner,
                                        tau_vec, K)
            elapsed = time.perf_counter() - hb["t0"]
            print(f"[t={elapsed:7.0f}s] stage={hb['stage']:2d} "
                  f"Newton iter={hb['newton_iter']:3d}  "
                  f"||F_inner||inf={F_inf:.4e}  "
                  f"1-R^2_f128={d:.6e}  phi_calls={hb['phi_calls']}",
                  flush=True)
            history.inner_trace.append({
                "stage": hb["stage"],
                "kind": "newton-iter",
                "iter": hb["newton_iter"],
                "F_inf": F_inf,
                "deficit_f128": d,
                "elapsed_s": elapsed,
                "phi_calls": hb["phi_calls"],
            })
            hb["last_hb"] = time.perf_counter()

        hb["newton_iter"] = 0
        try:
            sol_flat = newton_krylov(
                F_inner_flat, P_inner.ravel(),
                method=inner_method,
                inner_maxiter=inner_inner_maxiter, outer_k=inner_outer_k,
                rdiff=inner_rdiff, line_search="armijo",
                f_tol=inner_tol, maxiter=inner_max_iter,
                callback=newton_callback,
            )
        except NoConvergence as exc:
            sol_flat = np.asarray(exc.args[0]).ravel()
            print(f"[t={time.perf_counter() - hb['t0']:7.0f}s] "
                  f"stage={s} Newton did not reach inner_tol={inner_tol}; "
                  f"taking best iterate", flush=True)

        P_inner = _safe_clip(sol_flat.reshape((G_inner,) * K))

        # ------- stage diagnostics -------
        P_full_now = replace_inner(halo, P_inner, inner_lo, inner_hi)
        P_full_phi = phi_full(P_full_now)
        hb["phi_calls"] += 1
        F_inner_now = extract_inner(P_full_now - P_full_phi,
                                    inner_lo, inner_hi)
        F_inner_inf = float(np.max(np.abs(F_inner_now)))
        deficit = revelation_deficit_f128(P_inner, u_grid_inner,
                                          tau_vec, K)
        drift = float(np.max(np.abs(P_inner - P_inner_prev)))
        elapsed = time.perf_counter() - hb["t0"]
        print(f"[t={elapsed:7.0f}s] === STAGE {s} END === "
              f"||F_inner||inf={F_inner_inf:.4e}  "
              f"1-R^2_f128={deficit:.6e}  drift={drift:.4e}  "
              f"phi_calls={hb['phi_calls']}", flush=True)
        history.stages.append(StageRecord(
            stage=s, phi_calls=hb["phi_calls"],
            F_inner_inf=F_inner_inf, deficit_f128=deficit,
            inner_drift_inf=drift, elapsed_s=elapsed,
        ))

        if drift < stage_tol and s >= 2:
            print(f"[t={elapsed:7.0f}s] stage drift {drift:.4e} below "
                  f"stage_tol {stage_tol:.1e}; stopping", flush=True)
            break

    return P_inner, history
