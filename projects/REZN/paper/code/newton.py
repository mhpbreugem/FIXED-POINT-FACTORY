"""Newton-Krylov solver with 15-second wall-clock heartbeats.

Wraps scipy.optimize.newton_krylov, which provides:
  * Eisenstat-Walker inexact-Newton tolerance management
  * Armijo backtracking line search (less prone to wild divergent jumps
    than my hand-rolled GMRES + line search at K=4)
  * choice of LGMRES inner solver (better than restarted GMRES for
    poorly-conditioned Jacobians)

We instrument it with a Phi wrapper that tracks wall clock and emits
a heartbeat line whenever ``heartbeat_s`` seconds have passed. The
heartbeat fires inside the FD-Jacobian product, so it ticks during
slow inner Krylov iterations as well as between outer Newton steps.

Optional Picard pre-smoothing: a few damped Picard steps from the
no-learning seed average out the contour-integration kinks before
Newton starts, which materially improves convergence in our
heterogeneous K=4 setting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from .config import DTYPE
from .demand import EPS_PRICE
from .f128 import revelation_deficit_f128


PhiFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class _History:
    elapsed_s: List[float] = field(default_factory=list)
    F_inf: List[float] = field(default_factory=list)
    deficit_f128: List[float] = field(default_factory=list)
    phi_calls: List[int] = field(default_factory=list)
    label: List[str] = field(default_factory=list)


def _safe_clip(P: np.ndarray) -> np.ndarray:
    return np.clip(P, EPS_PRICE, 1.0 - EPS_PRICE)


def _picard_presmooth(phi: PhiFn, P: np.ndarray, *,
                      n_steps: int, alpha: float,
                      u_grid: np.ndarray, tau_vec: np.ndarray,
                      hb_state: dict, heartbeat_s: float,
                      history: _History) -> np.ndarray:
    print(f"[t={time.perf_counter() - hb_state['t0']:6.0f}s] "
          f"Picard pre-smooth: {n_steps} steps at alpha={alpha}", flush=True)
    for n in range(1, n_steps + 1):
        Phi_P = phi(P)
        hb_state["phi_calls"] += 1
        F = Phi_P - P
        F_inf = float(np.max(np.abs(F)))
        P = _safe_clip((1.0 - alpha) * P + alpha * Phi_P)
        now = time.perf_counter()
        if now - hb_state["last_hb"] >= heartbeat_s or n == n_steps:
            d = revelation_deficit_f128(P, u_grid, tau_vec, 4)
            elapsed = now - hb_state["t0"]
            print(f"[t={elapsed:6.0f}s] Picard step={n:3d}  "
                  f"||F||inf={F_inf:.4e}  1-R^2_f128={d:.6e}  "
                  f"phi_calls={hb_state['phi_calls']}", flush=True)
            history.elapsed_s.append(elapsed)
            history.F_inf.append(F_inf)
            history.deficit_f128.append(d)
            history.phi_calls.append(hb_state["phi_calls"])
            history.label.append(f"picard{n}")
            hb_state["last_hb"] = now
    return P


def newton_krylov_solve(
        phi: PhiFn, P0: np.ndarray,
        u_grid: np.ndarray, tau_vec: np.ndarray, K: int,
        *,
        max_iter: int = 30, tol: float = 1.0e-7,
        method: str = "lgmres",
        inner_maxiter: int = 60,
        outer_k: int = 30,           # LGMRES outer-vector memory depth
        rdiff: float = 1.0e-4,       # FD eps for Jvp; bigger absorbs kinks
        line_search: str = "armijo",
        presmooth_steps: int = 10,
        presmooth_alpha: float = 0.05,
        heartbeat_s: float = 15.0,
        ) -> Tuple[np.ndarray, _History]:
    """Solve P = Phi(P) via scipy.optimize.newton_krylov.

    Returns (P_solution, history). history fields are parallel lists.
    """
    shape = P0.shape

    history = _History()
    hb_state = {
        "t0": time.perf_counter(),
        "last_hb": time.perf_counter() - heartbeat_s,
        "phi_calls": 0,
        "newton_iter": 0,
    }

    # Initial diagnostics.
    P = _safe_clip(P0.astype(DTYPE, copy=True))
    Phi_P = phi(P)
    hb_state["phi_calls"] += 1
    F0 = Phi_P - P
    F0_inf = float(np.max(np.abs(F0)))
    d0 = revelation_deficit_f128(P, u_grid, tau_vec, K)
    print(f"[t=     0s] init  ||F||inf={F0_inf:.4e}  "
          f"1-R^2_f128={d0:.6e}  phi_calls=1", flush=True)
    history.elapsed_s.append(0.0)
    history.F_inf.append(F0_inf)
    history.deficit_f128.append(d0)
    history.phi_calls.append(1)
    history.label.append("init")
    hb_state["last_hb"] = time.perf_counter()

    # Optional Picard pre-smoothing
    if presmooth_steps > 0:
        P = _picard_presmooth(phi, P, n_steps=presmooth_steps,
                              alpha=presmooth_alpha, u_grid=u_grid,
                              tau_vec=tau_vec, hb_state=hb_state,
                              heartbeat_s=heartbeat_s, history=history)

    # scipy.optimize.newton_krylov flattens the iterate internally.
    # F_fn and the callback both receive (and must return) 1-D arrays.
    def F_fn(P_flat: np.ndarray) -> np.ndarray:
        P_arr = _safe_clip(P_flat.reshape(shape))
        Phi_P = phi(P_arr)
        hb_state["phi_calls"] += 1
        F = (P_arr - Phi_P).ravel()
        now = time.perf_counter()
        if now - hb_state["last_hb"] >= heartbeat_s:
            elapsed = now - hb_state["t0"]
            F_inf = float(np.max(np.abs(F)))
            d = revelation_deficit_f128(P_arr, u_grid, tau_vec, K)
            print(f"[t={elapsed:6.0f}s] Newton newton_iter~{hb_state['newton_iter']}  "
                  f"||F||inf={F_inf:.4e}  1-R^2_f128={d:.6e}  "
                  f"phi_calls={hb_state['phi_calls']}", flush=True)
            history.elapsed_s.append(elapsed)
            history.F_inf.append(F_inf)
            history.deficit_f128.append(d)
            history.phi_calls.append(hb_state["phi_calls"])
            history.label.append("newton-fjvp")
            hb_state["last_hb"] = now
        return F

    def newton_callback(x: np.ndarray, F: np.ndarray) -> None:
        # called once per outer Newton iter; both args are 1-D
        hb_state["newton_iter"] += 1
        elapsed = time.perf_counter() - hb_state["t0"]
        F_inf = float(np.max(np.abs(F)))
        d = revelation_deficit_f128(x.reshape(shape), u_grid, tau_vec, K)
        print(f"[t={elapsed:6.0f}s] Newton iter={hb_state['newton_iter']:3d} "
              f"done  ||F||inf={F_inf:.4e}  1-R^2_f128={d:.6e}  "
              f"phi_calls={hb_state['phi_calls']}", flush=True)
        history.elapsed_s.append(elapsed)
        history.F_inf.append(F_inf)
        history.deficit_f128.append(d)
        history.phi_calls.append(hb_state["phi_calls"])
        history.label.append(f"newton{hb_state['newton_iter']}")
        hb_state["last_hb"] = time.perf_counter()

    print(f"[t={time.perf_counter() - hb_state['t0']:6.0f}s] "
          f"Newton-Krylov start: method={method} rdiff={rdiff} "
          f"max_iter={max_iter}", flush=True)
    try:
        sol_flat = newton_krylov(
            F_fn, P.ravel(), method=method,
            inner_maxiter=inner_maxiter, outer_k=outer_k,
            rdiff=rdiff, line_search=line_search,
            f_tol=tol, maxiter=max_iter, callback=newton_callback,
        )
    except NoConvergence as exc:
        sol_flat = np.asarray(exc.args[0]).ravel()
        print(f"[t={time.perf_counter() - hb_state['t0']:6.0f}s] "
              f"newton_krylov did not reach tol={tol}; "
              f"returning best iterate", flush=True)

    sol = _safe_clip(sol_flat.reshape(shape))
    Phi_sol = phi(sol)
    hb_state["phi_calls"] += 1
    F_inf = float(np.max(np.abs(Phi_sol - sol)))
    d = revelation_deficit_f128(sol, u_grid, tau_vec, K)
    elapsed = time.perf_counter() - hb_state["t0"]
    print(f"[t={elapsed:6.0f}s] FINAL ||F||inf={F_inf:.4e}  "
          f"1-R^2_f128={d:.6e}  phi_calls={hb_state['phi_calls']}",
          flush=True)
    history.elapsed_s.append(elapsed)
    history.F_inf.append(F_inf)
    history.deficit_f128.append(d)
    history.phi_calls.append(hb_state["phi_calls"])
    history.label.append("final")

    return sol, history
