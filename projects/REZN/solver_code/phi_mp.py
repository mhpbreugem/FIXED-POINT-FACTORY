"""phi_mp.py — mpmath implementation of phi_K3_halo_smooth.

Mirrors the numba kernel in code/contour_K3_halo.py but uses arbitrary-
precision arithmetic via mpmath.  Intended for a final polishing phase:
warm-start from a float64 fixed-point (F~1e-7), then iterate in mpmath
until F < mp_tol (e.g. 1e-50).

Key algorithms
--------------
phi_newton_mp  (recommended)
    Inexact Newton-GMRES: one mpmath phi call per Newton step gives the
    accurate residual; float64 LGMRES + finite-difference Jacobian gives
    the Newton direction.  Convergence: ~5 steps from F~1e-7 to F<1e-50.
    Each step: ~5-10 min (mpmath) + ~1-3 min (float64 LGMRES).

phi_picard_mp  (fallback)
    Pure-Picard in mpmath with progressive dps and bisection market clearing.
    Only viable when the Picard contraction is strong (L << 0.99).

Public API
----------
phi_newton_mp(P_inner_np, halo_np, u_full_np, inner_lo, inner_hi,
              tau_vec_np, gamma_vec_np, W_vec_np, kernel_h,
              phi_float64_fn,
              dps, tol_str, max_newton, lgmres_tol, lgmres_inner_m,
              lgmres_outer, reporter)
    -> (P_inner_final_np, F_inf_final, n_iters)

phi_picard_mp(P_inner_np, halo_np, u_full_np, inner_lo, inner_hi,
              tau_vec_np, gamma_vec_np, W_vec_np, kernel_h,
              dps, tol_str, max_iters, alpha, reporter)
    -> (P_inner_final_np, F_inf_final, n_iters)
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# mpmath primitives
# ---------------------------------------------------------------------------

def _f_signal_mp(mp, u, v: int, tau):
    """Gaussian signal density f_v(u), v in {0, 1}."""
    mean = mp.mpf("0.5") if v == 1 else mp.mpf("-0.5")
    tau_ = mp.mpf(tau)
    coeff = mp.sqrt(tau_ / (2 * mp.pi))
    return coeff * mp.exp(-tau_ / 2 * (u - mean) ** 2)


def _logit_mp(mp, p):
    return mp.log(p) - mp.log(1 - p)


def _clear_crra_bisect_mp(mp, mu_vec, gamma_vec, W_vec, eps=None):
    """Bisection for price p* where sum_k x_k(p) = 0.

    Matches REZN's clear_crra bisection exactly but at mpmath precision.
    Uses int(dps * 3.5) + 10 steps — sufficient to halve the interval to
    10^(-dps) accuracy (since 2^(-n) < 10^(-dps) requires n > dps * log2(10)).
    """
    if eps is None:
        eps = mp.mpf(10) ** (-int(mp.dps * 0.8))
    a = eps
    b = 1 - eps
    K = len(mu_vec)
    logit_mu = [_logit_mp(mp, mu_vec[k]) for k in range(K)]

    def _excess(p):
        lp = _logit_mp(mp, p)
        return sum(
            _x_crra_for_excess(mp, logit_mu[k], lp, gamma_vec[k], W_vec[k])
            for k in range(K)
        )

    fa = _excess(a)
    if fa <= 0:
        return a
    fb = _excess(b)
    if fb >= 0:
        return b

    n_steps = int(mp.dps * 3.5) + 10
    for _ in range(n_steps):
        c = (a + b) / 2
        fc = _excess(c)
        if fc >= 0:
            a = c
        else:
            b = c

    return (a + b) / 2


def _x_crra_for_excess(mp, logit_mu_k, logit_p, gamma, W):
    """CRRA demand using precomputed logit values."""
    z = (logit_mu_k - logit_p) / gamma
    E = mp.exp(z)
    p = mp.mpf(1) / (1 + mp.exp(-logit_p))
    D = (1 - p) + p * E
    return W * (E - 1) / D


# ---------------------------------------------------------------------------
# Agent evidence (Gaussian-kernel Bayes integral over a 2-D slice)
# ---------------------------------------------------------------------------

def _agent_evidence_mp(mp, P_slice_mp, p_target, u_full_mp,
                       f0_u, f1_u, kernel_h, skip_thr):
    """Return (A0, A1): kernel-weighted sums over G_full x G_full slice."""
    inv_2h2 = mp.mpf(1) / (2 * kernel_h * kernel_h)
    A0 = mp.mpf(0)
    A1 = mp.mpf(0)
    G = len(u_full_mp)
    f0_a_list, f0_b_list = f0_u
    f1_a_list, f1_b_list = f1_u
    for ia in range(G):
        f0a = f0_a_list[ia]
        f1a = f1_a_list[ia]
        row = P_slice_mp[ia]
        for ib in range(G):
            diff = row[ib] - p_target
            if abs(diff) > skip_thr:
                continue
            w = mp.exp(-diff * diff * inv_2h2)
            A0 += w * f0a * f0_b_list[ib]
            A1 += w * f1a * f1_b_list[ib]
    return A0, A1


# ---------------------------------------------------------------------------
# Full phi evaluation in mpmath
# ---------------------------------------------------------------------------

def phi_K3_smooth_mp(mp, P_full_mp, u_full_mp, inner_lo, inner_hi,
                     tau_vec_mp, gamma_vec_mp, W_vec_mp, kernel_h_mp):
    """Compute one application of phi_K3_halo_smooth in mpmath.

    P_full_mp : 3-D list of lists of lists of mp.mpf (G_full × G_full × G_full)
    Returns a new P_full_mp (same structure, halo unchanged).
    """
    G = len(u_full_mp)

    f0 = [[_f_signal_mp(mp, u_full_mp[i], 0, tau_vec_mp[k]) for i in range(G)]
          for k in range(3)]
    f1 = [[_f_signal_mp(mp, u_full_mp[i], 1, tau_vec_mp[k]) for i in range(G)]
          for k in range(3)]

    eps_p = mp.mpf(10) ** (-int(mp.dps * 0.8))

    import math
    inv_2h2_f = 1.0 / (2.0 * float(kernel_h_mp) ** 2)
    skip_thr = mp.mpf(str(math.sqrt(mp.dps * math.log(10) / max(inv_2h2_f, 1e-30))))

    P_new = [[[P_full_mp[i][j][l] for l in range(G)] for j in range(G)]
             for i in range(G)]

    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full_mp[i][j][l]

                slice0 = [P_full_mp[i][a] for a in range(G)]
                A0_0, A1_0 = _agent_evidence_mp(
                    mp, slice0, p, u_full_mp,
                    f0_u=(f0[1], f0[2]), f1_u=(f1[1], f1[2]),
                    kernel_h=kernel_h_mp, skip_thr=skip_thr,
                )
                denom0 = f0[0][i] * A0_0 + f1[0][i] * A1_0
                mu0 = (f1[0][i] * A1_0 / denom0) if denom0 > 0 else mp.mpf("0.5")
                mu0 = max(eps_p, min(1 - eps_p, mu0))

                slice1 = [[P_full_mp[a][j][b] for b in range(G)] for a in range(G)]
                A0_1, A1_1 = _agent_evidence_mp(
                    mp, slice1, p, u_full_mp,
                    f0_u=(f0[0], f0[2]), f1_u=(f1[0], f1[2]),
                    kernel_h=kernel_h_mp, skip_thr=skip_thr,
                )
                denom1 = f0[1][j] * A0_1 + f1[1][j] * A1_1
                mu1 = (f1[1][j] * A1_1 / denom1) if denom1 > 0 else mp.mpf("0.5")
                mu1 = max(eps_p, min(1 - eps_p, mu1))

                slice2 = [[P_full_mp[a][b][l] for b in range(G)] for a in range(G)]
                A0_2, A1_2 = _agent_evidence_mp(
                    mp, slice2, p, u_full_mp,
                    f0_u=(f0[0], f0[1]), f1_u=(f1[0], f1[1]),
                    kernel_h=kernel_h_mp, skip_thr=skip_thr,
                )
                denom2 = f0[2][l] * A0_2 + f1[2][l] * A1_2
                mu2 = (f1[2][l] * A1_2 / denom2) if denom2 > 0 else mp.mpf("0.5")
                mu2 = max(eps_p, min(1 - eps_p, mu2))

                mu_vec = [mu0, mu1, mu2]
                p_star = _clear_crra_bisect_mp(mp, mu_vec, gamma_vec_mp, W_vec_mp, eps=eps_p)
                P_new[i][j][l] = p_star

    return P_new


# ---------------------------------------------------------------------------
# Helpers: convert between numpy and mpmath nested lists
# ---------------------------------------------------------------------------

def np_to_mp(mp, arr: np.ndarray):
    """Convert 3-D numpy array to nested list of mp.mpf."""
    if arr.ndim == 1:
        return [mp.mpf(str(x)) for x in arr]
    return [[[mp.mpf(str(arr[i, j, l]))
              for l in range(arr.shape[2])]
             for j in range(arr.shape[1])]
            for i in range(arr.shape[0])]


def mp_to_np(P_mp) -> np.ndarray:
    """Convert 3-D nested list of mp.mpf back to float64 numpy."""
    G = len(P_mp)
    out = np.zeros((G, G, G), dtype=np.float64)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                out[i, j, l] = float(P_mp[i][j][l])
    return out


def f_inf_mp(mp, P_new_mp, P_old_mp, inner_lo, inner_hi):
    """||phi(P) - P||_inf over inner block."""
    F = mp.mpf(0)
    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                d = abs(P_new_mp[i][j][l] - P_old_mp[i][j][l])
                if d > F:
                    F = d
    return F


# ---------------------------------------------------------------------------
# Newton-GMRES at mpmath precision (primary solver)
# ---------------------------------------------------------------------------

def phi_newton_mp(
    P_inner_np: np.ndarray,
    halo_np: np.ndarray,
    u_full_np: np.ndarray,
    inner_lo: int,
    inner_hi: int,
    tau_vec_np: np.ndarray,
    gamma_vec_np: np.ndarray,
    W_vec_np: np.ndarray,
    kernel_h: float,
    phi_float64_fn: Callable,   # phi_K3_halo_smooth(P_full_np, ...) -> P_full_np
    dps: int = 100,
    tol_str: str = "1e-50",
    max_newton: int = 20,
    lgmres_tol: float = 1e-10,
    lgmres_inner_m: int = 30,
    lgmres_outer: int = 10,
    reporter: Any = None,
) -> tuple[np.ndarray, float, int]:
    """Inexact Newton-GMRES for high-precision fixed-point polishing.

    Algorithm per Newton step:
      1. Compute residual F = phi(P) - P in mpmath (accurate, expensive).
      2. Normalise: F_scaled = F / ||F||_inf (cast to float64, O(1) entries).
      3. Run float64 LGMRES to solve (I - Dphi) * delta_scaled = F_scaled
         with finite-difference Jacobian-vector products in float64.
      4. Update: P_mp -= ||F||_inf * delta_scaled  (in mpmath).
      5. Check convergence against tol in mpmath.

    Convergence estimate (LGMRES tol=1e-10, L_local≈0.995):
      F: 1e-7 → 1e-14 → 1e-24 → 1e-34 → 1e-44 → 1e-54 (<1e-50 ✓)
    ~5-6 Newton steps, each costing one mpmath phi call (~5-10 min at G=20, dps=100).
    """
    try:
        import mpmath as _mp
    except ImportError:
        raise ImportError("mpmath is required for high-precision polishing.")

    from scipy.sparse.linalg import LinearOperator, lgmres as sp_lgmres

    target_dps = dps + 15
    _mp.mp.dps = target_dps
    tol = _mp.mpf(tol_str)

    G_full = halo_np.shape[0]
    G_inner = inner_hi - inner_lo
    n_inner = G_inner ** 3

    P_full_np = halo_np.copy()
    P_full_np[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi] = P_inner_np

    u_full_mp   = [_mp.mpf(str(x)) for x in u_full_np]
    tau_mp      = [_mp.mpf(str(x)) for x in tau_vec_np]
    gamma_mp    = [_mp.mpf(str(x)) for x in gamma_vec_np]
    W_mp        = [_mp.mpf(str(x)) for x in W_vec_np]
    kernel_mp   = _mp.mpf(str(kernel_h))

    print(f"[phi_mp/newton] dps={dps} tol={tol_str} max_newton={max_newton}", flush=True)
    print(f"[phi_mp/newton] G_full={G_full} inner=[{inner_lo},{inner_hi}] "
          f"inner_cells={n_inner}  lgmres_tol={lgmres_tol}", flush=True)

    t0 = time.perf_counter()
    F_inf = _mp.mpf("inf")
    F_float = float("inf")
    n_steps = 0

    # Initialize once — never reconstruct from float64 inside the loop.
    # Reconstructing from P_full_np each iteration would discard sub-float64
    # mpmath corrections once F < 1e-15.
    P_full_mp = np_to_mp(_mp.mp, P_full_np)

    for newton_it in range(max_newton):
        # ------------------------------------------------------------------
        # Step 1: accurate residual in mpmath
        # ------------------------------------------------------------------
        t_mp = time.perf_counter()
        P_new_mp = phi_K3_smooth_mp(
            _mp.mp, P_full_mp, u_full_mp, inner_lo, inner_hi,
            tau_mp, gamma_mp, W_mp, kernel_mp,
        )
        F_inf = f_inf_mp(_mp.mp, P_new_mp, P_full_mp, inner_lo, inner_hi)
        n_steps = newton_it + 1
        F_float = float(F_inf)
        elapsed_total = time.perf_counter() - t0
        elapsed_mp = time.perf_counter() - t_mp
        print(f"[phi_mp/newton] step={n_steps:3d}  F={F_float:.4e}  "
              f"t_mp={elapsed_mp:.0f}s  t_total={elapsed_total:.0f}s", flush=True)
        if reporter is not None:
            reporter.update(iter=n_steps, ftol=F_float)

        if F_inf < tol:
            print(f"[phi_mp/newton] converged at step={n_steps}  F={F_float:.4e}", flush=True)
            break

        # ------------------------------------------------------------------
        # Step 2: scaled float64 RHS — keeps LGMRES numerically stable
        # regardless of the magnitude of F.
        # ------------------------------------------------------------------
        F_inner_mp_list = []
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    F_inner_mp_list.append(P_new_mp[i][j][l] - P_full_mp[i][j][l])

        F_inf_mp = max(abs(x) for x in F_inner_mp_list)
        if F_inf_mp == 0:
            break  # already at zero

        F_scaled_np = np.array([float(x / F_inf_mp) for x in F_inner_mp_list],
                               dtype=np.float64)

        # ------------------------------------------------------------------
        # Step 3: float64 LGMRES with finite-difference Jacobian-vector products
        # ------------------------------------------------------------------
        # Precompute phi(P_full) in float64 once (used in every matvec)
        phi0_full = phi_float64_fn(P_full_np)

        eps_fd = 1e-6  # finite-difference step

        def matvec(v: np.ndarray) -> np.ndarray:
            v3 = v.reshape(G_inner, G_inner, G_inner)
            P_eps = P_full_np.copy()
            P_eps[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi] += eps_fd * v3
            phi_eps = phi_float64_fn(P_eps)
            # (I - Dphi) * v  ≈  v - (phi(P+eps*v) - phi(P)) / eps
            Jv_inner = (
                v3
                - (phi_eps[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi]
                   - phi0_full[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi])
                / eps_fd
            )
            return Jv_inner.ravel()

        J_op = LinearOperator((n_inner, n_inner), matvec=matvec, dtype=np.float64)

        t_lgmres = time.perf_counter()
        delta_scaled, info = sp_lgmres(
            J_op, F_scaled_np,
            atol=lgmres_tol,
            maxiter=lgmres_outer,
            inner_m=lgmres_inner_m,
        )
        elapsed_lgmres = time.perf_counter() - t_lgmres
        print(f"[phi_mp/newton]   LGMRES info={info} t={elapsed_lgmres:.0f}s "
              f"||r||={float(np.linalg.norm(J_op @ delta_scaled - F_scaled_np)):.2e}",
              flush=True)

        # ------------------------------------------------------------------
        # Step 4: apply Newton step in mpmath
        #   P_new = P - F_inf * delta_scaled
        # ------------------------------------------------------------------
        delta_mp = [_mp.mpf(str(x)) for x in delta_scaled]
        F_inf_scale = _mp.mpf(str(float(F_inf_mp)))
        idx = 0
        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    P_full_mp[i][j][l] += F_inf_scale * delta_mp[idx]
                    idx += 1

        # Extract updated float64 P for the next Jacobian evaluation
        P_full_np = mp_to_np(P_full_mp)

    # Extract final inner block
    P_inner_final = np.zeros((G_inner, G_inner, G_inner), dtype=np.float64)
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_inner_final[i, j, l] = float(
                    P_full_mp[inner_lo + i][inner_lo + j][inner_lo + l]
                )

    return P_inner_final, float(F_inf), n_steps


# ---------------------------------------------------------------------------
# Progressive dps helper (used by phi_picard_mp)
# ---------------------------------------------------------------------------

def _dps_for_F(F_float: float, target_dps: int) -> int:
    """Working dps appropriate for residual F (progressive precision)."""
    import math
    if F_float <= 0 or not math.isfinite(F_float) or F_float >= 1.0:
        return min(25, target_dps)
    needed = max(25, int(-math.log10(F_float) * 1.5) + 15)
    return min(needed, target_dps)


# ---------------------------------------------------------------------------
# Picard loop in mpmath (fallback / alternative)
# ---------------------------------------------------------------------------

def phi_picard_mp(
    P_inner_np: np.ndarray,
    halo_np: np.ndarray,
    u_full_np: np.ndarray,
    inner_lo: int,
    inner_hi: int,
    tau_vec_np: np.ndarray,
    gamma_vec_np: np.ndarray,
    W_vec_np: np.ndarray,
    kernel_h: float,
    dps: int = 100,
    tol_str: str = "1e-50",
    max_iters: int = 2000,
    alpha: float = 0.5,
    reporter: Any = None,
) -> tuple[np.ndarray, float, int]:
    """Pure-Picard with bisection market clearing + progressive dps.

    NOTE: For L_local ≈ 0.995 (typical REZN), this requires tens of
    thousands of iterations to reach F<1e-50.  Use phi_newton_mp instead.
    This fallback is useful when phi_float64_fn is unavailable or as a
    warm-up phase before Newton.
    """
    try:
        import mpmath as _mp
    except ImportError:
        raise ImportError("mpmath is required for high-precision polishing.")

    target_dps = dps + 15
    _mp.mp.dps = target_dps
    tol = _mp.mpf(tol_str)

    G_full = halo_np.shape[0]
    P_full_np = halo_np.copy()
    P_full_np[inner_lo:inner_hi, inner_lo:inner_hi, inner_lo:inner_hi] = P_inner_np

    print(f"[phi_mp/picard] dps={dps} tol={tol_str} alpha={alpha} max_iters={max_iters}",
          flush=True)
    print(f"[phi_mp/picard] G_full={G_full} inner=[{inner_lo},{inner_hi}] "
          f"inner_cells={(inner_hi-inner_lo)**3}  Newton market clearing + progressive dps",
          flush=True)

    t0 = time.perf_counter()
    F_inf = _mp.mpf("inf")
    F_float = float("inf")
    n_iters = 0
    cur_dps = _dps_for_F(1e-7, target_dps)

    _mp.mp.dps = cur_dps
    P_full_mp = np_to_mp(_mp.mp, P_full_np)
    u_full_mp = [_mp.mpf(str(x)) for x in u_full_np]
    tau_mp    = [_mp.mpf(str(x)) for x in tau_vec_np]
    gamma_mp  = [_mp.mpf(str(x)) for x in gamma_vec_np]
    W_mp      = [_mp.mpf(str(x)) for x in W_vec_np]
    kernel_mp = _mp.mpf(str(kernel_h))
    alpha_mp  = _mp.mpf(str(alpha))
    one_minus_alpha = _mp.mpf(1) - alpha_mp

    for it in range(max_iters):
        needed_dps = _dps_for_F(F_float, target_dps)
        if needed_dps != cur_dps:
            cur_dps = needed_dps
            _mp.mp.dps = cur_dps
            P_full_mp = np_to_mp(_mp.mp, mp_to_np(P_full_mp))
            u_full_mp = [_mp.mpf(str(x)) for x in u_full_np]
            tau_mp    = [_mp.mpf(str(x)) for x in tau_vec_np]
            gamma_mp  = [_mp.mpf(str(x)) for x in gamma_vec_np]
            W_mp      = [_mp.mpf(str(x)) for x in W_vec_np]
            kernel_mp = _mp.mpf(str(kernel_h))
            alpha_mp  = _mp.mpf(str(alpha))
            one_minus_alpha = _mp.mpf(1) - alpha_mp
            print(f"[phi_mp/picard] dps → {cur_dps} (F={F_float:.2e})", flush=True)

        P_new_mp = phi_K3_smooth_mp(
            _mp.mp, P_full_mp, u_full_mp, inner_lo, inner_hi,
            tau_mp, gamma_mp, W_mp, kernel_mp,
        )
        F_inf = f_inf_mp(_mp.mp, P_new_mp, P_full_mp, inner_lo, inner_hi)
        n_iters = it + 1

        for i in range(inner_lo, inner_hi):
            for j in range(inner_lo, inner_hi):
                for l in range(inner_lo, inner_hi):
                    P_full_mp[i][j][l] = (one_minus_alpha * P_full_mp[i][j][l]
                                          + alpha_mp * P_new_mp[i][j][l])

        F_float = float(F_inf)
        elapsed = time.perf_counter() - t0
        print(f"[phi_mp/picard] iter={n_iters:5d}  F={F_float:.4e}  "
              f"dps={cur_dps}  t={elapsed:.0f}s", flush=True)
        if reporter is not None:
            reporter.update(iter=n_iters, ftol=F_float)

        _mp.mp.dps = target_dps
        tol = _mp.mpf(tol_str)
        if F_inf < tol:
            print(f"[phi_mp/picard] converged at iter={n_iters}  F={F_float:.4e}", flush=True)
            break
        _mp.mp.dps = cur_dps

    G_inner = inner_hi - inner_lo
    P_inner_final = np.zeros((G_inner, G_inner, G_inner), dtype=np.float64)
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_inner_final[i, j, l] = float(
                    P_full_mp[inner_lo + i][inner_lo + j][inner_lo + l]
                )

    return P_inner_final, float(F_inf), n_iters
