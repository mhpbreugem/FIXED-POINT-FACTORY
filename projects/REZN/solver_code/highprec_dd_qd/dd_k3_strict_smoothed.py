"""Smoothed strict-h=0 operator: regularize the cell-membership flip.

In the original strict op, `all_roots_linear` returns roots where
sign(P_along[j]-p) != sign(P_along[j+1]-p) — a discrete decision.
We replace this with a SMOOTH weight that ramps from 0 to 1 as the
segment "decisively crosses" p.

Specifically, for segment [u_j, u_j+1] with P_along[j] = a+p, P_along[j+1]=b+p:
   classical: contribute if a*b < 0
   smoothed:  contribute always, weighted by w(a,b) = sigmoid(-a*b / eps^2)
              -- w ~ 1 when a*b is very negative (deep crossing)
              -- w ~ 0.5 when a*b ~ 0 (marginal)
              -- w ~ 0 when a*b is very positive (no crossing)
   root location: use linear formula u_root = u_j + (-a)/((b-a)/h)*h
                  EVEN if a, b same sign (gives root outside segment)
                  But weight w → 0 if outside, so no contribution.

Key property: as a or b changes sign (continuously), w transitions smoothly
so the operator Phi(P) is Lipschitz in P. As eps → 0, recover original
strict operator.

Strategy: solve at large eps first, then continuation eps → 0.
"""
import os, sys, time, json
import numpy as np
from numba import njit
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_pchip import make_cdf_uniform_grid
from lin_cdf_strict import (f_signal_jit, _find_interval, linterp_slice_eval,
                                 linterp_slice_eval_axis_b, linterp_deriv_1d,
                                 make_gl_for_u, MAX_ROOTS, phi_lin_strict_jit)
from lin_cdf_kern_tab import make_p_grid as make_p_grid_logit
from scipy.optimize import newton_krylov
try: from scipy.optimize._nonlin import NoConvergence
except: from scipy.optimize import NoConvergence
import matplotlib.pyplot as plt


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/smooth_strict"
os.makedirs(f"{OUT}/figs", exist_ok=True)


@njit(cache=True)
def all_roots_smoothed(u_grid, P_along, p_target, n, eps,
                          roots_out, slopes_out, weights_out):
    """Smoothed crossing detector. Returns all (root, slope, weight)
    triples for segments where weight > 1e-12."""
    nr = 0
    eps2 = eps * eps
    for j in range(n - 1):
        a = P_along[j] - p_target
        b = P_along[j+1] - p_target
        # weight: sigmoid of -a*b / eps^2
        # exp may overflow; clip the argument
        arg = -a * b / eps2
        if arg > 50: w = 1.0
        elif arg < -50: w = 0.0
        else: w = 1.0 / (1.0 + np.exp(-arg))
        if w < 1e-12: continue
        slope = (P_along[j+1] - P_along[j]) / (u_grid[j+1] - u_grid[j])
        if abs(slope) < 1e-300: continue
        u_root = u_grid[j] + (-a) / slope
        # Project to segment
        if u_root < u_grid[j]: u_root = u_grid[j]
        if u_root > u_grid[j+1]: u_root = u_grid[j+1]
        if nr < MAX_ROOTS:
            roots_out[nr] = u_root
            slopes_out[nr] = slope
            weights_out[nr] = w
            nr += 1
    return nr


@njit(cache=True)
def co_area_smoothed(u_grid, slice2d, p_target,
                         gl_u_a, gl_du_a, gl_u_b, gl_du_b,
                         tau, n_grid, nq, eps):
    A0 = 0.0; A1 = 0.0
    roots = np.empty(MAX_ROOTS); slopes = np.empty(MAX_ROOTS); ws = np.empty(MAX_ROOTS)
    # Term A^(b)
    for q in range(nq):
        u_a = gl_u_a[q]; w_a = gl_du_a[q]
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        P_along_b = linterp_slice_eval(u_grid, slice2d, u_a, n_grid)
        nr = all_roots_smoothed(u_grid, P_along_b, p_target, n_grid, eps,
                                       roots, slopes, ws)
        if nr == 0: continue
        for r in range(nr):
            u_b = roots[r]; dPdu_b = slopes[r]; w_r = ws[r]
            P_along_a = linterp_slice_eval_axis_b(u_grid, slice2d, u_b, n_grid)
            dPdu_a = linterp_deriv_1d(u_grid, P_along_a, u_a, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou * w_r / abs(dPdu_b)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    # Term A^(a)
    for q in range(nq):
        u_b = gl_u_b[q]; w_b = gl_du_b[q]
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        P_along_a = linterp_slice_eval_axis_b(u_grid, slice2d, u_b, n_grid)
        nr = all_roots_smoothed(u_grid, P_along_a, p_target, n_grid, eps,
                                       roots, slopes, ws)
        if nr == 0: continue
        for r in range(nr):
            u_a = roots[r]; dPdu_a = slopes[r]; w_r = ws[r]
            P_along_b = linterp_slice_eval(u_grid, slice2d, u_a, n_grid)
            dPdu_b = linterp_deriv_1d(u_grid, P_along_b, u_b, n_grid)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            wt = w_b * w_a_pou * w_r / abs(dPdu_a)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    return A0, A1


@njit(cache=True)
def build_mu_smoothed(P_vals, u_grid, p_grid, gl_u, gl_du, tau, n_grid, nq, eps):
    G = n_grid; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    for k_node in range(G):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
        slice2d = np.empty((G, G))
        for jj in range(G):
            for kk in range(G):
                slice2d[jj, kk] = P_vals[k_node, jj, kk]
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_smoothed(u_grid, slice2d, p, gl_u, gl_du, gl_u, gl_du,
                                          tau, G, nq, eps)
            den = f0k*A0 + f1k*A1
            if den > 1e-300:
                m = f1k*A1 / den
                if m < 1e-9: m = 1e-9
                elif m > 1-1e-9: m = 1-1e-9
                mu_table[ip, k_node] = m
            else:
                mu_table[ip, k_node] = 0.5
    return mu_table


from numba import njit
@njit(cache=True)
def _interp_mu(mu_table, p, p_grid, k_idx, G_p):
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    lo = 0; hi = G_p - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_grid[mid] <= p: lo = mid
        else: hi = mid
    w = (p - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    return (1.0 - w) * mu_table[lo, k_idx] + w * mu_table[hi, k_idx]


@njit(cache=True)
def crra_clear(mu0, mu1, mu2, gamma):
    """CRRA clearing: same as in existing solver."""
    # Use logistic form: P = sigma(gamma * (logit(mu1)+logit(mu2)+logit(mu3))/3) ??
    # Actually standard CRRA clearing for K=3: see existing dd_k3_solver.
    # For now use the simplest: P = mean(mu_i) (Hellwig clearing)
    # The actual CRRA formula:
    # Let q = (1-p)/p. Demand_i(p) = (mu_i - p) / (gamma*p*(1-p)).
    # Sum demands = 0 => sum(mu_i) = 3*p => p = mean(mu_i).
    return (mu0 + mu1 + mu2) / 3.0


@njit(cache=True)
def phi_smoothed(P_vals, u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq, eps):
    """One Phi pass with smoothed strict operator."""
    mu_table = build_mu_smoothed(P_vals, u_grid, p_grid, gl_u, gl_du, tau, G, nq, eps)
    G_p = p_grid.size
    P_new = np.empty_like(P_vals)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                mu0 = _interp_mu(mu_table, p_cell, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, p_cell, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, p_cell, p_grid, k, G_p)
                P_new[i, j, k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_new


def solve_eps_continuation(P_warm, u_grid, p_grid, gl_u, gl_du, tau, gamma,
                              eps_schedule=None, n_anderson=60, target=1e-10):
    """Anderson FP solve with eps continuation: start with eps_0, reduce
    eps -> 0 as iteration converges."""
    if eps_schedule is None:
        eps_schedule = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    G = u_grid.size; nq = gl_u.size
    P = P_warm.copy()
    full_hist = []
    for eps in eps_schedule:
        print(f"  --- eps = {eps:.0e} ---", flush=True)
        def F(xflat):
            return (phi_smoothed(xflat.reshape(G,G,G), u_grid, p_grid,
                                       gl_u, gl_du, tau, gamma, G, nq, eps)
                    - xflat.reshape(G,G,G)).ravel()
        x = P.ravel().copy(); Xh, Gh = [], []; Fs = []
        x_best = x.copy(); f_best = float("inf")
        for it in range(n_anderson):
            Fv = F(x); gx = Fv + x
            f = float(np.max(np.abs(Fv))); Fs.append(f); full_hist.append((eps, f))
            if f < f_best: f_best = f; x_best = x.copy()
            if f < target: break
            Xh.append(x.copy()); Gh.append(gx.copy())
            if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
            k = len(Xh)
            if k <= 1: x = 0.5*x + 0.5*gx  # damped Picard
            else:
                DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
                R_k = Gh[k-1] - Xh[k-1]
                try:
                    A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                    ga = np.linalg.solve(A, -DR.T @ R_k)
                    DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                    x = Gh[k-1] + DG @ ga
                except: x = gx
        print(f"      best |F| in this stage: {f_best:.3e} after {len(Fs)} iters", flush=True)
        P = x_best.reshape(G,G,G)
    return P, full_hist


def main():
    G = 11
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid_logit(121)
    nq = 16
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)

    # Warm up numba
    print("warmup numba...", flush=True)
    P0 = np.full((G,G,G), 0.5)
    t0 = time.time()
    _ = phi_smoothed(P0, u_grid, p_grid, gl_u, gl_du, 1.0, 1.0, G, nq, 1e-2)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    # Test cells
    results = {}
    for gamma, tau in [(100, 1.0), (100, 0.2), (1000, 1.0)]:
        fp = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g{gamma}_t{tau:.4f}.npz")
        P_warm = fp["P_strict"].astype(np.float64)
        print(f"\n=== smoothed strict at g={gamma}, t={tau} ===", flush=True)
        t0 = time.time()
        P_new, hist = solve_eps_continuation(P_warm, u_grid, p_grid, gl_u, gl_du,
                                                  tau, gamma,
                                                  eps_schedule=[3e-1, 1e-1, 3e-2, 1e-2, 3e-3],
                                                  n_anderson=40)
        wall = time.time() - t0
        # Check residual under the ORIGINAL strict operator
        P_check = phi_lin_strict_jit(P_new, u_grid, p_grid, gl_u, gl_du, tau, gamma, G, nq)
        F_orig = float(np.max(np.abs(P_check - P_new)))
        print(f"  total wall: {wall:.1f}s")
        print(f"  final |F|_orig (under original strict op) = {F_orig:.3e}")
        results[f"g{gamma}_t{tau:.4f}"] = dict(
            gamma=gamma, tau=tau, wall=wall,
            F_orig_warm=float(np.max(np.abs(phi_lin_strict_jit(P_warm, u_grid, p_grid,
                                                                          gl_u, gl_du, tau, gamma, G, nq) - P_warm))),
            F_orig_final=F_orig,
            hist=hist,
        )
        np.savez(f"{OUT}/smoothed_g{gamma}_t{tau:.4f}.npz",
                 P_warm=P_warm, P_smoothed=P_new)

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'hist'}
                  for k, v in results.items()},
                  open(f"{OUT}/results.json", "w"), indent=2, default=str)
    print("\nSaved results.json + per-cell NPZs")

    # Plot histories
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, r in results.items():
        h = r["hist"]
        Fs = [e[1] for e in h]
        ax.semilogy(Fs, label=tag, alpha=0.7)
    ax.set_xlabel("global iteration")
    ax.set_ylabel("|F|_inf (under smoothed op at current eps)")
    ax.set_title("eps-continuation Anderson convergence on smoothed strict-h=0")
    ax.legend(); ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/eps_continuation.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
