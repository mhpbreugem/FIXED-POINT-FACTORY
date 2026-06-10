"""K=3 CRRA REE solver in (L_priv, L_pub) coordinates with NO bounded cube.

u-grid: uniform in xi = tanh(tau*u/2), the private-log-odds-style transform.
        xi ∈ [-xi_max, +xi_max] with xi_max < 1 (slight margin from boundary).
        Corresponds to u ∈ [-(2/tau) atanh(xi_max), +(2/tau) atanh(xi_max)].
        At tau=0.1, xi_max=0.9 → u range ±29.4.
        NO truncation at U_MAX; we extend to wherever Gaussian density permits.

p-grid: uniform in L_pub(p) — the public-only posterior log-odds.
        L_pub computed bootstrap-style from the current cube empirical CDF.

Operator: strict-h=0 (zero kernel-band) via co-area + POU.

Precision: float64 first pass (numba), DD upgrade if time.

Live per-iteration printing: F_inf, omega, n_iter, wall.

CLI: python dd_k3_logodds.py [gamma] [tau] [G]
"""
import os, sys, time, json
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from lin_cdf_strict import (phi_lin_strict_jit, make_gl_for_u, build_mu_table_lin_strict,
                                  f_signal_jit)


REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/logodds"
os.makedirs(f"{OUT}/figs", exist_ok=True)


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------
def make_u_grid_logodds(G, tau, xi_max=0.9):
    """G uniform points in xi = tanh(tau*u/2), then u = (2/tau)*atanh(xi).
    Returns (u_grid, xi_grid)."""
    xi = np.linspace(-xi_max, xi_max, G)
    u = (2.0 / tau) * np.arctanh(xi)
    return u, xi


def compute_L_pub_from_P(P_cube, u_grid, tau, G_p=121, eps=1e-12):
    """Bootstrap L_pub(p) = log(f_1(p)/f_0(p)) from the cube empirical CDF.
    f_v(p) is the marginal price density under state v.
    Uses cube-cell integration with Gaussian weights f_v(u_a)*f_v(u_b)*f_v(u_k).
    Returns (p_grid_L_pub_uniform, L_pub_values)."""
    G = u_grid.size
    # Gaussian density f_v at each u_grid point
    sd = 1.0 / np.sqrt(tau)
    from scipy.stats import norm
    f0 = norm.pdf(u_grid, loc=-0.5, scale=sd)
    f1 = norm.pdf(u_grid, loc=+0.5, scale=sd)
    # Trap weights on the u-grid (in u-space, not xi)
    w = np.zeros(G)
    for i in range(G):
        if i == 0: w[i] = 0.5 * (u_grid[1] - u_grid[0])
        elif i == G-1: w[i] = 0.5 * (u_grid[-1] - u_grid[-2])
        else: w[i] = 0.5 * (u_grid[i+1] - u_grid[i-1])
    # Marginal density f_v(p) ~ sum over cube cells with P_cell ~ p of mass(cell) under v
    # Use logit-uniform p_grid for sampling
    p_eval = 1.0 / (1.0 + np.exp(-np.linspace(-6, 6, 201)))
    F_0 = np.zeros_like(p_eval); F_1 = np.zeros_like(p_eval)
    P_flat = P_cube.ravel()
    w_arr = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    f0_arr = (f0[:, None, None] * f0[None, :, None] * f0[None, None, :]).ravel()
    f1_arr = (f1[:, None, None] * f1[None, :, None] * f1[None, None, :]).ravel()
    for i, p in enumerate(p_eval):
        mask = P_flat <= p
        F_0[i] = np.sum(w_arr[mask] * f0_arr[mask])
        F_1[i] = np.sum(w_arr[mask] * f1_arr[mask])
    # Normalize
    F_0_max = F_0[-1]; F_1_max = F_1[-1]
    if F_0_max < eps or F_1_max < eps:
        return p_eval, np.zeros_like(p_eval)
    F_0_n = F_0 / F_0_max
    F_1_n = F_1 / F_1_max
    # Derivative ~ density
    f0_p = np.gradient(F_0_n, p_eval)
    f1_p = np.gradient(F_1_n, p_eval)
    L_pub = np.log(np.clip(f1_p, eps, None) / np.clip(f0_p, eps, None))
    # Enforce sign-flip symmetry: L_pub(1-p) = -L_pub(p).
    # Achieved by averaging L_pub(p) and -L_pub(1-p) on a symmetric eval grid.
    # The p_eval grid is constructed sigmoid-symmetric already.
    L_pub_sym = 0.5 * (L_pub - L_pub[::-1])
    return p_eval, L_pub_sym


def make_p_grid_Lpub(L_pub_vals, p_arr_for_Lpub, G_p, L_pub_max=None):
    """Pick G_p uniform L_pub points; return corresponding p values
    via interpolation of the inverse L_pub(p)."""
    if L_pub_max is None:
        # default: range to where L_pub is finite and well-defined
        L_pub_max = min(6.0, float(np.max(np.abs(L_pub_vals[np.isfinite(L_pub_vals)]))))
    L_target = np.linspace(-L_pub_max, L_pub_max, G_p)
    # Monotonize L_pub for invertibility
    L_pub_mono = np.maximum.accumulate(L_pub_vals)
    p_grid = np.interp(L_target, L_pub_mono, p_arr_for_Lpub)
    return p_grid


# ----------------------------------------------------------------------
# Strict-h=0 FP operator (delegate to existing lin_cdf_strict)
# ----------------------------------------------------------------------
def crra_clear_array(mu0, mu1, mu2, gamma):
    """K=3 CRRA: sum(mu_i) = 3p → p = mean(mu_i)."""
    return (mu0 + mu1 + mu2) / 3.0


@njit(cache=True)
def interp_mu_at(mu_table, p_grid, p, k):
    G_p = p_grid.size
    if p <= p_grid[0]: return mu_table[0, k]
    if p >= p_grid[-1]: return mu_table[-1, k]
    lo, hi = 0, G_p - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_grid[mid] <= p: lo = mid
        else: hi = mid
    w = (p - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    return (1.0 - w) * mu_table[lo, k] + w * mu_table[hi, k]


@njit(cache=True)
def crra_clear_float(m0, m1, m2, gamma, steps=80):
    """K=3 binary-state CRRA bisection on p. Solves
       sum_k (R_k - 1) / ((1-p) + R_k*p) = 0
    where R_k = exp((logit(mu_k) - logit(p)) / gamma).
    Matches dd_k3_ops.py:dd_crra_clear (float64 version).
    """
    eps = 1e-15
    m0c = max(min(m0, 1.0 - eps), eps)
    m1c = max(min(m1, 1.0 - eps), eps)
    m2c = max(min(m2, 1.0 - eps), eps)
    lm0 = np.log(m0c / (1.0 - m0c))
    lm1 = np.log(m1c / (1.0 - m1c))
    lm2 = np.log(m2c / (1.0 - m2c))
    a = 1e-12; b = 1.0 - 1e-12
    for _ in range(steps):
        m = 0.5 * (a + b)
        lp = np.log(m / (1.0 - m))
        e = 0.0
        for k in range(3):
            if k == 0: lmk = lm0
            elif k == 1: lmk = lm1
            else: lmk = lm2
            ar = (lmk - lp) / gamma
            if ar > 60.0: ar = 60.0
            if ar < -60.0: ar = -60.0
            R = np.exp(ar)
            den = (1.0 - m) + R * m
            e += (R - 1.0) / den
        if e > 0.0: a = m
        else: b = m
    return 0.5 * (a + b)


@njit(cache=True, parallel=True)
def phi_cube_logodds(P_cube, mu_table, p_grid, u_grid, gamma, G):
    """Cube clearing: for each (i,j,k), compute new P from mu lookups."""
    P_new = np.empty_like(P_cube)
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_cube[i, j, k]
                m0 = interp_mu_at(mu_table, p_grid, p_cell, i)
                m1 = interp_mu_at(mu_table, p_grid, p_cell, j)
                m2 = interp_mu_at(mu_table, p_grid, p_cell, k)
                P_new[i, j, k] = crra_clear_float(m0, m1, m2, gamma)
    return P_new


# ----------------------------------------------------------------------
# Anderson FP solver with live updates
# ----------------------------------------------------------------------
def solve_logodds(gamma, tau, G, G_p=81, n_iter=80, target=1e-10,
                       nq=24, xi_max=0.85, verbose=True, tag=""):
    u_grid, xi_grid = make_u_grid_logodds(G, tau, xi_max=xi_max)
    if verbose:
        sigma_sig = 1.0 / np.sqrt(tau)
        u_max_sig = u_grid[-1] / sigma_sig
        print(f"[{tag}] G={G}, tau={tau}, gamma={gamma}", flush=True)
        print(f"  u_grid range: [{u_grid[0]:+.3f}, {u_grid[-1]:+.3f}] = {u_max_sig:.1f}σ", flush=True)
        print(f"  xi_grid (uniform private log-odds): [{xi_grid[0]:+.3f}, {xi_grid[-1]:+.3f}]", flush=True)

    # Initial P_cube: sigmoid(sum L_priv) = sigmoid(tau * (u1+u2+u3))
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    L_init = tau * (U1 + U2 + U3)
    P = 1.0 / (1.0 + np.exp(-np.clip(L_init, -30, 30)))

    # Gaussian weights for density-weighted F
    from scipy.stats import norm
    sd = 1.0 / np.sqrt(tau)
    # Use the uninformative mixture (1/2 f_0 + 1/2 f_1)
    f_mix = 0.5 * norm.pdf(u_grid, loc=-0.5, scale=sd) + 0.5 * norm.pdf(u_grid, loc=+0.5, scale=sd)
    W3 = f_mix[:, None, None] * f_mix[None, :, None] * f_mix[None, None, :]
    W3 /= W3.max()  # normalize to [0, 1]

    # Initial L_pub from this P
    p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
    p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
    if verbose:
        print(f"  init L_pub range: [{np.min(L_pub):.2f}, {np.max(L_pub):.2f}]", flush=True)
        print(f"  p_grid range:     [{p_grid[0]:.4f}, {p_grid[-1]:.4f}]", flush=True)

    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], nq)
    Xh = []; Gh = []
    F_best = float("inf"); P_best = P.copy()
    t0 = time.time()
    hist = []
    for it in range(n_iter):
        t_it = time.time()
        # Build mu_table via strict-h=0 on current P
        mu_table = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u, gl_du, tau, G, nq)
        # Cube clearing
        P_new = phi_cube_logodds(P, mu_table, p_grid, u_grid, gamma, G)
        # Raw and weighted F
        diff = np.abs(P_new - P)
        F_raw = float(np.max(diff))
        F_w = float(np.max(diff * W3))
        wall_it = time.time() - t_it
        wall_tot = time.time() - t0
        if F_w < F_best:
            F_best = F_w; P_best = P.copy()
        hist.append((F_raw, F_w))
        if verbose:
            argmax_w = np.unravel_index(np.argmax(diff * W3), P.shape)
            i_, j_, k_ = argmax_w
            print(f"  [{wall_tot:6.1f}s] {tag}it{it:3d}: "
                  f"F_raw={F_raw:.3e} F_w={F_w:.3e} "
                  f"(weighted-max at xi={xi_grid[i_]:+.2f},{xi_grid[j_]:+.2f},{xi_grid[k_]:+.2f}) "
                  f"iter_wall={wall_it:.1f}s", flush=True)
        if F_w < target: break
        # Damped Anderson
        Xh.append(P.ravel().copy()); Gh.append(P_new.ravel().copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: P = 0.5 * P + 0.5 * P_new
        else:
            try:
                DR = np.column_stack([(Gh[i_]-Xh[i_])-(Gh[k-1]-Xh[k-1]) for i_ in range(k-1)])
                R_k = Gh[k-1] - Xh[k-1]
                A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i_]-Gh[k-1] for i_ in range(k-1)])
                x_a = Gh[k-1] + DG @ ga
                P = (0.5 * P + 0.5 * x_a.reshape(G, G, G))
            except Exception:
                P = 0.5 * P + 0.5 * P_new
        P = np.clip(P, 1e-12, 1 - 1e-12)
        # Permutation symmetry (operator preserves but Anderson may not)
        Psym = (P + np.transpose(P, (0,2,1)) + np.transpose(P, (1,0,2))
                + np.transpose(P, (1,2,0)) + np.transpose(P, (2,0,1))
                + np.transpose(P, (2,1,0))) / 6.0
        # Sign-flip symmetry P(u) + P(-u) = 1
        P = 0.5 * (Psym + (1.0 - Psym[::-1, ::-1, ::-1]))
        # Refresh L_pub once at iter 5 only (using symmetrized L_pub)
        if it == 5:
            p_eval, L_pub = compute_L_pub_from_P(P, u_grid, tau)
            p_grid = make_p_grid_Lpub(L_pub, p_eval, G_p)
            if verbose:
                print(f"      refresh L_pub: [{np.min(L_pub):.2f}, {np.max(L_pub):.2f}], "
                      f"p_grid: [{p_grid[0]:.4f}, {p_grid[-1]:.4f}]", flush=True)

    return P_best, F_best, hist, u_grid, xi_grid, p_grid


def save_and_plot(P, F_best, hist, u_grid, xi_grid, p_grid, gamma, tau, G):
    suffix = f"g{gamma:g}_t{tau:.4f}_G{G}"
    np.savez(f"{OUT}/fp_{suffix}.npz",
              P=P, u_grid=u_grid, xi_grid=xi_grid, p_grid=p_grid,
              gamma=gamma, tau=tau, G=G, F_best=F_best, hist=hist)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy([h[0] for h in hist], '-o', ms=4, label='F_raw')
    axes[0].semilogy([h[1] for h in hist], '-s', ms=4, label='F_w (density-weighted)')
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('|F|_inf')
    axes[0].set_title(f'{suffix}: convergence')
    axes[0].legend(); axes[0].grid(alpha=0.3, which='both')
    # P slice at middle u_k
    k0 = G // 2
    im = axes[1].imshow(P[:, :, k0], extent=[u_grid[0],u_grid[-1],u_grid[0],u_grid[-1]],
                          origin='lower', cmap='viridis')
    axes[1].set_xlabel('u1'); axes[1].set_ylabel('u2')
    axes[1].set_title(f'P(u1,u2,u3=0) at FP')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/{suffix}.png", dpi=120)
    plt.close()


def main():
    # Default config: tau=0.1, gamma=100, ladder G in [7,9,11,13,15]
    # Then gamma sweep at tau=0.1
    args = sys.argv[1:]
    if len(args) >= 3:
        gamma = float(args[0]); tau = float(args[1]); G = int(args[2])
        P, F, hist, u, xi, p = solve_logodds(gamma, tau, G, n_iter=60)
        save_and_plot(P, F, hist, u, xi, p, gamma, tau, G)
        return

    # G-ladder at tau=0.1, gamma=100
    print("\n" + "="*60)
    print("PHASE 1: G ladder at tau=0.1, gamma=100")
    print("="*60, flush=True)
    G_ladder = [7, 9, 11, 13, 15]
    for G in G_ladder:
        print(f"\n--- G = {G} ---", flush=True)
        P, F, hist, u, xi, p = solve_logodds(100.0, 0.1, G, n_iter=40, tag=f"G{G} ")
        save_and_plot(P, F, hist, u, xi, p, 100.0, 0.1, G)
        print(f"  ====> G={G} done, F_best = {F:.3e}", flush=True)

    # Gamma sweep at tau=0.1, G=11
    print("\n" + "="*60)
    print("PHASE 2: gamma sweep at tau=0.1, G=11")
    print("="*60, flush=True)
    for gamma in [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        print(f"\n--- gamma = {gamma} ---", flush=True)
        P, F, hist, u, xi, p = solve_logodds(gamma, 0.1, 11, n_iter=40, tag=f"g{gamma:g} ")
        save_and_plot(P, F, hist, u, xi, p, gamma, 0.1, 11)
        print(f"  ====> gamma={gamma} done, F_best = {F:.3e}", flush=True)

    print(f"\nAll done. Saved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
