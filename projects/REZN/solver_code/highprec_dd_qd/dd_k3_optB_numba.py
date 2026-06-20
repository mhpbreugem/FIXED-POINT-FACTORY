"""Option B (empirical CDF + PCHIP density) as a complete K=3 operator,
numba float64 implementation.

Phase A: for each u_k, sort cube cells by P, build cumulative weighted F_v
         over the sorted cells, fit PCHIP, evaluate derivative to get density
         a_v(p_grid, u_k). All numba.

Phase B: for each cube cell, look up mu via the table and CRRA-clear in numba.

The operator has a non-smooth Jacobian (cube cells reorder when P changes
through grid-crossing values), so we use Anderson + NK rather than direct
Newton.
"""
import os, sys, time
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit, prange
from cheby_numba import f_signal_jit, crra_clear_jit


@njit(cache=True)
def _pchip_slopes(xh, yh, n, mh):
    """Standard PCHIP (Fritsch-Carlson) slopes for nodes (xh[0..n-1], yh[0..n-1])."""
    for k in range(1, n-1):
        h0 = xh[k] - xh[k-1]; h1 = xh[k+1] - xh[k]
        if h0 <= 0.0 or h1 <= 0.0:
            mh[k] = 0.0; continue
        d0 = (yh[k] - yh[k-1])/h0
        d1 = (yh[k+1] - yh[k])/h1
        if d0 == 0.0 or d1 == 0.0 or (d0 > 0.0) != (d1 > 0.0):
            mh[k] = 0.0
        else:
            w1 = 2.0*h1 + h0; w2 = 2.0*h0 + h1
            mh[k] = (w1 + w2) / (w1/d0 + w2/d1)
    # endpoints (Fritsch-Carlson endpoint formula)
    for which in range(2):
        if which == 0:
            k = 0
            if n < 3: mh[k] = (yh[1]-yh[0])/(xh[1]-xh[0]); continue
            h0 = xh[1] - xh[0]; h1 = xh[2] - xh[1]
            d0 = (yh[1]-yh[0])/h0 if h0 > 0 else 0.0
            d1 = (yh[2]-yh[1])/h1 if h1 > 0 else 0.0
        else:
            k = n-1
            if n < 3: mh[k] = (yh[n-1]-yh[n-2])/(xh[n-1]-xh[n-2]); continue
            h0 = xh[n-1] - xh[n-2]; h1 = xh[n-2] - xh[n-3]
            d0 = (yh[n-1]-yh[n-2])/h0 if h0 > 0 else 0.0
            d1 = (yh[n-2]-yh[n-3])/h1 if h1 > 0 else 0.0
        if h0 + h1 <= 0.0: mh[k] = 0.0; continue
        m = ((2.0*h0 + h1)*d0 - h0*d1) / (h0 + h1)
        if (d0 > 0.0) != (m > 0.0): m = 0.0
        elif (d0 > 0.0) != (d1 > 0.0) and abs(m) > 3.0*abs(d0):
            m = 3.0*d0
        mh[k] = m


@njit(cache=True, inline="always")
def _pchip_deriv_eval(xh, yh, mh, n, q):
    """PCHIP derivative df/dp at query q on knots xh[0..n-1] yh[0..n-1]
    with precomputed slopes mh. Returns 0 if q outside knot range."""
    if q <= xh[0] or q >= xh[n-1]: return 0.0
    # bracket search
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if xh[mid] <= q: lo = mid
        else: hi = mid
    h = xh[hi] - xh[lo]
    if h <= 0.0: return 0.0
    t = (q - xh[lo]) / h
    # Cubic Hermite derivative basis (df/dt):
    H00p = 6.0*t*(t - 1.0)
    H10p = 3.0*t*t - 4.0*t + 1.0
    H01p = 6.0*t*(1.0 - t)
    H11p = 3.0*t*t - 2.0*t
    # df/dp = (1/h) * [yh[lo]*H00p + h*mh[lo]*H10p + yh[hi]*H01p + h*mh[hi]*H11p]
    return (yh[lo]*H00p + yh[hi]*H01p)/h + mh[lo]*H10p + mh[hi]*H11p


@njit(cache=True)
def build_mu_optB(P_vals, u_grid, p_grid, tau, w_trap, f0_u, f1_u, mu_table):
    """Build mu(p, u_k) via Option B: empirical CDF on cube cells + PCHIP density."""
    G = u_grid.size; G_p = p_grid.size
    Gsq = G*G
    P_arr = np.empty(Gsq); w_arr = np.empty(Gsq)
    f0_arr = np.empty(Gsq); f1_arr = np.empty(Gsq)
    # PCHIP buffers (sized for Gsq points but typically fewer unique after dedup)
    Pu = np.empty(Gsq); F0u = np.empty(Gsq); F1u = np.empty(Gsq)
    m0 = np.empty(Gsq); m1 = np.empty(Gsq)
    for k_node in range(G):
        # Collect slice cells
        s = 0
        for i in range(G):
            for j in range(G):
                P_arr[s] = P_vals[k_node, i, j]
                w_arr[s] = w_trap[i] * w_trap[j]
                f0_arr[s] = f0_u[i] * f0_u[j]
                f1_arr[s] = f1_u[i] * f1_u[j]
                s += 1
        # argsort (numba supports np.argsort)
        order = np.argsort(P_arr)
        # Cumulative weighted F_v at sorted positions
        # Aggregate ties: collect into Pu, F0u, F1u
        n_u = 0
        cum0 = 0.0; cum1 = 0.0
        prev_P = P_arr[order[0]] - 1.0   # ensures first iteration creates a knot
        for s in range(Gsq):
            ii = order[s]
            cum0 += w_arr[ii] * f0_arr[ii]
            cum1 += w_arr[ii] * f1_arr[ii]
            P_cur = P_arr[ii]
            if P_cur > prev_P + 1e-15:
                Pu[n_u] = P_cur; F0u[n_u] = cum0; F1u[n_u] = cum1
                n_u += 1
                prev_P = P_cur
            else:
                F0u[n_u-1] = cum0; F1u[n_u-1] = cum1
        if n_u < 4:
            for ip in range(G_p): mu_table[ip, k_node] = 0.5
            continue
        # PCHIP slopes for F0 and F1 over (Pu, F_v)
        _pchip_slopes(Pu, F0u, n_u, m0)
        _pchip_slopes(Pu, F1u, n_u, m1)
        f0k = f0_u[k_node]; f1k = f1_u[k_node]
        for ip in range(G_p):
            p = p_grid[ip]
            a0 = _pchip_deriv_eval(Pu, F0u, m0, n_u, p)
            a1 = _pchip_deriv_eval(Pu, F1u, m1, n_u, p)
            den = f0k*a0 + f1k*a1
            if den > 1e-300:
                m = f1k*a1 / den
                if m < 1e-9: m = 1e-9
                elif m > 1.0-1e-9: m = 1.0-1e-9
                mu_table[ip, k_node] = m
            else:
                mu_table[ip, k_node] = 0.5


@njit(cache=True, inline="always")
def _interp_mu(mu_table, p, p_grid, k_idx, G_p):
    """Linear interp along p_grid for cube clearing lookup."""
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    lo = 0; hi = G_p-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if p_grid[mid] <= p: lo = mid
        else: hi = mid
    w = (p - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
    return (1.0 - w)*mu_table[lo, k_idx] + w*mu_table[hi, k_idx]


@njit(cache=True, parallel=True)
def phi_optB(P_vals, u_grid, p_grid, tau, gamma, w_trap, f0_u, f1_u, P_new):
    """One Phi call: Option B mu-table + Phase B cube clearing."""
    G = u_grid.size; G_p = p_grid.size
    mu_table = np.empty((G_p, G))
    build_mu_optB(P_vals, u_grid, p_grid, tau, w_trap, f0_u, f1_u, mu_table)
    eps = 1e-9
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                pc = P_vals[i, j, k]
                if pc < eps: pc = eps
                elif pc > 1.0-eps: pc = 1.0-eps
                mu0 = _interp_mu(mu_table, pc, p_grid, i, G_p)
                mu1 = _interp_mu(mu_table, pc, p_grid, j, G_p)
                mu2 = _interp_mu(mu_table, pc, p_grid, k, G_p)
                if mu0 < eps: mu0 = eps
                elif mu0 > 1.0-eps: mu0 = 1.0-eps
                if mu1 < eps: mu1 = eps
                elif mu1 > 1.0-eps: mu1 = 1.0-eps
                if mu2 < eps: mu2 = eps
                elif mu2 > 1.0-eps: mu2 = 1.0-eps
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)


# ============================ Convenience driver ============================
def make_helpers(u_grid, tau):
    G = u_grid.size
    du = np.diff(u_grid)
    w_trap = np.empty(G); w_trap[0] = 0.5*du[0]; w_trap[-1] = 0.5*du[-1]
    w_trap[1:-1] = 0.5*(du[:-1] + du[1:])
    f0_u = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
    f1_u = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
    return w_trap, f0_u, f1_u


def phi(P, u_grid, p_grid, tau, gamma):
    w_trap, f0_u, f1_u = make_helpers(u_grid, tau)
    P_new = np.empty_like(P)
    phi_optB(P, u_grid, p_grid, float(tau), float(gamma),
                w_trap, f0_u, f1_u, P_new)
    return P_new


# ============================ Tester ============================
if __name__ == "__main__":
    from lin_cdf_strict import make_cdf_uniform_grid, make_p_grid
    import dd_k3_lookup_alts as ALT
    G = 11; G_p = 121
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    P0 = np.full((G,G,G), 0.5)
    print("JIT warmup...", flush=True); t0 = time.time()
    phi(P0, u_grid, p_grid, 1.0, 1.0)
    print(f"  done {time.time()-t0:.1f}s", flush=True)
    # Cross-check vs scipy version
    import glob
    fp_files = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    if fp_files:
        d = np.load(fp_files[0])
        P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
        if P.shape[0] == G:
            gamma = float(d["gamma"]); tau = float(d["tau"])
            print(f"\nXcheck on {fp_files[0]}: gamma={gamma}, tau={tau}", flush=True)
            mu_numba = np.empty((G_p, G))
            w_trap, f0_u, f1_u = make_helpers(u_grid, tau)
            t0 = time.time()
            for _ in range(5):
                build_mu_optB(P, u_grid, p_grid, tau, w_trap, f0_u, f1_u, mu_numba)
            t_numba = (time.time()-t0)/5
            mu_scipy = ALT.empirical_cdf_density(P, u_grid, p_grid, tau)
            d_max = float(np.max(np.abs(mu_numba - mu_scipy)))
            print(f"  numba build_mu_optB: {t_numba*1000:.1f}ms/call",
                  f" vs scipy: {1e-3}s. max|d|={d_max:.3e}", flush=True)
            t0 = time.time()
            for _ in range(5):
                P_new = phi(P, u_grid, p_grid, tau, gamma)
            t_phi = (time.time()-t0)/5
            F_inf = float(np.max(np.abs(P_new - P)))
            print(f"  full Phi: {t_phi*1000:.1f}ms/call. |F|={F_inf:.3e}", flush=True)
