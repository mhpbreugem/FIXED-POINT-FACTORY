"""Alternative mu-table builders for the K=3 CRRA lookup function.
Five independent options + cross-comparison against strict-h=0.

A: cube-node kernel-band       -- replace bilinear-interp + GL nodes by cube nodes
B: empirical CDF + density     -- PCHIP CDF of f_v over P, differentiate
C: Monte Carlo Bayes            -- sample (u_a, u_b, u_c) ~ N(+-0.5, 1/tau)^3, KDE
D: level-set Monte Carlo        -- rejection-sample on {P=p}, sum f_v
E: distribution of mu over u_k -- diagnostic, not a builder
"""
import os, sys, time
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit
from scipy.interpolate import PchipInterpolator
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)
from lin_cdf_kern_tab import build_mu_table_lin_kern
from cheby_numba import f_signal_jit


# ============================ Option A: cube-node KB ============================

@njit(cache=True)
def build_mu_cube_kern(P_vals, u_grid, p_grid, tau, kernel_h, w_trap, mu_table):
    """Kernel-band using ONLY cube nodes (no bilinear interp, no GL nodes).
    The cube node P values are exact, so the only kinks in the integrand are
    at the cube grid lines themselves (predictable, can be probed with various h)."""
    G = u_grid.size; G_p = p_grid.size
    inv2h2 = 0.5 / (kernel_h * kernel_h)
    for k_node in range(G):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau)
        f1k = f_signal_jit(u_k, 1, tau)
        for ip in range(G_p):
            p = p_grid[ip]
            A0 = 0.0; A1 = 0.0
            for i in range(G):
                f0a = f_signal_jit(u_grid[i], 0, tau)
                f1a = f_signal_jit(u_grid[i], 1, tau)
                wi = w_trap[i]
                for j in range(G):
                    f0b = f_signal_jit(u_grid[j], 0, tau)
                    f1b = f_signal_jit(u_grid[j], 1, tau)
                    diff = P_vals[k_node, i, j] - p
                    K = np.exp(-diff*diff*inv2h2)
                    w = wi * w_trap[j] * K
                    A0 += w * f0a * f0b
                    A1 += w * f1a * f1b
            den = f0k * A0 + f1k * A1
            if den > 1e-300: mu_table[ip, k_node] = f1k * A1 / den
            else: mu_table[ip, k_node] = 0.5


def cube_node_mu(P_vals, u_grid, p_grid, tau, kernel_h):
    G = u_grid.size
    du = np.diff(u_grid)
    w_trap = np.empty(G); w_trap[0] = 0.5*du[0]; w_trap[-1] = 0.5*du[-1]
    w_trap[1:-1] = 0.5*(du[:-1] + du[1:])
    mu = np.empty((p_grid.size, G))
    build_mu_cube_kern(P_vals.astype(np.float64), u_grid, p_grid, float(tau),
                          float(kernel_h), w_trap, mu)
    return mu


# ============================ Option B: empirical CDF + density ============================

def empirical_cdf_density(P_vals, u_grid, p_grid, tau):
    """For each u_k, build PCHIP CDF F_v(p | u_k) from cube cells, differentiate
    to get density a_v(p, u_k). Then mu = f1*a1 / (f0*a0 + f1*a1).
    Bandwidth-free (smoothness comes from PCHIP cubic Hermite + spline-derivative)."""
    G = u_grid.size; G_p = p_grid.size
    du = np.diff(u_grid)
    w_trap = np.empty(G); w_trap[0] = 0.5*du[0]; w_trap[-1] = 0.5*du[-1]
    w_trap[1:-1] = 0.5*(du[:-1] + du[1:])
    f0 = np.array([f_signal_jit(u, 0, tau) for u in u_grid])
    f1 = np.array([f_signal_jit(u, 1, tau) for u in u_grid])
    mu_table = np.empty((G_p, G))
    for k_node in range(G):
        # Collect all cube cells in this slice
        P_slice = P_vals[k_node, :, :].ravel()
        i_idx, j_idx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        i_idx = i_idx.ravel(); j_idx = j_idx.ravel()
        w_cell = (w_trap[i_idx] * w_trap[j_idx])
        f0_cell = f0[i_idx] * f0[j_idx]
        f1_cell = f1[i_idx] * f1[j_idx]
        # Sort by P
        order = np.argsort(P_slice)
        P_s = P_slice[order]; w_s = w_cell[order]
        f0_s = f0_cell[order]; f1_s = f1_cell[order]
        # CDFs (cumulative weighted f_v)
        F0 = np.cumsum(w_s * f0_s); F1 = np.cumsum(w_s * f1_s)
        # Build PCHIP for F_v(P) over unique sorted P values
        # Handle ties by aggregating
        Pu, inv = np.unique(P_s, return_inverse=True)
        if len(Pu) < 4:
            mu_table[:, k_node] = 0.5; continue
        F0u = np.zeros(len(Pu)); F1u = np.zeros(len(Pu))
        for s in range(len(P_s)):
            F0u[inv[s]] = F0[s]; F1u[inv[s]] = F1[s]
        # Density via derivative of PCHIP
        try:
            pchip0 = PchipInterpolator(Pu, F0u, extrapolate=False)
            pchip1 = PchipInterpolator(Pu, F1u, extrapolate=False)
            a0 = pchip0.derivative()(p_grid)
            a1 = pchip1.derivative()(p_grid)
            # Bounds: outside [Pu[0], Pu[-1]] density is 0
            mask = (p_grid >= Pu[0]) & (p_grid <= Pu[-1])
            f0k = f_signal_jit(u_grid[k_node], 0, tau)
            f1k = f_signal_jit(u_grid[k_node], 1, tau)
            num = f1k * a1
            den = f0k * a0 + num
            with np.errstate(divide="ignore", invalid="ignore"):
                mu_col = np.where(np.abs(den) > 1e-30, num / den, 0.5)
            mu_col = np.nan_to_num(mu_col, nan=0.5, posinf=0.5, neginf=0.5)
            mu_col = np.clip(mu_col, 1e-9, 1-1e-9)
            mu_col[~mask] = 0.5
            mu_table[:, k_node] = mu_col
        except Exception:
            mu_table[:, k_node] = 0.5
    return mu_table


# ============================ Option C: Monte Carlo Bayes ============================

def mc_bayes_mu(P_vals, u_grid, p_grid, tau, N_samples=200_000, kde_h_p=0.02,
                kde_h_u=0.5, seed=0):
    """Monte Carlo: sample (u_a, u_b, u_c) ~ N(vm, 1/tau)^3 for v=0,1 separately
    (vm=-0.5 or +0.5). For each sample compute P via trilinear interp on the cube.
    Then 2D KDE of (P, u_k) per v. mu = f1_KDE / (f0_KDE + f1_KDE) (proportional to
    posterior odds, since the v-specific samples already include f_v in their density).
    Returns mu_table of shape (G_p, G).
    """
    rng = np.random.default_rng(seed)
    G = u_grid.size; G_p = p_grid.size
    sigma = 1.0 / np.sqrt(tau)
    # Trilinear interp helper (vectorized on a sample batch)
    def trilin_P(uvals):
        """Trilinear interp of P_vals on u_grid at sample points uvals (N, 3)."""
        out = np.empty(uvals.shape[0])
        for s in range(uvals.shape[0]):
            ua, ub, uc = uvals[s]
            # find brackets
            def find(u):
                if u <= u_grid[0]: return 0, 0.0
                if u >= u_grid[-1]: return G-2, 1.0
                lo = 0; hi = G-1
                while hi - lo > 1:
                    m = (lo+hi)//2
                    if u_grid[m] <= u: lo = m
                    else: hi = m
                w = (u - u_grid[lo]) / (u_grid[hi] - u_grid[lo])
                return lo, w
            ia, wa = find(ua); ib, wb = find(ub); ic, wc = find(uc)
            v = 0.0
            for da, fa in [(0, 1-wa), (1, wa)]:
                for db, fb in [(0, 1-wb), (1, wb)]:
                    for dc, fc in [(0, 1-wc), (1, wc)]:
                        v += fa*fb*fc * P_vals[ia+da, ib+db, ic+dc]
            out[s] = v
        return out
    mu_table = np.full((G_p, G), 0.5)
    for v in [0, 1]:
        vm = -0.5 if v == 0 else 0.5
        # Sample
        samps = rng.normal(loc=vm, scale=sigma, size=(N_samples, 3))
        Ps = trilin_P(samps)
        # Pick "u_k" axis: use samps[:, 2] (the third component, by symmetry any)
        # For each (p, u_k) target: KDE density ~ Σ exp(-(P-p)^2/(2h_p^2)) exp(-(u-u_k)^2/(2h_u^2))
        inv2hp2 = 0.5 / (kde_h_p**2)
        inv2hu2 = 0.5 / (kde_h_u**2)
        for ip in range(G_p):
            p_t = p_grid[ip]
            wP = np.exp(-(Ps - p_t)**2 * inv2hp2)
            for k_node in range(G):
                u_t = u_grid[k_node]
                wU = np.exp(-(samps[:, 2] - u_t)**2 * inv2hu2)
                w_tot = wP * wU
                den_v = w_tot.sum()
                if den_v < 1e-300: continue
                if v == 0:
                    mu_table[ip, k_node] = -den_v        # stash f0 contribution
                else:
                    f0_part = -mu_table[ip, k_node]
                    f1_part = den_v
                    mu_table[ip, k_node] = f1_part / (f0_part + f1_part)
    return mu_table


# ============================ Option D: level-set MC ============================

def levelset_mc_mu(P_vals, u_grid, p_grid, tau, N_samples=2_000_000,
                       band_delta=0.01, seed=0):
    """Rejection sample (u_a, u_b, u_c) ~ N(0, 2/tau)^3 (centered on 0).
    Keep samples with |P - p_target| < band_delta. Compute A_v = Σ f_v(u_a) f_v(u_b)
    f_v(u_k) over kept samples, normalized. mu = f1*A1 / (f0*A0 + f1*A1).
    For each (p, u_k) we partition kept samples by which u_k-bracket they fall in.
    """
    rng = np.random.default_rng(seed)
    G = u_grid.size; G_p = p_grid.size
    sigma_q = np.sqrt(2.0/tau)
    # Sample proposal once
    samps = rng.normal(scale=sigma_q, size=(N_samples, 3))
    # P at samples via trilinear
    Ps = np.empty(N_samples)
    for s in range(N_samples):
        ua, ub, uc = samps[s]
        def find(u):
            if u <= u_grid[0]: return 0, 0.0
            if u >= u_grid[-1]: return G-2, 1.0
            lo = 0; hi = G-1
            while hi - lo > 1:
                m = (lo+hi)//2
                if u_grid[m] <= u: lo = m
                else: hi = m
            w = (u - u_grid[lo]) / (u_grid[hi] - u_grid[lo])
            return lo, w
        ia, wa = find(ua); ib, wb = find(ub); ic, wc = find(uc)
        v = 0.0
        for da, fa in [(0, 1-wa), (1, wa)]:
            for db, fb in [(0, 1-wb), (1, wb)]:
                for dc, fc in [(0, 1-wc), (1, wc)]:
                    v += fa*fb*fc * P_vals[ia+da, ib+db, ic+dc]
        Ps[s] = v
    # Importance weights to "un-bias" the q proposal
    iw = np.exp(-0.5*tau*(samps[:, 0]**2 + samps[:, 1]**2 + samps[:, 2]**2)) \
            / np.exp(-0.25*tau*(samps[:, 0]**2 + samps[:, 1]**2 + samps[:, 2]**2))
    # f_v(u_a) f_v(u_b) f_v(u_k)
    coef = np.sqrt(tau / (2*np.pi))
    f0_a = coef * np.exp(-0.5*tau*(samps[:, 0]+0.5)**2)
    f1_a = coef * np.exp(-0.5*tau*(samps[:, 0]-0.5)**2)
    f0_b = coef * np.exp(-0.5*tau*(samps[:, 1]+0.5)**2)
    f1_b = coef * np.exp(-0.5*tau*(samps[:, 1]-0.5)**2)
    f0_k = coef * np.exp(-0.5*tau*(samps[:, 2]+0.5)**2)
    f1_k = coef * np.exp(-0.5*tau*(samps[:, 2]-0.5)**2)
    # For each (p, u_k) compute density
    mu_table = np.empty((G_p, G))
    sigma_u = (u_grid[-1] - u_grid[0]) / G    # u-bin width
    for ip in range(G_p):
        p_t = p_grid[ip]
        in_band = np.abs(Ps - p_t) < band_delta
        if not in_band.any():
            mu_table[ip, :] = 0.5; continue
        kept = np.where(in_band)[0]
        u_kept = samps[kept, 2]
        for k_node in range(G):
            u_t = u_grid[k_node]
            wU = np.exp(-0.5 * ((u_kept - u_t)/sigma_u)**2)
            A0 = np.sum(iw[kept] * wU * f0_a[kept] * f0_b[kept] * f0_k[kept])
            A1 = np.sum(iw[kept] * wU * f1_a[kept] * f1_b[kept] * f1_k[kept])
            den = A0 + A1
            mu_table[ip, k_node] = A1 / den if den > 1e-300 else 0.5
    return mu_table


# ============================ Option E: distribution of mu over u_k ============================

def mu_distribution_diagnostic(mu_table, p_grid):
    """Across u_k axis at each p, compute the distribution statistics
    (mean, std, percentiles)."""
    G_p, G = mu_table.shape
    mean = mu_table.mean(axis=1)
    std = mu_table.std(axis=1)
    p25 = np.percentile(mu_table, 25, axis=1)
    p75 = np.percentile(mu_table, 75, axis=1)
    return dict(mean=mean.tolist(), std=std.tolist(),
                p25=p25.tolist(), p75=p75.tolist())
