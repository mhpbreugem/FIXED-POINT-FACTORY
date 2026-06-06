"""Two-step test:

Step 1: Replay the smooth (Gaussian-kernel) K=3 operator from the saved
        Pa_G9.npy FP. Expect Phi(Pa) - Pa = ~7e-16 (1 iter to machine eps).

Step 2: Build a tabulated-mu version of the SAME smooth operator:
          mu_table(p, u_k) = bayes(u_k, A0(p,u_k), A1(p,u_k))
          built once over a 1D p-grid; per-cube-cell clearing uses it.
        Test: does the table version also nail to machine eps from Pa?
"""
import os, sys, time, json
import numpy as np
# Use the package via its parent path so relative imports work
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')

from reznsrc.contour_K3_halo import (init_no_learning_K3, phi_K3_halo_smooth,
                                        _agent_evidence_K3_smooth, _bayes)
from reznsrc.signals import f_signal
from reznsrc.demand import clear_crra
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

# Grid setup -- matches verify_test1.py exactly
K = 3; pad = 2; UMAX = 4.0
TAU = 2.0; GAMMA = 0.1; C = 0.45
G_inner = 9
du = 2 * UMAX / (G_inner - 1)
Gf = G_inner + 2 * pad   # 13
uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
lo, hi = pad, pad + G_inner
tv = np.full(K, TAU); gv = np.full(K, GAMMA); wv = np.full(K, 1.0)
h = C * du ** 0.5

print(f'Grid: G_inner={G_inner}, G_full={Gf}, du={du}, h={h}')
print(f'    inner cells = [{lo}:{hi}], halo padding = {pad}')
print(f'    tau={TAU}, gamma={GAMMA}\n')

# Load saved FP
Pa = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_verify_coarea/Pa_G9.npy')
print(f'Pa_G9 shape: {Pa.shape}, range: [{Pa.min():.6e}, {Pa.max():.6e}]')

# Pa is inner-only? or with halo? Check shape
if Pa.shape == (G_inner, G_inner, G_inner):
    # Build halo from no-learning and embed Pa in the interior
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P_full[lo:hi, lo:hi, lo:hi] = Pa
    print(f'Pa is inner-only; embedded into G_full={Gf} with no-learning halo')
elif Pa.shape == (Gf, Gf, Gf):
    P_full = Pa.copy()
    print(f'Pa is full G={Gf}; using directly')
else:
    print(f'!!! Unexpected shape; trying inner G={G_inner} cube')
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P_full[lo:hi, lo:hi, lo:hi] = Pa

# ===== STEP 1: Replay smooth operator =====
print('\n=== STEP 1: replay phi_K3_halo_smooth ===')
print('JIT warmup...')
_ = phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, wv, h)
t0 = time.time()
P_new = phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, wv, h)
dt = time.time() - t0
F = P_new[lo:hi, lo:hi, lo:hi] - P_full[lo:hi, lo:hi, lo:hi]
Ferr = float(np.max(np.abs(F)))
print(f'1 iter of smooth phi: ||F||_inf = {Ferr:.3e} (target: ~7e-16)')
print(f'wall time: {dt*1000:.1f} ms')

# Newton-Krylov nail just to confirm
def resid(x):
    P = P_full.copy()
    P[lo:hi, lo:hi, lo:hi] = x.reshape((G_inner,)*3)
    return (phi_K3_halo_smooth(P, uf, lo, hi, tv, gv, wv, h)
            - P)[lo:hi, lo:hi, lo:hi].ravel()
x0 = P_full[lo:hi, lo:hi, lo:hi].ravel().copy()
print(f'\nFirst residual via resid(): {float(np.max(np.abs(resid(x0)))):.3e}')

# ===== STEP 2: Tabulated mu version of the SAME smooth operator =====
print('\n=== STEP 2: tabulated-mu version (lookup table) ===')

from numba import njit, prange

@njit(cache=True)
def build_mu_table_smooth(P_full, u_full, p_grid, tau_vec, kernel_h,
                            n_full, G_p):
    """Build mu(p, u_k_index) using the SAME Gaussian-kernel co-area as
    phi_K3_halo_smooth. By S_3 symmetry of P (we assume it), one table per
    axis suffices. Here we build the table from agent 0's perspective:
    slice = P[axis0_idx, :, :], tau_o0=tau[1], tau_o1=tau[2]."""
    mu_table = np.empty((G_p, n_full))
    inv_2h2 = 0.5 / (kernel_h * kernel_h)
    for k_node in range(n_full):
        u_k = u_full[k_node]
        # Pre-compute f_v(u_a) f_v(u_b) tables (G_full x G_full each)
        f0a = np.empty(n_full); f1a = np.empty(n_full)
        f0b = np.empty(n_full); f1b = np.empty(n_full)
        for ia in range(n_full):
            f0a[ia] = f_signal(u_full[ia], 0, tau_vec[1])
            f1a[ia] = f_signal(u_full[ia], 1, tau_vec[1])
        for ib in range(n_full):
            f0b[ib] = f_signal(u_full[ib], 0, tau_vec[2])
            f1b[ib] = f_signal(u_full[ib], 1, tau_vec[2])
        slice2 = P_full[k_node, :, :]
        for ip in range(G_p):
            p = p_grid[ip]
            A0 = 0.0; A1 = 0.0
            for ia in range(n_full):
                for ib in range(n_full):
                    diff = slice2[ia, ib] - p
                    w = np.exp(-diff * diff * inv_2h2)
                    A0 += w * f0a[ia] * f0b[ib]
                    A1 += w * f1a[ia] * f1b[ib]
            mu_table[ip, k_node] = _bayes(u_k, tau_vec[0], A0, A1)
    return mu_table


@njit(cache=True, inline='always')
def _interp_p(mu_table, p, p_grid, k_idx, G_p):
    if p <= p_grid[0]: return mu_table[0, k_idx]
    if p >= p_grid[G_p-1]: return mu_table[G_p-1, k_idx]
    # binary search
    lo_i, hi_i = 0, G_p - 1
    while hi_i - lo_i > 1:
        mid = (lo_i + hi_i) // 2
        if p_grid[mid] <= p: lo_i = mid
        else: hi_i = mid
    w = (p - p_grid[lo_i]) / (p_grid[hi_i] - p_grid[lo_i])
    return (1.0 - w) * mu_table[lo_i, k_idx] + w * mu_table[hi_i, k_idx]


@njit(cache=True, parallel=True)
def phi_K3_halo_smooth_tab(P_full, u_full, p_grid, lo_inner, hi_inner,
                              tau_vec, gamma_vec, W_vec, kernel_h):
    n_full = u_full.size
    G_p = p_grid.size
    mu_table = build_mu_table_smooth(P_full, u_full, p_grid, tau_vec,
                                        kernel_h, n_full, G_p)
    P_new = P_full.copy()
    for i in prange(lo_inner, hi_inner):
        mu_vec = np.empty(3)
        for j in range(lo_inner, hi_inner):
            for l in range(lo_inner, hi_inner):
                p = P_full[i, j, l]
                mu_vec[0] = _interp_p(mu_table, p, p_grid, i, G_p)
                mu_vec[1] = _interp_p(mu_table, p, p_grid, j, G_p)
                mu_vec[2] = _interp_p(mu_table, p, p_grid, l, G_p)
                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new


# Build p-grid: logit-uniform
G_p = 51
p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, G_p)))
print('JIT warmup of tab version...')
_ = phi_K3_halo_smooth_tab(P_full, uf, p_grid, lo, hi, tv, gv, wv, h)

print('\nTAB version: 1 iter from Pa:')
for G_p_try in [21, 31, 51, 81, 121, 201]:
    p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, G_p_try)))
    t0 = time.time()
    P_new_tab = phi_K3_halo_smooth_tab(P_full, uf, p_grid, lo, hi,
                                          tv, gv, wv, h)
    dt = time.time() - t0
    F = P_new_tab[lo:hi, lo:hi, lo:hi] - P_full[lo:hi, lo:hi, lo:hi]
    Ferr = float(np.max(np.abs(F)))
    print(f'  G_p={G_p_try:>3d}: ||F||_inf = {Ferr:.3e}  ({dt*1000:.1f} ms)')

# Compare original vs tab at G_p=121 across the inner cube
P_new = phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, wv, h)
p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, 121)))
P_new_tab = phi_K3_halo_smooth_tab(P_full, uf, p_grid, lo, hi, tv, gv, wv, h)
diff = float(np.max(np.abs(P_new[lo:hi,lo:hi,lo:hi] - P_new_tab[lo:hi,lo:hi,lo:hi])))
print(f'\nmax|phi_smooth - phi_smooth_tab(G_p=121)| in inner cube: {diff:.3e}')

# Newton-Krylov nail with tab version
print('\n=== STEP 2b: Newton-Krylov nail with tab operator (G_p=121) ===')
p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, 121)))
def resid_tab(x):
    P = P_full.copy()
    P[lo:hi, lo:hi, lo:hi] = x.reshape((G_inner,)*3)
    return (phi_K3_halo_smooth_tab(P, uf, p_grid, lo, hi, tv, gv, wv, h)
            - P)[lo:hi, lo:hi, lo:hi].ravel()
x0 = P_full[lo:hi, lo:hi, lo:hi].ravel().copy()
print(f'Initial resid_tab: {float(np.max(np.abs(resid_tab(x0)))):.3e}')
try:
    x_sol = newton_krylov(resid_tab, x0, f_tol=1e-13, maxiter=50, verbose=False)
    f_final = float(np.max(np.abs(resid_tab(x_sol))))
    print(f'After Newton-Krylov nail: {f_final:.3e}')
except NoConvergence as e:
    x_sol = e.args[0]
    f_final = float(np.max(np.abs(resid_tab(x_sol))))
    print(f'Newton-Krylov did NOT converge: residual = {f_final:.3e}')
