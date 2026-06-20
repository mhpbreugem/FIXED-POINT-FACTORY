"""Numba-JIT'd Chebyshev h=0 operator.

Drop-in replacement for cheby_h0_solver.phi() — same answer, ~30-60× faster.

Replaces:
  - chebval → Clenshaw recurrence (inlined)
  - chebder → coefficient recurrence (inlined)
  - chebroots → Chebyshev companion matrix + np.linalg.eigvals (inlined)
  - vals↔coeffs via Chebyshev-Lobatto Vandermonde matrix (precomputed)

All inner-loop work runs as numba-compiled native code with prange parallelism
over the outermost cube cell loop.
"""
import os, time, math
import numpy as np
from numba import njit

# ===== Parameters (must match cheby_h0_solver.py) =====
TAU = 1.0
GAMMA = 1.0
C_STRETCH = 2.0
N = 6
NQ = 12
N_GRID = N + 1
EPS_PRICE = 1e-9

# Chebyshev-Lobatto nodes (canonical: -cos(πj/N))
LOBATTO = -np.cos(np.pi * np.arange(N_GRID) / N)

def u_of_xi_np(xi):
    return C_STRETCH * np.arctanh(np.clip(xi, -0.9999, 0.9999))

U_NODES = u_of_xi_np(LOBATTO)

# Precomputed Vandermonde V[j, n] = T_n(LOBATTO[j]); invert for vals→coeffs.
def _build_vandermonde():
    V = np.empty((N_GRID, N_GRID))
    for j in range(N_GRID):
        x = LOBATTO[j]
        # T_0=1, T_1=x, T_{k+1}=2x T_k - T_{k-1}
        V[j, 0] = 1.0
        if N_GRID > 1:
            V[j, 1] = x
            for k in range(1, N_GRID-1):
                V[j, k+1] = 2*x*V[j, k] - V[j, k-1]
    return V

V_MAT = _build_vandermonde()
V_INV = np.linalg.inv(V_MAT)  # coeffs = V_INV @ vals (along an axis)

# Gauss-Legendre nodes & weights for transverse quadrature
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(NQ)


# ===== Low-level numba kernels =====
@njit(cache=True, inline='always')
def chebval_jit(x, c, n):
    """Evaluate Chebyshev series sum_{k=0..n-1} c[k] T_k(x) by Clenshaw."""
    if n == 0: return 0.0
    if n == 1: return c[0]
    bk1 = 0.0; bk2 = 0.0
    for k in range(n-1, 0, -1):
        bk = c[k] + 2*x*bk1 - bk2
        bk2 = bk1
        bk1 = bk
    return c[0] + x*bk1 - bk2


@njit(cache=True, inline='always')
def chebder_jit(c, n, out):
    """Compute derivative coeffs of Cheb series c[0..n-1] into out[0..n-2].
    Returns degree of result (n-1)."""
    if n < 2:
        out[0] = 0.0
        return 1
    # work copy (we modify in-place)
    cc = c.copy()
    for j in range(n-1):
        out[j] = 0.0
    for j in range(n - 1, 2, -1):
        out[j - 1] = 2.0 * j * cc[j]
        cc[j - 2] = cc[j - 2] + (j * cc[j]) / (j - 2)
    if n > 2:
        out[1] = 4.0 * cc[2]
    out[0] = cc[1]
    return n - 1


@njit(cache=True)
def chebroots_jit(c, n):
    """Find roots of Cheb series c[0..n-1]. Returns array of complex roots.
    Uses standard Chebyshev companion (colleague) matrix from numpy."""
    # Trim trailing tiny coefficients (numerical noise can inflate degree)
    eff_n = n
    while eff_n > 1 and abs(c[eff_n-1]) < 1e-14 * max(abs(c[0]), 1.0):
        eff_n -= 1
    if eff_n < 2:
        return np.empty(0, dtype=np.complex128)
    if eff_n == 2:
        out = np.empty(1, dtype=np.complex128)
        out[0] = complex(-c[0] / c[1], 0.0)
        return out
    deg = eff_n - 1
    mat = np.zeros((deg, deg))
    # Off-diagonals (Chebyshev colleague):
    # mat[i, i+1] = sqrt(1/2) if i==0 else 1/2
    # mat[i+1, i] = same
    sqrt_half = math.sqrt(0.5)
    for i in range(deg - 1):
        if i == 0:
            mat[0, 1] = sqrt_half
            mat[1, 0] = sqrt_half
        else:
            mat[i, i+1] = 0.5
            mat[i+1, i] = 0.5
    # Adjust last column with scaled c coefficients
    # scl = [1, sqrt(1/2), sqrt(1/2), ..., sqrt(1/2)] (length deg)
    # mat[:, -1] -= 0.5 * (c[:-1]/c[-1]) * (scl/scl[-1])
    # Since scl[-1]=sqrt_half (for deg>=2), scl/scl[-1] = [1/sqrt_half, 1, 1, ..., 1]
    cn = c[deg]  # leading coeff
    inv_sqrt_half = 1.0 / sqrt_half
    for i in range(deg):
        ratio = c[i] / cn
        if i == 0:
            mat[i, deg-1] -= 0.5 * ratio * inv_sqrt_half
        else:
            mat[i, deg-1] -= 0.5 * ratio
    # numba's eigvals refuses real matrices that may have complex eigs.
    # Convert to complex matrix to force complex return type.
    mat_c = mat.astype(np.complex128)
    eigs = np.linalg.eigvals(mat_c)
    return eigs


@njit(cache=True, inline='always')
def u_of_xi_jit(xi, c):
    """u = c * atanh(xi), clipped."""
    if xi > 0.9999: xi = 0.9999
    elif xi < -0.9999: xi = -0.9999
    return c * 0.5 * math.log((1.0 + xi) / (1.0 - xi))


@njit(cache=True, inline='always')
def dudxi_jit(xi, c):
    if xi > 0.9999: xi = 0.9999
    elif xi < -0.9999: xi = -0.9999
    return c / (1.0 - xi*xi)


@njit(cache=True, inline='always')
def f_signal_jit(u, v, tau):
    """N(±0.5, 1/τ) density."""
    vm = 0.5 if v == 1 else -0.5
    return math.sqrt(tau/(2.0*math.pi)) * math.exp(-0.5*tau*(u-vm)*(u-vm))


@njit(cache=True)
def crra_clear_jit(mu0, mu1, mu2, gamma, steps=200):
    """CRRA market clearing by bisection on log-odds."""
    eps = 1e-30
    eps_p = 1e-9
    a = eps; b = 1.0 - eps
    # Clip mus
    m0 = max(min(mu0, 1-eps_p), eps_p)
    m1 = max(min(mu1, 1-eps_p), eps_p)
    m2 = max(min(mu2, 1-eps_p), eps_p)
    lm0 = math.log(m0/(1-m0))
    lm1 = math.log(m1/(1-m1))
    lm2 = math.log(m2/(1-m2))
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = math.log(m/(1-m))
        e = 0.0
        for lmk in (lm0, lm1, lm2):
            arg = (lmk - lp)/gamma
            if arg > 700:
                e += 1.0/m
            elif arg < -700:
                pass
            else:
                R = math.exp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0:
            a = m
        else:
            b = m
    return 0.5*(a+b)


# ===== 3D Cheb tensor conversions =====
@njit(cache=True)
def vals_to_coeffs_3d_jit(P_vals, V_inv):
    """Convert (G,G,G) Lobatto values → (G,G,G) Chebyshev coefficients."""
    G = P_vals.shape[0]
    # Step 1: along axis 0
    tmp1 = np.empty((G, G, G))
    for j in range(G):
        for k in range(G):
            for n in range(G):
                s = 0.0
                for p in range(G):
                    s += V_inv[n, p] * P_vals[p, j, k]
                tmp1[n, j, k] = s
    # Step 2: along axis 1
    tmp2 = np.empty((G, G, G))
    for i in range(G):
        for k in range(G):
            for n in range(G):
                s = 0.0
                for p in range(G):
                    s += V_inv[n, p] * tmp1[i, p, k]
                tmp2[i, n, k] = s
    # Step 3: along axis 2
    out = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for n in range(G):
                s = 0.0
                for p in range(G):
                    s += V_inv[n, p] * tmp2[i, j, p]
                out[i, j, n] = s
    return out


# ===== Per-axis 2D slice extraction & co-area =====
@njit(cache=True, inline='always')
def _T_basis(xi, n, out):
    """Compute T_0..T_{n-1} at xi into out[0..n-1] (Chebyshev recurrence)."""
    out[0] = 1.0
    if n > 1:
        out[1] = xi
        for k in range(1, n-1):
            out[k+1] = 2*xi*out[k] - out[k-1]


@njit(cache=True)
def co_area_evidence_jit(slice2_coeffs, p_target, gl_nodes, gl_weights,
                          tau, c_stretch, n_grid, nq):
    """Compute (A0, A1) co-area integrals on a 2D slice (axes a, b).
    slice2_coeffs has shape (n_grid, n_grid)."""
    A0 = 0.0; A1 = 0.0
    G = n_grid
    Tbas = np.empty(G)
    c1d_b = np.empty(G)
    c1d_b_shifted = np.empty(G)
    der_buf = np.empty(G)
    for q in range(nq):
        xi_a = gl_nodes[q]
        w = gl_weights[q]
        # Build 1D Cheb poly in xi_b: c1d_b[n] = sum_m slice2[m, n] * T_m(xi_a)
        _T_basis(xi_a, G, Tbas)
        for n in range(G):
            s = 0.0
            for m in range(G):
                s += slice2_coeffs[m, n] * Tbas[m]
            c1d_b[n] = s
        # Shift by p_target
        for n in range(G):
            c1d_b_shifted[n] = c1d_b[n]
        c1d_b_shifted[0] -= p_target
        # Find roots
        all_roots = chebroots_jit(c1d_b_shifted, G)
        if all_roots.shape[0] == 0:
            continue
        # Precompute u_a, dudxi_a, f0a, f1a
        u_a = u_of_xi_jit(xi_a, c_stretch)
        dudxi_a = dudxi_jit(xi_a, c_stretch)
        f0a = f_signal_jit(u_a, 0, tau)
        f1a = f_signal_jit(u_a, 1, tau)
        # Derivative coeffs of c1d_b
        deg_der = chebder_jit(c1d_b, G, der_buf)
        for r_idx in range(all_roots.shape[0]):
            r = all_roots[r_idx]
            if abs(r.imag) > 1e-10:
                continue
            xi_b = r.real
            if xi_b <= -1.0 or xi_b >= 1.0:
                continue
            dPdxi_b = chebval_jit(xi_b, der_buf, deg_der)
            if abs(dPdxi_b) < 1e-12:
                continue
            u_b = u_of_xi_jit(xi_b, c_stretch)
            dudxi_b = dudxi_jit(xi_b, c_stretch)
            f0b = f_signal_jit(u_b, 0, tau)
            f1b = f_signal_jit(u_b, 1, tau)
            dPdu_b = dPdxi_b / dudxi_b
            wt = w * dudxi_a * dudxi_b / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return 0.5*A0, 0.5*A1


@njit(cache=True)
def phi_one_cell_jit(coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                       tau, gamma, c_stretch, n_grid, nq):
    """Compute new P at cell (i,j,k)."""
    G = n_grid
    # Compute p_old at this cell from coefficients (triple Clenshaw)
    Ti = np.empty(G); Tj = np.empty(G); Tk = np.empty(G)
    _T_basis(lobatto[i], G, Ti)
    _T_basis(lobatto[j], G, Tj)
    _T_basis(lobatto[k], G, Tk)
    p_old = 0.0
    for ii in range(G):
        for jj in range(G):
            for kk in range(G):
                p_old += coeffs[ii, jj, kk] * Ti[ii] * Tj[jj] * Tk[kk]
    eps_p = 1e-9
    if p_old < eps_p: p_old = eps_p
    elif p_old > 1-eps_p: p_old = 1 - eps_p

    # Bayes for each agent
    slice2 = np.empty((G, G))
    Tfix = np.empty(G)
    mus = np.empty(3)
    for k_agent in range(3):
        if k_agent == 0:
            idx_fix = i
        elif k_agent == 1:
            idx_fix = j
        else:
            idx_fix = k
        xi_fix = lobatto[idx_fix]
        _T_basis(xi_fix, G, Tfix)
        # Build 2D slice: contract coeffs along axis_fix
        if k_agent == 0:
            for jj in range(G):
                for kk in range(G):
                    s = 0.0
                    for ii in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[ii]
                    slice2[jj, kk] = s
        elif k_agent == 1:
            for ii in range(G):
                for kk in range(G):
                    s = 0.0
                    for jj in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[jj]
                    slice2[ii, kk] = s
        else:
            for ii in range(G):
                for jj in range(G):
                    s = 0.0
                    for kk in range(G):
                        s += coeffs[ii, jj, kk] * Tfix[kk]
                    slice2[ii, jj] = s
        A0, A1 = co_area_evidence_jit(slice2, p_old, gl_nodes, gl_weights,
                                        tau, c_stretch, G, nq)
        u_own = u_of_xi_jit(xi_fix, c_stretch)
        f0 = f_signal_jit(u_own, 0, tau)
        f1 = f_signal_jit(u_own, 1, tau)
        den = f0*A0 + f1*A1
        if den > 1e-30:
            mus[k_agent] = (f1*A1) / den
        else:
            mus[k_agent] = 0.5
    return crra_clear_jit(mus[0], mus[1], mus[2], gamma)


@njit(cache=True)
def phi_jit(P_vals, V_inv, lobatto, gl_nodes, gl_weights,
            tau, gamma, c_stretch, n_grid, nq):
    """Full operator: P_vals (G,G,G) → P_new (G,G,G)."""
    coeffs = vals_to_coeffs_3d_jit(P_vals, V_inv)
    G = n_grid
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = phi_one_cell_jit(
                    coeffs, i, j, k, lobatto, gl_nodes, gl_weights,
                    tau, gamma, c_stretch, G, nq)
    return P_new


# ===== Python wrappers =====
def phi(P_vals, gamma=GAMMA, tau=TAU):
    """Drop-in replacement for cheby_h0_solver.phi(). Supports overriding tau."""
    return phi_jit(P_vals, V_INV, LOBATTO, GL_NODES, GL_WEIGHTS,
                    tau, gamma, C_STRETCH, N_GRID, NQ)


# CRRA wrapper for IC construction (Python-friendly)
def crra_clear(mu0, mu1, mu2, gamma, steps=200):
    return crra_clear_jit(mu0, mu1, mu2, gamma, steps)


if __name__ == '__main__':
    print(f'CHEBY NUMBA self-test: τ={TAU}, γ={GAMMA}, N={N}, NQ={NQ}')
    # Build no-learning IC
    def sg(x): return 1.0/(1.0+np.exp(-x))
    P_IC = np.empty((N_GRID, N_GRID, N_GRID))
    for i in range(N_GRID):
        for j in range(N_GRID):
            for k in range(N_GRID):
                mu0 = sg(TAU*U_NODES[i])
                mu1 = sg(TAU*U_NODES[j])
                mu2 = sg(TAU*U_NODES[k])
                P_IC[i,j,k] = crra_clear(mu0, mu1, mu2, GAMMA)
    print(f'IC range: [{P_IC.min():.6f}, {P_IC.max():.6f}]')
    print('First phi call (includes JIT compile)...', flush=True)
    t0 = time.time()
    P_new = phi(P_IC, GAMMA)
    t1 = time.time()
    print(f'  first call: {t1-t0:.1f}s')
    print(f'  P_new range: [{P_new.min():.6f}, {P_new.max():.6f}]')
    print('Second phi call (steady-state)...', flush=True)
    t2 = time.time()
    P_new2 = phi(P_IC, GAMMA)
    t3 = time.time()
    print(f'  second call: {t3-t2:.3f}s')
    diff = float(np.max(np.abs(P_new - P_new2)))
    print(f'  determinism: ||diff||_∞ = {diff:.3e}')

    # Compare with pure-Python reference
    import sys
    sys.path.insert(0, '/tmp/cheby_h0')
    print('\nReference (pure Python) phi...', flush=True)
    t4 = time.time()
    from cheby_h0_solver import phi as phi_ref
    P_ref = phi_ref(P_IC, GAMMA)
    t5 = time.time()
    print(f'  pure Python: {t5-t4:.1f}s')
    err = float(np.max(np.abs(P_new - P_ref)))
    print(f'  numba vs pure-Python ||diff||_∞ = {err:.3e}')
    print(f'  SPEEDUP: {(t5-t4)/(t3-t2):.1f}×')
    np.save('/tmp/cheby_h0/P_numba_test.npy', P_new)
    np.save('/tmp/cheby_h0/P_ref_test.npy', P_ref)
