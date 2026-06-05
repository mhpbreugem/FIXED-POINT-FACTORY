"""Minimal Chebyshev spectral h=0 solver for K=3 CRRA REE.

Architecture:
  - Coordinates: ξ_k = tanh(u_k/c), c=2.0
  - Storage: P as values at Chebyshev-Lobatto nodes, shape (N+1, N+1, N+1)
  - Operator: For each cell, compute new P via:
      For each agent k:
        Get slice P[i_k=fixed, :, :] (2D Chebyshev field)
        For each GL quadrature node ξ_a^q in transverse axis a:
          Get 1D polynomial of P along contour axis b (chebfit)
          Find all roots where P = p_target (chebroots)
          At each root: f_v(u_a) * f_v(u_b) / |dP/du_b|
          Accumulate co-area integral with GL weight
      Bayes for each agent
      CRRA clearing

Pure Newton with forward-diff Jacobian.

Status: minimal proof-of-concept. NO symmetry reduction yet.
NO sigmoid lift (chebroots needs polynomial).
"""
import os, time, math, json
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, chebroots, chebder
from scipy.optimize import brentq

# ===== Parameters =====
TAU = 1.0
GAMMA = 1.0   # CRRA risk aversion
C_STRETCH = 2.0  # ξ = tanh(u / c)
N = 6   # Chebyshev order per axis
NQ = 12  # Gauss-Legendre nodes for transverse quadrature
N_GRID = N + 1  # 7 grid points per axis
EPS_PRICE = 1e-9

# ===== Coordinate maps =====
def u_of_xi(xi):
    return C_STRETCH * np.arctanh(np.clip(xi, -0.9999, 0.9999))

def dudxi(xi):
    # du/dξ = c/(1-ξ²)
    return C_STRETCH / (1.0 - np.clip(xi, -0.9999, 0.9999)**2)

def xi_of_u(u):
    return np.tanh(u / C_STRETCH)

# Chebyshev-Lobatto nodes
LOBATTO = -np.cos(np.pi * np.arange(N_GRID) / N)
U_NODES = u_of_xi(LOBATTO)

# Signal density
def f_signal(u, v):
    vm = 0.5 if v == 1 else -0.5
    return np.sqrt(TAU/(2*np.pi)) * np.exp(-0.5*TAU*(u-vm)**2)

# CRRA clearing (vectorized over single call)
def crra_clear(mu0, mu1, mu2, gamma, steps=200):
    eps = 1e-30
    a = eps; b = 1.0 - eps
    me = [max(min(m, 1-EPS_PRICE), EPS_PRICE) for m in (mu0, mu1, mu2)]
    lm = [np.log(m/(1-m)) for m in me]
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = np.log(m/(1-m))
        e = 0.0
        for lmk in lm:
            arg = (lmk - lp)/gamma
            if arg > 700: e += 1.0/m
            elif arg < -700: pass  # R=0, demand = -1/(1-m)*something, but tiny
            else:
                R = np.exp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)

# ===== Chebyshev tensor utilities =====
# Convert P (values at Lobatto grid) ↔ Chebyshev coefficients
def vals_to_coeffs_3d(P_vals):
    """Convert 3D Lobatto-grid values to 3D Chebyshev coefficients."""
    # Apply DCT-like operation along each axis
    # For Lobatto nodes, chebfit on each 1D fiber gives coefficients
    G = P_vals.shape[0]
    coeffs = P_vals.copy()
    # Axis 0
    for j in range(G):
        for k in range(G):
            coeffs[:, j, k] = chebfit(LOBATTO, coeffs[:, j, k], N)
    # Axis 1
    for i in range(G):
        for k in range(G):
            coeffs[i, :, k] = chebfit(LOBATTO, coeffs[i, :, k], N)
    # Axis 2
    for i in range(G):
        for j in range(G):
            coeffs[i, j, :] = chebfit(LOBATTO, coeffs[i, j, :], N)
    return coeffs

def coeffs_to_vals_3d(coeffs):
    """Convert 3D Chebyshev coefficients back to Lobatto-grid values."""
    G = coeffs.shape[0]
    vals = coeffs.copy()
    for j in range(G):
        for k in range(G):
            vals[:, j, k] = chebval(LOBATTO, vals[:, j, k])
    for i in range(G):
        for k in range(G):
            vals[i, :, k] = chebval(LOBATTO, vals[i, :, k])
    for i in range(G):
        for j in range(G):
            vals[i, j, :] = chebval(LOBATTO, vals[i, j, :])
    return vals

# ===== Operator phi (the heart) =====
def evaluate_P_slice_at_xi(coeffs, axis_fix, idx_fix, xi_a, xi_b):
    """Evaluate P at (axis_fix=fixed_node, ξ_a, ξ_b) for axes a, b.
    axis_fix ∈ {0,1,2}; idx_fix is the Lobatto node index on that axis.
    Returns scalar P value.
    """
    # Get the 2D slice of coefficients at the fixed node
    # First contract along axis_fix using T_n(ξ_fix) values
    xi_fix = LOBATTO[idx_fix]
    # Compute T_n(xi_fix) for n=0..N
    T_fix = np.array([chebval(xi_fix, np.eye(1, N_GRID, n).ravel()) for n in range(N_GRID)])
    if axis_fix == 0:
        slice_coeffs = np.einsum('ijk,i->jk', coeffs, T_fix)
    elif axis_fix == 1:
        slice_coeffs = np.einsum('ijk,j->ik', coeffs, T_fix)
    else:
        slice_coeffs = np.einsum('ijk,k->ij', coeffs, T_fix)
    # Now slice_coeffs is 2D Cheb tensor in (axes a,b)
    # Evaluate at (xi_a, xi_b)
    T_a = np.array([chebval(xi_a, np.eye(1, N_GRID, n).ravel()) for n in range(N_GRID)])
    T_b = np.array([chebval(xi_b, np.eye(1, N_GRID, n).ravel()) for n in range(N_GRID)])
    return float(np.einsum('mn,m,n->', slice_coeffs, T_a, T_b))

def get_1d_chebyshev_in_xi_b(slice_coeffs_2d, xi_a):
    """Given 2D slice Chebyshev coeffs (axes a, b), and fixed ξ_a,
    return 1D Chebyshev coefficients in ξ_b."""
    T_a = np.array([chebval(xi_a, np.eye(1, N_GRID, n).ravel()) for n in range(N_GRID)])
    return slice_coeffs_2d.T @ T_a  # shape (N+1,)

def co_area_evidence(slice_coeffs_2d, p_target, gl_nodes, gl_weights):
    """Compute (A_0, A_1) on a 2D slice by GL quadrature in axis_a +
    chebroots in axis_b.
    slice_coeffs_2d has shape (N+1, N+1) (axes a, b).
    """
    A0, A1 = 0.0, 0.0
    for q, w in zip(gl_nodes, gl_weights):
        xi_a = q  # GL node in [-1, 1]
        # Get 1D Chebyshev poly along axis_b at this xi_a
        c1d_b = get_1d_chebyshev_in_xi_b(slice_coeffs_2d, xi_a)
        # Roots where P(xi_b) = p_target
        c1d_b_shifted = c1d_b.copy(); c1d_b_shifted[0] -= p_target
        try:
            all_roots = chebroots(c1d_b_shifted)
        except Exception:
            continue
        # Keep real roots in (-1, 1)
        real_roots = [float(r.real) for r in all_roots
                       if abs(r.imag) < 1e-10 and -1 < r.real < 1]
        if not real_roots: continue
        u_a = u_of_xi(xi_a); dudxi_a = dudxi(xi_a)
        f0a = f_signal(u_a, 0); f1a = f_signal(u_a, 1)
        # Derivative coefficients
        c1d_b_der = chebder(c1d_b)
        for xi_b_root in real_roots:
            dPdxi_b = chebval(xi_b_root, c1d_b_der)
            if abs(dPdxi_b) < 1e-12: continue
            u_b = u_of_xi(xi_b_root); dudxi_b = dudxi(xi_b_root)
            f0b = f_signal(u_b, 0); f1b = f_signal(u_b, 1)
            dPdu_b = dPdxi_b / dudxi_b
            wt = w * dudxi_a * dudxi_b / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return 0.5*A0, 0.5*A1  # average over... hmm actually no PoU here

def phi_one_cell(coeffs, i, j, k, gl_nodes, gl_weights, gamma):
    """Compute the new P value at cell (i, j, k)."""
    # Current p at this cell (from coefficients)
    p_old = float(evaluate_P_slice_at_xi(coeffs, 0, i,
                                            LOBATTO[j], LOBATTO[k]))
    p_old = max(min(p_old, 1-EPS_PRICE), EPS_PRICE)
    # Get 2D slices for each agent
    # Agent 1: axis_fix = 0 (fix u_1), integrate over (xi_2, xi_3)
    # Agent 2: axis_fix = 1 (fix u_2), integrate over (xi_1, xi_3)
    # Agent 3: axis_fix = 2 (fix u_3), integrate over (xi_1, xi_2)
    # Get the 2D slice coefficients for each
    mus = []
    for k_agent, idx_fix in enumerate([i, j, k]):
        axis_fix = k_agent
        xi_fix = LOBATTO[idx_fix]
        T_fix = np.array([chebval(xi_fix, np.eye(1, N_GRID, n).ravel()) for n in range(N_GRID)])
        if axis_fix == 0:
            slice2 = np.einsum('ijk,i->jk', coeffs, T_fix)
        elif axis_fix == 1:
            slice2 = np.einsum('ijk,j->ik', coeffs, T_fix)
        else:
            slice2 = np.einsum('ijk,k->ij', coeffs, T_fix)
        A0, A1 = co_area_evidence(slice2, p_old, gl_nodes, gl_weights)
        u_own = u_of_xi(xi_fix)
        f0 = f_signal(u_own, 0); f1 = f_signal(u_own, 1)
        den = f0*A0 + f1*A1
        mu = (f1*A1)/den if den > 1e-30 else 0.5
        mus.append(mu)
    return crra_clear(mus[0], mus[1], mus[2], gamma)

def phi(P_vals, gamma):
    """Apply the operator to the full cube of values."""
    coeffs = vals_to_coeffs_3d(P_vals)
    G = N_GRID
    # GL nodes for transverse quadrature
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(NQ)  # on [-1, 1]
    P_new = P_vals.copy()
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = phi_one_cell(coeffs, i, j, k, gl_nodes, gl_weights, gamma)
    return P_new

# ===== Quick test =====
if __name__ == '__main__':
    print(f'Cheby h=0 MIZN solver: τ={TAU}, γ={GAMMA}, N={N}, NQ={NQ}, c={C_STRETCH}')
    print(f'Lobatto nodes ξ: {LOBATTO}')
    print(f'u-nodes: {U_NODES}')
    # No-learning IC
    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T_grid = TAU*(U1+U2+U3)
    mu0_grid = sg(TAU*U1); mu1_grid = sg(TAU*U2); mu2_grid = sg(TAU*U3)
    P_IC = np.empty_like(U1)
    for i in range(N_GRID):
        for j in range(N_GRID):
            for k in range(N_GRID):
                P_IC[i,j,k] = crra_clear(mu0_grid[i,j,k], mu1_grid[i,j,k], mu2_grid[i,j,k], GAMMA)
    print(f'IC P range: [{P_IC.min():.4f}, {P_IC.max():.4f}]')
    # Apply phi once
    print('Applying phi once...', flush=True)
    t = time.time()
    P_new = phi(P_IC, GAMMA)
    print(f'  phi took {time.time()-t:.1f}s')
    print(f'  P_new range: [{P_new.min():.4f}, {P_new.max():.4f}]')
    ferr = float(np.max(np.abs(P_new - P_IC)))
    print(f'  ||F||_∞ = ||phi(P_IC) - P_IC||_∞ = {ferr:.3e}')
    # Save
    np.save('/tmp/cheby_h0/P_IC.npy', P_IC)
    np.save('/tmp/cheby_h0/P_after_one_phi.npy', P_new)
    print('saved')
