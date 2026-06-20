"""V5 KERNEL CO-AREA on sigma-delta grid -- FAST NUMPY float64 version.

Same algorithm as flint_sd_v5_kernel.py but pure-numpy with vectorized
inner sums. Per-cell kernel sum becomes an einsum / matrix product over
the (a, b) grid axes; outer (i, j, k) loop in Python.

Verified at G=10 to match the flint V5 result (when run with the same h).

Usage: import phi_sigdelta_v5_np and call from a driver.
"""
import numpy as np
import math

TAU_DEFAULT = 2.0

def make_grids(G_full, TOT_u, TOT_S, TOT_d):
    """Return xi_inner (no boundary), u_phys, S_phys, d_phys for axes."""
    xi_full = np.linspace(-1.0, 1.0, G_full)
    return xi_full

def precompute(G_full, INNER_LO, INNER_HI, TOT_u, TOT_S, TOT_d, TAU):
    """Precompute density / Jacobian arrays needed by the kernel sum.

    Returns a dict with arrays:
      xi_full[i]               (G_full,) ξ-values
      u_full[i], S_full[i], d_full[i]  physical values (boundary -> ±inf placeholder)
      J_u[i], J_S[i], J_d[i]   Jacobians 1/(1-xi^2); 0 at the boundary
      f0_u[i], f1_u[i]         signal densities for own-u_1 axis
      sg_grid_full[i,j]        sigmoid table not needed -- using direct f formulas
      f_v_S[j], f_v_d[k]       not needed -- u_2 and u_3 depend on BOTH (S, d)
      f_v_2d_0/1[j, k]         f_v(u_2(S[j], d[k])) * f_v(u_3(S[j], d[k])) for v=0, 1
      J_2d[j, k]               J_S[j] * J_d[k]
      f0_S/d, f1_S/d           not directly used
    """
    xi_full = np.linspace(-1.0, 1.0, G_full)
    # physical values; boundary (xi = +-1) gives inf, we mark with 0 in f-grids
    safe = np.clip(xi_full, -0.9999999, 0.9999999)
    u_full = TOT_u * np.arctanh(safe)
    S_full = TOT_S * np.arctanh(safe)
    d_full = TOT_d * np.arctanh(safe)
    interior = np.abs(xi_full) < 1 - 1e-15
    J_u = np.where(interior, 1.0/(1.0-xi_full**2), 0.0)
    J_S = np.where(interior, 1.0/(1.0-xi_full**2), 0.0)
    J_d = np.where(interior, 1.0/(1.0-xi_full**2), 0.0)
    # 2D inner-axis grids (for agent 1: axes are S, d)
    Sgrid, Dgrid = np.meshgrid(S_full, d_full, indexing='ij')
    U2_2d = 0.5 * (Sgrid + Dgrid)
    U3_2d = 0.5 * (Sgrid - Dgrid)
    coef = math.sqrt(TAU / (2.0 * math.pi))
    f0_2d = coef * np.exp(-0.5*TAU*(U2_2d + 0.5)**2) * coef * np.exp(-0.5*TAU*(U3_2d + 0.5)**2)
    f1_2d = coef * np.exp(-0.5*TAU*(U2_2d - 0.5)**2) * coef * np.exp(-0.5*TAU*(U3_2d - 0.5)**2)
    # zero out boundary rows / cols (xi = +-1)
    bd = ~interior
    f0_2d[bd, :] = 0; f0_2d[:, bd] = 0
    f1_2d[bd, :] = 0; f1_2d[:, bd] = 0
    J_2d = np.outer(J_S, J_d)
    # 1D own-signal densities
    f0_u = coef * np.exp(-0.5*TAU*(u_full + 0.5)**2)
    f1_u = coef * np.exp(-0.5*TAU*(u_full - 0.5)**2)
    f0_u[bd] = 0; f1_u[bd] = 0
    # Pre-baked agent-1 weight tensors
    W0_a1 = f0_2d * J_2d  # (G_full, G_full)
    W1_a1 = f1_2d * J_2d
    return dict(xi_full=xi_full, u_full=u_full, S_full=S_full, d_full=d_full,
                interior=interior, J_u=J_u, J_S=J_S, J_d=J_d,
                f0_u=f0_u, f1_u=f1_u, W0_a1=W0_a1, W1_a1=W1_a1,
                INNER_LO=INNER_LO, INNER_HI=INNER_HI, TAU=TAU,
                TOT_u=TOT_u, TOT_S=TOT_S, TOT_d=TOT_d, coef=coef)


def _interp_sigma(P, i_u, k_d, Sigma_target, xi_full, TOT_S, idx_cache=None):
    """Linear interp of P[i_u, :, k_d] at xi corresponding to Sigma_target."""
    G = xi_full.size
    if Sigma_target > 1e10: return P[i_u, G-1, k_d]
    if Sigma_target < -1e10: return P[i_u, 0, k_d]
    xi_t = math.tanh(Sigma_target / TOT_S)
    if xi_t <= xi_full[0]: return P[i_u, 0, k_d]
    if xi_t >= xi_full[-1]: return P[i_u, G-1, k_d]
    j = int(np.searchsorted(xi_full, xi_t) - 1)
    j = max(0, min(G-2, j))
    denom = xi_full[j+1] - xi_full[j]
    if denom == 0: return P[i_u, j, k_d]
    frac = (xi_t - xi_full[j]) / denom
    return (1-frac) * P[i_u, j, k_d] + frac * P[i_u, j+1, k_d]


def _build_slice_oblique(P, u_cell, sign_for_other, pre):
    """Build P_slice[a, b] of shape (G_full, G_full) where
      a indexes u_1 (axis 0 of P)
      b indexes delta (axis 2 of P)
      Sigma_req(b) = 2*u_cell + sign_for_other*delta[b]
    using linear Sigma-interp.
    """
    G = pre['xi_full'].size
    xi_full = pre['xi_full']; TOT_S = pre['TOT_S']; d_full = pre['d_full']
    interior = pre['interior']
    # Compute Sigma_req at each b
    Sigma_req = np.empty(G)
    # at boundary b, use sign * 1e10
    for b in range(G):
        if interior[b]:
            Sigma_req[b] = 2*u_cell + sign_for_other * d_full[b]
        else:
            Sigma_req[b] = 2*u_cell + sign_for_other * math.copysign(1e10, xi_full[b])
    # for each b, find xi_t and the bracketing interval
    P_slc = np.empty((G, G))
    for b in range(G):
        St = Sigma_req[b]
        if St > 1e10:
            P_slc[:, b] = P[:, G-1, b]; continue
        if St < -1e10:
            P_slc[:, b] = P[:, 0, b]; continue
        xi_t = math.tanh(St / TOT_S)
        if xi_t <= xi_full[0]: P_slc[:, b] = P[:, 0, b]; continue
        if xi_t >= xi_full[-1]: P_slc[:, b] = P[:, G-1, b]; continue
        j = int(np.searchsorted(xi_full, xi_t) - 1)
        j = max(0, min(G-2, j))
        denom = xi_full[j+1] - xi_full[j]
        if denom == 0:
            P_slc[:, b] = P[:, j, b]; continue
        frac = (xi_t - xi_full[j]) / denom
        P_slc[:, b] = (1-frac) * P[:, j, b] + frac * P[:, j+1, b]
    return P_slc


def phi_sigdelta_v5_np(P, pre, gamma, h, clearing='crra'):
    """One Picard step of V5 kernel co-area sigma-delta operator (numpy float64).
    P: shape (G_full,)*3, full-grid including boundary halo.
    Returns P_new (boundary unchanged here -- caller should re-apply set_boundary).
    """
    G_full = P.shape[0]
    INNER_LO = pre['INNER_LO']; INNER_HI = pre['INNER_HI']
    TAU = pre['TAU']
    xi_full = pre['xi_full']; u_full = pre['u_full']; S_full = pre['S_full']; d_full = pre['d_full']
    J_u = pre['J_u']; J_S = pre['J_S']; J_d = pre['J_d']
    f0_u = pre['f0_u']; f1_u = pre['f1_u']
    W0_a1 = pre['W0_a1']; W1_a1 = pre['W1_a1']
    interior = pre['interior']
    TOT_u = pre['TOT_u']; TOT_S = pre['TOT_S']; TOT_d = pre['TOT_d']
    coef = pre['coef']
    inv_2h2 = 0.5 / (h * h)
    eps_p = 1e-40

    P_new = P.copy()

    # ----- Agent 1 ----- vectorized over (j, k) for each i
    # For each i (in inner range), slice = P[i, :, :] (G_full, G_full)
    # For each inner cell (j, k), p_target = P[i, j, k]
    # diff[j, k, a, b] = P[i, a, b] - P[i, j, k]
    # K = exp(-diff^2 * inv_2h2)
    # A0[j, k] = einsum('jkab,ab->jk', K, W0_a1)
    # A1[j, k] = einsum('jkab,ab->jk', K, W1_a1)

    # To save memory at large G, loop over i but vectorize over (j, k)
    G_inner = INNER_HI - INNER_LO
    A0_a1 = np.zeros((G_inner, G_inner, G_inner))
    A1_a1 = np.zeros((G_inner, G_inner, G_inner))
    for i in range(INNER_LO, INNER_HI):
        if not interior[i]: continue
        slc = P[i, :, :]  # (G_full, G_full)
        p_inner = P[i, INNER_LO:INNER_HI, INNER_LO:INNER_HI]  # (G_inner, G_inner)
        # diff[j, k, a, b]
        diff = slc[None, None, :, :] - p_inner[:, :, None, None]
        K = np.exp(-diff*diff*inv_2h2)
        # contract over (a, b)
        A0_a1[i-INNER_LO] = np.einsum('jkab,ab->jk', K, W0_a1)
        A1_a1[i-INNER_LO] = np.einsum('jkab,ab->jk', K, W1_a1)
    f0_u_inner = f0_u[INNER_LO:INNER_HI]; f1_u_inner = f1_u[INNER_LO:INNER_HI]
    # mu0[ii, j, k] = f1_u[i]*A1 / (f0_u[i]*A0 + f1_u[i]*A1)
    den = f0_u_inner[:, None, None] * A0_a1 + f1_u_inner[:, None, None] * A1_a1
    num = f1_u_inner[:, None, None] * A1_a1
    mu0 = np.where(den > 0, num/den, 0.5)
    mu0 = np.clip(mu0, eps_p, 1-eps_p)

    # ----- Agents 2 and 3 ----- loop over the inner block; vectorize inner kernel sum
    mu1 = np.empty((G_inner, G_inner, G_inner))
    mu2 = np.empty((G_inner, G_inner, G_inner))
    for j in range(G_inner):
        S_j = S_full[INNER_LO + j]
        for k in range(G_inner):
            d_k = d_full[INNER_LO + k]
            u_2_cell = 0.5 * (S_j + d_k)
            u_3_cell = 0.5 * (S_j - d_k)
            # Agent 2: sign = -1 for delta
            P_slc_a2 = _build_slice_oblique(P, u_2_cell, -1.0, pre)  # (G_full, G_full) = (u_1, delta)
            # Per inner i, p_target = P[i, j, k] (already inner)
            # Density weights: for each (a, b): f_v(u_1[a]) * f_v(u_other[b])
            # u_other for agent 2 = u_2_cell - delta[b]  (varies with b)
            # For each b, f_v_other = f_v(u_2_cell - d_full[b])
            # Precompute u_other_b and f_v_other_b
            u_other_b = u_2_cell - d_full  # (G_full,)
            f0_oth = coef * np.exp(-0.5*TAU*(u_other_b + 0.5)**2) * interior.astype(float)
            f1_oth = coef * np.exp(-0.5*TAU*(u_other_b - 0.5)**2) * interior.astype(float)
            # Build W0_a2[a, b] = f0_u[a] * f0_oth[b] * J_u[a] * J_d[b]
            W0_a2 = (f0_u[:, None] * f0_oth[None, :]) * np.outer(J_u, J_d)
            W1_a2 = (f1_u[:, None] * f1_oth[None, :]) * np.outer(J_u, J_d)
            # Inner sum over i:
            p_inner_a2 = P[INNER_LO:INNER_HI, j+INNER_LO, k+INNER_LO]  # (G_inner,) -- p_target at fixed (j,k) for each i
            # Wait that's the SAME (j, k) for all i. But the inner cell varies i.
            # We want mu1 at each inner cell (i, j, k); p_target = P[i, j, k].
            for ii in range(G_inner):
                i = ii + INNER_LO
                if not interior[i]: mu1[ii, j, k] = 0.5; continue
                p_t = P[i, j+INNER_LO, k+INNER_LO]
                diff = P_slc_a2 - p_t  # (G_full, G_full)
                K = np.exp(-diff*diff*inv_2h2)
                A0 = (K * W0_a2).sum(); A1 = (K * W1_a2).sum()
                f0o = f0_u[i]; f1o = f1_u[i]
                f0_own_2 = coef * math.exp(-0.5*TAU*(u_2_cell + 0.5)**2)
                f1_own_2 = coef * math.exp(-0.5*TAU*(u_2_cell - 0.5)**2)
                den2 = f0_own_2 * A0 + f1_own_2 * A1
                mu1[ii, j, k] = f1_own_2 * A1 / den2 if den2 > 0 else 0.5
            # Agent 3: sign = +1
            P_slc_a3 = _build_slice_oblique(P, u_3_cell, +1.0, pre)
            u_other_b3 = u_3_cell + d_full
            f0_oth3 = coef * np.exp(-0.5*TAU*(u_other_b3 + 0.5)**2) * interior.astype(float)
            f1_oth3 = coef * np.exp(-0.5*TAU*(u_other_b3 - 0.5)**2) * interior.astype(float)
            W0_a3 = (f0_u[:, None] * f0_oth3[None, :]) * np.outer(J_u, J_d)
            W1_a3 = (f1_u[:, None] * f1_oth3[None, :]) * np.outer(J_u, J_d)
            for ii in range(G_inner):
                i = ii + INNER_LO
                if not interior[i]: mu2[ii, j, k] = 0.5; continue
                p_t = P[i, j+INNER_LO, k+INNER_LO]
                diff = P_slc_a3 - p_t
                K = np.exp(-diff*diff*inv_2h2)
                A0 = (K * W0_a3).sum(); A1 = (K * W1_a3).sum()
                f0_own_3 = coef * math.exp(-0.5*TAU*(u_3_cell + 0.5)**2)
                f1_own_3 = coef * math.exp(-0.5*TAU*(u_3_cell - 0.5)**2)
                den3 = f0_own_3 * A0 + f1_own_3 * A1
                mu2[ii, j, k] = f1_own_3 * A1 / den3 if den3 > 0 else 0.5
    mu1 = np.clip(mu1, eps_p, 1-eps_p)
    mu2 = np.clip(mu2, eps_p, 1-eps_p)

    # ----- clearing -----
    if clearing == 'cara':
        pi_ = (np.log(mu0/(1-mu0)) + np.log(mu1/(1-mu1)) + np.log(mu2/(1-mu2))) / 3.0
        P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = 1.0/(1.0+np.exp(-pi_))
    else:
        # CRRA: per-cell bisection. Vectorize across all inner cells.
        P_inner_new = _crra_clear_sym_vec(mu0, mu1, mu2, gamma, 1.0, steps=120)
        P_new[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_inner_new
    return P_new


def _crra_clear_sym_vec(mu0, mu1, mu2, gamma, W, steps=120):
    """Bisection clearing for CRRA with symmetric weights, vectorized."""
    eps = 1e-30
    a = np.full_like(mu0, eps)
    b = np.full_like(mu0, 1 - eps)
    lm0 = np.log(mu0/(1-mu0)); lm1 = np.log(mu1/(1-mu1)); lm2 = np.log(mu2/(1-mu2))
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = np.log(m/(1-m))
        R0 = np.exp((lm0-lp)/gamma)
        R1 = np.exp((lm1-lp)/gamma)
        R2 = np.exp((lm2-lp)/gamma)
        e = W*(R0-1)/((1-m)+R0*m) + W*(R1-1)/((1-m)+R1*m) + W*(R2-1)/((1-m)+R2*m)
        a = np.where(e > 0, m, a)
        b = np.where(e > 0, b, m)
    return 0.5*(a+b)


def set_boundary_np(P):
    G = P.shape[0]
    P[0, :, :] = 0.0
    P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0
    P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G-1] = P[:, :, G-2]
    return P
