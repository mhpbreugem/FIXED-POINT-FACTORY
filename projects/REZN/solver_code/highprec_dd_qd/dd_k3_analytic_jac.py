"""Analytic Jacobian for the K=3 Lin-CDF R4 kernel-band operator.

Per-Phi cost ~5s; FD Jacobian is 84 Phi calls = 7 min. Analytic Jacobian
does ~1 Phi-equivalent of work for the FULL Jacobian via chain rule, so
~5s. ~80x speedup.

Operator:
  Phase A: mu_R4[ip, k_col] = sum_h w_R4[h] * mu_h[ip, k_col]
           where mu_h[ip, k_col] = f1k*A1_h / (f0k*A0_h + f1k*A1_h)
           and A_v_h[ip, k_col] = sum_{q_a, q_b} ww * K_h(P_at_node-p) * f_v(u_qa) * f_v(u_qb)
           and P_at_node[q_a, q_b] = bilinear interp of P[k_col, :, :] at (gl_u[qa], gl_u[qb])

  Phase B: P_new[i,j,k] = CRRA(mu_a, mu_b, mu_c, gamma)
           where mu_a = interp(mu_R4[:, i], p_cell)
                 mu_b = interp(mu_R4[:, j], p_cell)
                 mu_c = interp(mu_R4[:, k], p_cell)
                 p_cell = P[i, j, k]
"""
import os, sys, time
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numba import njit, prange
from cheby_numba import f_signal_jit, crra_clear_jit
from lin_cdf_kern_tab import build_mu_table_lin_kern, make_cdf_uniform_grid, make_p_grid, make_gl_for_u
from lin_cdf_richardson import richardson_weights, phi_lin_richardson
import math


@njit(cache=True, inline="always")
def _find_bracket(u_grid, q, n):
    if q <= u_grid[0]: return 0, 0.0
    if q >= u_grid[n-1]: return n-2, 1.0
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if u_grid[mid] <= q: lo = mid
        else: hi = mid
    w = (q - u_grid[lo]) / (u_grid[hi] - u_grid[lo])
    return lo, w


@njit(cache=True)
def _crra_dmu(mu0, mu1, mu2, gamma, steps=200):
    """Returns (p_new, dp/dmu0, dp/dmu1, dp/dmu2) via implicit derivative
    of the CRRA market-clearing bisection. dp/dmu is computed analytically
    after solving for p."""
    eps_p = 1e-9
    m0 = max(min(mu0, 1-eps_p), eps_p)
    m1 = max(min(mu1, 1-eps_p), eps_p)
    m2 = max(min(mu2, 1-eps_p), eps_p)
    lm0 = math.log(m0/(1-m0))
    lm1 = math.log(m1/(1-m1))
    lm2 = math.log(m2/(1-m2))
    # Bisection
    a = 1e-30; b = 1-1e-30
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = math.log(m/(1-m))
        e = 0.0
        for lmk in (lm0, lm1, lm2):
            arg = (lmk - lp)/gamma
            if arg > 700: e += 1.0/m
            elif arg < -700: pass
            else:
                R = math.exp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    p = 0.5*(a+b)
    # Implicit derivative: F(p, mu) = sum_k g_k(p, mu_k) = 0
    # dp/dmu_k = -(∂F/∂mu_k) / (∂F/∂p)
    lp = math.log(p/(1-p))
    dFdp = 0.0
    dFdm = [0.0, 0.0, 0.0]
    for kk in range(3):
        m_k = (m0, m1, m2)[kk]
        lm_k = (lm0, lm1, lm2)[kk]
        arg = (lm_k - lp)/gamma
        if arg > 700 or arg < -700:
            continue
        R = math.exp(arg)
        denom_k = (1-p) + R*p
        g_k = (R - 1)/denom_k
        # d g_k / d p = derivative of (R-1)/((1-p) + R*p) wrt p
        # R does not depend on p (here lm_k - lp varies but we already use the bisection solution)
        # Actually R = exp((lm_k - lp)/gamma), lp = log(p/(1-p))
        # d R/ dp = R * (-1/gamma) * (1/(p(1-p)))
        # d denom_k/dp = -1 + p * dR/dp + R
        # d g_k/dp = [(dR/dp)(denom_k) - (R-1)(d denom_k/dp)] / denom_k^2
        dRdp = R * (-1.0/gamma) * (1.0/(p*(1-p)))
        ddenomdp = -1.0 + p*dRdp + R
        dgdp = (dRdp*denom_k - (R-1)*ddenomdp) / (denom_k*denom_k)
        dFdp += dgdp
        # d g_k / d mu_k:
        #   d lm_k/d mu_k = 1/(mu_k(1-mu_k))
        #   d R/d mu_k = R * (1/gamma) * d lm_k/d mu_k = R/(gamma * mu_k(1-mu_k))
        #   d denom_k/d mu_k = p * dR/d mu_k
        dlm_dmu_k = 1.0/(m_k*(1-m_k))
        dRdmu = R * (1.0/gamma) * dlm_dmu_k
        ddenomdmu = p * dRdmu
        dgdmu = (dRdmu*denom_k - (R-1)*ddenomdmu) / (denom_k*denom_k)
        dFdm[kk] = dgdmu
    if abs(dFdp) < 1e-300:
        return p, 0.0, 0.0, 0.0
    return p, -dFdm[0]/dFdp, -dFdm[1]/dFdp, -dFdm[2]/dFdp


@njit(cache=True, parallel=True)
def _build_mu_and_djac(P, u_grid, p_grid, gl_u, gl_du, tau, NQK, hs, w_R4,
                            mu_R4, dmu_dP):
    """Build mu_R4 table and the dmu_R4/dP tensor (per k_col).
       dmu_dP[k_col, ip, i, j] = d mu_R4[ip, k_col] / d P[k_col, i, j]."""
    G = u_grid.size; G_p = p_grid.size; n_h = hs.size
    inv_2h2 = np.empty(n_h)
    for hh in range(n_h):
        inv_2h2[hh] = 0.5 / (hs[hh]*hs[hh])
    # f at GL nodes
    f0_gl = np.empty(NQK); f1_gl = np.empty(NQK)
    for q in range(NQK):
        f0_gl[q] = f_signal_jit(gl_u[q], 0, tau)
        f1_gl[q] = f_signal_jit(gl_u[q], 1, tau)
    for k_col in prange(G):
        u_k = u_grid[k_col]
        f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
        slice2 = P[k_col]  # (G, G)
        # Accumulators across h
        A0_h = np.zeros(n_h); A1_h = np.zeros(n_h)
        # dA[h, ip, i, j] would be big; we accumulate dmu_R4/dP directly
        # Use temp dA0_h and dA1_h per (ip, i, j)
        # Iterate ip outer, accumulate over (q_a, q_b) and corners
        dA0_h = np.zeros((n_h, G_p, G, G))
        dA1_h = np.zeros((n_h, G_p, G, G))
        A0_full = np.zeros((n_h, G_p)); A1_full = np.zeros((n_h, G_p))
        for q_a in range(NQK):
            ua = gl_u[q_a]; wa = gl_du[q_a]
            f0a = f0_gl[q_a]; f1a = f1_gl[q_a]
            ia, w_a = _find_bracket(u_grid, ua, G); omw_a = 1.0 - w_a
            for q_b in range(NQK):
                ub = gl_u[q_b]; wb = gl_du[q_b]
                f0b = f0_gl[q_b]; f1b = f1_gl[q_b]
                ib, w_b = _find_bracket(u_grid, ub, G); omw_b = 1.0 - w_b
                # P_at_node value
                P_at = (omw_a*omw_b*slice2[ia, ib] + w_a*omw_b*slice2[ia+1, ib]
                          + omw_a*w_b*slice2[ia, ib+1] + w_a*w_b*slice2[ia+1, ib+1])
                ww = wa * wb
                ff0 = f0a * f0b; ff1 = f1a * f1b
                # for each h, ip
                for hh in range(n_h):
                    inv2h2 = inv_2h2[hh]
                    for ip in range(G_p):
                        diff = P_at - p_grid[ip]
                        K = math.exp(-diff*diff*inv2h2)
                        contrib_A0 = ww * K * ff0
                        contrib_A1 = ww * K * ff1
                        A0_full[hh, ip] += contrib_A0
                        A1_full[hh, ip] += contrib_A1
                        # dK/dP_at = -diff*inv2h2 (the factor 2 cancels) * K
                        # Wait: K = exp(-d^2 * inv2h2), so dK/dd = -2*d*inv2h2*K = -d/h^2 * K
                        # ...where inv2h2 = 1/(2h^2), so -2*d*inv2h2 = -d/h^2. Same.
                        # d K / dP_at = -diff/h^2 * K = -2*diff*inv2h2*K
                        dKdP_at = -2.0*diff*inv2h2*K
                        # dP_at/dP[k_col, i, j] = bilinear weights
                        # contribute to (ia, ib), (ia+1, ib), (ia, ib+1), (ia+1, ib+1)
                        weights_4 = (omw_a*omw_b, w_a*omw_b, omw_a*w_b, w_a*w_b)
                        idx_4 = ((ia, ib), (ia+1, ib), (ia, ib+1), (ia+1, ib+1))
                        for cc in range(4):
                            i_in, j_in = idx_4[cc]
                            w_corner = weights_4[cc]
                            dA0_h[hh, ip, i_in, j_in] += ww * dKdP_at * w_corner * ff0
                            dA1_h[hh, ip, i_in, j_in] += ww * dKdP_at * w_corner * ff1
        # Combine into mu_R4 and dmu_R4/dP for this k_col
        for ip in range(G_p):
            mu_R = 0.0
            # accumulators for dmu_R4/dP at this (ip)
            dmuR4_dP_local = np.zeros((G, G))
            for hh in range(n_h):
                A0 = A0_full[hh, ip]; A1 = A1_full[hh, ip]
                denom = f0k*A0 + f1k*A1
                if denom > 1e-300:
                    mu_h_val = f1k*A1 / denom
                else:
                    mu_h_val = 0.5
                mu_R += w_R4[hh] * mu_h_val
                # dmu_h/dA0 = -f0k * mu_h_val / denom (since d(f1*A1/(f0*A0+f1*A1))/dA0 = -f0*f1*A1/denom^2 = -mu*f0/denom)
                # dmu_h/dA1 = f1k * (1 - mu_h_val) / denom
                if denom > 1e-300:
                    dmu_dA0 = -f0k * mu_h_val / denom
                    dmu_dA1 = f1k * (1.0 - mu_h_val) / denom
                else:
                    dmu_dA0 = 0.0; dmu_dA1 = 0.0
                for i_in in range(G):
                    for j_in in range(G):
                        dmuR4_dP_local[i_in, j_in] += w_R4[hh] * (
                            dmu_dA0 * dA0_h[hh, ip, i_in, j_in]
                            + dmu_dA1 * dA1_h[hh, ip, i_in, j_in])
            mu_R4[ip, k_col] = mu_R
            for i_in in range(G):
                for j_in in range(G):
                    dmu_dP[k_col, ip, i_in, j_in] = dmuR4_dP_local[i_in, j_in]


@njit(cache=True, inline="always")
def _interp_mu_and_grad(mu_col, p_grid, p_cell, G_p):
    """Return (mu_val, dmu/dp_cell, ip_lo, ip_hi, w_p) — linear interp."""
    if p_cell <= p_grid[0]:
        return mu_col[0], 0.0, 0, 0, 0.0
    if p_cell >= p_grid[G_p-1]:
        return mu_col[G_p-1], 0.0, G_p-1, G_p-1, 0.0
    lo = 0; hi = G_p-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if p_grid[mid] <= p_cell: lo = mid
        else: hi = mid
    h = p_grid[hi] - p_grid[lo]
    w = (p_cell - p_grid[lo]) / h
    mu_val = (1-w)*mu_col[lo] + w*mu_col[hi]
    dmu_dp = (mu_col[hi] - mu_col[lo]) / h
    return mu_val, dmu_dp, lo, hi, w


@njit(cache=True, parallel=True)
def phi_and_jac(P, u_grid, p_grid, gl_u, gl_du, tau, gamma, NQK, hs, w_R4,
                 mu_R4, dmu_dP, P_new, J_full):
    """Returns P_new and J_full (G^3 x G^3 dense Jacobian dPhi/dP)."""
    G = u_grid.size; G_p = p_grid.size
    _build_mu_and_djac(P, u_grid, p_grid, gl_u, gl_du, tau, NQK, hs, w_R4,
                            mu_R4, dmu_dP)
    G3 = G*G*G
    for ii in prange(G):
        for jj in range(G):
            for kk in range(G):
                idx_out = ii*G*G + jj*G + kk
                p_cell = P[ii, jj, kk]
                eps_p = 1e-9
                if p_cell < eps_p: p_cell = eps_p
                elif p_cell > 1-eps_p: p_cell = 1-eps_p
                # mu_a uses column ii
                mu_a, dmu_a_dp, ip_lo_a, ip_hi_a, w_p_a = _interp_mu_and_grad(
                    mu_R4[:, ii], p_grid, p_cell, G_p)
                mu_b, dmu_b_dp, ip_lo_b, ip_hi_b, w_p_b = _interp_mu_and_grad(
                    mu_R4[:, jj], p_grid, p_cell, G_p)
                mu_c, dmu_c_dp, ip_lo_c, ip_hi_c, w_p_c = _interp_mu_and_grad(
                    mu_R4[:, kk], p_grid, p_cell, G_p)
                p_new, dCRRA_dmu_a, dCRRA_dmu_b, dCRRA_dmu_c = _crra_dmu(
                    mu_a, mu_b, mu_c, gamma)
                P_new[ii, jj, kk] = p_new
                # Jacobian row for (ii, jj, kk)
                # Direct term: dP_new/dp_cell * δ(input = (ii,jj,kk))
                direct_chain = dCRRA_dmu_a*dmu_a_dp + dCRRA_dmu_b*dmu_b_dp + dCRRA_dmu_c*dmu_c_dp
                # Initialize row to zeros
                for inp in range(G3):
                    J_full[idx_out, inp] = 0.0
                # Add direct term
                J_full[idx_out, idx_out] += direct_chain
                # Indirect: via mu_a uses column ii (so dmu_R4[:, ii]/dP[ii, i_in, j_in])
                # dmu_a/dP[ii, i_in, j_in] = w_p factor * dmu_R4 at (lo or hi, ii)
                for i_in in range(G):
                    for j_in in range(G):
                        # the input slice index is (ii, i_in, j_in); the index in flat is ii*G^2 + i_in*G + j_in
                        idx_in = ii*G*G + i_in*G + j_in
                        # contribution from mu_a
                        J_full[idx_out, idx_in] += dCRRA_dmu_a * (
                            (1.0-w_p_a)*dmu_dP[ii, ip_lo_a, i_in, j_in]
                            + w_p_a*dmu_dP[ii, ip_hi_a, i_in, j_in])
                # via mu_b uses column jj
                for i_in in range(G):
                    for j_in in range(G):
                        idx_in = jj*G*G + i_in*G + j_in
                        J_full[idx_out, idx_in] += dCRRA_dmu_b * (
                            (1.0-w_p_b)*dmu_dP[jj, ip_lo_b, i_in, j_in]
                            + w_p_b*dmu_dP[jj, ip_hi_b, i_in, j_in])
                # via mu_c uses column kk
                for i_in in range(G):
                    for j_in in range(G):
                        idx_in = kk*G*G + i_in*G + j_in
                        J_full[idx_out, idx_in] += dCRRA_dmu_c * (
                            (1.0-w_p_c)*dmu_dP[kk, ip_lo_c, i_in, j_in]
                            + w_p_c*dmu_dP[kk, ip_hi_c, i_in, j_in])


# ============================ Test ============================
if __name__ == "__main__":
    G = 7; G_p = 121; NQK = 16
    HS = np.array([0.5, 0.4, 0.3, 0.2])
    gamma = 100.0; tau = 0.1
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    w_R4 = richardson_weights(tuple(HS))
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    mu_R4 = np.empty((G_p, G))
    dmu_dP = np.empty((G, G_p, G, G))
    P_new = np.empty_like(P)
    G3 = G*G*G
    J_full = np.empty((G3, G3))
    print(f"Analytic Jacobian build at G={G}, G_p={G_p}, NQK={NQK}...", flush=True)
    print("  JIT warmup..."); t0 = time.time()
    phi_and_jac(P, u_grid, p_grid, gl_u, gl_du, tau, gamma, NQK, HS, w_R4,
                  mu_R4, dmu_dP, P_new, J_full)
    print(f"    warm done {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    for _ in range(3):
        phi_and_jac(P, u_grid, p_grid, gl_u, gl_du, tau, gamma, NQK, HS, w_R4,
                      mu_R4, dmu_dP, P_new, J_full)
    print(f"  analytic Phi+J: {(time.time()-t0)/3*1000:.0f}ms per call", flush=True)
    # Compare to FD Jacobian (one column)
    print("Cross-check vs FD at one column...", flush=True)
    eps = 1e-7
    k_test = 50  # arbitrary index
    i_t, j_t, k_t = k_test // (G*G), (k_test // G) % G, k_test % G
    P_pert = P.copy(); P_pert[i_t, j_t, k_t] += eps
    mu_R4_p = np.empty((G_p, G)); dmu_p = np.empty((G, G_p, G, G))
    P_new_p = np.empty_like(P); J_full_p = np.empty((G3, G3))
    phi_and_jac(P_pert, u_grid, p_grid, gl_u, gl_du, tau, gamma, NQK, HS, w_R4,
                  mu_R4_p, dmu_p, P_new_p, J_full_p)
    fd_col = (P_new_p - P_new).ravel() / eps
    analytic_col = J_full[:, k_test]
    diff = np.abs(fd_col - analytic_col)
    print(f"  max|FD - analytic| = {diff.max():.3e}", flush=True)
    print(f"  med|FD - analytic| = {np.median(diff):.3e}", flush=True)
    print(f"  FD col range:       [{fd_col.min():.3e}, {fd_col.max():.3e}]", flush=True)
    print(f"  analytic col range: [{analytic_col.min():.3e}, {analytic_col.max():.3e}]",
          flush=True)
