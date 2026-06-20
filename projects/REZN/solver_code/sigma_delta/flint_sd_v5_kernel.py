"""flint sigma-delta V5 -- KERNEL CO-AREA (matches u-grid kernel approach).

The raw contour scan (V3) under-resolves the Jensen gap at finite G (deficit
~0.02 at gamma=0.1 vs the kernel co-area's 0.28). V5 replaces the scan
with a Gaussian-band kernel sum, the SAME approach the u-grid headline
operator (phi_K3_halo_smooth) uses to nail the genuine PR equilibrium.

For each agent and price target p:
  A_v(p) ~ Sum_{a,b} K_h(P[a,b] - p) * f_v(u_a) * f_v(u_b) * J(xi_a, xi_b)

where:
  K_h(x) = exp(-x^2 / (2 h^2)) is the Gaussian kernel (bandwidth h)
  J(xi_a, xi_b) = 1 / ((1-xi_a^2) * (1-xi_b^2)) is the JACOBIAN for the
                  xi -> physical change of variables (du = TOT/(1-xi^2) dxi)
  Constants (TOT, dxi) cancel in the Bayes ratio.

For agent 1 (own = u_1): sum over the (Sigma, delta) slice P[i, :, :]
  with u_2(xi_S, xi_d) = (Sigma + delta)/2, u_3 = (Sigma - delta)/2.

For agent 2 (own = u_2): build the slice P_slice(u_1, delta) via Sigma-interp
  (Sigma_required = 2*u_2 - delta), then kernel-sum over the (u_1, delta) plane.

Bandwidth: h = C_H * sqrt(dxi) with C_H = 0.45 (matches u-grid choice).
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

def mp(x):
    if isinstance(x, str): return arb(x)
    return arb(float(x))

def fsig(u, vm, tau, coef):
    d = u - vm
    return coef * (mp('-0.5') * tau * d * d).exp()

def crra_demand(mu, p, gamma, W):
    one = mp(1)
    lm = (mu / (one - mu)).log(); lp = (p / (one - p)).log()
    R = ((lm - lp) / gamma).exp()
    return W * (R - one) / ((one - p) + R * p)

def crra_clear_sym(mus, gamma, W, steps=200):
    eps = mp('1e-40'); one = mp(1); two = mp(2); zero = mp(0)
    a, b = eps, one - eps
    for _ in range(steps):
        m = (a + b) / two
        ex = zero
        for mu in mus: ex = ex + crra_demand(mu, m, gamma, W)
        if float(ex) > 0: a = m
        else: b = m
    return (a + b) / two

def interp_along_Sigma(P, i_u, k_d, Sigma_target, xi_S, TOT_S_mp):
    G = len(xi_S); one = mp(1)
    St = float(Sigma_target)
    if St > 1e10: return P[i_u][G-1][k_d]
    if St < -1e10: return P[i_u][0][k_d]
    arg = Sigma_target / TOT_S_mp
    xi_t = arg.tanh(); xt = float(xi_t)
    if xt <= float(xi_S[0]): return P[i_u][0][k_d]
    if xt >= float(xi_S[-1]): return P[i_u][G-1][k_d]
    for j in range(G - 1):
        xj = float(xi_S[j]); xj1 = float(xi_S[j+1])
        if xj <= xt <= xj1:
            denom = xi_S[j+1] - xi_S[j]
            if float(denom) == 0: return P[i_u][j][k_d]
            frac = (xi_t - xi_S[j]) / denom
            return (one - frac) * P[i_u][j][k_d] + frac * P[i_u][j+1][k_d]
    return P[i_u][G-1][k_d]

def evidence_agent1_kernel(P, p_target, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm, i_u, coef, h, J_S_arr, J_d_arr):
    """Kernel co-area for agent 1 (slice P[i_u, :, :] in (Sigma, delta)).
    Sum K_h(P-p_target) * f_v(u_2) * f_v(u_3) * J(xi_S, xi_d) over all (j, k).
    """
    G_S = len(xi_S); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    inv_2h2 = mp('1') / (mp('2') * h * h)
    # Loop over the inner+halo grid; J_S_arr[j] = 1/(1-xi_S[j]^2) precomputed as arb
    # (handle boundary by skipping J that's huge)
    for j in range(G_S):
        if abs(float(xi_S[j])) >= 1 - 1e-15: continue  # skip boundary cells
        Sigma_j = TOT_S_mp * xi_S[j].atanh()
        for k in range(G_d):
            if abs(float(xi_d[k])) >= 1 - 1e-15: continue
            delta_k = TOT_d_mp * xi_d[k].atanh()
            diff = P[i_u][j][k] - p_target
            K = (-diff * diff * inv_2h2).exp()
            u2 = (Sigma_j + delta_k) / two
            u3 = (Sigma_j - delta_k) / two
            f2 = fsig(u2, vm, tau, coef); f3 = fsig(u3, vm, tau, coef)
            A = A + K * f2 * f3 * J_S_arr[j] * J_d_arr[k]
    return A

def evidence_agent_oblique_kernel(P, p_target, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                   tau, vm, u_cell, sign_for_other, coef, h, J_u_arr, J_d_arr):
    """Kernel co-area for agents 2,3 -- slice extracted via Sigma-interp then kernel sum
    over (u_1, delta). sign_for_other = -1 for agent 2 (own=u_2), +1 for agent 3."""
    G_u = len(xi_u1); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    inv_2h2 = mp('1') / (mp('2') * h * h)
    # Build P_slice(i, k) via Sigma-interp at Sigma_req = 2*u_cell + sign*delta_k
    P_slc = [[mp(0) for _ in range(G_d)] for _ in range(G_u)]
    for i in range(G_u):
        for k in range(G_d):
            if abs(float(xi_d[k])) < 1 - 1e-15:
                delta_k = TOT_d_mp * xi_d[k].atanh()
            else:
                delta_k = mp(1e10) * xi_d[k]
            Sigma_req = two * u_cell + sign_for_other * delta_k
            P_slc[i][k] = interp_along_Sigma(P, i, k, Sigma_req, xi_S, TOT_S_mp)
    # Kernel-sum over (u_1, delta)
    for i in range(G_u):
        if abs(float(xi_u1[i])) >= 1 - 1e-15: continue
        u1 = TOT_u_mp * xi_u1[i].atanh()
        f1 = fsig(u1, vm, tau, coef)
        for k in range(G_d):
            if abs(float(xi_d[k])) >= 1 - 1e-15: continue
            delta_k = TOT_d_mp * xi_d[k].atanh()
            u_other = u_cell + sign_for_other * delta_k
            f_oth = fsig(u_other, vm, tau, coef)
            diff = P_slc[i][k] - p_target
            K = (-diff * diff * inv_2h2).exp()
            A = A + K * f1 * f_oth * J_u_arr[i] * J_d_arr[k]
    return A

def phi_sigdelta_v5(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                     INNER_LO, INNER_HI, h, clearing='crra'):
    one = mp(1); two = mp(2); vm0 = mp('-0.5'); vm1 = mp('0.5'); eps_p = mp('1e-40')
    coef = (tau / (two * arb.pi())).sqrt()
    G_u = len(xi_u1)
    # precompute Jacobian arrays J[j] = 1/(1-xi[j]^2) (arb)
    def jac(xi_arr):
        J = []
        for x in xi_arr:
            xf = float(x)
            if abs(xf) >= 1 - 1e-15:
                J.append(mp(0))
            else:
                J.append(one / (one - x*x))
        return J
    J_u = jac(xi_u1); J_S = jac(xi_S); J_d = jac(xi_d)

    P_new = [[[P[i][j][k] for k in range(G_u)] for j in range(G_u)] for i in range(G_u)]
    for i in range(INNER_LO, INNER_HI):
        if abs(float(xi_u1[i])) >= 1 - 1e-15: continue
        u1_cell = TOT_u_mp * xi_u1[i].atanh()
        for j in range(INNER_LO, INNER_HI):
            if abs(float(xi_S[j])) >= 1 - 1e-15: continue
            Sigma_cell = TOT_S_mp * xi_S[j].atanh()
            for k in range(INNER_LO, INNER_HI):
                if abs(float(xi_d[k])) >= 1 - 1e-15: continue
                d_cell = TOT_d_mp * xi_d[k].atanh()
                p_cell = P[i][j][k]
                u2_cell = (Sigma_cell + d_cell) / two
                u3_cell = (Sigma_cell - d_cell) / two
                A1_0 = evidence_agent1_kernel(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm0, i, coef, h, J_S, J_d)
                A1_1 = evidence_agent1_kernel(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm1, i, coef, h, J_S, J_d)
                f0_u1 = fsig(u1_cell, vm0, tau, coef); f1_u1 = fsig(u1_cell, vm1, tau, coef)
                den = f0_u1 * A1_0 + f1_u1 * A1_1
                mu0 = f1_u1 * A1_1 / den if float(den) > 0 else mp('0.5')
                A2_0 = evidence_agent_oblique_kernel(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                      tau, vm0, u2_cell, -one, coef, h, J_u, J_d)
                A2_1 = evidence_agent_oblique_kernel(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                      tau, vm1, u2_cell, -one, coef, h, J_u, J_d)
                f0_u2 = fsig(u2_cell, vm0, tau, coef); f1_u2 = fsig(u2_cell, vm1, tau, coef)
                den = f0_u2 * A2_0 + f1_u2 * A2_1
                mu1 = f1_u2 * A2_1 / den if float(den) > 0 else mp('0.5')
                A3_0 = evidence_agent_oblique_kernel(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                      tau, vm0, u3_cell, one, coef, h, J_u, J_d)
                A3_1 = evidence_agent_oblique_kernel(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                      tau, vm1, u3_cell, one, coef, h, J_u, J_d)
                f0_u3 = fsig(u3_cell, vm0, tau, coef); f1_u3 = fsig(u3_cell, vm1, tau, coef)
                den = f0_u3 * A3_0 + f1_u3 * A3_1
                mu2 = f1_u3 * A3_1 / den if float(den) > 0 else mp('0.5')
                def clipmu(x):
                    xf = float(x)
                    if xf < float(eps_p): return eps_p
                    if xf > float(one - eps_p): return one - eps_p
                    return x
                mus = [clipmu(mu) for mu in (mu0, mu1, mu2)]
                if clearing == 'crra':
                    P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
                else:
                    pi_ = ((mus[0]/(one-mus[0])).log() + (mus[1]/(one-mus[1])).log() + (mus[2]/(one-mus[2])).log()) / mp(3)
                    P_new[i][j][k] = one / (one + (-pi_).exp())
    return P_new

def set_boundary(P):
    G = len(P); one = mp(1); zero = mp(0)
    for j in range(G):
        for k in range(G):
            P[0][j][k] = zero; P[G-1][j][k] = one
    for i in range(G):
        for k in range(G):
            P[i][0][k] = zero; P[i][G-1][k] = one
    for i in range(G):
        for j in range(G):
            P[i][j][0] = P[i][j][1]; P[i][j][G-1] = P[i][j][G-2]
    return P

def f_inf(A, B, INNER_LO, INNER_HI):
    m = mp(0)
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if float(d) > float(m): m = d
    return m

def to_np(P):
    G = len(P)
    return np.array([[[float(P[i][j][k]) for k in range(G)] for j in range(G)] for i in range(G)])
