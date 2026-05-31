"""V6 = V5 kernel co-area with CUBIC (Catmull-Rom) Sigma-interp for agents 2,3.

Bug in V5: linear Sigma-interp builds P_slice(u_1, delta) for the oblique
slice; P(Sigma) is sigmoid-shaped (highly nonlinear), so linear interp has
O(dxi^2) error per knot. This biased the kernel sum for agents 2,3, drifting
the FP away from FR (V5 CARA G=25 gave slope=0.628 instead of 1.0).

Fix: 4-point Catmull-Rom cubic interp. On uniform xi-grid, the cubic uses
knots [j-1, j, j+1, j+2] surrounding the target xi_t. At boundary edges
(j=0 or G-2), fall back to linear. Cubic error is O(dxi^4) per cell --
much smaller than linear's O(dxi^2).
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

# Reuse most primitives from V5
from flint_sd_v5_kernel import (mp, fsig, crra_clear_sym,
                                 evidence_agent1_kernel, set_boundary, f_inf, to_np)


def interp_along_Sigma_cubic(P, i_u, k_d, Sigma_target, xi_S, TOT_S_mp):
    """Catmull-Rom cubic interp along the Sigma axis of P[i_u, :, k_d].
    Uniform xi-grid: knots [j-1, j, j+1, j+2] around xi_t. Linear fallback at edges."""
    G = len(xi_S); one = mp(1)
    St = float(Sigma_target)
    if St > 1e10: return P[i_u][G-1][k_d]
    if St < -1e10: return P[i_u][0][k_d]
    arg = Sigma_target / TOT_S_mp
    xi_t = arg.tanh(); xt = float(xi_t)
    if xt <= float(xi_S[0]): return P[i_u][0][k_d]
    if xt >= float(xi_S[-1]): return P[i_u][G-1][k_d]
    # find bracketing interval [j, j+1]
    for j in range(G - 1):
        if float(xi_S[j]) <= xt <= float(xi_S[j+1]):
            break
    denom = xi_S[j+1] - xi_S[j]
    if float(denom) == 0: return P[i_u][j][k_d]
    t_frac = (xi_t - xi_S[j]) / denom
    t = t_frac
    # Get 4 P values (use linear fallback if at boundary)
    if j == 0 or j == G - 2:
        # linear fallback
        return (one - t_frac) * P[i_u][j][k_d] + t_frac * P[i_u][j+1][k_d]
    P0 = P[i_u][j-1][k_d]; P1 = P[i_u][j][k_d]
    P2 = P[i_u][j+1][k_d]; P3 = P[i_u][j+2][k_d]
    # Catmull-Rom cubic (uniform spacing): 0.5*(2P1 + (-P0+P2)t + (2P0-5P1+4P2-P3)t^2 + (-P0+3P1-3P2+P3)t^3)
    two = mp(2); three = mp(3); four = mp(4); five = mp(5); half = mp('0.5')
    c0 = two * P1
    c1 = (-P0 + P2) * t
    c2 = (two*P0 - five*P1 + four*P2 - P3) * t * t
    c3 = (-P0 + three*P1 - three*P2 + P3) * t * t * t
    return half * (c0 + c1 + c2 + c3)


def evidence_agent_oblique_kernel_cubic(P, p_target, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                         tau, vm, u_cell, sign_for_other, coef, h, J_u_arr, J_d_arr):
    """Same as V5 but uses CUBIC Sigma-interp."""
    G_u = len(xi_u1); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    inv_2h2 = mp('1') / (mp('2') * h * h)
    P_slc = [[mp(0) for _ in range(G_d)] for _ in range(G_u)]
    for i in range(G_u):
        for k in range(G_d):
            if abs(float(xi_d[k])) < 1 - 1e-15:
                delta_k = TOT_d_mp * xi_d[k].atanh()
            else:
                delta_k = mp(1e10) * xi_d[k]
            Sigma_req = two * u_cell + sign_for_other * delta_k
            P_slc[i][k] = interp_along_Sigma_cubic(P, i, k, Sigma_req, xi_S, TOT_S_mp)
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


def phi_sigdelta_v6(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                     INNER_LO, INNER_HI, h, clearing='crra'):
    one = mp(1); two = mp(2); vm0 = mp('-0.5'); vm1 = mp('0.5'); eps_p = mp('1e-40')
    coef = (tau / (two * arb.pi())).sqrt()
    G_u = len(xi_u1)
    def jac(xi_arr):
        J = []
        for x in xi_arr:
            xf = float(x)
            if abs(xf) >= 1 - 1e-15: J.append(mp(0))
            else: J.append(one / (one - x*x))
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
                # Agent 1: same as V5 (no Sigma-interp needed)
                A1_0 = evidence_agent1_kernel(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm0, i, coef, h, J_S, J_d)
                A1_1 = evidence_agent1_kernel(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm1, i, coef, h, J_S, J_d)
                f0_u1 = fsig(u1_cell, vm0, tau, coef); f1_u1 = fsig(u1_cell, vm1, tau, coef)
                den = f0_u1 * A1_0 + f1_u1 * A1_1
                mu0 = f1_u1 * A1_1 / den if float(den) > 0 else mp('0.5')
                # Agent 2: CUBIC interp
                A2_0 = evidence_agent_oblique_kernel_cubic(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                            tau, vm0, u2_cell, -one, coef, h, J_u, J_d)
                A2_1 = evidence_agent_oblique_kernel_cubic(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                            tau, vm1, u2_cell, -one, coef, h, J_u, J_d)
                f0_u2 = fsig(u2_cell, vm0, tau, coef); f1_u2 = fsig(u2_cell, vm1, tau, coef)
                den = f0_u2 * A2_0 + f1_u2 * A2_1
                mu1 = f1_u2 * A2_1 / den if float(den) > 0 else mp('0.5')
                # Agent 3: CUBIC interp
                A3_0 = evidence_agent_oblique_kernel_cubic(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                            tau, vm0, u3_cell, one, coef, h, J_u, J_d)
                A3_1 = evidence_agent_oblique_kernel_cubic(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
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
