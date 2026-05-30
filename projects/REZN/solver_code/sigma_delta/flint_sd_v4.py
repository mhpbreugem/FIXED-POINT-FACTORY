"""flint sigma-delta V4 -- adds the missing Jacobian + co-area weights.

The bug fixed here: the contour scan was summing f_v(u_a) * f_v(u_b) at each
crossing WITHOUT the proper geometric weight. The proper integral in (Sigma,
delta) space, after substitution to uniform xi-grid, requires per-crossing
weight:

  w = 1 / [(1 - xi_a^2) * (1 - xi_b^2) * |dP/dxi_scan|]

where:
  - (1 - xi^2) is the Jacobian factor d(physical)/d(xi) = TOT / (1 - xi^2)
    (constants TOT cancel in the Bayes ratio)
  - |dP/dxi_scan| is the co-area 1/|grad P| weight; approximated by
    |nxt - prev| / dxi (uniform dxi cancels in the ratio).

So in scan code:
  weight = 1 / ((1 - xi_off_a^2) * (1 - xi_off_b^2) * |nxt - prev|)

For PASS A (scan Sigma at fixed delta):
  xi_off_a = xi_S_off (interpolated crossing)
  xi_off_b = xi_d[k] (the fixed row's xi)
For PASS B (scan delta at fixed Sigma): swap roles.

For agent 1 at FR, the contour is Sigma=const, so xi_S_off=const and the
1/(1-xi_S_off^2) factor is constant -- f_1/f_0 ratio is unaffected.
For agents 2, 3 (oblique slices), the contour has BOTH xi_u and xi_d varying,
and the Jacobian DOES matter.

V4 PRESERVES V3's duplicate-crossing fix and ADDS the Jacobian/co-area weight.
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

def _crossing_with_weight(dp_f, dn_f, prev, nxt, p_target, xi_lo, xi_hi, xi_fixed):
    """Returns (xi_off_arb, weight_arb) if this iteration owns the crossing,
    else (None, None). Avoids double-count (V3 fix) and computes Jacobian +
    co-area weight (V4 addition).

    weight = 1 / ((1 - xi_off_scan^2) * (1 - xi_fixed^2) * |dP/dxi|)
           ~ 1 / ((1 - xi_off^2) * (1 - xi_fixed^2) * |nxt - prev|)
    (constant dxi cancels in Bayes ratio)
    """
    one = mp(1)
    xi_off = None
    if dp_f == 0.0 and dn_f == 0.0:
        return None, None
    if dp_f == 0.0 and dn_f != 0.0:
        xi_off = xi_lo
    elif dp_f != 0.0 and dn_f == 0.0:
        return None, None  # next iter owns it
    elif dp_f * dn_f < 0.0:
        dp = prev - p_target
        denom = nxt - prev
        if float(denom) == 0: return None, None
        frac = -dp / denom
        fr_f = float(frac)
        if fr_f < 0: frac = mp(0)
        elif fr_f > 1: frac = one
        xi_off = (one - frac) * xi_lo + frac * xi_hi
    else:
        return None, None
    # weight: 1 / ((1 - xi_off^2) * (1 - xi_fixed^2) * |nxt - prev|)
    one_minus_off2 = one - xi_off * xi_off
    one_minus_fix2 = one - xi_fixed * xi_fixed
    slope = nxt - prev
    slope_abs = slope if float(slope) >= 0 else -slope
    if float(slope_abs) == 0.0:
        return None, None
    # both (1-xi^2) factors are >0 inside the box; if very small (near
    # boundary), skip to avoid 1/0 blow-up.
    if float(one_minus_off2) < 1e-15 or float(one_minus_fix2) < 1e-15:
        return None, None
    weight = one / (one_minus_off2 * one_minus_fix2 * slope_abs)
    return xi_off, weight

def evidence_agent1(P, p_target, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm, i_u, coef):
    G_S = len(xi_S); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    pt = float(p_target)
    # Pass A: scan Sigma at fixed delta
    for k in range(G_d):
        xi_d_k = xi_d[k]
        if abs(float(xi_d_k)) >= 1 - 1e-15: continue
        prev = P[i_u][0][k]
        for j in range(G_S - 1):
            nxt = P[i_u][j+1][k]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            xi_off, w = _crossing_with_weight(dp_f, dn_f, prev, nxt, p_target, xi_S[j], xi_S[j+1], xi_d_k)
            if xi_off is not None and abs(float(xi_off)) < 1 - 1e-15:
                Sigma_off = TOT_S_mp * xi_off.atanh()
                delta_off = TOT_d_mp * xi_d_k.atanh()
                u2_off = (Sigma_off + delta_off) / two
                u3_off = (Sigma_off - delta_off) / two
                A = A + w * fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
            prev = nxt
    # Pass B: scan delta at fixed Sigma
    for j in range(G_S):
        xi_S_j = xi_S[j]
        if abs(float(xi_S_j)) >= 1 - 1e-15: continue
        prev = P[i_u][j][0]
        for k in range(G_d - 1):
            nxt = P[i_u][j][k+1]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            xi_off, w = _crossing_with_weight(dp_f, dn_f, prev, nxt, p_target, xi_d[k], xi_d[k+1], xi_S_j)
            if xi_off is not None and abs(float(xi_off)) < 1 - 1e-15:
                delta_off = TOT_d_mp * xi_off.atanh()
                Sigma_off = TOT_S_mp * xi_S_j.atanh()
                u2_off = (Sigma_off + delta_off) / two
                u3_off = (Sigma_off - delta_off) / two
                A = A + w * fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
            prev = nxt
    return A / two

def evidence_agent_oblique(P, p_target, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                            tau, vm, u_cell, sign_for_other, coef):
    G_u = len(xi_u1); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    pt = float(p_target)
    P_slc = [[mp(0) for _ in range(G_d)] for _ in range(G_u)]
    for i in range(G_u):
        for k in range(G_d):
            if abs(float(xi_d[k])) < 1 - 1e-15:
                delta_k = TOT_d_mp * xi_d[k].atanh()
            else:
                delta_k = mp(1e10) * xi_d[k]
            Sigma_req = two * u_cell + sign_for_other * delta_k
            P_slc[i][k] = interp_along_Sigma(P, i, k, Sigma_req, xi_S, TOT_S_mp)
    # scan u_1 at fixed delta
    for k in range(G_d):
        xi_d_k = xi_d[k]
        if abs(float(xi_d_k)) >= 1 - 1e-15: continue
        delta_k = TOT_d_mp * xi_d_k.atanh()
        u_other = u_cell + sign_for_other * delta_k
        prev = P_slc[0][k]
        for i in range(G_u - 1):
            nxt = P_slc[i+1][k]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            xi_off, w = _crossing_with_weight(dp_f, dn_f, prev, nxt, p_target, xi_u1[i], xi_u1[i+1], xi_d_k)
            if xi_off is not None and abs(float(xi_off)) < 1 - 1e-15:
                u1_off = TOT_u_mp * xi_off.atanh()
                A = A + w * fsig(u1_off, vm, tau, coef) * fsig(u_other, vm, tau, coef)
            prev = nxt
    # scan delta at fixed u_1
    for i in range(G_u):
        xi_u_i = xi_u1[i]
        if abs(float(xi_u_i)) >= 1 - 1e-15: continue
        u1_at = TOT_u_mp * xi_u_i.atanh()
        prev = P_slc[i][0]
        for k in range(G_d - 1):
            nxt = P_slc[i][k+1]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            xi_off, w = _crossing_with_weight(dp_f, dn_f, prev, nxt, p_target, xi_d[k], xi_d[k+1], xi_u_i)
            if xi_off is not None and abs(float(xi_off)) < 1 - 1e-15:
                delta_off = TOT_d_mp * xi_off.atanh()
                u_other_off = u_cell + sign_for_other * delta_off
                A = A + w * fsig(u1_at, vm, tau, coef) * fsig(u_other_off, vm, tau, coef)
            prev = nxt
    return A / two

def phi_sigdelta_v4(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W,
                     INNER_LO, INNER_HI, clearing='crra'):
    one = mp(1); two = mp(2); vm0 = mp('-0.5'); vm1 = mp('0.5'); eps_p = mp('1e-40')
    coef = (tau / (two * arb.pi())).sqrt()
    G_u = len(xi_u1)
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
                A1_0 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm0, i, coef)
                A1_1 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm1, i, coef)
                f0_u1 = fsig(u1_cell, vm0, tau, coef); f1_u1 = fsig(u1_cell, vm1, tau, coef)
                den = f0_u1 * A1_0 + f1_u1 * A1_1
                mu0 = f1_u1 * A1_1 / den if float(den) > 0 else mp('0.5')
                A2_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm0, u2_cell, -one, coef)
                A2_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm1, u2_cell, -one, coef)
                f0_u2 = fsig(u2_cell, vm0, tau, coef); f1_u2 = fsig(u2_cell, vm1, tau, coef)
                den = f0_u2 * A2_0 + f1_u2 * A2_1
                mu1 = f1_u2 * A2_1 / den if float(den) > 0 else mp('0.5')
                A3_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm0, u3_cell, one, coef)
                A3_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm1, u3_cell, one, coef)
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
