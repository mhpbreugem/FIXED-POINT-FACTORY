"""CARA variant of phi_sigma_delta.py on the (u_1, Σ, δ) ξ-grid.

Identical inference (axis-aligned + Σ-interp for off-axis agents) and
EXACT FR boundary conditions:
  u_1 = ±∞  → P = 0 / 1
  Σ   = ±∞  → P = 0 / 1   (S = u_1+Σ → ±∞ ⇒ Λ → 0/1)
  δ   = ±∞  → zero-order extrap

The ONLY change vs phi_sigma_delta.py is the clearing:
  CARA log-odds: pi = mean_k logit(mu_k); P_new = sigmoid(pi)
i.e. linear demand in log-odds, no Jensen curvature term.

Expected: with the proper FR-consistent BCs, the analytic FR
P*(u_1, Σ, δ) = sigmoid(tau*(u_1+Σ)) (δ-flat) IS a fixed point of this
operator to machine precision -- realizing Hellwig numerically.
"""
import math
import numpy as np
from numba import njit, prange

# reuse the helper functions from phi_sigma_delta
from phi_sigma_delta import (
    set_boundary, fsig, fsig_log, interp_along_Sigma,
    evidence_agent1, evidence_agent2, evidence_agent3, finf_interior,
)


@njit
def cara_clear_logodds(mu0, mu1, mu2):
    """CARA equal-weight log-odds clearing: pi = mean(logit(mu_k)), p = sigmoid(pi)."""
    eps = 1e-30
    m0 = max(eps, min(1 - eps, mu0))
    m1 = max(eps, min(1 - eps, mu1))
    m2 = max(eps, min(1 - eps, mu2))
    pi = (math.log(m0 / (1 - m0)) + math.log(m1 / (1 - m1)) + math.log(m2 / (1 - m2))) / 3.0
    return 1.0 / (1.0 + math.exp(-pi))


@njit(parallel=True)
def phi_sigmadelta_cara(P, xi_u1, xi_Sigma, xi_delta,
                         TOT_u, TOT_Sigma, TOT_delta,
                         tau, W,
                         inner_lo_u, inner_hi_u,
                         inner_lo_S, inner_hi_S,
                         inner_lo_d, inner_hi_d):
    G_u = P.shape[0]; G_S = P.shape[1]; G_d = P.shape[2]
    P_new = P.copy()
    for i in prange(inner_lo_u, inner_hi_u):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_cell = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        for j in range(inner_lo_S, inner_hi_S):
            if abs(xi_Sigma[j]) < 1 - 1e-15:
                Sigma_cell = TOT_Sigma * math.atanh(xi_Sigma[j])
            else:
                continue
            for k in range(inner_lo_d, inner_hi_d):
                if abs(xi_delta[k]) < 1 - 1e-15:
                    delta_cell = TOT_delta * math.atanh(xi_delta[k])
                else:
                    continue
                p_cell = P[i, j, k]
                u2_cell = 0.5 * (Sigma_cell + delta_cell)
                u3_cell = 0.5 * (Sigma_cell - delta_cell)

                # Agent 1 evidence (axis-aligned)
                A1_0 = evidence_agent1(P, p_cell, xi_Sigma, xi_delta, TOT_Sigma, TOT_delta, tau, -0.5, i)
                A1_1 = evidence_agent1(P, p_cell, xi_Sigma, xi_delta, TOT_Sigma, TOT_delta, tau, +0.5, i)
                f0_u1 = fsig(u1_cell, -0.5, tau); f1_u1 = fsig(u1_cell, +0.5, tau)
                num1 = f1_u1 * A1_1; den1 = f0_u1 * A1_0 + num1
                mu0 = (num1 / den1) if den1 > 0 else 0.5

                # Agent 2 evidence (Σ-interp + 2D scan in (u_1, δ))
                A2_0 = evidence_agent2(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, -0.5, u2_cell)
                A2_1 = evidence_agent2(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, +0.5, u2_cell)
                f0_u2 = fsig(u2_cell, -0.5, tau); f1_u2 = fsig(u2_cell, +0.5, tau)
                num2 = f1_u2 * A2_1; den2 = f0_u2 * A2_0 + num2
                mu1 = (num2 / den2) if den2 > 0 else 0.5

                # Agent 3 evidence (mirror)
                A3_0 = evidence_agent3(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, -0.5, u3_cell)
                A3_1 = evidence_agent3(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, +0.5, u3_cell)
                f0_u3 = fsig(u3_cell, -0.5, tau); f1_u3 = fsig(u3_cell, +0.5, tau)
                num3 = f1_u3 * A3_1; den3 = f0_u3 * A3_0 + num3
                mu2 = (num3 / den3) if den3 > 0 else 0.5

                # clip + CARA clear
                eps_p = 1e-12
                mu0 = max(eps_p, min(1-eps_p, mu0))
                mu1 = max(eps_p, min(1-eps_p, mu1))
                mu2 = max(eps_p, min(1-eps_p, mu2))
                P_new[i, j, k] = cara_clear_logodds(mu0, mu1, mu2)
    return P_new
