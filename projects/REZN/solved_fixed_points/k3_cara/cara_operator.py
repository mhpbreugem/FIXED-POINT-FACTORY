"""K=3 CARA (constant absolute risk aversion) REE contour operator.

Reuses the *consistent co-area smoothed inference* of the reference CRRA
operator (reference_operator.py / phi_K3_halo_smooth) EXACTLY -- the
Gaussian-band evidence -> Bayes posterior mu_i is identical. The ONLY change
is the market-clearing / demand step.

Primary CARA model (the gamma->inf CRRA limit): LINEARIZED-log-odds demand

    x_i = W*(m_i - pi)/a ,   m_i = logit(mu_i),  pi = logit(p)

Clearing  sum_i W_i x_i = 0  with homogeneous W,a  =>  pi = mean_i(m_i)
                                                  =>  p = sigmoid(mean logit mu).

This is the clean theoretical high-gamma limit of CRRA: demand linear in the
log-odds gap with NO (1/2 - p) curvature term -> no Jensen gap -> full
revelation.

Alternative ("true CARA") exponential-utility binary-asset demand, provided
for comparison:  x_i = (mu_i - p)/(a*p*(1-p)) ; clearing solved by bisection.

Everything else (signal density, co-area band evidence, Bayes) is imported
from the reference operator so there is a single source of truth.
"""
from __future__ import annotations

import os
import sys
import numpy as np

# pull the reference operator (single source of truth for the inference)
_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(
    "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points",
    "methodology"))
from reference_operator import (  # noqa: E402
    f_signal, lam, logit, EPS, bayes, agent_evidence_smooth, deficit,
)


# ----------------------------------------------------------------------
# CARA market clearing
# ----------------------------------------------------------------------

def clear_cara_logodds(mu_vec, W_vec):
    """Primary CARA clearing: linear-in-log-odds demand.

    x_i = W_i (m_i - pi)/a ,  m_i = logit(mu_i).  With clearing
    sum W_i x_i = 0 the risk-aversion a cancels and

        pi = (sum_i W_i m_i)/(sum_i W_i)   =>   p = sigmoid(pi).

    This is the gamma->inf limit of CRRA: NO (1/2-p) curvature -> price is the
    (weighted) average belief in log-odds space, no Jensen gap.
    """
    m = np.array([logit(min(max(mu, EPS), 1.0 - EPS)) for mu in mu_vec])
    W = np.asarray(W_vec, dtype=float)
    pi = float(np.sum(W * m) / np.sum(W))
    return float(min(max(lam(pi), EPS), 1.0 - EPS))


def clear_cara_exp(mu_vec, a_vec, W_vec):
    """Alternative 'true CARA' exponential-utility demand for a binary asset.

        x_i = (mu_i - p)/(a_i * p * (1-p))

    (mean-variance / CARA-on-binary-payoff form). Aggregate excess demand is
    strictly decreasing in p, so guarded bisection nails it.
    """
    a = np.asarray(a_vec, dtype=float)
    W = np.asarray(W_vec, dtype=float)
    mu = np.asarray(mu_vec, dtype=float)

    def excess(p):
        return float(np.sum(W * (mu - p) / (a * p * (1.0 - p))))

    lo, hi = EPS, 1.0 - EPS
    if excess(lo) <= 0.0:
        return lo
    if excess(hi) >= 0.0:
        return hi
    for _ in range(80):
        c = 0.5 * (lo + hi)
        if excess(c) >= 0.0:
            lo = c
        else:
            hi = c
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# The CARA fixed-point operator Phi (one full grid sweep)
# ----------------------------------------------------------------------

def phi_cara(P_full, u_full, lo, hi, tau_vec, a_vec, W_vec, h,
             model="logodds"):
    """One application of the CARA operator.

    Inference (co-area band evidence -> Bayes mu_i) is identical to the CRRA
    reference; only the clearing call differs.
    """
    P_new = P_full.copy()
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                A0, A1 = agent_evidence_smooth(P_full[i, :, :], p, u_full,
                                               tau_vec[1], tau_vec[2], h)
                mu0 = bayes(u_full[i], tau_vec[0], A0, A1)
                A0, A1 = agent_evidence_smooth(P_full[:, j, :], p, u_full,
                                               tau_vec[0], tau_vec[2], h)
                mu1 = bayes(u_full[j], tau_vec[1], A0, A1)
                A0, A1 = agent_evidence_smooth(P_full[:, :, l], p, u_full,
                                               tau_vec[0], tau_vec[1], h)
                mu2 = bayes(u_full[l], tau_vec[2], A0, A1)
                if model == "logodds":
                    P_new[i, j, l] = clear_cara_logodds([mu0, mu1, mu2], W_vec)
                else:
                    P_new[i, j, l] = clear_cara_exp([mu0, mu1, mu2],
                                                    a_vec, W_vec)
    return P_new


def init_no_learning_cara(u_full, tau_vec, a_vec, W_vec, model="logodds"):
    """Halo / Newton warm start: agents use only their own signal."""
    G = u_full.size
    P = np.empty((G, G, G))
    for i in range(G):
        m0 = float(lam(tau_vec[0] * u_full[i]))
        for j in range(G):
            m1 = float(lam(tau_vec[1] * u_full[j]))
            for l in range(G):
                m2 = float(lam(tau_vec[2] * u_full[l]))
                if model == "logodds":
                    P[i, j, l] = clear_cara_logodds([m0, m1, m2], W_vec)
                else:
                    P[i, j, l] = clear_cara_exp([m0, m1, m2], a_vec, W_vec)
    return P


# ----------------------------------------------------------------------
# Nail the CARA fixed point
# ----------------------------------------------------------------------

def _grid(G_inner, tau, umax=4.0, pad=2, C=0.45):
    du = 2 * umax / (G_inner - 1)
    h = C * du ** 0.5
    G_full = G_inner + 2 * pad
    u_full = np.array([-umax + (q - pad) * du for q in range(G_full)])
    lo, hi = pad, pad + G_inner
    ui = u_full[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = tau * (U1 + U2 + U3)
    return du, h, u_full, lo, hi, T


def nail_cara(G_inner=9, tau=2.0, a=1.0, C=0.45, umax=4.0, pad=2,
              model="logodds", W=1.0, max_iter=400, tol=1e-13):
    """Nail the CARA fixed point Phi(P)=P.

    For the log-odds model Phi is a *contraction-style* smooth map; plain
    Picard iteration converges geometrically and is well-conditioned in
    float64, so we drive it to <1e-13 and return (deficit, ||F||_inf, P).
    """
    K = 3
    tau_vec = np.full(K, tau)
    a_vec = np.full(K, a)
    W_vec = np.full(K, W)
    du, h, u_full, lo, hi, T = _grid(G_inner, tau, umax, pad, C)
    slc = (slice(lo, hi),) * K

    Pf = init_no_learning_cara(u_full, tau_vec, a_vec, W_vec, model)
    halo = Pf.copy()
    Finf = np.inf
    iters = 0
    for it in range(max_iter):
        Pn = phi_cara(Pf, u_full, lo, hi, tau_vec, a_vec, W_vec, h, model)
        # keep halo fixed
        Pn_h = halo.copy()
        Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h
        iters = it + 1
        if Finf < tol:
            break

    P_inner = Pf[slc].copy()
    return {
        "deficit": deficit(P_inner, T),
        "Finf": Finf,
        "iters": iters,
        "P_inner": P_inner,
        "du": du, "h": h, "G_inner": G_inner,
    }
