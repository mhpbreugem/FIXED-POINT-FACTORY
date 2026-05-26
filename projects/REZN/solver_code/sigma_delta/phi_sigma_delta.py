"""Φ map on the (u_1, Σ, δ) cube with rotated coordinates.

Σ = u_2 + u_3,   δ = u_2 - u_3,   so   u_2 = (Σ+δ)/2,   u_3 = (Σ-δ)/2.

Grid (ξ-coords, all uniform in (-1, +1)):
  ξ_u1 axis → u_1 = TOT_u · atanh(ξ_u1)
  ξ_Σ axis  → Σ   = TOT_Σ · atanh(ξ_Σ)
  ξ_δ axis  → δ   = TOT_δ · atanh(ξ_δ)

Boundaries (face cells):
  u_1 = ±∞  → P_FR limit (0 or 1)
  Σ   = ±∞  → P_FR limit (0 or 1) — exact since S = u_1+Σ→±∞
  δ   = ±∞  → zero-order extrapolation (= nearest interior cell)

Φ algorithm at interior cell (i, j, k):
  Agent-1 evidence: slice P[i, :, :] is axis-aligned in (Σ, δ); standard
    contour scan finds crossings of p_target = P[i,j,k].
  Agent-2 evidence: u_2 fixed at u_2_cell = (Σ[j]+δ[k])/2. The slice has
    axes (u_1, δ) with Σ_required = 2·u_2_cell − δ for each δ point. We
    interpolate P along Σ to extract the slice values, then run a 2D
    contour scan over (u_1, δ) at fixed Σ_required(δ).
  Agent-3 evidence: u_3 fixed; mirror of agent 2 with Σ_required = 2·u_3_cell + δ.

Float-64 implementation first.
"""
import os, sys, time, math
import numpy as np
from numba import njit, prange

# ===== boundary helpers =====
def set_boundary(P, TOT_u, TOT_Sigma, TOT_delta, xi_u1, xi_Sigma, xi_delta):
    """Apply face boundaries: u_1=±∞ FR (Λ@±∞), Σ=±∞ FR, δ=±∞ extrap."""
    G_u = P.shape[0]; G_S = P.shape[1]; G_d = P.shape[2]
    # u_1 boundary face (i=0 → u_1=-∞ → P=0; i=G_u-1 → P=1)
    P[0, :, :] = 0.0
    P[G_u-1, :, :] = 1.0
    # Σ boundary face (j=0 → Σ=-∞ → P=0; j=G_S-1 → P=1)
    P[:, 0, :] = 0.0
    P[:, G_S-1, :] = 1.0
    # δ boundary face: zero-order extrapolation from nearest interior
    P[:, :, 0] = P[:, :, 1]
    P[:, :, G_d-1] = P[:, :, G_d-2]
    # corner conflict resolution: majority rule
    # at any cell with multiple axes at boundary, use majority
    # we keep face boundaries above; corners can be inconsistent but
    # they're rarely read in the contour scan
    return P


# ===== signal density =====
@njit
def fsig_log(u, vm, tau):
    """log f_v(u) = log sqrt(τ/2π) - 0.5*τ*(u-vm)^2"""
    return 0.5 * math.log(tau / (2.0 * math.pi)) - 0.5 * tau * (u - vm) ** 2

@njit
def fsig(u, vm, tau):
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * (u - vm) ** 2)


# ===== CRRA demand and clearing =====
@njit
def crra_demand(mu, p, gamma, W):
    if mu <= 1e-30: mu = 1e-30
    if mu >= 1 - 1e-30: mu = 1 - 1e-30
    if p <= 1e-30: p = 1e-30
    if p >= 1 - 1e-30: p = 1 - 1e-30
    lm = math.log(mu / (1 - mu))
    lp = math.log(p / (1 - p))
    R = math.exp((lm - lp) / gamma)
    return W * (R - 1) / ((1 - p) + R * p)

@njit
def crra_clear_sym(mu0, mu1, mu2, gamma, W, max_steps=80):
    eps = 1e-30
    a, b = eps, 1.0 - eps
    for _ in range(max_steps):
        m = 0.5 * (a + b)
        ex = crra_demand(mu0, m, gamma, W) + crra_demand(mu1, m, gamma, W) + crra_demand(mu2, m, gamma, W)
        if ex > 0: a = m
        else: b = m
    return 0.5 * (a + b)


# ===== Σ-interpolation helper =====
@njit
def interp_along_Sigma(P, i_u, i_delta, Sigma_target, xi_Sigma, TOT_Sigma):
    """Linear interp in ξ_Σ of P[i_u, :, i_delta] at the ξ corresponding to
    Sigma_target. Zero-order at the boundary."""
    G_S = xi_Sigma.shape[0]
    if Sigma_target <= -1e10:
        return P[i_u, 0, i_delta]
    if Sigma_target >= 1e10:
        return P[i_u, G_S-1, i_delta]
    xi_t = math.tanh(Sigma_target / TOT_Sigma)
    if xi_t <= xi_Sigma[0]:
        return P[i_u, 0, i_delta]
    if xi_t >= xi_Sigma[-1]:
        return P[i_u, G_S-1, i_delta]
    # linear search (G_S small, no need for bisect)
    for j in range(G_S - 1):
        if xi_Sigma[j] <= xi_t <= xi_Sigma[j+1]:
            denom = xi_Sigma[j+1] - xi_Sigma[j]
            if denom == 0:
                return P[i_u, j, i_delta]
            frac = (xi_t - xi_Sigma[j]) / denom
            return (1.0 - frac) * P[i_u, j, i_delta] + frac * P[i_u, j+1, i_delta]
    return P[i_u, G_S-1, i_delta]


# ===== agent-1 evidence (axis-aligned in Σ, δ) =====
@njit
def evidence_agent1(P, p_target, xi_Sigma, xi_delta, TOT_Sigma, TOT_delta,
                    tau, vm, i_u):
    """At fixed u_1 (index i_u), contour-scan P[i_u, :, :] over (ξ_Σ, ξ_δ)
    for p_target. f_v evaluated at u_2 = (Σ+δ)/2 and u_3 = (Σ-δ)/2 at
    each crossing."""
    G_S = xi_Sigma.shape[0]; G_d = xi_delta.shape[0]
    A = 0.0
    # scan axis Σ at fixed δ
    for k in range(G_d):
        prev = P[i_u, 0, k]
        for j in range(G_S - 1):
            nxt = P[i_u, j+1, k]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_Sigma_off = (1 - frac) * xi_Sigma[j] + frac * xi_Sigma[j+1]
                    if abs(xi_Sigma_off) < 1 - 1e-15:
                        Sigma_off = TOT_Sigma * math.atanh(xi_Sigma_off)
                        delta_off = TOT_delta * math.atanh(xi_delta[k]) if abs(xi_delta[k]) < 1 - 1e-15 else 0.0
                        u2_off = 0.5 * (Sigma_off + delta_off)
                        u3_off = 0.5 * (Sigma_off - delta_off)
                        A += fsig(u2_off, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    # scan axis δ at fixed Σ
    for j in range(G_S):
        prev = P[i_u, j, 0]
        for k in range(G_d - 1):
            nxt = P[i_u, j, k+1]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_delta_off = (1 - frac) * xi_delta[k] + frac * xi_delta[k+1]
                    if abs(xi_delta_off) < 1 - 1e-15:
                        delta_off = TOT_delta * math.atanh(xi_delta_off)
                        Sigma_off = TOT_Sigma * math.atanh(xi_Sigma[j]) if abs(xi_Sigma[j]) < 1 - 1e-15 else 0.0
                        u2_off = 0.5 * (Sigma_off + delta_off)
                        u3_off = 0.5 * (Sigma_off - delta_off)
                        A += fsig(u2_off, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== agent-2 evidence (slice fixing u_2; needs Σ-interp) =====
@njit
def evidence_agent2(P, p_target, xi_u1, xi_Sigma, xi_delta,
                    TOT_u, TOT_Sigma, TOT_delta, tau, vm, u_2_cell):
    """Agent 2 fixes u_2 = u_2_cell. Slice has axes (ξ_u1, ξ_δ), with
    Σ_required(δ) = 2·u_2_cell − δ extracted by Σ-interp at each (u_1, δ)
    grid point. Then 2D contour scan over (u_1, δ).

    f_v at crossings: f_v(u_1_cross) · f_v(u_3_at_cross) where
    u_3 = u_2_cell − δ_cross.
    """
    G_u = xi_u1.shape[0]; G_d = xi_delta.shape[0]
    A = 0.0
    # Pre-build the (G_u, G_d) slice via Σ-interp
    P_slice = np.zeros((G_u, G_d))
    for i in range(G_u):
        for k in range(G_d):
            if abs(xi_delta[k]) < 1 - 1e-15:
                delta_k = TOT_delta * math.atanh(xi_delta[k])
            else:
                delta_k = math.copysign(1e10, xi_delta[k])
            Sigma_req = 2.0 * u_2_cell - delta_k
            P_slice[i, k] = interp_along_Sigma(P, i, k, Sigma_req, xi_Sigma, TOT_Sigma)

    # scan axis u_1 at fixed δ
    for k in range(G_d):
        if abs(xi_delta[k]) < 1 - 1e-15:
            delta_k = TOT_delta * math.atanh(xi_delta[k])
        else:
            continue  # skip δ boundary (extrapolated zero-order)
        u3_at = u_2_cell - delta_k
        prev = P_slice[0, k]
        for i in range(G_u - 1):
            nxt = P_slice[i+1, k]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_u1_off = (1 - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(xi_u1_off) < 1 - 1e-15:
                        u1_off = TOT_u * math.atanh(xi_u1_off)
                        A += fsig(u1_off, vm, tau) * fsig(u3_at, vm, tau)
            prev = nxt
    # scan axis δ at fixed u_1
    for i in range(G_u):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_at = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        prev = P_slice[i, 0]
        for k in range(G_d - 1):
            nxt = P_slice[i, k+1]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_delta_off = (1 - frac) * xi_delta[k] + frac * xi_delta[k+1]
                    if abs(xi_delta_off) < 1 - 1e-15:
                        delta_off = TOT_delta * math.atanh(xi_delta_off)
                        u3_off = u_2_cell - delta_off
                        A += fsig(u1_at, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== agent-3 evidence (mirror of agent 2) =====
@njit
def evidence_agent3(P, p_target, xi_u1, xi_Sigma, xi_delta,
                    TOT_u, TOT_Sigma, TOT_delta, tau, vm, u_3_cell):
    """Mirror of agent 2: u_3 fixed, Σ_required(δ) = 2·u_3_cell + δ."""
    G_u = xi_u1.shape[0]; G_d = xi_delta.shape[0]
    A = 0.0
    P_slice = np.zeros((G_u, G_d))
    for i in range(G_u):
        for k in range(G_d):
            if abs(xi_delta[k]) < 1 - 1e-15:
                delta_k = TOT_delta * math.atanh(xi_delta[k])
            else:
                delta_k = math.copysign(1e10, xi_delta[k])
            Sigma_req = 2.0 * u_3_cell + delta_k
            P_slice[i, k] = interp_along_Sigma(P, i, k, Sigma_req, xi_Sigma, TOT_Sigma)
    for k in range(G_d):
        if abs(xi_delta[k]) < 1 - 1e-15:
            delta_k = TOT_delta * math.atanh(xi_delta[k])
        else:
            continue
        u2_at = u_3_cell + delta_k
        prev = P_slice[0, k]
        for i in range(G_u - 1):
            nxt = P_slice[i+1, k]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_u1_off = (1 - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(xi_u1_off) < 1 - 1e-15:
                        u1_off = TOT_u * math.atanh(xi_u1_off)
                        A += fsig(u1_off, vm, tau) * fsig(u2_at, vm, tau)
            prev = nxt
    for i in range(G_u):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_at = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        prev = P_slice[i, 0]
        for k in range(G_d - 1):
            nxt = P_slice[i, k+1]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_delta_off = (1 - frac) * xi_delta[k] + frac * xi_delta[k+1]
                    if abs(xi_delta_off) < 1 - 1e-15:
                        delta_off = TOT_delta * math.atanh(xi_delta_off)
                        u2_off = u_3_cell + delta_off
                        A += fsig(u1_at, vm, tau) * fsig(u2_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== full Φ on (u_1, Σ, δ) =====
@njit(parallel=True)
def phi_sigmadelta(P, xi_u1, xi_Sigma, xi_delta,
                    TOT_u, TOT_Sigma, TOT_delta,
                    tau, gamma, W,
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
                # Agent 1 evidence
                A1_0 = evidence_agent1(P, p_cell, xi_Sigma, xi_delta, TOT_Sigma, TOT_delta, tau, -0.5, i)
                A1_1 = evidence_agent1(P, p_cell, xi_Sigma, xi_delta, TOT_Sigma, TOT_delta, tau, +0.5, i)
                f0_u1 = fsig(u1_cell, -0.5, tau); f1_u1 = fsig(u1_cell, +0.5, tau)
                num1 = f1_u1 * A1_1; den1 = f0_u1 * A1_0 + num1
                mu0 = (num1 / den1) if den1 > 0 else 0.5
                # Agent 2
                A2_0 = evidence_agent2(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, -0.5, u2_cell)
                A2_1 = evidence_agent2(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, +0.5, u2_cell)
                f0_u2 = fsig(u2_cell, -0.5, tau); f1_u2 = fsig(u2_cell, +0.5, tau)
                num2 = f1_u2 * A2_1; den2 = f0_u2 * A2_0 + num2
                mu1 = (num2 / den2) if den2 > 0 else 0.5
                # Agent 3
                A3_0 = evidence_agent3(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, -0.5, u3_cell)
                A3_1 = evidence_agent3(P, p_cell, xi_u1, xi_Sigma, xi_delta, TOT_u, TOT_Sigma, TOT_delta, tau, +0.5, u3_cell)
                f0_u3 = fsig(u3_cell, -0.5, tau); f1_u3 = fsig(u3_cell, +0.5, tau)
                num3 = f1_u3 * A3_1; den3 = f0_u3 * A3_0 + num3
                mu2 = (num3 / den3) if den3 > 0 else 0.5
                # clip
                eps_p = 1e-12
                mu0 = max(eps_p, min(1-eps_p, mu0))
                mu1 = max(eps_p, min(1-eps_p, mu1))
                mu2 = max(eps_p, min(1-eps_p, mu2))
                P_new[i, j, k] = crra_clear_sym(mu0, mu1, mu2, gamma, W)
    return P_new


@njit
def finf_interior(A, B, ilo_u, ihi_u, ilo_S, ihi_S, ilo_d, ihi_d):
    m = 0.0
    for i in range(ilo_u, ihi_u):
        for j in range(ilo_S, ihi_S):
            for k in range(ilo_d, ihi_d):
                d = abs(A[i,j,k] - B[i,j,k])
                if d > m: m = d
    return m


if __name__ == '__main__':
    # smoke test at G=10 in each axis
    G = 10
    TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
    TAU = 2.0; GAMMA = 0.1; W = 1.0
    G_FULL = G + 2  # +2 for boundary cells per axis
    INNER_LO, INNER_HI = 1, G + 1
    dxi = 2.0 / (G + 1)
    xi_inner = np.linspace(-1+dxi, 1-dxi, G)
    xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])
    xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

    # FR ansatz on the cube
    P = np.zeros((G_FULL,)*3)
    for i in range(G_FULL):
        for j in range(G_FULL):
            for k in range(G_FULL):
                if abs(xi_u1[i]) < 1 - 1e-15:
                    u1 = TOT_u * math.atanh(xi_u1[i])
                else:
                    u1 = math.copysign(1e10, xi_u1[i])
                if abs(xi_S[j]) < 1 - 1e-15:
                    Sigma = TOT_S * math.atanh(xi_S[j])
                else:
                    Sigma = math.copysign(1e10, xi_S[j])
                S = u1 + Sigma  # S = u_1+u_2+u_3 = u_1+Σ (independent of δ in FR)
                if S > 50: P[i,j,k] = 1.0
                elif S < -50: P[i,j,k] = 0.0
                else: P[i,j,k] = 1.0 / (1.0 + math.exp(-TAU * S))
    set_boundary(P, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)

    print(f"smoke: G={G}, G_FULL={G_FULL}")
    print(f"FR P[0,5,5]={P[0,5,5]:.4f}, P[5,5,5]={P[5,5,5]:.4f}, P[10,5,5]={P[10,5,5]:.4f}")
    t0 = time.time()
    P_new = phi_sigmadelta(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d,
                            TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    print(f"first Φ (JIT compile): {time.time()-t0:.1f}s")
    P_new = set_boundary(P_new, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
    res = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    print(f"  F_inf at FR_ansatz: {res:.3e}")
    t0 = time.time()
    P_new = phi_sigmadelta(P_new, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d,
                            TAU, GAMMA, W,
                            INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    print(f"second Φ (compiled): {time.time()-t0:.1f}s")
