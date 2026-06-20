"""Symmetry-folded Φ on (u_1, Σ̂, δ̂) cube.

Folding via:
  δ-sym (u_2 ↔ u_3):    P(i, j, k) = P(i, j, G-1-k)
  v↔1-v combined:       P(i, j, k) = 1 − P(G-1-i, G-1-j, k)

Stored domain: full (i ∈ [0,G-1]) × (j ∈ [mid, G-1]) × (k ∈ [mid, G-1])
where mid = G//2 (ξ ≥ 0 for both Σ̂ and δ̂).

Access function P_at(i, j, k) handles reflection + value flip transparently.
"""
import os, sys, time, math
import numpy as np
from numba import njit, prange

# ===== signal density =====
@njit
def fsig(u, vm, tau):
    return math.sqrt(tau / (2.0 * math.pi)) * math.exp(-0.5 * tau * (u - vm) ** 2)


# ===== CRRA =====
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


# ===== folded access =====
@njit(inline='always')
def P_at(P_stored, i_full, j_full, k_full, G):
    """Get P at (i, j, k) full-cube indices from folded P_stored.

    P_stored shape: (G, mid+1, mid+1) where mid = G//2.
    Stored covers (i ∈ [0,G-1], j ∈ [mid,G-1], k ∈ [mid,G-1]).

    δ-symmetry: P(i, j, k) = P(i, j, G-1-k) for k < mid.
    Combined sym: P(i, j, k) = 1 - P(G-1-i, G-1-j, k) for j < mid.
    """
    mid = G // 2
    # δ reflection
    if k_full < mid:
        k_eff = G - 1 - k_full
    else:
        k_eff = k_full
    # Σ + u_1 reflection (with value flip)
    flip = False
    if j_full < mid:
        i_eff = G - 1 - i_full
        j_eff = G - 1 - j_full
        flip = True
    else:
        i_eff = i_full
        j_eff = j_full
    val = P_stored[i_eff, j_eff - mid, k_eff - mid]
    if flip:
        val = 1.0 - val
    return val


# ===== axis-aligned slice for agent 1 =====
@njit
def evidence_agent1_folded(P_stored, p_target, xi_S, xi_d,
                            TOT_S, TOT_d, tau, vm, i_full, G):
    """At fixed u_1 (index i_full), contour-scan P[i_full, :, :] for p_target."""
    A = 0.0
    # scan axis Σ at fixed δ
    for k in range(G):
        prev = P_at(P_stored, i_full, 0, k, G)
        for j in range(G - 1):
            nxt = P_at(P_stored, i_full, j+1, k, G)
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_S_off = (1 - frac) * xi_S[j] + frac * xi_S[j+1]
                    if abs(xi_S_off) < 1 - 1e-15:
                        Sigma_off = TOT_S * math.atanh(xi_S_off)
                        d_off = TOT_d * math.atanh(xi_d[k]) if abs(xi_d[k]) < 1 - 1e-15 else 0.0
                        u2_off = 0.5 * (Sigma_off + d_off)
                        u3_off = 0.5 * (Sigma_off - d_off)
                        A += fsig(u2_off, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    # scan axis δ at fixed Σ
    for j in range(G):
        prev = P_at(P_stored, i_full, j, 0, G)
        for k in range(G - 1):
            nxt = P_at(P_stored, i_full, j, k+1, G)
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_d_off = (1 - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(xi_d_off) < 1 - 1e-15:
                        d_off = TOT_d * math.atanh(xi_d_off)
                        Sigma_off = TOT_S * math.atanh(xi_S[j]) if abs(xi_S[j]) < 1 - 1e-15 else 0.0
                        u2_off = 0.5 * (Sigma_off + d_off)
                        u3_off = 0.5 * (Sigma_off - d_off)
                        A += fsig(u2_off, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== Σ-interp helper (folded) =====
@njit
def interp_along_Sigma_folded(P_stored, i_full, k_full, Sigma_target,
                                xi_S, TOT_S, G):
    if Sigma_target <= -1e10:
        return P_at(P_stored, i_full, 0, k_full, G)
    if Sigma_target >= 1e10:
        return P_at(P_stored, i_full, G-1, k_full, G)
    xi_t = math.tanh(Sigma_target / TOT_S)
    if xi_t <= xi_S[0]:
        return P_at(P_stored, i_full, 0, k_full, G)
    if xi_t >= xi_S[-1]:
        return P_at(P_stored, i_full, G-1, k_full, G)
    for j in range(G - 1):
        if xi_S[j] <= xi_t <= xi_S[j+1]:
            denom = xi_S[j+1] - xi_S[j]
            if denom == 0:
                return P_at(P_stored, i_full, j, k_full, G)
            frac = (xi_t - xi_S[j]) / denom
            return ((1.0 - frac) * P_at(P_stored, i_full, j, k_full, G)
                    + frac * P_at(P_stored, i_full, j+1, k_full, G))
    return P_at(P_stored, i_full, G-1, k_full, G)


# ===== agent 2 evidence (folded, oblique slice) =====
@njit
def evidence_agent2_folded(P_stored, p_target, xi_u1, xi_S, xi_d,
                            TOT_u, TOT_S, TOT_d, tau, vm, u_2_cell, G):
    # Build P_slice[i_full, k_full] = interp at Σ = 2*u_2 - δ_k
    P_slice = np.empty((G, G))
    for i in range(G):
        for k in range(G):
            if abs(xi_d[k]) < 1 - 1e-15:
                d_k = TOT_d * math.atanh(xi_d[k])
            else:
                d_k = math.copysign(1e10, xi_d[k])
            Sigma_req = 2.0 * u_2_cell - d_k
            P_slice[i, k] = interp_along_Sigma_folded(P_stored, i, k, Sigma_req,
                                                       xi_S, TOT_S, G)
    A = 0.0
    # scan u_1 at fixed δ
    for k in range(G):
        if abs(xi_d[k]) < 1 - 1e-15:
            d_k = TOT_d * math.atanh(xi_d[k])
        else:
            continue
        u3_at = u_2_cell - d_k
        prev = P_slice[0, k]
        for i in range(G - 1):
            nxt = P_slice[i+1, k]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_u_off = (1 - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(xi_u_off) < 1 - 1e-15:
                        u1_off = TOT_u * math.atanh(xi_u_off)
                        A += fsig(u1_off, vm, tau) * fsig(u3_at, vm, tau)
            prev = nxt
    # scan δ at fixed u_1
    for i in range(G):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_at = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        prev = P_slice[i, 0]
        for k in range(G - 1):
            nxt = P_slice[i, k+1]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_d_off = (1 - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(xi_d_off) < 1 - 1e-15:
                        d_off = TOT_d * math.atanh(xi_d_off)
                        u3_off = u_2_cell - d_off
                        A += fsig(u1_at, vm, tau) * fsig(u3_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== agent 3 evidence (mirror) =====
@njit
def evidence_agent3_folded(P_stored, p_target, xi_u1, xi_S, xi_d,
                            TOT_u, TOT_S, TOT_d, tau, vm, u_3_cell, G):
    P_slice = np.empty((G, G))
    for i in range(G):
        for k in range(G):
            if abs(xi_d[k]) < 1 - 1e-15:
                d_k = TOT_d * math.atanh(xi_d[k])
            else:
                d_k = math.copysign(1e10, xi_d[k])
            Sigma_req = 2.0 * u_3_cell + d_k
            P_slice[i, k] = interp_along_Sigma_folded(P_stored, i, k, Sigma_req,
                                                       xi_S, TOT_S, G)
    A = 0.0
    for k in range(G):
        if abs(xi_d[k]) < 1 - 1e-15:
            d_k = TOT_d * math.atanh(xi_d[k])
        else:
            continue
        u2_at = u_3_cell + d_k
        prev = P_slice[0, k]
        for i in range(G - 1):
            nxt = P_slice[i+1, k]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_u_off = (1 - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(xi_u_off) < 1 - 1e-15:
                        u1_off = TOT_u * math.atanh(xi_u_off)
                        A += fsig(u1_off, vm, tau) * fsig(u2_at, vm, tau)
            prev = nxt
    for i in range(G):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_at = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        prev = P_slice[i, 0]
        for k in range(G - 1):
            nxt = P_slice[i, k+1]
            dp, dn = prev - p_target, nxt - p_target
            if not (dp == 0.0 and dn == 0.0) and dp * dn <= 0.0:
                denom = nxt - prev
                if denom != 0.0:
                    frac = -dp / denom
                    if frac < 0: frac = 0.0
                    if frac > 1: frac = 1.0
                    xi_d_off = (1 - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(xi_d_off) < 1 - 1e-15:
                        d_off = TOT_d * math.atanh(xi_d_off)
                        u2_off = u_3_cell + d_off
                        A += fsig(u1_at, vm, tau) * fsig(u2_off, vm, tau)
            prev = nxt
    return 0.5 * A


# ===== full Φ on folded storage =====
@njit(parallel=True)
def phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d,
                tau, gamma, W, G):
    """Φ acts on stored cells only. Inner = i ∈ [1, G-1), j ∈ [mid, G-1),
    k ∈ [mid, G-1) plus the symmetry axis j=mid or k=mid which is still
    interior to the full cube."""
    mid = G // 2
    half = mid + 1  # j_stored, k_stored ∈ [0, mid]
    P_new = P_stored.copy()
    # Iterate over stored interior: i ∈ [1, G-1], j_s ∈ [0, mid-1], k_s ∈ [0, mid-1]
    # (excluding the FR boundary at j_full = G-1 ⇔ j_s = mid, k_full = G-1 ⇔ k_s = mid,
    #  and the u_1=±1 boundaries at i=0 and i=G-1).
    for i in prange(1, G-1):
        if abs(xi_u1[i]) < 1 - 1e-15:
            u1_cell = TOT_u * math.atanh(xi_u1[i])
        else:
            continue
        for j_s in range(0, mid):  # j_full = mid + j_s ∈ [mid, G-2]
            j_full = mid + j_s
            if abs(xi_S[j_full]) < 1 - 1e-15:
                Sigma_cell = TOT_S * math.atanh(xi_S[j_full])
            else:
                continue
            for k_s in range(0, mid):
                k_full = mid + k_s
                if abs(xi_d[k_full]) < 1 - 1e-15:
                    d_cell = TOT_d * math.atanh(xi_d[k_full])
                else:
                    continue
                p_cell = P_stored[i, j_s, k_s]
                u2_cell = 0.5 * (Sigma_cell + d_cell)
                u3_cell = 0.5 * (Sigma_cell - d_cell)
                # Agent 1
                A1_0 = evidence_agent1_folded(P_stored, p_cell, xi_S, xi_d, TOT_S, TOT_d, tau, -0.5, i, G)
                A1_1 = evidence_agent1_folded(P_stored, p_cell, xi_S, xi_d, TOT_S, TOT_d, tau, +0.5, i, G)
                f0_u1 = fsig(u1_cell, -0.5, tau); f1_u1 = fsig(u1_cell, +0.5, tau)
                num1 = f1_u1 * A1_1; den1 = f0_u1 * A1_0 + num1
                mu0 = (num1 / den1) if den1 > 0 else 0.5
                # Agent 2
                A2_0 = evidence_agent2_folded(P_stored, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u, TOT_S, TOT_d, tau, -0.5, u2_cell, G)
                A2_1 = evidence_agent2_folded(P_stored, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u, TOT_S, TOT_d, tau, +0.5, u2_cell, G)
                f0_u2 = fsig(u2_cell, -0.5, tau); f1_u2 = fsig(u2_cell, +0.5, tau)
                num2 = f1_u2 * A2_1; den2 = f0_u2 * A2_0 + num2
                mu1 = (num2 / den2) if den2 > 0 else 0.5
                # Agent 3
                A3_0 = evidence_agent3_folded(P_stored, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u, TOT_S, TOT_d, tau, -0.5, u3_cell, G)
                A3_1 = evidence_agent3_folded(P_stored, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u, TOT_S, TOT_d, tau, +0.5, u3_cell, G)
                f0_u3 = fsig(u3_cell, -0.5, tau); f1_u3 = fsig(u3_cell, +0.5, tau)
                num3 = f1_u3 * A3_1; den3 = f0_u3 * A3_0 + num3
                mu2 = (num3 / den3) if den3 > 0 else 0.5
                eps_p = 1e-12
                mu0 = max(eps_p, min(1-eps_p, mu0))
                mu1 = max(eps_p, min(1-eps_p, mu1))
                mu2 = max(eps_p, min(1-eps_p, mu2))
                P_new[i, j_s, k_s] = crra_clear_sym(mu0, mu1, mu2, gamma, W)
    return P_new


def set_boundary_folded(P_stored, G):
    """Boundary cells in the folded storage.

    Stored has shape (G, mid+1, mid+1). Full-cube boundary translates:
      - u_1 = -∞ (i=0)         → P = 0
      - u_1 = +∞ (i=G-1)       → P = 1
      - Σ̂ = +∞  (j_full=G-1, j_s=mid) → P = 1
      - Σ̂ = 0   (j_full=mid, j_s=0)   → 'reflection axis', value held
      - δ̂ = +∞  (k_full=G-1, k_s=mid) → zero-order extrap
      - δ̂ = 0   (k_full=mid, k_s=0)   → 'reflection axis', value held
    """
    mid = G // 2
    # u_1 = ±∞
    P_stored[0, :, :] = 0.0
    P_stored[G-1, :, :] = 1.0
    # Σ̂ = +∞
    P_stored[:, mid, :] = 1.0
    # δ̂ = +∞: zero-order extrap from k_s = mid-1
    P_stored[:, :, mid] = P_stored[:, :, mid-1]
    return P_stored


def unfold_full(P_stored, G):
    """Reconstruct full G^3 cube from folded storage for inspection/plotting."""
    P_full = np.zeros((G, G, G))
    mid = G // 2
    for i in range(G):
        for j in range(G):
            for k in range(G):
                if k < mid:
                    k_eff = G - 1 - k
                else:
                    k_eff = k
                if j < mid:
                    i_eff = G - 1 - i
                    j_eff = G - 1 - j
                    flip = True
                else:
                    i_eff = i
                    j_eff = j
                    flip = False
                val = P_stored[i_eff, j_eff - mid, k_eff - mid]
                P_full[i, j, k] = (1.0 - val) if flip else val
    return P_full


@njit
def finf_interior_folded(P_new, P_old, G):
    mid = G // 2
    m = 0.0
    for i in range(1, G-1):
        for j_s in range(0, mid):
            for k_s in range(0, mid):
                d = abs(P_new[i, j_s, k_s] - P_old[i, j_s, k_s])
                if d > m: m = d
    return m


if __name__ == '__main__':
    G = 41
    mid = G // 2
    TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
    TAU = 2.0; GAMMA = 0.1; W = 1.0
    xi_full = np.linspace(-1.0, 1.0, G)
    xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

    # Build initial P_stored (folded shape)
    P_stored = np.zeros((G, mid+1, mid+1))
    # Set FR ansatz in stored region: i ∈ [1, G-1], j_s ∈ [0, mid], k_s ∈ [0, mid]
    for i in range(G):
        for j_s in range(mid+1):
            for k_s in range(mid+1):
                j_full = j_s + mid; k_full = k_s + mid
                if abs(xi_u1[i]) < 1 - 1e-15: u1 = TOT_u * math.atanh(xi_u1[i])
                else: u1 = math.copysign(1e10, xi_u1[i])
                if abs(xi_S[j_full]) < 1 - 1e-15: Sigma = TOT_S * math.atanh(xi_S[j_full])
                else: Sigma = math.copysign(1e10, xi_S[j_full])
                S = u1 + Sigma
                if S > 50: P_stored[i, j_s, k_s] = 1.0
                elif S < -50: P_stored[i, j_s, k_s] = 0.0
                else: P_stored[i, j_s, k_s] = 1.0 / (1.0 + math.exp(-TAU * S))
    P_stored = set_boundary_folded(P_stored, G)

    print(f'G={G}, stored shape={P_stored.shape}, cells={P_stored.size}')
    t0 = time.time()
    P_new = phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
    print(f'first Φ (JIT compile): {time.time()-t0:.1f}s')
    P_new = set_boundary_folded(P_new, G)
    res = finf_interior_folded(P_new, P_stored, G)
    print(f'F_inf at FR_ansatz = {res:.3e}')
    t0 = time.time()
    P_new = phi_folded(P_new, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
    print(f'second Φ (compiled): {time.time()-t0:.1f}s')
