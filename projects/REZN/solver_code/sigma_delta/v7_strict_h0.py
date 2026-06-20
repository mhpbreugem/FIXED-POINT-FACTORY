"""V7 STRICTLY h=0 on sigma-delta frame: marching squares + bilinear gradient
+ 3-pt Gauss-Legendre per contour segment. NO kernel, NO bandwidth. Exact
co-area weighting 1/|grad P|.

This is the proven approach from k3_strict_h0_exact/strict_h0_operator.py,
ported to the (u_1, Sigma, delta) xi-grid.

For each inner cell (i, j, k):
  Agent 1 (own u_1): slice P[i, :, :] in (xi_S, xi_d). March squares finds
    contour segments. Integrate f_v(u_2)*f_v(u_3)/|grad_phys P| * dl_phys
    along each segment.
  Agent 2 (own u_2): Sigma_required(delta) = 2*u_2_cell - delta. Build slice
    P_slc(u_1[a], delta[b]) via CUBIC Sigma-interp. March on (xi_u, xi_d).
    Integrate f_v(u_1)*f_v(u_3=u_2_cell-delta).
  Agent 3 (own u_3): symmetric with sign flip.

Bayes ratio mu_k = f_v(u_own)*A_1/(f_0(u_own)*A_0 + f_1(u_own)*A_1).

Float64 implementation for SPEED -- if it works at G=10, then high-prec
flint version can be built. NO Jacobian factor in the kernel sum (we use
the actual physical co-area integration).
"""
import os, sys, math
import numpy as np
from scipy.interpolate import CubicSpline

TAU = 2.0
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
VM0 = -0.5; VM1 = +0.5

def f_signal(u, vm, tau=TAU):
    return math.sqrt(tau/(2*math.pi)) * math.exp(-0.5*tau*(u-vm)**2)
def f_signal_arr(u, vm, tau=TAU):
    return np.sqrt(tau/(2*np.pi)) * np.exp(-0.5*tau*(u-vm)**2)

# ---- Marching squares + co-area integration on a 2D slice ----
# slice: P (N+1, N+1) array of P-values at corners
# coord_to_phys: function (xi_a, xi_b) -> (phys_a, phys_b) for converting ξ-coords to physical
# d phys_a/d xi_a, d phys_b/d xi_b: Jacobian factors at each (xi_a, xi_b)
# integrand_phys(phys_a, phys_b) -> (f_0_val, f_1_val): the density product per unit physical area
# Returns (A0, A1)

GL_NODES = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
GL_W = np.array([5/9, 8/9, 5/9])

def _interp_edge(pa, pb, p):
    d = pb - pa
    return (p - pa) / d if d != 0 else 0.5

def _marching_segments(case, P00, P10, P01, P11, p):
    def eb(): return _interp_edge(P00, P10, p)
    def er(): return _interp_edge(P10, P11, p)
    def et(): return _interp_edge(P01, P11, p)
    def el(): return _interp_edge(P00, P01, p)
    if case == 1 or case == 14: return [(0, el(), et(), 1)]
    if case == 2 or case == 13: return [(et(), 1, 1, er())]
    if case == 4 or case == 11: return [(eb(), 0, 1, er())]
    if case == 8 or case == 7:  return [(0, el(), eb(), 0)]
    if case == 3 or case == 12: return [(0, el(), 1, er())]
    if case == 6 or case == 9:  return [(eb(), 0, et(), 1)]
    if case == 5 or case == 10:
        center = (P00 + P10 + P01 + P11) * 0.25
        if case == 5:
            if center >= p: return [(eb(), 0, 1, er()), (0, el(), et(), 1)]
            else:           return [(eb(), 0, 0, el()), (et(), 1, 1, er())]
        else:
            if center >= p: return [(0, el(), eb(), 0), (et(), 1, 1, er())]
            else:           return [(0, el(), et(), 1), (eb(), 0, 1, er())]
    return []

def march_co_area(P_slc, xi_a_arr, xi_b_arr, phys_a_arr, phys_b_arr,
                  Ja_arr, Jb_arr, p_target, integrand_phys):
    """Marching squares co-area integration on a 2D slice P_slc[a, b].
    xi_a_arr, xi_b_arr: ξ-coords of the grid points (len N).
    phys_a_arr, phys_b_arr: physical coords.
    Ja_arr[a] = d phys_a / d xi_a (Jacobian at point a).
    Jb_arr[b] = d phys_b / d xi_b.
    integrand_phys(phys_a, phys_b) -> (f0_val, f1_val): density product f_v_a * f_v_b at (phys_a, phys_b).
    Returns (A0, A1) physical-coord co-area integrals.
    """
    Na, Nb = P_slc.shape
    A0 = 0.0; A1 = 0.0
    for ia in range(Na - 1):
        dxi_a = xi_a_arr[ia+1] - xi_a_arr[ia]
        Ja_avg = 0.5*(Ja_arr[ia] + Ja_arr[ia+1])  # avg Jacobian over cell (a-dir)
        d_phys_a = phys_a_arr[ia+1] - phys_a_arr[ia]
        for ib in range(Nb - 1):
            P00 = P_slc[ia,   ib  ]; P10 = P_slc[ia+1, ib  ]
            P01 = P_slc[ia,   ib+1]; P11 = P_slc[ia+1, ib+1]
            c00 = P00 >= p_target; c10 = P10 >= p_target
            c01 = P01 >= p_target; c11 = P11 >= p_target
            case = (8 if c00 else 0) | (4 if c10 else 0) | (2 if c11 else 0) | (1 if c01 else 0)
            if case == 0 or case == 15: continue
            d_phys_b = phys_b_arr[ib+1] - phys_b_arr[ib]
            Jb_avg = 0.5*(Jb_arr[ib] + Jb_arr[ib+1])
            segs = _marching_segments(case, P00, P10, P01, P11, p_target)
            for (s0, t0, s1, t1) in segs:
                # endpoints in physical
                # NOTE: s, t are local fracs in [0,1]; physical is bilinear in ξ within the cell
                # For agent 1 (ξ-coords for both axes), physical_a depends only on ξ_a
                # We approximate physical coords linearly in (s, t) within the cell
                # (acceptable since Δphys/dxi is the local Jacobian and within one cell
                #  the variation is small for moderate G)
                ds = s1 - s0; dt = t1 - t0
                seg_dxi_a = ds * dxi_a
                seg_dxi_b = dt * (xi_b_arr[ib+1] - xi_b_arr[ib])
                # physical segment length: dl_phys = sqrt((Ja*dxi_a)^2 + (Jb*dxi_b)^2)
                # using midpoint Jacobians
                # but for accuracy, compute at each Gauss node
                for k in range(3):
                    r = (GL_NODES[k] + 1.0) * 0.5
                    s = s0 + r * ds; t = t0 + r * dt
                    xi_a = xi_a_arr[ia] + s * dxi_a
                    xi_b = xi_b_arr[ib] + t * (xi_b_arr[ib+1] - xi_b_arr[ib])
                    # physical coords at node (linear interp in ξ within cell)
                    p_a = phys_a_arr[ia] + s * d_phys_a
                    p_b = phys_b_arr[ib] + t * d_phys_b
                    # Jacobian at node (linear interp)
                    Ja = Ja_arr[ia] + s * (Ja_arr[ia+1] - Ja_arr[ia])
                    Jb = Jb_arr[ib] + t * (Jb_arr[ib+1] - Jb_arr[ib])
                    # bilinear gradient of P in (ξ_a, ξ_b)
                    dPds_xi = (P10 - P00)*(1-t) + (P11 - P01)*t  # ∂P/∂(s)
                    dPdt_xi = (P01 - P00)*(1-s) + (P11 - P10)*s  # ∂P/∂(t)
                    # ∂P/∂ξ_a = dPds_xi / dxi_a
                    dPdxi_a = dPds_xi / dxi_a
                    dPdxi_b = dPdt_xi / (xi_b_arr[ib+1] - xi_b_arr[ib])
                    # ∂P/∂phys_a = ∂P/∂ξ_a / Ja
                    dPdpa = dPdxi_a / Ja
                    dPdpb = dPdxi_b / Jb
                    gnorm = math.sqrt(dPdpa*dPdpa + dPdpb*dPdpb)
                    if gnorm <= 0: continue
                    # physical segment differential
                    dseg_pa = ds * (Ja * dxi_a)   # d phys_a along segment
                    dseg_pb = dt * (Jb * (xi_b_arr[ib+1] - xi_b_arr[ib]))
                    dl_phys = math.sqrt(dseg_pa*dseg_pa + dseg_pb*dseg_pb)
                    # weight: GL × (jacobian [-1,1]->[0,1] = 0.5) × dl_phys
                    w = GL_W[k] * 0.5 * dl_phys
                    f0_val, f1_val = integrand_phys(p_a, p_b)
                    A0 += w * f0_val / gnorm
                    A1 += w * f1_val / gnorm
    return A0, A1


# ---- Per-agent evidence ----

def evidence_agent1_strict(P_full, i_u, p_target, xi_arr, u_arr, S_arr, d_arr, JS, Jd):
    """Agent 1: slice P[i_u, :, :] in (ξ_S, ξ_d). Marching squares + co-area.
    Integrand at (Σ, δ): f_v((Σ+δ)/2) * f_v((Σ-δ)/2).
    Returns (A0, A1)."""
    P_slc = P_full[i_u, :, :]  # (G_full, G_full) in (ξ_S, ξ_d)
    def integrand_phys(S, d):
        u2 = 0.5*(S+d); u3 = 0.5*(S-d)
        f0 = f_signal(u2, VM0) * f_signal(u3, VM0)
        f1 = f_signal(u2, VM1) * f_signal(u3, VM1)
        return f0, f1
    A0, A1 = march_co_area(P_slc, xi_arr, xi_arr, S_arr, d_arr, JS, Jd,
                            p_target, integrand_phys)
    # The change of variables (u_2, u_3) -> (Σ, δ) has Jacobian 0.5
    return 0.5*A0, 0.5*A1


def _build_oblique_slice(P_full, u_cell, sign_for_other, xi_arr, S_arr, TOT_S_):
    """Build P_slc[a, b] over (ξ_u, ξ_d) at fixed u_2 (sign=-1) or u_3 (sign=+1),
    using CUBIC Sigma-interp along axis 1 of P_full.
    Σ_req(δ_b) = 2*u_cell + sign*δ_b. At each (a, b), interp P_full[a, :, b] at Σ_req.
    """
    G = len(xi_arr)
    P_slc = np.zeros((G, G))
    for b in range(G):
        if abs(xi_arr[b]) >= 1 - 1e-12:
            delta_b = math.copysign(1e10, xi_arr[b])
        else:
            delta_b = TOT_d * math.atanh(xi_arr[b])
        Sigma_req = 2*u_cell + sign_for_other * delta_b
        # Convert to xi_t for interp
        if Sigma_req > 1e10: P_slc[:, b] = P_full[:, G-1, b]; continue
        if Sigma_req < -1e10: P_slc[:, b] = P_full[:, 0, b]; continue
        xi_t = math.tanh(Sigma_req / TOT_S_)
        if xi_t <= xi_arr[0]: P_slc[:, b] = P_full[:, 0, b]; continue
        if xi_t >= xi_arr[-1]: P_slc[:, b] = P_full[:, G-1, b]; continue
        # cubic interp along Sigma axis at xi_t
        for a in range(G):
            P_col = P_full[a, :, b]
            cs = CubicSpline(xi_arr, P_col, bc_type='natural', extrapolate=False)
            P_slc[a, b] = float(cs(xi_t))
    return P_slc


def evidence_agent_oblique_strict(P_full, p_target, xi_arr, u_arr, S_arr, d_arr,
                                    JS, Jd, u_cell, sign_for_other, TOT_S_):
    """Agent 2 (sign=-1, own=u_2) or Agent 3 (sign=+1, own=u_3).
    Build oblique slice P_slc(u_1[a], δ[b]), march squares on (ξ_u, ξ_d) plane.
    Integrand: f_v(u_1)*f_v(u_other), where u_other = u_cell + sign*δ_b.
    """
    P_slc = _build_oblique_slice(P_full, u_cell, sign_for_other, xi_arr, S_arr, TOT_S_)
    def integrand_phys(u1, delta):
        u_other = u_cell + sign_for_other * delta
        f0 = f_signal(u1, VM0) * f_signal(u_other, VM0)
        f1 = f_signal(u1, VM1) * f_signal(u_other, VM1)
        return f0, f1
    A0, A1 = march_co_area(P_slc, xi_arr, xi_arr, u_arr, d_arr, JS, Jd,
                            p_target, integrand_phys)
    # Note: du_3 = -dδ at fixed u_2 for agent 2, so |jacobian|=1, no extra factor
    return A0, A1


# ---- Full Phi ----

def phi_strict_sigdelta(P_full, INNER_LO, INNER_HI, xi_arr, u_arr, S_arr, d_arr,
                          JS, Jd, gamma, clearing='crra'):
    G = len(xi_arr)
    P_new = P_full.copy()
    coef = math.sqrt(TAU/(2*math.pi))
    for i in range(INNER_LO, INNER_HI):
        if abs(xi_arr[i]) >= 1 - 1e-12: continue
        u1_cell = TOT_u * math.atanh(xi_arr[i])
        f0_u1 = coef * math.exp(-0.5*TAU*(u1_cell-VM0)**2)
        f1_u1 = coef * math.exp(-0.5*TAU*(u1_cell-VM1)**2)
        for j in range(INNER_LO, INNER_HI):
            if abs(xi_arr[j]) >= 1 - 1e-12: continue
            Sigma_cell = TOT_S * math.atanh(xi_arr[j])
            for k in range(INNER_LO, INNER_HI):
                if abs(xi_arr[k]) >= 1 - 1e-12: continue
                d_cell = TOT_d * math.atanh(xi_arr[k])
                p_cell = P_full[i, j, k]
                u2_cell = 0.5*(Sigma_cell + d_cell)
                u3_cell = 0.5*(Sigma_cell - d_cell)
                f0_u2 = coef * math.exp(-0.5*TAU*(u2_cell-VM0)**2)
                f1_u2 = coef * math.exp(-0.5*TAU*(u2_cell-VM1)**2)
                f0_u3 = coef * math.exp(-0.5*TAU*(u3_cell-VM0)**2)
                f1_u3 = coef * math.exp(-0.5*TAU*(u3_cell-VM1)**2)
                # Agent 1
                A0, A1 = evidence_agent1_strict(P_full, i, p_cell, xi_arr, u_arr, S_arr, d_arr, JS, Jd)
                num = f1_u1 * A1; den = f0_u1 * A0 + num
                mu0 = num/den if den > 0 else 0.5
                # Agent 2
                A0, A1 = evidence_agent_oblique_strict(P_full, p_cell, xi_arr, u_arr, S_arr, d_arr,
                                                       JS, Jd, u2_cell, -1.0, TOT_S)
                num = f1_u2 * A1; den = f0_u2 * A0 + num
                mu1 = num/den if den > 0 else 0.5
                # Agent 3
                A0, A1 = evidence_agent_oblique_strict(P_full, p_cell, xi_arr, u_arr, S_arr, d_arr,
                                                       JS, Jd, u3_cell, +1.0, TOT_S)
                num = f1_u3 * A1; den = f0_u3 * A0 + num
                mu2 = num/den if den > 0 else 0.5
                eps_p = 1e-30
                mu0 = max(eps_p, min(1-eps_p, mu0))
                mu1 = max(eps_p, min(1-eps_p, mu1))
                mu2 = max(eps_p, min(1-eps_p, mu2))
                if clearing == 'cara':
                    pi_ = (math.log(mu0/(1-mu0)) + math.log(mu1/(1-mu1)) + math.log(mu2/(1-mu2))) / 3.0
                    P_new[i, j, k] = 1.0/(1.0+math.exp(-pi_))
                else:
                    P_new[i, j, k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_new


def crra_clear(mu0, mu1, mu2, gamma, steps=200):
    eps = 1e-30
    a, b = eps, 1-eps
    lm0 = math.log(mu0/(1-mu0)); lm1 = math.log(mu1/(1-mu1)); lm2 = math.log(mu2/(1-mu2))
    def dd(lm, p, gf):
        lp = math.log(p/(1-p))
        R = math.exp((lm-lp)/gf); return (R-1)/((1-p)+R*p)
    for _ in range(steps):
        m = 0.5*(a+b)
        e = dd(lm0, m, gamma) + dd(lm1, m, gamma) + dd(lm2, m, gamma)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)


def set_boundary(P):
    G = P.shape[0]
    P[0, :, :] = 0.0; P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0; P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]; P[:, :, G-1] = P[:, :, G-2]
    return P


def make_grids(G_FULL):
    xi_arr = np.linspace(-1.0, 1.0, G_FULL)
    safe = np.clip(xi_arr, -0.9999999, 0.9999999)
    u_arr = TOT_u * np.arctanh(safe)
    S_arr = TOT_S * np.arctanh(safe)
    d_arr = TOT_d * np.arctanh(safe)
    interior = np.abs(xi_arr) < 1 - 1e-12
    Ju = np.where(interior, TOT_u/(1-xi_arr**2), 0.0)
    JS = np.where(interior, TOT_S/(1-xi_arr**2), 0.0)
    Jd = np.where(interior, TOT_d/(1-xi_arr**2), 0.0)
    return xi_arr, u_arr, S_arr, d_arr, Ju, JS, Jd
