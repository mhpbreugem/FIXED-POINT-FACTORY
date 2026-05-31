"""V8 = sigma-delta port of k3_hfree_smooth -- STRICTLY h=0, no kernel,
no bandwidth, no smoothing parameter.

P represented by natural cubic spline along each grid line in xi-coords.
A_v(p) = int_{P=p} f_v(u_a) f_v(u_b) / |grad_phys P| dsigma computed via
partition-of-unity over two parameterizations:

  A_v(p) = A_v^(a)(p) + A_v^(b)(p)
  A_v^(a)(p) = sum over xi_b nodes of {roots xi_a*: P(xi_a*, xi_b)=p}
                f_v(phys_a*) f_v(phys_b) / |dP/d phys_a| * w_a * gl_weight(xi_b)
  A_v^(b)(p) = symmetric with axes swapped
  w_a = (dP/d phys_a)^2 / ((dP/d phys_a)^2 + (dP/d phys_b)^2)
  w_b = 1 - w_a

The partition-of-unity weight kills the 1/|grad| blow-up at turning points
(the contour having zero slope in one direction is well-resolved by the
other parameterization).

Implementation: natural cubic spline (uniform xi-grid spacing), Newton
root-find on the spline, then evaluation at each root for the integrand.
NO kernel anywhere.
"""
import os, sys, math
import numpy as np

TAU = 2.0
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
VM0 = -0.5; VM1 = +0.5
COEF = math.sqrt(TAU / (2*math.pi))
EPS_PRICE = 1e-12

def f_signal(u, vm, tau=TAU):
    return COEF * math.exp(-0.5*tau*(u-vm)**2)


# 3-pt GL on [-1, 1]
GL_NODES = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
GL_W = np.array([5/9, 8/9, 5/9])

# ============== natural cubic spline (uniform spacing) ==============

def natural_spline_M(y, h):
    """Second derivatives M at the n knots of a natural cubic spline.
    Uniform spacing h."""
    n = y.size
    M = np.zeros(n)
    if n < 3: return M
    rhs = np.zeros(n)
    for i in range(1, n-1):
        rhs[i] = 6.0/(h*h) * (y[i-1] - 2*y[i] + y[i+1])
    # Thomas: diag=4, off=1 with natural BC M[0]=M[n-1]=0
    c = np.zeros(n); d = np.zeros(n)
    b0 = 4.0
    c[1] = 1.0/b0; d[1] = rhs[1]/b0
    for i in range(2, n-1):
        m = 4.0 - c[i-1]
        c[i] = 1.0/m
        d[i] = (rhs[i] - d[i-1])/m
    for i in range(n-2, 0, -1):
        M[i] = d[i] - c[i]*M[i+1]
    return M

def spline_eval(y, M, h, u0, t):
    """Evaluate natural cubic spline at coordinate t, knots at u0+k*h. Returns (val, deriv)."""
    n = y.size
    x = (t-u0)/h
    i = int(math.floor(x))
    if i < 0: i = 0
    if i > n-2: i = n-2
    xi = u0 + i*h
    a = (xi+h-t)/h; b = (t-xi)/h
    yi = y[i]; yi1 = y[i+1]
    Mi = M[i]; Mi1 = M[i+1]
    val = (a*yi + b*yi1
           + ((a**3 - a)*Mi + (b**3 - b)*Mi1) * (h*h)/6.0)
    der = ((yi1-yi)/h
           - (3*a*a-1)/6*h*Mi
           + (3*b*b-1)/6*h*Mi1)
    return val, der

def spline_roots(y, M, h, u0, p_target, sub=8, max_roots=4):
    """Find all roots of spline(t)=p_target in [u0, u0+(n-1)*h].
    Returns list of (t_root, derivative_at_root)."""
    n = y.size
    nseg = (n-1)*sub
    roots = []
    t_prev = u0
    v_prev, _ = spline_eval(y, M, h, u0, t_prev)
    step = (h*(n-1))/nseg
    for s in range(1, nseg+1):
        t_cur = u0 + s*step
        v_cur, _ = spline_eval(y, M, h, u0, t_cur)
        dp = v_prev - p_target
        dc = v_cur - p_target
        if not (dp == 0.0 and dc == 0.0) and dp*dc <= 0:
            # bracket; Newton polish from midpoint
            t = 0.5*(t_prev + t_cur)
            for _ in range(30):
                val, der = spline_eval(y, M, h, u0, t)
                fval = val - p_target
                if abs(der) < 1e-14: break
                tn = t - fval/der
                # clamp to bracket
                if tn < t_prev: tn = t_prev
                if tn > t_cur: tn = t_cur
                if abs(tn - t) < 1e-12: break
                t = tn
            _, der = spline_eval(y, M, h, u0, t)
            roots.append((t, der))
            if len(roots) >= max_roots: return roots
        t_prev = t_cur; v_prev = v_cur
    return roots


# ============== set_boundary ==============

def set_boundary(P):
    G = P.shape[0]
    P[0, :, :] = 0.0; P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0; P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]; P[:, :, G-1] = P[:, :, G-2]
    return P


# ============== Agent 1 evidence (slice P[i, :, :], direct, no Sigma-interp) ==============

def evidence_agent1_hfree(P, i_u, p_target, xi_arr, dxi, S_arr, d_arr, Js, Jd):
    """A_v(p) = co-area integral via partition-of-unity over (xi_S, xi_d) parameterizations.
    Returns (A0, A1).

    Sigma-axis: parametrize by xi_b (fixed xi_S, find roots in xi_d via spline).
    delta-axis: parametrize by xi_a (fixed xi_d, find roots in xi_S via spline).

    Actually: re-do with consistent naming.
    Let axis a = Sigma (xi_S), axis b = delta (xi_d).
    A_v^(b-fixed)(p) = sum over xi_b nodes of {roots in xi_a} f_v(u_2)*f_v(u_3) / |dP/dSigma| * w_a
    A_v^(a-fixed)(p) = sum over xi_a nodes of {roots in xi_b} ... / |dP/d delta| * w_b
    """
    G = xi_arr.size
    # The interpolant in xi-coords (uniform spacing dxi)
    xi0 = xi_arr[0]   # = -1
    # 3-pt GL nodes on [-1, 1] mapped to xi interior range [xi[1], xi[-2]]
    A_a = np.array([0.0, 0.0])  # parameterize by xi_b (fixed), roots in xi_a
    # Loop over xi_b "nodes" -- use grid nodes (interior cells only)
    for kb in range(1, G-1):
        xib = xi_arr[kb]
        delta_b = d_arr[kb]
        # spline of P[i_u, :, kb] vs xi_a (Sigma direction)
        P_line = P[i_u, :, kb]
        M = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, M, dxi, xi0, p_target, sub=8)
        for (xi_a_star, dP_dxi_a) in roots:
            # convert to physical
            if abs(xi_a_star) >= 1 - 1e-12: continue
            Sigma_a = TOT_S * math.atanh(xi_a_star)
            # |dP/dSigma| at this root: dP/dxi_a / (dSigma/dxi_a) = dP/dxi_a * (1-xi_a^2)/TOT_S
            dP_dSigma = dP_dxi_a * (1 - xi_a_star*xi_a_star) / TOT_S
            if abs(dP_dSigma) < 1e-14: continue
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal(u_2, VM0) * f_signal(u_3, VM0)
            f1 = f_signal(u_2, VM1) * f_signal(u_3, VM1)
            # partition weight: w_a-parameterization is good when |dP/dSigma| dominates
            # compute dP/d_delta at the same point (need spline of P[i_u, ja_nearest, :] -- bilinear approx)
            # For simplicity use bilinear: |dP/d_delta| ~ (P[i_u, ja, kb+1]-P[i_u, ja, kb-1])/(2 dxi) * dxi/d_delta
            ja = int(round((xi_a_star - xi0)/dxi))
            ja = max(1, min(G-2, ja))
            dP_dxi_b_local = (P[i_u, ja, kb+1] - P[i_u, ja, kb-1]) / (2*dxi)
            dP_ddelta = dP_dxi_b_local * (1 - xi_arr[kb]*xi_arr[kb]) / TOT_d
            w_a = dP_dSigma*dP_dSigma / max(dP_dSigma*dP_dSigma + dP_ddelta*dP_ddelta, 1e-30)
            # integrand weight in xi_b: GL-quadrature emulating midpoint*dxi_b
            # We use a simple trapezoid (each interior node weight = dxi_b)
            w_node_b = dxi
            contrib = w_node_b * w_a / abs(dP_dSigma)
            A_a[0] += contrib * f0
            A_a[1] += contrib * f1
    # symmetric: parameterize by xi_a (fixed), roots in xi_b
    A_b = np.array([0.0, 0.0])
    for ka in range(1, G-1):
        xia = xi_arr[ka]
        Sigma_a = S_arr[ka]
        P_line = P[i_u, ka, :]
        M = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, M, dxi, xi0, p_target, sub=8)
        for (xi_b_star, dP_dxi_b) in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * math.atanh(xi_b_star)
            dP_ddelta = dP_dxi_b * (1 - xi_b_star*xi_b_star) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal(u_2, VM0) * f_signal(u_3, VM0)
            f1 = f_signal(u_2, VM1) * f_signal(u_3, VM1)
            # dP/dSigma estimate at the same point
            kb = int(round((xi_b_star - xi0)/dxi))
            kb = max(1, min(G-2, kb))
            dP_dxi_a_local = (P[i_u, ka+1, kb] - P[i_u, ka-1, kb]) / (2*dxi)
            dP_dSigma_loc = dP_dxi_a_local * (1 - xi_arr[ka]*xi_arr[ka]) / TOT_S
            w_b = dP_ddelta*dP_ddelta / max(dP_dSigma_loc*dP_dSigma_loc + dP_ddelta*dP_ddelta, 1e-30)
            w_node_a = dxi
            contrib = w_node_a * w_b / abs(dP_ddelta)
            A_b[0] += contrib * f0
            A_b[1] += contrib * f1
    # Total: A_a + A_b, plus 1/2 from (u2, u3)->(Sigma, delta) Jacobian
    A0_tot = 0.5 * (A_a[0] + A_b[0])
    A1_tot = 0.5 * (A_a[1] + A_b[1])
    return A0_tot, A1_tot


# ============== Agent 2/3 evidence (oblique slice via cubic Sigma-interp) ==============

def evidence_agent_oblique_hfree(P, p_target, xi_arr, dxi, u_arr, S_arr, d_arr,
                                   Ju, Js, Jd, u_cell, sign_for_other):
    """Build slice P_slc(xi_u, xi_d) at fixed u_2=u_cell (sign=-1) or u_3=u_cell (sign=+1)
    via cubic Sigma-interp. Then apply the same partition-of-unity co-area.
    """
    G = xi_arr.size
    xi0 = xi_arr[0]
    # Build P_slc[a, b]: at each (xi_u[a], xi_d[b]), the Sigma_required = 2*u_cell + sign*delta_b
    P_slc = np.zeros((G, G))
    # For each xi_d[b], find Sigma_req, then for each a, spline-interp P[a, :, b] at xi_t
    for b in range(G):
        if abs(xi_arr[b]) >= 1 - 1e-12:
            # boundary: use the extreme P value
            delta_b = math.copysign(1e10, xi_arr[b])
        else:
            delta_b = TOT_d * math.atanh(xi_arr[b])
        Sigma_req = 2*u_cell + sign_for_other * delta_b
        # Convert to xi_t
        if Sigma_req > 1e10: P_slc[:, b] = P[:, G-1, b]; continue
        if Sigma_req < -1e10: P_slc[:, b] = P[:, 0, b]; continue
        xi_t = math.tanh(Sigma_req / TOT_S)
        if xi_t <= xi_arr[0]: P_slc[:, b] = P[:, 0, b]; continue
        if xi_t >= xi_arr[-1]: P_slc[:, b] = P[:, G-1, b]; continue
        # cubic spline along Sigma axis at each a, eval at xi_t
        for a in range(G):
            P_col = P[a, :, b]
            M = natural_spline_M(P_col, dxi)
            val, _ = spline_eval(P_col, M, dxi, xi0, xi_t)
            P_slc[a, b] = val
    # Now P_slc(xi_u, xi_d); apply same partition-of-unity co-area
    # Axis a = xi_u (-> u_1 phys), axis b = xi_d (-> delta phys)
    A_a = np.array([0.0, 0.0])
    for kb in range(1, G-1):
        xib = xi_arr[kb]
        delta_b = d_arr[kb]
        u_other = u_cell + sign_for_other * delta_b
        f0_oth = f_signal(u_other, VM0); f1_oth = f_signal(u_other, VM1)
        P_line = P_slc[:, kb]
        M = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, M, dxi, xi0, p_target, sub=8)
        for (xi_a_star, dP_dxi_a) in roots:
            if abs(xi_a_star) >= 1 - 1e-12: continue
            u_1 = TOT_u * math.atanh(xi_a_star)
            dP_du1 = dP_dxi_a * (1 - xi_a_star**2) / TOT_u
            if abs(dP_du1) < 1e-14: continue
            f0_u1 = f_signal(u_1, VM0); f1_u1 = f_signal(u_1, VM1)
            # dP/d_delta from neighbors
            ja = int(round((xi_a_star - xi0)/dxi))
            ja = max(1, min(G-2, ja))
            dP_dxi_b_local = (P_slc[ja, kb+1] - P_slc[ja, kb-1]) / (2*dxi)
            dP_ddelta = dP_dxi_b_local * (1 - xi_arr[kb]**2) / TOT_d
            w_a = dP_du1*dP_du1 / max(dP_du1*dP_du1 + dP_ddelta*dP_ddelta, 1e-30)
            w_node_b = dxi
            contrib = w_node_b * w_a / abs(dP_du1)
            A_a[0] += contrib * f0_u1 * f0_oth
            A_a[1] += contrib * f1_u1 * f1_oth
    A_b = np.array([0.0, 0.0])
    for ka in range(1, G-1):
        xia = xi_arr[ka]
        u_1 = u_arr[ka]
        f0_u1 = f_signal(u_1, VM0); f1_u1 = f_signal(u_1, VM1)
        P_line = P_slc[ka, :]
        M = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, M, dxi, xi0, p_target, sub=8)
        for (xi_b_star, dP_dxi_b) in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * math.atanh(xi_b_star)
            u_other = u_cell + sign_for_other * delta_b
            dP_ddelta = dP_dxi_b * (1 - xi_b_star**2) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            f0_oth = f_signal(u_other, VM0); f1_oth = f_signal(u_other, VM1)
            kb = int(round((xi_b_star - xi0)/dxi))
            kb = max(1, min(G-2, kb))
            dP_dxi_a_local = (P_slc[ka+1, kb] - P_slc[ka-1, kb]) / (2*dxi)
            dP_du1_loc = dP_dxi_a_local * (1 - xi_arr[ka]**2) / TOT_u
            w_b = dP_ddelta*dP_ddelta / max(dP_du1_loc*dP_du1_loc + dP_ddelta*dP_ddelta, 1e-30)
            w_node_a = dxi
            contrib = w_node_a * w_b / abs(dP_ddelta)
            A_b[0] += contrib * f0_u1 * f0_oth
            A_b[1] += contrib * f1_u1 * f1_oth
    return A_a[0] + A_b[0], A_a[1] + A_b[1]


# ============== Full Phi ==============

def crra_clear(mu0, mu1, mu2, gamma, steps=120):
    eps = 1e-30
    a, b = eps, 1-eps
    lm0 = math.log(mu0/(1-mu0)); lm1 = math.log(mu1/(1-mu1)); lm2 = math.log(mu2/(1-mu2))
    def dd(lm, p):
        lp = math.log(p/(1-p))
        R = math.exp((lm-lp)/gamma); return (R-1)/((1-p)+R*p)
    for _ in range(steps):
        m = 0.5*(a+b)
        if dd(lm0, m) + dd(lm1, m) + dd(lm2, m) > 0: a = m
        else: b = m
    return 0.5*(a+b)


def phi_hfree_sigdelta(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                        Ju, Js, Jd, gamma, clearing='crra'):
    G = xi_arr.size
    P_new = P.copy()
    for i in range(INNER_LO, INNER_HI):
        if abs(xi_arr[i]) >= 1 - 1e-12: continue
        u1_cell = u_arr[i]
        f0_u1 = f_signal(u1_cell, VM0); f1_u1 = f_signal(u1_cell, VM1)
        for j in range(INNER_LO, INNER_HI):
            if abs(xi_arr[j]) >= 1 - 1e-12: continue
            Sigma_cell = S_arr[j]
            for k in range(INNER_LO, INNER_HI):
                if abs(xi_arr[k]) >= 1 - 1e-12: continue
                d_cell = d_arr[k]
                p_cell = P[i, j, k]
                u2_cell = 0.5*(Sigma_cell + d_cell)
                u3_cell = 0.5*(Sigma_cell - d_cell)
                f0_u2 = f_signal(u2_cell, VM0); f1_u2 = f_signal(u2_cell, VM1)
                f0_u3 = f_signal(u3_cell, VM0); f1_u3 = f_signal(u3_cell, VM1)
                # Agent 1
                A0, A1 = evidence_agent1_hfree(P, i, p_cell, xi_arr, dxi, S_arr, d_arr, Js, Jd)
                num = f1_u1*A1; den = f0_u1*A0 + num
                mu0 = num/den if den > 0 else 0.5
                # Agent 2 (sign=-1)
                A0, A1 = evidence_agent_oblique_hfree(P, p_cell, xi_arr, dxi, u_arr, S_arr, d_arr,
                                                       Ju, Js, Jd, u2_cell, -1.0)
                num = f1_u2*A1; den = f0_u2*A0 + num
                mu1 = num/den if den > 0 else 0.5
                # Agent 3 (sign=+1)
                A0, A1 = evidence_agent_oblique_hfree(P, p_cell, xi_arr, dxi, u_arr, S_arr, d_arr,
                                                       Ju, Js, Jd, u3_cell, +1.0)
                num = f1_u3*A1; den = f0_u3*A0 + num
                mu2 = num/den if den > 0 else 0.5
                # clip
                eps_p = EPS_PRICE
                mu0 = max(eps_p, min(1-eps_p, mu0))
                mu1 = max(eps_p, min(1-eps_p, mu1))
                mu2 = max(eps_p, min(1-eps_p, mu2))
                if clearing == 'cara':
                    pi_ = (math.log(mu0/(1-mu0)) + math.log(mu1/(1-mu1)) + math.log(mu2/(1-mu2))) / 3
                    P_new[i, j, k] = 1.0/(1.0+math.exp(-pi_))
                else:
                    P_new[i, j, k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_new
