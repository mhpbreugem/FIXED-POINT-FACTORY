"""V9 = V8 + Gauss-Legendre quadrature DECOUPLED from the grid (the proper
hfree_smooth fix). Strictly h=0, no kernel, no bandwidth.

Bug in V8: integration over the fixed axis (e.g. xi_d when finding roots in
xi_S) was a sum over GRID NODES with weight dxi. That re-coupled the
quadrature to the grid and reintroduced kinks at grid-aligned prices.

Fix: use Nq Gauss-Legendre nodes in the fixed axis, decoupled from grid.
At each GL node xi_fixed_GL (NOT on the grid), build 1D cubic spline
P_line(xi_other) by spline-interp through the column at each grid xi_other,
then root-find on P_line.

Tensor cubic spline: P(xi_a, xi_b) is C2 in both axes via natural cubic
splines along each grid line.
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

# ---- Gauss-Legendre on [-1, 1] (will rescale to [-(1-EPSB), 1-EPSB]) ----
# Use Nq=16 for good accuracy
GL_NODES_16 = np.array([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878317, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
    +0.0950125098376374, +0.2816035507792589, +0.4580167776572274, +0.6178762444026438,
    +0.7554044083550030, +0.8656312023878317, +0.9445750230732326, +0.9894009349916499])
GL_W_16 = np.array([
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541])

# Map GL nodes to xi-domain [-(1-EPSB), 1-EPSB]
EPSB = 0.01
XI_GL = (1.0 - EPSB) * GL_NODES_16   # in xi-coords
W_GL = (1.0 - EPSB) * GL_W_16        # GL weights (length factor)
NQ = len(XI_GL)


# ============== natural cubic spline (uniform spacing) ==============

def natural_spline_M(y, h):
    n = y.size
    M = np.zeros(n)
    if n < 3: return M
    rhs = np.zeros(n)
    for i in range(1, n-1):
        rhs[i] = 6.0/(h*h) * (y[i-1] - 2*y[i] + y[i+1])
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
            t = 0.5*(t_prev + t_cur)
            for _ in range(30):
                val, der = spline_eval(y, M, h, u0, t)
                fval = val - p_target
                if abs(der) < 1e-14: break
                tn = t - fval/der
                if tn < t_prev: tn = t_prev
                if tn > t_cur: tn = t_cur
                if abs(tn - t) < 1e-13: break
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


# ============== Agent 1 evidence with GL-decoupled quadrature ==============

def evidence_agent1_v9(P, i_u, p_target, xi_arr, dxi):
    """A_v(p) for agent 1 (slice P[i_u, :, :] in xi_S, xi_d). Tensor cubic spline
    + partition-of-unity + GL quadrature in the fixed axis.
    """
    G = xi_arr.size; xi0 = xi_arr[0]
    A0 = 0.0; A1 = 0.0
    # ---- Pass: fix xi_d (GL nodes), root-find in xi_S ----
    for q in range(NQ):
        xi_b = XI_GL[q]; w_b = W_GL[q]
        delta_b = TOT_d * math.atanh(xi_b)
        # Build P_line[ka] = P[i_u, ka, xi_b] via spline interp in xi_d at each ka
        P_line = np.empty(G)
        Mb_cache = [natural_spline_M(P[i_u, ka, :], dxi) for ka in range(G)]
        for ka in range(G):
            val, _ = spline_eval(P[i_u, ka, :], Mb_cache[ka], dxi, xi0, xi_b)
            P_line[ka] = val
        # Spline along xi_a, find roots
        Ma = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, Ma, dxi, xi0, p_target, sub=8)
        for (xi_a_star, dP_dxi_a) in roots:
            if abs(xi_a_star) >= 1 - 1e-12: continue
            Sigma_a = TOT_S * math.atanh(xi_a_star)
            dP_dSigma = dP_dxi_a * (1 - xi_a_star**2) / TOT_S
            if abs(dP_dSigma) < 1e-14: continue
            # Local dP/d_delta at this (xi_a_star, xi_b) point
            # Build vertical spline at xi_a_star, evaluate derivative at xi_b
            # P_at_xi_a_star[kb] = spline_eval at xi_a_star of P[i_u, :, kb]
            P_vertcol = np.empty(G)
            for kb in range(G):
                Mv = natural_spline_M(P[i_u, :, kb], dxi)
                val, _ = spline_eval(P[i_u, :, kb], Mv, dxi, xi0, xi_a_star)
                P_vertcol[kb] = val
            Mvc = natural_spline_M(P_vertcol, dxi)
            _, dP_dxi_b_loc = spline_eval(P_vertcol, Mvc, dxi, xi0, xi_b)
            dP_ddelta = dP_dxi_b_loc * (1 - xi_b**2) / TOT_d
            w_a = dP_dSigma*dP_dSigma / max(dP_dSigma**2 + dP_ddelta**2, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal(u_2, VM0) * f_signal(u_3, VM0)
            f1 = f_signal(u_2, VM1) * f_signal(u_3, VM1)
            contrib = w_b * w_a / abs(dP_dSigma)
            A0 += contrib * f0
            A1 += contrib * f1
    # ---- Symmetric pass: fix xi_a (GL nodes), root-find in xi_d ----
    for q in range(NQ):
        xi_a = XI_GL[q]; w_a_GL = W_GL[q]
        Sigma_a = TOT_S * math.atanh(xi_a)
        P_line = np.empty(G)
        Ma_cache = [natural_spline_M(P[i_u, :, kb], dxi) for kb in range(G)]
        for kb in range(G):
            val, _ = spline_eval(P[i_u, :, kb], Ma_cache[kb], dxi, xi0, xi_a)
            P_line[kb] = val
        Mb = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, Mb, dxi, xi0, p_target, sub=8)
        for (xi_b_star, dP_dxi_b) in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * math.atanh(xi_b_star)
            dP_ddelta = dP_dxi_b * (1 - xi_b_star**2) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            # Local dP/dSigma at this (xi_a, xi_b_star)
            P_horiz = np.empty(G)
            for ka in range(G):
                Mh = natural_spline_M(P[i_u, ka, :], dxi)
                val, _ = spline_eval(P[i_u, ka, :], Mh, dxi, xi0, xi_b_star)
                P_horiz[ka] = val
            Mhc = natural_spline_M(P_horiz, dxi)
            _, dP_dxi_a_loc = spline_eval(P_horiz, Mhc, dxi, xi0, xi_a)
            dP_dSigma_loc = dP_dxi_a_loc * (1 - xi_a**2) / TOT_S
            w_b = dP_ddelta**2 / max(dP_dSigma_loc**2 + dP_ddelta**2, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal(u_2, VM0) * f_signal(u_3, VM0)
            f1 = f_signal(u_2, VM1) * f_signal(u_3, VM1)
            contrib = w_a_GL * w_b / abs(dP_ddelta)
            A0 += contrib * f0
            A1 += contrib * f1
    # Jacobian factor 1/2 for (u_2, u_3) -> (Sigma, delta)
    return 0.5*A0, 0.5*A1


# ============== Agent 2/3 evidence (Sigma-interp + GL co-area) ==============

def evidence_agent_oblique_v9(P, p_target, xi_arr, dxi, u_cell, sign_for_other):
    """Same GL-decoupled approach as agent 1, but slice extracted via Sigma-interp.
    For each (xi_u, xi_d) point, P_slc = spline-eval of P[xi_u, :, xi_d] at
    xi_t = tanh(Sigma_req/TOT_S), where Sigma_req = 2*u_cell + sign*delta(xi_d).
    """
    G = xi_arr.size; xi0 = xi_arr[0]
    A0 = 0.0; A1 = 0.0
    # ---- Pass: fix xi_d (GL), root-find in xi_u ----
    for q in range(NQ):
        xi_b = XI_GL[q]; w_b = W_GL[q]
        delta_b = TOT_d * math.atanh(xi_b)
        u_other = u_cell + sign_for_other * delta_b
        f0_oth = f_signal(u_other, VM0); f1_oth = f_signal(u_other, VM1)
        Sigma_req = 2*u_cell + sign_for_other * delta_b
        # Convert to xi_t
        if abs(Sigma_req) > TOT_S * math.atanh(1 - 1e-12):
            continue   # off-grid (use boundary?)
        xi_t = math.tanh(Sigma_req / TOT_S)
        if xi_t <= xi_arr[0] or xi_t >= xi_arr[-1]: continue
        # Build P_line[ka] = P_slc(xi_a = grid, xi_b = xi_b, with Sigma interp at xi_t)
        # Tensor: at each ka, P_col_in_Sigma = P[ka, :, kb_via_spline_in_xi_b]
        # First, for each ka, spline P[ka, :, :] in xi_d at xi_b to get a 1D in Sigma,
        # then spline-eval in Sigma at xi_t to get P_slc(ka, xi_b)
        P_line = np.empty(G)
        for ka in range(G):
            # For this ka, get the 1D spline along Sigma at the GL xi_b
            P_col_2D = P[ka, :, :]  # (G, G) - axes (Sigma, delta)
            # Step 1: at each Sigma index j, spline P_col_2D[j, :] in xi_d, eval at xi_b
            P_along_Sigma = np.empty(G)
            for j in range(G):
                Md = natural_spline_M(P_col_2D[j, :], dxi)
                val, _ = spline_eval(P_col_2D[j, :], Md, dxi, xi0, xi_b)
                P_along_Sigma[j] = val
            # Step 2: spline P_along_Sigma vs xi_S, eval at xi_t
            Ms = natural_spline_M(P_along_Sigma, dxi)
            val, _ = spline_eval(P_along_Sigma, Ms, dxi, xi0, xi_t)
            P_line[ka] = val
        # Now P_line[ka] = P_slc(xi_u[ka], xi_b). Spline in xi_u, find roots
        Mu = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, Mu, dxi, xi0, p_target, sub=8)
        for (xi_u_star, dP_dxi_u) in roots:
            if abs(xi_u_star) >= 1 - 1e-12: continue
            u_1 = TOT_u * math.atanh(xi_u_star)
            dP_du1 = dP_dxi_u * (1 - xi_u_star**2) / TOT_u
            if abs(dP_du1) < 1e-14: continue
            f0_u1 = f_signal(u_1, VM0); f1_u1 = f_signal(u_1, VM1)
            # Local dP/d_delta at this (xi_u_star, xi_b) point
            # Build P_vertcol along xi_d at xi_u_star (using full Sigma-interp at each xi_d_idx)
            # Approx: use the same P_line approach but iterate over xi_d's GL/grid -> needs more work
            # For now: use the local FINITE DIFFERENCE in xi_d on P_line built at slightly shifted xi_b
            # Simpler approx: assume P_slc smooth in xi_d, use bilinear gradient from neighbors
            # We'll use a small finite-diff via shift in xi_b
            delta_xi = 1e-4
            xi_b_plus = min(xi_b + delta_xi, 1 - 1e-9)
            # rebuild P_line at xi_b_plus (just at the relevant ka cells around xi_u_star)
            # For speed: build P_line_plus only at a few nearest ka, then approximate spline-deriv
            # Use ka_near
            ka_near = int(round((xi_u_star - xi0) / dxi))
            ka_near = max(1, min(G-2, ka_near))
            # P_plus at ka_near using the same procedure
            P_along_Sigma2 = np.empty(G)
            for j in range(G):
                Md = natural_spline_M(P[ka_near, j, :], dxi)
                val, _ = spline_eval(P[ka_near, j, :], Md, dxi, xi0, xi_b_plus)
                P_along_Sigma2[j] = val
            Ms2 = natural_spline_M(P_along_Sigma2, dxi)
            val_plus, _ = spline_eval(P_along_Sigma2, Ms2, dxi, xi0, xi_t)
            dP_dxi_b_loc = (val_plus - P_line[ka_near]) / delta_xi
            dP_ddelta = dP_dxi_b_loc * (1 - xi_b**2) / TOT_d
            w_a = dP_du1**2 / max(dP_du1**2 + dP_ddelta**2, 1e-30)
            contrib = w_b * w_a / abs(dP_du1)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    # ---- Symmetric pass: fix xi_u (GL), root-find in xi_d ----
    for q in range(NQ):
        xi_a = XI_GL[q]; w_a_GL = W_GL[q]
        u_1 = TOT_u * math.atanh(xi_a)
        f0_u1 = f_signal(u_1, VM0); f1_u1 = f_signal(u_1, VM1)
        # For each xi_d grid index kb, compute P_slc(xi_a, xi_d[kb]) via Sigma-interp
        P_line = np.empty(G)
        for kb in range(G):
            delta_b = TOT_d * math.atanh(np.clip(xi_arr[kb], -0.9999999, 0.9999999)) if abs(xi_arr[kb]) < 1-1e-12 else math.copysign(1e10, xi_arr[kb])
            Sigma_req = 2*u_cell + sign_for_other * delta_b
            if abs(Sigma_req) > TOT_S * math.atanh(1-1e-12):
                P_line[kb] = (1.0 if Sigma_req > 0 else 0.0); continue
            xi_t = math.tanh(Sigma_req / TOT_S)
            if xi_t <= xi_arr[0]: P_line[kb] = 0.0; continue
            if xi_t >= xi_arr[-1]: P_line[kb] = 1.0; continue
            # P_slc(xi_a, xi_d[kb]) = spline-eval in (xi_u, xi_Sigma) at (xi_a, xi_t)
            # First: at each xi_S index j, spline P[:, j, kb] in xi_u, eval at xi_a
            P_along_xi_u = np.empty(G)
            for j in range(G):
                Mu = natural_spline_M(P[:, j, kb], dxi)
                val, _ = spline_eval(P[:, j, kb], Mu, dxi, xi0, xi_a)
                P_along_xi_u[j] = val
            # Second: spline P_along_xi_u vs xi_S, eval at xi_t
            Ms = natural_spline_M(P_along_xi_u, dxi)
            val, _ = spline_eval(P_along_xi_u, Ms, dxi, xi0, xi_t)
            P_line[kb] = val
        # Spline P_line vs xi_d, find roots
        Md = natural_spline_M(P_line, dxi)
        roots = spline_roots(P_line, Md, dxi, xi0, p_target, sub=8)
        for (xi_b_star, dP_dxi_b) in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * math.atanh(xi_b_star)
            u_other = u_cell + sign_for_other * delta_b
            f0_oth = f_signal(u_other, VM0); f1_oth = f_signal(u_other, VM1)
            dP_ddelta = dP_dxi_b * (1 - xi_b_star**2) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            # Local dP/du_1: approximate via shift in xi_a (small finite-diff)
            delta_xi = 1e-4
            xi_a_plus = min(xi_a + delta_xi, 1 - 1e-9)
            kb_near = int(round((xi_b_star - xi0)/dxi))
            kb_near = max(1, min(G-2, kb_near))
            delta_kb = TOT_d * math.atanh(np.clip(xi_arr[kb_near], -0.9999999, 0.9999999)) if abs(xi_arr[kb_near]) < 1-1e-12 else 0
            Sigma_req2 = 2*u_cell + sign_for_other * delta_kb
            xi_t2 = math.tanh(Sigma_req2 / TOT_S) if abs(Sigma_req2) < TOT_S*math.atanh(1-1e-12) else math.copysign(0.9999, Sigma_req2)
            xi_t2 = max(xi_arr[0]+1e-9, min(xi_arr[-1]-1e-9, xi_t2))
            P_along_xi_u_plus = np.empty(G)
            for j in range(G):
                Mu = natural_spline_M(P[:, j, kb_near], dxi)
                val, _ = spline_eval(P[:, j, kb_near], Mu, dxi, xi0, xi_a_plus)
                P_along_xi_u_plus[j] = val
            Ms2 = natural_spline_M(P_along_xi_u_plus, dxi)
            val_plus, _ = spline_eval(P_along_xi_u_plus, Ms2, dxi, xi0, xi_t2)
            dP_dxi_a_loc = (val_plus - P_line[kb_near]) / delta_xi
            dP_du1_loc = dP_dxi_a_loc * (1 - xi_a**2) / TOT_u
            w_b = dP_ddelta**2 / max(dP_du1_loc**2 + dP_ddelta**2, 1e-30)
            contrib = w_a_GL * w_b / abs(dP_ddelta)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    return A0, A1


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


def phi_v9(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
            gamma, clearing='crra'):
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
                A0, A1 = evidence_agent1_v9(P, i, p_cell, xi_arr, dxi)
                den = f0_u1*A0 + f1_u1*A1
                mu0 = f1_u1*A1/den if den > 0 else 0.5
                A0, A1 = evidence_agent_oblique_v9(P, p_cell, xi_arr, dxi, u2_cell, -1.0)
                den = f0_u2*A0 + f1_u2*A1
                mu1 = f1_u2*A1/den if den > 0 else 0.5
                A0, A1 = evidence_agent_oblique_v9(P, p_cell, xi_arr, dxi, u3_cell, +1.0)
                den = f0_u3*A0 + f1_u3*A1
                mu2 = f1_u3*A1/den if den > 0 else 0.5
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
