"""V11 = STRICT h=0 HARDWIRED variant of V10.

CONTRACT (audit invariant):
  - NO kernel anywhere
  - NO bandwidth parameter (no h, no sigma, no smoothing scale)
  - Only h-like quantities:
      * dxi: uniform xi-grid spacing (resolution parameter, not smoothing)
      * EPSB: tiny buffer to avoid evaluating atanh at xi=+-1
  - A_v(p) computed via partition-of-unity + EXACT root-find on C2 cubic spline

Original V10 (numba JIT of V9):

Numba constraints:
- All arrays numpy
- spline_roots fills a pre-allocated array (no Python list)
- All inner loops in @njit functions

Expected speedup: 100x-300x over pure-python V9.
"""
import os, sys, math
import numpy as np
from numba import njit, prange

TAU = 2.0
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
VM0 = -0.5; VM1 = +0.5
COEF = math.sqrt(TAU / (2*math.pi))
EPS_PRICE = 1e-12

# GL-16 nodes/weights on [-1,1]
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

# STRICT h=0: EPSB = boundary-buffer ONLY, NOT a smoothing bandwidth.
EPSB = 1.0e-12
XI_GL = (1.0 - EPSB) * GL_NODES_16
W_GL = (1.0 - EPSB) * GL_W_16
NQ = 16


@njit(cache=True, inline='always', fastmath=False)
def f_signal_(u, vm, coef, tau):
    return coef * math.exp(-0.5*tau*(u-vm)*(u-vm))


@njit(cache=True, fastmath=False)
def natural_spline_M(y, h):
    n = y.size
    M = np.zeros(n)
    if n < 3: return M
    rhs = np.zeros(n)
    for i in range(1, n-1):
        rhs[i] = 6.0/(h*h) * (y[i-1] - 2*y[i] + y[i+1])
    c = np.zeros(n); d = np.zeros(n)
    c[1] = 1.0/4.0; d[1] = rhs[1]/4.0
    for i in range(2, n-1):
        m = 4.0 - c[i-1]
        c[i] = 1.0/m
        d[i] = (rhs[i] - d[i-1])/m
    for i in range(n-2, 0, -1):
        M[i] = d[i] - c[i]*M[i+1]
    return M


@njit(cache=True, inline='always', fastmath=False)
def spline_eval_val(y, M, h, u0, t):
    n = y.size
    x = (t-u0)/h
    i = int(math.floor(x))
    if i < 0: i = 0
    if i > n-2: i = n-2
    xi = u0 + i*h
    a = (xi+h-t)/h; b = (t-xi)/h
    return (a*y[i] + b*y[i+1]
            + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h)/6.0)


@njit(cache=True, inline='always', fastmath=False)
def spline_eval_pair(y, M, h, u0, t):
    """Returns (val, deriv) at t."""
    n = y.size
    x = (t-u0)/h
    i = int(math.floor(x))
    if i < 0: i = 0
    if i > n-2: i = n-2
    xi = u0 + i*h
    a = (xi+h-t)/h; b = (t-xi)/h
    val = (a*y[i] + b*y[i+1]
           + ((a*a*a - a)*M[i] + (b*b*b - b)*M[i+1]) * (h*h)/6.0)
    der = ((y[i+1]-y[i])/h
           - (3*a*a-1)/6*h*M[i]
           + (3*b*b-1)/6*h*M[i+1])
    return val, der


@njit(cache=True, fastmath=False)
def spline_roots_fill(y, M, h, u0, p_target, sub, out_roots, out_ders):
    """Fill out_roots[0:cnt] and out_ders[0:cnt] with roots of spline=p_target.
    Returns cnt. Max roots = out_roots.size."""
    n = y.size
    nseg = (n-1)*sub
    cnt = 0
    max_roots = out_roots.size
    t_prev = u0
    v_prev = spline_eval_val(y, M, h, u0, t_prev)
    step = (h*(n-1))/nseg
    for s in range(1, nseg+1):
        t_cur = u0 + s*step
        v_cur = spline_eval_val(y, M, h, u0, t_cur)
        dp = v_prev - p_target
        dc = v_cur - p_target
        if not (dp == 0.0 and dc == 0.0) and dp*dc <= 0:
            t = 0.5*(t_prev + t_cur)
            for _ in range(30):
                val, der = spline_eval_pair(y, M, h, u0, t)
                fval = val - p_target
                if abs(der) < 1e-14: break
                tn = t - fval/der
                if tn < t_prev: tn = t_prev
                if tn > t_cur: tn = t_cur
                if abs(tn - t) < 1e-13: break
                t = tn
            _, der = spline_eval_pair(y, M, h, u0, t)
            out_roots[cnt] = t
            out_ders[cnt] = der
            cnt += 1
            if cnt >= max_roots: return cnt
        t_prev = t_cur; v_prev = v_cur
    return cnt


@njit(cache=True, fastmath=False)
def precompute_Ms(P, axis, G, dxi):
    """Build M[i, j, k, n] = spline second-derivatives along the given axis.
    Returns array shape (G, G, n)."""
    pass


@njit(cache=True, fastmath=False)
def evidence_agent1_v10(P, i_u, p_target, xi_arr, dxi, xi_gl, w_gl, nq, TOT_S_, TOT_d_, vm0, vm1, coef, tau):
    """Agent 1 A_v at price p_target on slice P[i_u, :, :]. Tensor cubic spline + GL quadrature."""
    G = xi_arr.size
    xi0 = xi_arr[0]
    A0 = 0.0; A1 = 0.0
    # Precompute Ms along the b-axis (xi_d) for each row ka
    Mb_cache = np.empty((G, G))
    for ka in range(G):
        Mb_cache[ka] = natural_spline_M(P[i_u, ka, :], dxi)
    Ma_cache = np.empty((G, G))
    for kb in range(G):
        Ma_cache[kb] = natural_spline_M(P[i_u, :, kb], dxi)
    out_roots = np.empty(4)
    out_ders = np.empty(4)

    # Pass A: fix xi_d at GL nodes, root-find in xi_S
    for q in range(nq):
        xi_b = xi_gl[q]; w_b = w_gl[q]
        delta_b = TOT_d_ * math.atanh(xi_b)
        # Build P_line[ka] = spline-eval of P[i_u, ka, :] at xi_b
        P_line = np.empty(G)
        for ka in range(G):
            P_line[ka] = spline_eval_val(P[i_u, ka, :], Mb_cache[ka], dxi, xi0, xi_b)
        Ma = natural_spline_M(P_line, dxi)
        n_roots = spline_roots_fill(P_line, Ma, dxi, xi0, p_target, 8, out_roots, out_ders)
        for r in range(n_roots):
            xi_a_star = out_roots[r]; dP_dxi_a = out_ders[r]
            if abs(xi_a_star) >= 1 - 1e-12: continue
            Sigma_a = TOT_S_ * math.atanh(xi_a_star)
            dP_dSigma = dP_dxi_a * (1 - xi_a_star*xi_a_star) / TOT_S_
            if abs(dP_dSigma) < 1e-14: continue
            # Local dP/d_delta: vertical spline at xi_a_star, deriv at xi_b
            P_vert = np.empty(G)
            for kb in range(G):
                P_vert[kb] = spline_eval_val(P[i_u, :, kb], Ma_cache[kb], dxi, xi0, xi_a_star)
            Mvc = natural_spline_M(P_vert, dxi)
            _, dP_dxi_b_loc = spline_eval_pair(P_vert, Mvc, dxi, xi0, xi_b)
            dP_ddelta = dP_dxi_b_loc * (1 - xi_b*xi_b) / TOT_d_
            w_a = dP_dSigma*dP_dSigma / max(dP_dSigma*dP_dSigma + dP_ddelta*dP_ddelta, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal_(u_2, vm0, coef, tau) * f_signal_(u_3, vm0, coef, tau)
            f1 = f_signal_(u_2, vm1, coef, tau) * f_signal_(u_3, vm1, coef, tau)
            contrib = w_b * w_a / abs(dP_dSigma)
            A0 += contrib * f0
            A1 += contrib * f1
    # Pass B: fix xi_S at GL nodes, root-find in xi_d
    for q in range(nq):
        xi_a = xi_gl[q]; w_a_GL = w_gl[q]
        Sigma_a = TOT_S_ * math.atanh(xi_a)
        P_line = np.empty(G)
        for kb in range(G):
            P_line[kb] = spline_eval_val(P[i_u, :, kb], Ma_cache[kb], dxi, xi0, xi_a)
        Mb = natural_spline_M(P_line, dxi)
        n_roots = spline_roots_fill(P_line, Mb, dxi, xi0, p_target, 8, out_roots, out_ders)
        for r in range(n_roots):
            xi_b_star = out_roots[r]; dP_dxi_b = out_ders[r]
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d_ * math.atanh(xi_b_star)
            dP_ddelta = dP_dxi_b * (1 - xi_b_star*xi_b_star) / TOT_d_
            if abs(dP_ddelta) < 1e-14: continue
            P_horiz = np.empty(G)
            for ka in range(G):
                P_horiz[ka] = spline_eval_val(P[i_u, ka, :], Mb_cache[ka], dxi, xi0, xi_b_star)
            Mhc = natural_spline_M(P_horiz, dxi)
            _, dP_dxi_a_loc = spline_eval_pair(P_horiz, Mhc, dxi, xi0, xi_a)
            dP_dSigma_loc = dP_dxi_a_loc * (1 - xi_a*xi_a) / TOT_S_
            w_b = dP_ddelta*dP_ddelta / max(dP_dSigma_loc*dP_dSigma_loc + dP_ddelta*dP_ddelta, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b)
            u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal_(u_2, vm0, coef, tau) * f_signal_(u_3, vm0, coef, tau)
            f1 = f_signal_(u_2, vm1, coef, tau) * f_signal_(u_3, vm1, coef, tau)
            contrib = w_a_GL * w_b / abs(dP_ddelta)
            A0 += contrib * f0
            A1 += contrib * f1
    return 0.5*A0, 0.5*A1


@njit(cache=True, fastmath=False)
def evidence_agent_oblique_v10(P, p_target, xi_arr, dxi, xi_gl, w_gl, nq,
                                 u_cell, sign_for_other,
                                 TOT_u_, TOT_S_, TOT_d_, vm0, vm1, coef, tau):
    """Same GL-decoupled approach but slice extracted via tensor cubic Sigma-interp."""
    G = xi_arr.size; xi0 = xi_arr[0]
    A0 = 0.0; A1 = 0.0
    out_roots = np.empty(4); out_ders = np.empty(4)

    # Pass A: fix xi_d at GL nodes, root-find in xi_u
    for q in range(nq):
        xi_b = xi_gl[q]; w_b = w_gl[q]
        delta_b = TOT_d_ * math.atanh(xi_b)
        u_other = u_cell + sign_for_other * delta_b
        f0_oth = f_signal_(u_other, vm0, coef, tau)
        f1_oth = f_signal_(u_other, vm1, coef, tau)
        Sigma_req = 2*u_cell + sign_for_other * delta_b
        xi_t = math.tanh(Sigma_req / TOT_S_)
        if xi_t <= xi_arr[0] or xi_t >= xi_arr[-1]: continue
        # Build P_line[ka] = P_slc(xi_u[ka], xi_b) via tensor spline:
        #   for each ka, at each j build spline P[ka, j, :] in xi_d, eval at xi_b -> P_AS[j];
        #   then spline P_AS in xi_S, eval at xi_t -> P_line[ka].
        P_line = np.empty(G)
        for ka in range(G):
            P_AS = np.empty(G)
            for j in range(G):
                Md = natural_spline_M(P[ka, j, :], dxi)
                P_AS[j] = spline_eval_val(P[ka, j, :], Md, dxi, xi0, xi_b)
            Ms = natural_spline_M(P_AS, dxi)
            P_line[ka] = spline_eval_val(P_AS, Ms, dxi, xi0, xi_t)
        Mu = natural_spline_M(P_line, dxi)
        n_roots = spline_roots_fill(P_line, Mu, dxi, xi0, p_target, 8, out_roots, out_ders)
        for r in range(n_roots):
            xi_u_star = out_roots[r]; dP_dxi_u = out_ders[r]
            if abs(xi_u_star) >= 1 - 1e-12: continue
            u_1 = TOT_u_ * math.atanh(xi_u_star)
            dP_du1 = dP_dxi_u * (1 - xi_u_star*xi_u_star) / TOT_u_
            if abs(dP_du1) < 1e-14: continue
            f0_u1 = f_signal_(u_1, vm0, coef, tau); f1_u1 = f_signal_(u_1, vm1, coef, tau)
            # Local dP/d_delta via finite-diff: shift xi_b by eps
            xi_b_plus = xi_b + 1e-4
            if xi_b_plus >= 1: xi_b_plus = xi_b - 1e-4
            Sigma_req_p = 2*u_cell + sign_for_other * TOT_d_ * math.atanh(xi_b_plus)
            xi_t_p = math.tanh(Sigma_req_p / TOT_S_)
            if xi_t_p <= xi_arr[0]: xi_t_p = xi_arr[0] + 1e-9
            if xi_t_p >= xi_arr[-1]: xi_t_p = xi_arr[-1] - 1e-9
            # Approximate P_line_plus only at the ka near xi_u_star
            ka_near = int(round((xi_u_star - xi0)/dxi))
            if ka_near < 1: ka_near = 1
            if ka_near > G-2: ka_near = G-2
            P_AS2 = np.empty(G)
            for j in range(G):
                Md = natural_spline_M(P[ka_near, j, :], dxi)
                P_AS2[j] = spline_eval_val(P[ka_near, j, :], Md, dxi, xi0, xi_b_plus)
            Ms2 = natural_spline_M(P_AS2, dxi)
            P_line_plus_at_ka = spline_eval_val(P_AS2, Ms2, dxi, xi0, xi_t_p)
            dP_dxi_b_loc = (P_line_plus_at_ka - P_line[ka_near]) / (xi_b_plus - xi_b)
            dP_ddelta = dP_dxi_b_loc * (1 - xi_b*xi_b) / TOT_d_
            w_a = dP_du1*dP_du1 / max(dP_du1*dP_du1 + dP_ddelta*dP_ddelta, 1e-30)
            contrib = w_b * w_a / abs(dP_du1)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    # Pass B: fix xi_u at GL nodes, root-find in xi_d (loop over grid xi_d)
    for q in range(nq):
        xi_a = xi_gl[q]; w_a_GL = w_gl[q]
        u_1 = TOT_u_ * math.atanh(xi_a)
        f0_u1 = f_signal_(u_1, vm0, coef, tau); f1_u1 = f_signal_(u_1, vm1, coef, tau)
        # For each grid xi_d[kb], compute P_slc(xi_a, xi_d[kb]) via Sigma-interp
        P_line = np.empty(G)
        for kb in range(G):
            if abs(xi_arr[kb]) >= 1 - 1e-12:
                P_line[kb] = (1.0 if xi_arr[kb] > 0 else 0.0); continue
            delta_b = TOT_d_ * math.atanh(xi_arr[kb])
            Sigma_req = 2*u_cell + sign_for_other * delta_b
            xi_t = math.tanh(Sigma_req / TOT_S_)
            if xi_t <= xi_arr[0]: P_line[kb] = 0.0; continue
            if xi_t >= xi_arr[-1]: P_line[kb] = 1.0; continue
            P_AU = np.empty(G)
            for j in range(G):
                Mu = natural_spline_M(P[:, j, kb], dxi)
                P_AU[j] = spline_eval_val(P[:, j, kb], Mu, dxi, xi0, xi_a)
            Ms = natural_spline_M(P_AU, dxi)
            P_line[kb] = spline_eval_val(P_AU, Ms, dxi, xi0, xi_t)
        Md = natural_spline_M(P_line, dxi)
        n_roots = spline_roots_fill(P_line, Md, dxi, xi0, p_target, 8, out_roots, out_ders)
        for r in range(n_roots):
            xi_b_star = out_roots[r]; dP_dxi_b = out_ders[r]
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d_ * math.atanh(xi_b_star)
            u_other = u_cell + sign_for_other * delta_b
            f0_oth = f_signal_(u_other, vm0, coef, tau)
            f1_oth = f_signal_(u_other, vm1, coef, tau)
            dP_ddelta = dP_dxi_b * (1 - xi_b_star*xi_b_star) / TOT_d_
            if abs(dP_ddelta) < 1e-14: continue
            # Local dP/du_1 via finite-diff in xi_a
            xi_a_plus = xi_a + 1e-4
            if xi_a_plus >= 1: xi_a_plus = xi_a - 1e-4
            kb_near = int(round((xi_b_star - xi0)/dxi))
            if kb_near < 1: kb_near = 1
            if kb_near > G-2: kb_near = G-2
            if abs(xi_arr[kb_near]) >= 1 - 1e-12:
                continue
            delta_kb = TOT_d_ * math.atanh(xi_arr[kb_near])
            Sigma_req_p = 2*u_cell + sign_for_other * delta_kb
            xi_t_p = math.tanh(Sigma_req_p / TOT_S_)
            if xi_t_p <= xi_arr[0]: xi_t_p = xi_arr[0] + 1e-9
            if xi_t_p >= xi_arr[-1]: xi_t_p = xi_arr[-1] - 1e-9
            P_AU2 = np.empty(G)
            for j in range(G):
                Mu = natural_spline_M(P[:, j, kb_near], dxi)
                P_AU2[j] = spline_eval_val(P[:, j, kb_near], Mu, dxi, xi0, xi_a_plus)
            Ms2 = natural_spline_M(P_AU2, dxi)
            P_line_plus_at_kb = spline_eval_val(P_AU2, Ms2, dxi, xi0, xi_t_p)
            dP_dxi_a_loc = (P_line_plus_at_kb - P_line[kb_near]) / (xi_a_plus - xi_a)
            dP_du1_loc = dP_dxi_a_loc * (1 - xi_a*xi_a) / TOT_u_
            w_b = dP_ddelta*dP_ddelta / max(dP_du1_loc*dP_du1_loc + dP_ddelta*dP_ddelta, 1e-30)
            contrib = w_a_GL * w_b / abs(dP_ddelta)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    return A0, A1


@njit(cache=True, fastmath=False)
def crra_clear_nb(mu0, mu1, mu2, gamma, steps):
    eps = 1e-30
    a = eps; b = 1.0 - eps
    lm0 = math.log(mu0/(1-mu0))
    lm1 = math.log(mu1/(1-mu1))
    lm2 = math.log(mu2/(1-mu2))
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = math.log(m/(1-m))
        R0 = math.exp((lm0-lp)/gamma)
        R1 = math.exp((lm1-lp)/gamma)
        R2 = math.exp((lm2-lp)/gamma)
        e = (R0-1)/((1-m)+R0*m) + (R1-1)/((1-m)+R1*m) + (R2-1)/((1-m)+R2*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)


@njit(cache=True, parallel=True, fastmath=False)
def phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
             xi_gl, w_gl, nq, gamma, clearing_cara,
             TOT_u_, TOT_S_, TOT_d_, vm0, vm1, coef, tau, eps_p):
    G = xi_arr.size
    P_new = P.copy()
    for i in prange(INNER_LO, INNER_HI):
        if abs(xi_arr[i]) >= 1 - 1e-12: continue
        u1_cell = u_arr[i]
        f0_u1 = f_signal_(u1_cell, vm0, coef, tau)
        f1_u1 = f_signal_(u1_cell, vm1, coef, tau)
        for j in range(INNER_LO, INNER_HI):
            if abs(xi_arr[j]) >= 1 - 1e-12: continue
            Sigma_cell = S_arr[j]
            for k in range(INNER_LO, INNER_HI):
                if abs(xi_arr[k]) >= 1 - 1e-12: continue
                d_cell = d_arr[k]
                p_cell = P[i, j, k]
                u2_cell = 0.5*(Sigma_cell + d_cell)
                u3_cell = 0.5*(Sigma_cell - d_cell)
                f0_u2 = f_signal_(u2_cell, vm0, coef, tau)
                f1_u2 = f_signal_(u2_cell, vm1, coef, tau)
                f0_u3 = f_signal_(u3_cell, vm0, coef, tau)
                f1_u3 = f_signal_(u3_cell, vm1, coef, tau)
                A0a, A1a = evidence_agent1_v10(P, i, p_cell, xi_arr, dxi, xi_gl, w_gl, nq,
                                                 TOT_S_, TOT_d_, vm0, vm1, coef, tau)
                den = f0_u1*A0a + f1_u1*A1a
                mu0 = f1_u1*A1a/den if den > 0 else 0.5
                A0b, A1b = evidence_agent_oblique_v10(P, p_cell, xi_arr, dxi, xi_gl, w_gl, nq,
                                                       u2_cell, -1.0, TOT_u_, TOT_S_, TOT_d_,
                                                       vm0, vm1, coef, tau)
                den = f0_u2*A0b + f1_u2*A1b
                mu1 = f1_u2*A1b/den if den > 0 else 0.5
                A0c, A1c = evidence_agent_oblique_v10(P, p_cell, xi_arr, dxi, xi_gl, w_gl, nq,
                                                       u3_cell, +1.0, TOT_u_, TOT_S_, TOT_d_,
                                                       vm0, vm1, coef, tau)
                den = f0_u3*A0c + f1_u3*A1c
                mu2 = f1_u3*A1c/den if den > 0 else 0.5
                if mu0 < eps_p: mu0 = eps_p
                if mu0 > 1-eps_p: mu0 = 1-eps_p
                if mu1 < eps_p: mu1 = eps_p
                if mu1 > 1-eps_p: mu1 = 1-eps_p
                if mu2 < eps_p: mu2 = eps_p
                if mu2 > 1-eps_p: mu2 = 1-eps_p
                if clearing_cara:
                    pi_ = (math.log(mu0/(1-mu0)) + math.log(mu1/(1-mu1)) + math.log(mu2/(1-mu2))) / 3.0
                    P_new[i, j, k] = 1.0/(1.0+math.exp(-pi_))
                else:
                    P_new[i, j, k] = crra_clear_nb(mu0, mu1, mu2, gamma, 120)
    return P_new


def set_boundary(P):
    G = P.shape[0]
    P[0, :, :] = 0.0; P[G-1, :, :] = 1.0
    P[:, 0, :] = 0.0; P[:, G-1, :] = 1.0
    P[:, :, 0] = P[:, :, 1]; P[:, :, G-1] = P[:, :, G-2]
    return P
