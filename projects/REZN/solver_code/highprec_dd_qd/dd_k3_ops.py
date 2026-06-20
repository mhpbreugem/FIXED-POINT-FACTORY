"""Double-double K=3 CRRA Lin-CDF Richardson kernel-band operator.

Port of /tmp/cheby_h0/lin_cdf_richardson.py + lin_cdf_kern_tab.py to DD.

Operator (one Phi call):
  Phase A (Richardson): for each h in hs, build a mu^h(p, u_k) DD table
    via the kernel-band integral:
        A_v^h(p, u_k) = int int K_h(P-p) f_v(u_a) f_v(u_b) du_a du_b
        mu^h(p, u_k) = f_1(u_k) A_1 / (f_0(u_k) A_0 + f_1(u_k) A_1)
    then combine: mu = sum_h w_h mu^h (cancels O(h^2) thru O(h^{2*(n-1)})).
  Phase B (cube clearing): for each (i,j,k) in G^3,
    p_cell = P[i,j,k]; mu_a = interp(mu, p_cell, i); ... ; CRRA clear in DD.

All inner arithmetic in DD primitives from dd_ops. Float64 grids; DD math.
"""
import sys
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
import numpy as np
from numba import njit, prange
import dd_ops as D
from dd_ops import (dd_add, dd_mul, dd_div, dd_exp, dd_log, dd_sqrt, PIH, PIL)


# ============================ DD f_signal, CRRA ============================

@njit(cache=True, inline="always")
def dd_f_signal(uh, ul, v, th, tl):
    """sqrt(tau/(2 pi)) * exp(-tau/2 * (u-vm)^2)  where vm=0.5 if v else -0.5."""
    vm = 0.5 if v == 1 else -0.5
    # (u - vm)
    dh, dl = dd_add(uh, ul, -vm, 0.0)
    # (u-vm)^2
    d2h, d2l = dd_mul(dh, dl, dh, dl)
    # -tau/2 * d2
    htau_h, htau_l = dd_mul(-0.5, 0.0, th, tl)
    eh, el = dd_mul(htau_h, htau_l, d2h, d2l)
    expv_h, expv_l = dd_exp(eh, el)
    # coef = sqrt(tau / (2 pi))
    tp_h, tp_l = dd_mul(2.0, 0.0, PIH, PIL)
    rh, rl = dd_div(th, tl, tp_h, tp_l)
    ch, cl = dd_sqrt(rh, rl)
    return dd_mul(ch, cl, expv_h, expv_l)


@njit(cache=True)
def dd_crra_clear(m0h, m0l, m1h, m1l, m2h, m2l, gh, gl, steps=120):
    """CRRA bisection on log-odds, all DD. Inputs already clipped to (eps, 1-eps).
    Solves sum_k (R_k - 1) / ((1-m) + R_k m) = 0  with R_k = exp((lm_k - lp)/gamma).
    Returns (mh, ml).
    """
    # log-odds of mu_k
    onemh, onemh_l = dd_add(1.0, 0.0, -m0h, -m0l); q_h, q_l = dd_div(m0h, m0l, onemh, onemh_l)
    lm0h, lm0l = dd_log(q_h, q_l)
    onemh, onemh_l = dd_add(1.0, 0.0, -m1h, -m1l); q_h, q_l = dd_div(m1h, m1l, onemh, onemh_l)
    lm1h, lm1l = dd_log(q_h, q_l)
    onemh, onemh_l = dd_add(1.0, 0.0, -m2h, -m2l); q_h, q_l = dd_div(m2h, m2l, onemh, onemh_l)
    lm2h, lm2l = dd_log(q_h, q_l)
    # bracket on p in (0, 1)
    ah = 1e-30; al = 0.0; bh = 1.0 - 1e-30; bl = 0.0
    for _ in range(steps):
        # m = 0.5*(a+b)
        sh, sl = dd_add(ah, al, bh, bl)
        mh, ml = dd_mul(sh, sl, 0.5, 0.0)
        # lp = log(m/(1-m))
        oh, ol = dd_add(1.0, 0.0, -mh, -ml); qh, ql = dd_div(mh, ml, oh, ol)
        lph, lpl = dd_log(qh, ql)
        # e = sum_k (R_k - 1) / ((1-m) + R_k m)
        eh = 0.0; el = 0.0
        for k in range(3):
            if k == 0: lmkh, lmkl = lm0h, lm0l
            elif k == 1: lmkh, lmkl = lm1h, lm1l
            else: lmkh, lmkl = lm2h, lm2l
            arh, arl = dd_add(lmkh, lmkl, -lph, -lpl)
            arh, arl = dd_div(arh, arl, gh, gl)
            Rh, Rl = dd_exp(arh, arl)
            numh, numl = dd_add(Rh, Rl, -1.0, 0.0)
            Rmh, Rml = dd_mul(Rh, Rl, mh, ml)
            denh, denl = dd_add(oh, ol, Rmh, Rml)   # (1-m) + R*m
            th, tl = dd_div(numh, numl, denh, denl)
            eh, el = dd_add(eh, el, th, tl)
        if eh > 0.0:
            ah = mh; al = ml
        else:
            bh = mh; bl = ml
    sh, sl = dd_add(ah, al, bh, bl)
    return dd_mul(sh, sl, 0.5, 0.0)


# ============================ DD interp on slice2d ============================

@njit(cache=True, inline="always")
def _find_interval(u_grid, q, n):
    if q <= u_grid[0]: return 0
    if q >= u_grid[n-1]: return n-2
    lo = 0; hi = n-1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if u_grid[mid] <= q: lo = mid
        else: hi = mid
    return lo


@njit(cache=True)
def dd_linterp_slice_2d_at_nodes(u_grid, slice2H, slice2L, gl_u_nodes,
                                      n_grid, nqk, outH, outL):
    """Bilinear interp of a DD slice (slice2H, slice2L) onto NQK x NQK GL nodes.
    u_grid and gl_u_nodes are float64. weights w_a, w_b computed in float64 first
    (small effect) then propagated through DD adds/muls to keep DD precision in
    the table entries."""
    for q_a in range(nqk):
        u_a = gl_u_nodes[q_a]
        if u_a <= u_grid[0]:
            ia = 0; w_a = 0.0
        elif u_a >= u_grid[n_grid-1]:
            ia = n_grid-2; w_a = 1.0
        else:
            ia = _find_interval(u_grid, u_a, n_grid)
            w_a = (u_a - u_grid[ia]) / (u_grid[ia+1] - u_grid[ia])
        omw_a = 1.0 - w_a
        for q_b in range(nqk):
            u_b = gl_u_nodes[q_b]
            if u_b <= u_grid[0]:
                ib = 0; w_b = 0.0
            elif u_b >= u_grid[n_grid-1]:
                ib = n_grid-2; w_b = 1.0
            else:
                ib = _find_interval(u_grid, u_b, n_grid)
                w_b = (u_b - u_grid[ib]) / (u_grid[ib+1] - u_grid[ib])
            omw_b = 1.0 - w_b
            # Bilinear in DD: (1-w_a)(1-w_b)*S00 + w_a(1-w_b)*S10 + (1-w_a)w_b*S01 + w_a*w_b*S11
            w0h, w0l = dd_mul(omw_a, 0.0, omw_b, 0.0)
            w1h, w1l = dd_mul(w_a,   0.0, omw_b, 0.0)
            w2h, w2l = dd_mul(omw_a, 0.0, w_b,   0.0)
            w3h, w3l = dd_mul(w_a,   0.0, w_b,   0.0)
            t0h, t0l = dd_mul(w0h, w0l, slice2H[ia,   ib  ], slice2L[ia,   ib  ])
            t1h, t1l = dd_mul(w1h, w1l, slice2H[ia+1, ib  ], slice2L[ia+1, ib  ])
            t2h, t2l = dd_mul(w2h, w2l, slice2H[ia,   ib+1], slice2L[ia,   ib+1])
            t3h, t3l = dd_mul(w3h, w3l, slice2H[ia+1, ib+1], slice2L[ia+1, ib+1])
            sh, sl = dd_add(t0h, t0l, t1h, t1l)
            sh, sl = dd_add(sh, sl, t2h, t2l)
            sh, sl = dd_add(sh, sl, t3h, t3l)
            outH[q_a, q_b] = sh; outL[q_a, q_b] = sl


@njit(cache=True, inline="always")
def dd_interp_mu_p(muH, muL, p_grid, ph, pl, k_idx, G_p):
    """1D linear interp of mu_table along p axis at column k. Returns DD."""
    # find interval using hi part
    if ph <= p_grid[0]:
        return muH[0, k_idx], muL[0, k_idx]
    if ph >= p_grid[G_p-1]:
        return muH[G_p-1, k_idx], muL[G_p-1, k_idx]
    lo = 0; hi = G_p-1
    while hi - lo > 1:
        mid = (lo + hi)//2
        if p_grid[mid] <= ph: lo = mid
        else: hi = mid
    # w = (p - p[lo]) / (p[hi] - p[lo])    in DD for precision
    plh_h, plh_l = dd_add(ph, pl, -p_grid[lo], 0.0)
    den = p_grid[hi] - p_grid[lo]   # float64 is fine (the grid is exact)
    wh, wl = dd_div(plh_h, plh_l, den, 0.0)
    omw_h, omw_l = dd_add(1.0, 0.0, -wh, -wl)
    a_h, a_l = dd_mul(omw_h, omw_l, muH[lo, k_idx], muL[lo, k_idx])
    b_h, b_l = dd_mul(wh, wl,         muH[hi, k_idx], muL[hi, k_idx])
    return dd_add(a_h, a_l, b_h, b_l)


# ============================ DD mu table builder ============================

@njit(cache=True, parallel=True)
def dd_build_mu_table_lin_kern(P_H, P_L, u_grid, p_grid, gl_u_nodes,
                                gl_du_weights, th, tl, n_grid, nqk,
                                kernel_h, muH, muL):
    """Builds mu^h(p, u_k) table in DD for kernel bandwidth kernel_h.
    NOTE: parallelized over k_node (== axis 0 of the cube slice)."""
    G = n_grid; G_p = p_grid.size
    # inv_2h2 = 1/(2 h^2) in DD (small but matters as multiplier on dist^2)
    h2 = kernel_h * kernel_h
    inv2h2 = 0.5 / h2
    # pre-compute f_v(u_q) for q in nqk, in DD
    f0H = np.empty(nqk); f0L = np.empty(nqk)
    f1H = np.empty(nqk); f1L = np.empty(nqk)
    for q in range(nqk):
        f0H[q], f0L[q] = dd_f_signal(gl_u_nodes[q], 0.0, 0, th, tl)
        f1H[q], f1L[q] = dd_f_signal(gl_u_nodes[q], 0.0, 1, th, tl)
    for k_node in prange(G):
        u_k = u_grid[k_node]
        f0kH, f0kL = dd_f_signal(u_k, 0.0, 0, th, tl)
        f1kH, f1kL = dd_f_signal(u_k, 0.0, 1, th, tl)
        # extract 2D slice P_vals[k_node, :, :]   (DD)
        sH = np.empty((G, G)); sL = np.empty((G, G))
        for jj in range(G):
            for kk in range(G):
                sH[jj, kk] = P_H[k_node, jj, kk]; sL[jj, kk] = P_L[k_node, jj, kk]
        # interpolate slice onto GL tensor nodes
        PaH = np.empty((nqk, nqk)); PaL = np.empty((nqk, nqk))
        dd_linterp_slice_2d_at_nodes(u_grid, sH, sL, gl_u_nodes, G, nqk, PaH, PaL)
        for ip in range(G_p):
            p_ = p_grid[ip]    # float64 grid -> exact
            A0h = 0.0; A0l = 0.0; A1h = 0.0; A1l = 0.0
            for q_a in range(nqk):
                wa = gl_du_weights[q_a]
                f0ah = f0H[q_a]; f0al = f0L[q_a]
                f1ah = f1H[q_a]; f1al = f1L[q_a]
                for q_b in range(nqk):
                    wb = gl_du_weights[q_b]
                    # diff = P_at_nodes - p   (DD - exact)
                    diffH, diffL = dd_add(PaH[q_a, q_b], PaL[q_a, q_b], -p_, 0.0)
                    d2H, d2L = dd_mul(diffH, diffL, diffH, diffL)
                    # exp_arg = -d2 * inv2h2
                    ah_, al_ = dd_mul(-inv2h2, 0.0, d2H, d2L)
                    wH, wL = dd_exp(ah_, al_)
                    # contribution: w_a*w_b * w * f_v(u_a) * f_v(u_b)
                    waw = wa * wb   # float64 OK
                    # f_v(u_a)*f_v(u_b) in DD
                    p00H, p00L = dd_mul(f0ah, f0al, f0H[q_b], f0L[q_b])
                    p11H, p11L = dd_mul(f1ah, f1al, f1H[q_b], f1L[q_b])
                    # * wH * waw
                    wwH, wwL = dd_mul(waw, 0.0, wH, wL)
                    c0H, c0L = dd_mul(wwH, wwL, p00H, p00L)
                    c1H, c1L = dd_mul(wwH, wwL, p11H, p11L)
                    A0h, A0l = dd_add(A0h, A0l, c0H, c0L)
                    A1h, A1l = dd_add(A1h, A1l, c1H, c1L)
            # mu = f1k * A1 / (f0k * A0 + f1k * A1)
            t0H, t0L = dd_mul(f0kH, f0kL, A0h, A0l)
            t1H, t1L = dd_mul(f1kH, f1kL, A1h, A1l)
            denH, denL = dd_add(t0H, t0L, t1H, t1L)
            if denH > 1e-300:
                mh, ml = dd_div(t1H, t1L, denH, denL)
                muH[ip, k_node] = mh; muL[ip, k_node] = ml
            else:
                muH[ip, k_node] = 0.5; muL[ip, k_node] = 0.0


# ============================ DD Phase B (cube clearing) ============================

@njit(cache=True, parallel=True)
def dd_phi_cube_clear(P_H, P_L, muH, muL, p_grid, gh, gl, G, G_p,
                          PnH, PnL):
    """For each cube cell (i,j,k): clip p_cell, interp mu0/mu1/mu2 at p_cell on the
    respective axes, CRRA-clear in DD."""
    eps = 1e-9
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                # clip p_cell in (eps, 1-eps), keeping DD
                ph = P_H[i, j, k]; pl = P_L[i, j, k]
                if ph < eps: ph = eps; pl = 0.0
                elif ph > 1.0-eps: ph = 1.0-eps; pl = 0.0
                m0H, m0L = dd_interp_mu_p(muH, muL, p_grid, ph, pl, i, G_p)
                m1H, m1L = dd_interp_mu_p(muH, muL, p_grid, ph, pl, j, G_p)
                m2H, m2L = dd_interp_mu_p(muH, muL, p_grid, ph, pl, k, G_p)
                # clip mu to (eps, 1-eps) in DD
                if m0H < eps: m0H = eps; m0L = 0.0
                elif m0H > 1.0-eps: m0H = 1.0-eps; m0L = 0.0
                if m1H < eps: m1H = eps; m1L = 0.0
                elif m1H > 1.0-eps: m1H = 1.0-eps; m1L = 0.0
                if m2H < eps: m2H = eps; m2L = 0.0
                elif m2H > 1.0-eps: m2H = 1.0-eps; m2L = 0.0
                PnH[i, j, k], PnL[i, j, k] = dd_crra_clear(m0H, m0L, m1H, m1L,
                                                                m2H, m2L, gh, gl)


# ============================ Richardson driver ============================

def richardson_weights(hs):
    n = len(hs); H = np.array(hs, dtype=float)
    V = np.array([[H[i]**(2*k) for k in range(n)] for i in range(n)])
    e0 = np.zeros(n); e0[0] = 1.0
    return np.linalg.solve(V.T, e0)


def phi_dd(P_H, P_L, u_grid, p_grid, gl_u_nodes, gl_du_weights,
           th, tl, gh, gl, hs, weights):
    """One DD Richardson Phi call: returns (P_new_H, P_new_L)."""
    G = u_grid.size; G_p = p_grid.size
    muH = np.zeros((G_p, G)); muL = np.zeros((G_p, G))
    mu_h_H = np.empty((G_p, G)); mu_h_L = np.empty((G_p, G))
    for hi, h in enumerate(hs):
        mu_h_H[:] = 0.0; mu_h_L[:] = 0.0
        dd_build_mu_table_lin_kern(P_H, P_L, u_grid, p_grid, gl_u_nodes,
                                       gl_du_weights, th, tl, G,
                                       gl_u_nodes.size, float(h), mu_h_H, mu_h_L)
        w = float(weights[hi])
        for ip in range(G_p):
            for k in range(G):
                tH, tL = D.dd_mul(w, 0.0, mu_h_H[ip, k], mu_h_L[ip, k])
                aH, aL = D.dd_add(muH[ip, k], muL[ip, k], tH, tL)
                muH[ip, k] = aH; muL[ip, k] = aL
    PnH = np.empty_like(P_H); PnL = np.empty_like(P_L)
    dd_phi_cube_clear(P_H, P_L, muH, muL, p_grid, gh, gl, G, G_p, PnH, PnL)
    return PnH, PnL
