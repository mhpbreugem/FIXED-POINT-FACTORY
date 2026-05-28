# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython port of the strict h=0 K=4 contour-Phi operator.

Mirrors /tmp/rezn-source/code/contour_K4_halo.py (Numba) exactly:
  - _scan_axis_halo / _agent_evidence_halo : contour-crossing evidence
  - _bayes                                  : posterior update
  - clear_crra (bisection)                  : per-realisation market clearing
  - phi_K4_halo                             : full 4D Phi update of inner cells
  - init_no_learning_halo                   : no-learning IC over the padded grid

float64 typed memoryviews throughout. No smoothing (strict h=0).
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, sqrt, M_PI

cdef double EPS_PRICE = 1.0e-12

# ----------------------------------------------------------------------
# primitives (mirror signals.py / demand.py)
# ----------------------------------------------------------------------
cdef inline double c_lam(double z) nogil:
    cdef double e
    if z >= 0.0:
        e = exp(-z)
        return 1.0 / (1.0 + e)
    e = exp(z)
    return e / (1.0 + e)

cdef inline double c_logit(double p) nogil:
    return log(p) - log(1.0 - p)

cdef inline double c_f_signal(double u, int v, double tau) nogil:
    cdef double mean = 0.5 if v == 1 else -0.5
    cdef double d = u - mean
    return sqrt(tau / (2.0 * M_PI)) * exp(-0.5 * tau * d * d)

cdef inline double c_x_crra(double mu, double p, double gamma, double W) nogil:
    cdef double z = (c_logit(mu) - c_logit(p)) / gamma
    cdef double e
    if z >= 0.0:
        e = exp(-z)
        return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = exp(z)
    return W * (e - 1.0) / ((1.0 - p) + p * e)

cdef inline double c_excess_crra(double* mu_vec, double* gamma_vec,
                                 double* W_vec, double p, int K) nogil:
    cdef double s = 0.0
    cdef int k
    for k in range(K):
        s += c_x_crra(mu_vec[k], p, gamma_vec[k], W_vec[k])
    return s

cdef inline double c_clear_crra(double* mu_vec, double* gamma_vec,
                                double* W_vec, int K) nogil:
    cdef double a = EPS_PRICE
    cdef double b = 1.0 - EPS_PRICE
    cdef double fa = c_excess_crra(mu_vec, gamma_vec, W_vec, a, K)
    cdef double fb = c_excess_crra(mu_vec, gamma_vec, W_vec, b, K)
    cdef double c, fc
    cdef int it
    if fa <= 0.0:
        return a
    if fb >= 0.0:
        return b
    for it in range(60):
        c = 0.5 * (a + b)
        fc = c_excess_crra(mu_vec, gamma_vec, W_vec, c, K)
        if fc >= 0.0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        if (b - a) < 1.0e-14:
            break
    return 0.5 * (a + b)

cdef inline double c_bayes(double u_own, double tau_own, double A0, double A1) nogil:
    cdef double f0 = c_f_signal(u_own, 0, tau_own)
    cdef double f1 = c_f_signal(u_own, 1, tau_own)
    cdef double num = f1 * A1
    cdef double den = f0 * A0 + num
    cdef double mu
    if den <= 0.0:
        return 0.5
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > 1.0 - EPS_PRICE:
        return 1.0 - EPS_PRICE
    return mu

# ----------------------------------------------------------------------
# contour scan on one 3D slice (mirror _scan_axis_halo / _agent_evidence_halo)
# P_slice is the agent's (G_full,G_full,G_full) slice, contiguous copy.
# ----------------------------------------------------------------------
cdef void c_scan_axis(double[:, :, ::1] P_slice, double p_target, int axis,
                      int a_idx, int b_idx, double[::1] u_full,
                      double tau_a, double tau_b, double tau_off,
                      double* acc) nogil:
    cdef int G_full = u_full.shape[0]
    cdef double u_a = u_full[a_idx]
    cdef double u_b = u_full[b_idx]
    cdef double f0_a = c_f_signal(u_a, 0, tau_a)
    cdef double f1_a = c_f_signal(u_a, 1, tau_a)
    cdef double f0_b = c_f_signal(u_b, 0, tau_b)
    cdef double f1_b = c_f_signal(u_b, 1, tau_b)
    cdef double prev_v, next_v, d_prev, d_next, denom, frac, u_off, f0_off, f1_off
    cdef int i

    if axis == 0:
        prev_v = P_slice[0, a_idx, b_idx]
    elif axis == 1:
        prev_v = P_slice[a_idx, 0, b_idx]
    else:
        prev_v = P_slice[a_idx, b_idx, 0]

    for i in range(G_full - 1):
        if axis == 0:
            next_v = P_slice[i + 1, a_idx, b_idx]
        elif axis == 1:
            next_v = P_slice[a_idx, i + 1, b_idx]
        else:
            next_v = P_slice[a_idx, b_idx, i + 1]
        d_prev = prev_v - p_target
        d_next = next_v - p_target
        if d_prev == 0.0 and d_next == 0.0:
            prev_v = next_v
            continue
        if d_prev * d_next <= 0.0:
            denom = next_v - prev_v
            if denom == 0.0:
                prev_v = next_v
                continue
            frac = (p_target - prev_v) / denom
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            u_off = (1.0 - frac) * u_full[i] + frac * u_full[i + 1]
            f0_off = c_f_signal(u_off, 0, tau_off)
            f1_off = c_f_signal(u_off, 1, tau_off)
            acc[0] += f0_a * f0_b * f0_off
            acc[1] += f1_a * f1_b * f1_off
        prev_v = next_v

cdef void c_agent_evidence(double[:, :, ::1] P_slice, double p_target,
                           double[::1] u_full,
                           double tau_o0, double tau_o1, double tau_o2,
                           double* acc) nogil:
    cdef int G_full = u_full.shape[0]
    cdef double a0 = 0.0
    cdef double a1 = 0.0
    cdef int a_idx, b_idx

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            c_scan_axis(P_slice, p_target, 0, a_idx, b_idx, u_full,
                        tau_o1, tau_o2, tau_o0, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            c_scan_axis(P_slice, p_target, 1, a_idx, b_idx, u_full,
                        tau_o0, tau_o2, tau_o1, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        for b_idx in range(G_full):
            c_scan_axis(P_slice, p_target, 2, a_idx, b_idx, u_full,
                        tau_o0, tau_o1, tau_o2, acc)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = a0 / 3.0
    acc[1] = a1 / 3.0

# ----------------------------------------------------------------------
# public: phi_K4_halo
# ----------------------------------------------------------------------
def phi_K4_halo_cy(cnp.ndarray[cnp.float64_t, ndim=4] P_full_in,
                   cnp.ndarray[cnp.float64_t, ndim=1] u_full_in,
                   int inner_lo, int inner_hi,
                   cnp.ndarray[cnp.float64_t, ndim=1] tau_vec_in,
                   cnp.ndarray[cnp.float64_t, ndim=1] gamma_vec_in,
                   cnp.ndarray[cnp.float64_t, ndim=1] W_vec_in):
    cdef cnp.ndarray[cnp.float64_t, ndim=4] P_new_arr = P_full_in.copy()
    cdef double[:, :, :, ::1] P_full = np.ascontiguousarray(P_full_in)
    cdef double[:, :, :, ::1] P_new = P_new_arr
    cdef double[::1] u_full = u_full_in
    cdef double[::1] tau_vec = tau_vec_in
    cdef double[::1] gamma_vec = gamma_vec_in
    cdef double[::1] W_vec = W_vec_in
    cdef int G_full = u_full.shape[0]
    cdef int i, j, l, m
    cdef double p
    cdef double mu_vec[4]
    cdef double gv[4]
    cdef double wv[4]
    cdef double acc[2]
    cdef int k

    for k in range(4):
        gv[k] = gamma_vec[k]
        wv[k] = W_vec[k]

    # Per-agent contiguous slice buffers
    cdef cnp.ndarray[cnp.float64_t, ndim=3] s0 = np.empty((G_full, G_full, G_full))
    cdef cnp.ndarray[cnp.float64_t, ndim=3] s1 = np.empty((G_full, G_full, G_full))
    cdef cnp.ndarray[cnp.float64_t, ndim=3] s2 = np.empty((G_full, G_full, G_full))
    cdef cnp.ndarray[cnp.float64_t, ndim=3] s3 = np.empty((G_full, G_full, G_full))
    cdef double[:, :, ::1] sl0 = s0
    cdef double[:, :, ::1] sl1 = s1
    cdef double[:, :, ::1] sl2 = s2
    cdef double[:, :, ::1] sl3 = s3

    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                for m in range(inner_lo, inner_hi):
                    p = P_full[i, j, l, m]

                    # Agent 0: slice axes (1,2,3) at fixed i -> P_full[i,:,:,:]
                    _fill_slice0(P_full, i, sl0, G_full)
                    c_agent_evidence(sl0, p, u_full,
                                     tau_vec[1], tau_vec[2], tau_vec[3], acc)
                    mu_vec[0] = c_bayes(u_full[i], tau_vec[0], acc[0], acc[1])

                    # Agent 1: P_full[:,j,:,:]
                    _fill_slice1(P_full, j, sl1, G_full)
                    c_agent_evidence(sl1, p, u_full,
                                     tau_vec[0], tau_vec[2], tau_vec[3], acc)
                    mu_vec[1] = c_bayes(u_full[j], tau_vec[1], acc[0], acc[1])

                    # Agent 2: P_full[:,:,l,:]
                    _fill_slice2(P_full, l, sl2, G_full)
                    c_agent_evidence(sl2, p, u_full,
                                     tau_vec[0], tau_vec[1], tau_vec[3], acc)
                    mu_vec[2] = c_bayes(u_full[l], tau_vec[2], acc[0], acc[1])

                    # Agent 3: P_full[:,:,:,m]
                    _fill_slice3(P_full, m, sl3, G_full)
                    c_agent_evidence(sl3, p, u_full,
                                     tau_vec[0], tau_vec[1], tau_vec[2], acc)
                    mu_vec[3] = c_bayes(u_full[m], tau_vec[3], acc[0], acc[1])

                    P_new[i, j, l, m] = c_clear_crra(mu_vec, gv, wv, 4)
    return P_new_arr

# slice fillers: copy the 3D agent slice into a contiguous buffer so that
# axis-scans see a (G,G,G) C-contiguous array (mirrors Numba P_full[i,:,:,:]).
cdef inline void _fill_slice0(double[:, :, :, ::1] P, int i,
                              double[:, :, ::1] out, int G) nogil:
    cdef int a, b, c
    for a in range(G):
        for b in range(G):
            for c in range(G):
                out[a, b, c] = P[i, a, b, c]

cdef inline void _fill_slice1(double[:, :, :, ::1] P, int j,
                              double[:, :, ::1] out, int G) nogil:
    cdef int a, b, c
    for a in range(G):
        for b in range(G):
            for c in range(G):
                out[a, b, c] = P[a, j, b, c]

cdef inline void _fill_slice2(double[:, :, :, ::1] P, int l,
                              double[:, :, ::1] out, int G) nogil:
    cdef int a, b, c
    for a in range(G):
        for b in range(G):
            for c in range(G):
                out[a, b, c] = P[a, b, l, c]

cdef inline void _fill_slice3(double[:, :, :, ::1] P, int m,
                              double[:, :, ::1] out, int G) nogil:
    cdef int a, b, c
    for a in range(G):
        for b in range(G):
            for c in range(G):
                out[a, b, c] = P[a, b, c, m]

# ----------------------------------------------------------------------
# public: init_no_learning_halo
# ----------------------------------------------------------------------
def init_no_learning_halo_cy(cnp.ndarray[cnp.float64_t, ndim=1] u_full_in,
                             cnp.ndarray[cnp.float64_t, ndim=1] tau_vec_in,
                             cnp.ndarray[cnp.float64_t, ndim=1] gamma_vec_in,
                             cnp.ndarray[cnp.float64_t, ndim=1] W_vec_in):
    cdef double[::1] u_full = u_full_in
    cdef double[::1] tau_vec = tau_vec_in
    cdef double[::1] gamma_vec = gamma_vec_in
    cdef double[::1] W_vec = W_vec_in
    cdef int G_full = u_full.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=4] P_arr = np.empty(
        (G_full, G_full, G_full, G_full))
    cdef double[:, :, :, ::1] P = P_arr
    cdef int i, j, l, m, k
    cdef double m0, m1, m2, m3
    cdef double mu_vec[4]
    cdef double gv[4]
    cdef double wv[4]
    for k in range(4):
        gv[k] = gamma_vec[k]
        wv[k] = W_vec[k]

    for i in range(G_full):
        m0 = c_lam(tau_vec[0] * u_full[i])
        for j in range(G_full):
            m1 = c_lam(tau_vec[1] * u_full[j])
            for l in range(G_full):
                m2 = c_lam(tau_vec[2] * u_full[l])
                for m in range(G_full):
                    m3 = c_lam(tau_vec[3] * u_full[m])
                    mu_vec[0] = m0
                    mu_vec[1] = m1
                    mu_vec[2] = m2
                    mu_vec[3] = m3
                    P[i, j, l, m] = c_clear_crra(mu_vec, gv, wv, 4)
    return P_arr
