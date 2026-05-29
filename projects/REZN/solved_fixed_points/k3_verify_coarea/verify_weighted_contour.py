"""Co-area-WEIGHTED strict (h=0) contour scan for K=3 CRRA REE operator.

This is method (c) of TEST 1. It implements a variant of phi_K3_halo where
each contour crossing of the level set {P=p} is weighted by 1/|grad P| at the
crossing (the co-area Jacobian), turning the naive grid-edge-crossing sum into
a CONSISTENT co-area quadrature.

Geometry. For a 2-D slice P_slice[ia, ib] (ia = swept "off" axis index along
which we look for a crossing of p_target between i and i+1; ib = a_idx the
on-grid "a" axis), a crossing along the off-axis at fractional position frac
between rows i and i+1 of column a_idx contributes f_a(u_a) * f_off(u_off).

The naive scan implicitly assumes the contour element ds equals du along the
off-axis. The true co-area surface measure is
    ds = du_off / |cos theta|   along the crossed edge,
and the co-area integral integrand picks up 1/|grad P|.  In a uniform line
integral along the on-grid axis (a_idx steps by du, sum over a_idx), the
correct discretization of  A_v = int_{P=p} f_v / |grad P| d(transverse)
is to weight each crossing by  du_a / |grad P|  -- but du_a is the constant
on-grid spacing common to all crossings, so up to the global constant du it is
    weight = 1 / |grad P| ,
with grad P = (dP/du_off, dP/du_a) evaluated at the crossing.

  dP/du_off : the along-scan slope = (next_v - prev_v)/du   (the edge we cross)
  dP/du_a   : the transverse slope, estimated from neighboring scan lines
              a_idx-1 and a_idx+1 at the SAME off-position (interpolated).

The overall global du constant cancels in the Bayes ratio A1/A0 and (because it
multiplies BOTH passes equally) in the pass-average, so it does not affect the
posterior mu; we keep weight = 1/|grad P| for clarity.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
from numba import njit, prange
from code.signals import f_signal, lam
from code.demand import clear_crra, EPS_PRICE


@njit(cache=True, fastmath=False)
def _interp_at(P_slice, axis, line_idx, i, frac):
    """Value of P on a neighbouring scan line at the same off-position.

    axis: off-sweep axis (0 -> sweep rows; 1 -> sweep cols).
    line_idx: index of the on-grid axis for the neighbouring line.
    i, frac: crossing is between off-index i and i+1 at fraction frac.
    Returns NaN if line_idx out of range.
    """
    G = P_slice.shape[0]
    if line_idx < 0 or line_idx >= G:
        return np.nan
    if axis == 0:
        v0 = P_slice[i, line_idx]
        v1 = P_slice[i + 1, line_idx]
    else:
        v0 = P_slice[line_idx, i]
        v1 = P_slice[line_idx, i + 1]
    return (1.0 - frac) * v0 + frac * v1


@njit(cache=True, fastmath=False)
def _scan_axis_K3_w(P_slice, p_target, axis, a_idx, u_full,
                    tau_a, tau_off, du, acc):
    """As _scan_axis_K3 but each crossing weighted by 1/|grad P|."""
    G_full = u_full.size
    u_a = u_full[a_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)

    prev_v = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
    for i in range(G_full - 1):
        next_v = (P_slice[i + 1, a_idx] if axis == 0
                  else P_slice[a_idx, i + 1])
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
            f0_off = f_signal(u_off, 0, tau_off)
            f1_off = f_signal(u_off, 1, tau_off)

            # along-scan slope (off-axis): dP/du_off
            g_off = denom / du

            # transverse slope (on-grid a-axis): central diff of the
            # interpolated P value on neighbouring scan lines.
            v_p = _interp_at(P_slice, axis, a_idx + 1, i, frac)
            v_m = _interp_at(P_slice, axis, a_idx - 1, i, frac)
            if v_p == v_p and v_m == v_m:           # both finite
                g_tr = (v_p - v_m) / (2.0 * du)
            elif v_p == v_p:
                here = (1.0 - frac) * (P_slice[i, a_idx] if axis == 0
                                       else P_slice[a_idx, i]) + \
                       frac * (P_slice[i + 1, a_idx] if axis == 0
                               else P_slice[a_idx, i + 1])
                g_tr = (v_p - here) / du
            elif v_m == v_m:
                here = (1.0 - frac) * (P_slice[i, a_idx] if axis == 0
                                       else P_slice[a_idx, i]) + \
                       frac * (P_slice[i + 1, a_idx] if axis == 0
                               else P_slice[a_idx, i + 1])
                g_tr = (here - v_m) / du
            else:
                g_tr = 0.0

            gmag = np.sqrt(g_off * g_off + g_tr * g_tr)
            if gmag < 1.0e-12:
                gmag = 1.0e-12
            w = 1.0 / gmag
            acc[0] += w * f0_a * f0_off
            acc[1] += w * f1_a * f1_off
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_K3_w(P_slice, p_target, u_full, tau_o0, tau_o1, du, acc):
    G_full = u_full.size
    a0 = 0.0
    a1 = 0.0
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3_w(P_slice, p_target, 0, a_idx, u_full,
                        tau_o1, tau_o0, du, acc)
    a0 += acc[0]
    a1 += acc[1]
    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G_full):
        _scan_axis_K3_w(P_slice, p_target, 1, a_idx, u_full,
                        tau_o0, tau_o1, du, acc)
    a0 += acc[0]
    a1 += acc[1]
    acc[0] = a0 / 2.0
    acc[1] = a1 / 2.0


@njit(cache=True, fastmath=False, inline="always")
def _bayes(u_own, tau_own, A0, A1):
    f0 = f_signal(u_own, 0, tau_own)
    f1 = f_signal(u_own, 1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > 1.0 - EPS_PRICE:
        return 1.0 - EPS_PRICE
    return mu


@njit(cache=True, fastmath=False, parallel=True)
def phi_K3_halo_weighted(P_full, u_full, inner_lo, inner_hi,
                         tau_vec, gamma_vec, W_vec, du):
    """Co-area-weighted strict (h=0) contour scan operator."""
    P_new = P_full.copy()
    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]
                _agent_evidence_K3_w(P_full[i, :, :], p, u_full,
                                     tau_vec[1], tau_vec[2], du, acc)
                mu_vec[0] = _bayes(u_full[i], tau_vec[0], acc[0], acc[1])
                _agent_evidence_K3_w(P_full[:, j, :], p, u_full,
                                     tau_vec[0], tau_vec[2], du, acc)
                mu_vec[1] = _bayes(u_full[j], tau_vec[1], acc[0], acc[1])
                _agent_evidence_K3_w(P_full[:, :, l], p, u_full,
                                     tau_vec[0], tau_vec[1], du, acc)
                mu_vec[2] = _bayes(u_full[l], tau_vec[2], acc[0], acc[1])
                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new
