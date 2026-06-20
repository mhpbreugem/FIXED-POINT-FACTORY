"""Audit V11 oblique agent (agent 2 & 3) — the one that uses a crude
finite-difference for dP/dδ instead of the analytic spline derivative.

Build a SciPy-reference oblique evidence with analytic derivatives along
the oblique slice, compare on the worst cell.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import brentq
from v11_strict_h0_hardwired import (evidence_agent_oblique_v10, phi_v10,
    f_signal_, XI_GL, W_GL, NQ, EPSB,
    TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

P_sd = np.load(os.path.join(HERE, 'fixed_mapping_P_sd.npy'))
G = P_sd.shape[0]
xi_arr = np.linspace(-1.0, 1.0, G); dxi = float(xi_arr[1]-xi_arr[0])
INNER_LO, INNER_HI = 1, G - 1
u_arr = TOT_u * np.arctanh(np.clip(xi_arr, -0.9999999, 0.9999999))
S_arr = TOT_S * np.arctanh(np.clip(xi_arr, -0.9999999, 0.9999999))
d_arr = TOT_d * np.arctanh(np.clip(xi_arr, -0.9999999, 0.9999999))

# Worst cell
i, j, k = 1, 4, 3
u1_c = u_arr[i]; S_c = S_arr[j]; d_c = d_arr[k]
u2_c = 0.5*(S_c + d_c); u3_c = 0.5*(S_c - d_c)
p_t = float(P_sd[i,j,k])
print(f'Cell ({i},{j},{k})  u=({u1_c:+.3f},{u2_c:+.3f},{u3_c:+.3f})  p_target={p_t:.6f}')

# JIT warmup
print('JIT warmup...', flush=True); t=time.time()
_ = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  done {time.time()-t:.0f}s', flush=True)

# V11 oblique for agent 2 (sign_for_other=-1, u_cell=u_2)
A0_v11_2, A1_v11_2 = evidence_agent_oblique_v10(P_sd, p_t, xi_arr, dxi, XI_GL, W_GL, NQ,
                                                  u2_c, -1.0, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU)
print(f'\nAgent 2 (u_cell={u2_c:+.3f}, sign=-1):')
print(f'  V11 A_v: A0={A0_v11_2:.6e}  A1={A1_v11_2:.6e}')

# Reference using analytic spline derivatives
def ref_oblique_Av(P, p_target, xi_arr, dxi, xi_gl, w_gl, nq, u_cell, sign_for_other):
    """Reference oblique evidence using scipy cubic splines with ANALYTIC
    derivatives instead of finite-difference for dP/dδ."""
    G = xi_arr.size

    def P_slc(xi_u_val, xi_d_val):
        """Evaluate P at oblique point: (ξ_u, ξ_d) → ξ_Σ = tanh((2u_cell + sign*δ)/TOT_S)."""
        delta_b = TOT_d * np.arctanh(np.clip(xi_d_val, -0.9999999, 0.9999999))
        Sigma_req = 2*u_cell + sign_for_other * delta_b
        xi_t = np.tanh(Sigma_req / TOT_S)
        if xi_t <= xi_arr[0]: return 0.0, 0.0, 0.0  # (val, dP/dxi_u, dP/dxi_d)
        if xi_t >= xi_arr[-1]: return 1.0, 0.0, 0.0
        # Use tensor evaluation; for each j (Σ), build δ-spline at fixed (ξ_u, j),
        # then evaluate at ξ_d → P_AS[j], then Σ-spline of P_AS, eval at xi_t.
        # For derivatives, we need:
        #   dP/dxi_u: vary ξ_u, keep (xi_t, ξ_d) fixed → spline along ξ_u
        #   dP/dxi_d: vary ξ_d but ALSO ξ_t changes since Σ_req = 2u_cell + sign*δ_b
        # Actually for the oblique evidence, what we want is dP_slc/d(ξ_u) at fixed ξ_d,
        # and dP_slc/d(ξ_d) at fixed ξ_u (along the slice).
        # Build the full P_slc evaluation:
        ka_grid = np.arange(G)
        # For each ka, build δ-spline of P[ka, :, :] eval at (xi_t, xi_d):
        # Approach: build P_AS[ka, j] = δ-spline of P[ka, j, :] eval at xi_d, then
        # Σ-spline of P_AS[ka, :] at xi_t = P_slice[ka]. Then ξ_u-spline of P_slice
        # at xi_u_val gives val, deriv wrt ξ_u.
        P_slice = np.zeros(G)
        P_slice_p_d = np.zeros(G)
        for ka in range(G):
            P_AS = np.zeros(G)
            P_AS_p_d = np.zeros(G)
            for jj in range(G):
                cs_d = CubicSpline(xi_arr, P[ka, jj, :], bc_type='natural')
                P_AS[jj] = float(cs_d(xi_d_val))
                P_AS_p_d[jj] = float(cs_d(xi_d_val, 1))
            cs_s = CubicSpline(xi_arr, P_AS, bc_type='natural')
            cs_s_pd = CubicSpline(xi_arr, P_AS_p_d, bc_type='natural')
            P_slice[ka] = float(cs_s(xi_t))
            P_slice_p_d[ka] = float(cs_s_pd(xi_t))
        # ξ_u spline of P_slice
        cs_u = CubicSpline(xi_arr, P_slice, bc_type='natural')
        val = float(cs_u(xi_u_val))
        dval_dxi_u = float(cs_u(xi_u_val, 1))
        # ξ_d derivative: combination of two effects. ∂P/∂ξ_d|_{ξ_u} =
        # (∂/∂ξ_d) of (Σ-spline(P_AS(ξ_d), ξ_t(ξ_d))) at fixed ξ_u
        # = ∂P_slice/∂ξ_d + ∂P_slice/∂ξ_t · dξ_t/dξ_d
        # The first term: from spline_eval(P_AS(ξ_d), ξ_t) the ∂/∂ξ_d = spline eval of ∂P_AS/∂ξ_d at ξ_t
        cs_u_pd = CubicSpline(xi_arr, P_slice_p_d, bc_type='natural')
        dP_dxi_d_part1 = float(cs_u_pd(xi_u_val))
        # The second term: dξ_t/dξ_d
        sech2 = 1 - xi_t**2
        d_sigma_d_xi_d = sign_for_other * TOT_d / (1 - xi_d_val**2)
        d_xi_t_d_xi_d = (sech2/TOT_S) * d_sigma_d_xi_d
        # dP_slice/dxi_t at given xi_u_val: build P_slice_p_t[ka] = cs_s deriv at xi_t
        P_slice_p_t = np.zeros(G)
        for ka in range(G):
            P_AS = np.zeros(G)
            for jj in range(G):
                cs_d = CubicSpline(xi_arr, P[ka, jj, :], bc_type='natural')
                P_AS[jj] = float(cs_d(xi_d_val))
            cs_s = CubicSpline(xi_arr, P_AS, bc_type='natural')
            P_slice_p_t[ka] = float(cs_s(xi_t, 1))
        cs_u_pt = CubicSpline(xi_arr, P_slice_p_t, bc_type='natural')
        dP_dxi_t_at_u = float(cs_u_pt(xi_u_val))
        dP_dxi_d_total = dP_dxi_d_part1 + dP_dxi_t_at_u * d_xi_t_d_xi_d
        return val, dval_dxi_u, dP_dxi_d_total

    A0 = 0.0; A1 = 0.0

    # Pass A: fix ξ_d at GL, root-find in ξ_u
    for q in range(nq):
        xi_b = xi_gl[q]; w_b = w_gl[q]
        delta_b = TOT_d * np.arctanh(xi_b)
        u_other = u_cell + sign_for_other * delta_b
        f0_oth = f_signal_(u_other, VM0, COEF, TAU)
        f1_oth = f_signal_(u_other, VM1, COEF, TAU)
        # Build P_line[ka] across ka
        P_line = np.zeros(G)
        for ka in range(G):
            v, _, _ = P_slc(xi_arr[ka], xi_b)
            P_line[ka] = v
        cs_line = CubicSpline(xi_arr, P_line, bc_type='natural')
        ts_d = np.linspace(xi_arr[0], xi_arr[-1], 500)
        sgn = np.sign(cs_line(ts_d) - p_target)
        roots = []
        for idx in range(len(sgn)-1):
            if sgn[idx] != sgn[idx+1] and sgn[idx] != 0:
                try:
                    roots.append(brentq(lambda x: float(cs_line(x)) - p_target,
                                        ts_d[idx], ts_d[idx+1], xtol=1e-12))
                except ValueError: pass
        for xi_u_star in roots:
            if abs(xi_u_star) >= 1 - 1e-12: continue
            u_1 = TOT_u * np.arctanh(xi_u_star)
            # ANALYTIC dP/dξ_u and dP/dξ_d at the contour point
            _, dP_dxi_u, dP_dxi_d = P_slc(xi_u_star, xi_b)
            dP_du1 = dP_dxi_u * (1 - xi_u_star**2) / TOT_u
            if abs(dP_du1) < 1e-14: continue
            dP_ddelta = dP_dxi_d * (1 - xi_b**2) / TOT_d
            f0_u1 = f_signal_(u_1, VM0, COEF, TAU); f1_u1 = f_signal_(u_1, VM1, COEF, TAU)
            w_a = dP_du1**2 / max(dP_du1**2 + dP_ddelta**2, 1e-30)
            contrib = w_b * w_a / abs(dP_du1)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    # Pass B: fix ξ_u at GL, root-find in ξ_d (iterate over grid)
    for q in range(nq):
        xi_a = xi_gl[q]; w_a_gl = w_gl[q]
        u_1 = TOT_u * np.arctanh(xi_a)
        f0_u1 = f_signal_(u_1, VM0, COEF, TAU); f1_u1 = f_signal_(u_1, VM1, COEF, TAU)
        P_line = np.zeros(G)
        for kb in range(G):
            if abs(xi_arr[kb]) >= 1 - 1e-12:
                P_line[kb] = 1.0 if xi_arr[kb] > 0 else 0.0; continue
            v, _, _ = P_slc(xi_a, xi_arr[kb])
            P_line[kb] = v
        cs_line = CubicSpline(xi_arr, P_line, bc_type='natural')
        ts_d = np.linspace(xi_arr[0], xi_arr[-1], 500)
        sgn = np.sign(cs_line(ts_d) - p_target)
        roots = []
        for idx in range(len(sgn)-1):
            if sgn[idx] != sgn[idx+1] and sgn[idx] != 0:
                try:
                    roots.append(brentq(lambda x: float(cs_line(x)) - p_target,
                                        ts_d[idx], ts_d[idx+1], xtol=1e-12))
                except ValueError: pass
        for xi_b_star in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * np.arctanh(xi_b_star)
            u_other = u_cell + sign_for_other * delta_b
            f0_oth = f_signal_(u_other, VM0, COEF, TAU)
            f1_oth = f_signal_(u_other, VM1, COEF, TAU)
            _, dP_dxi_u_loc, dP_dxi_d = P_slc(xi_a, xi_b_star)
            dP_du1_loc = dP_dxi_u_loc * (1 - xi_a**2) / TOT_u
            dP_ddelta = dP_dxi_d * (1 - xi_b_star**2) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            w_b = dP_ddelta**2 / max(dP_du1_loc**2 + dP_ddelta**2, 1e-30)
            contrib = w_a_gl * w_b / abs(dP_ddelta)
            A0 += contrib * f0_u1 * f0_oth
            A1 += contrib * f1_u1 * f1_oth
    return A0, A1

print('  Computing REF (analytic-deriv) oblique A_v... (slow)')
t0 = time.time()
A0_ref, A1_ref = ref_oblique_Av(P_sd, p_t, xi_arr, dxi, XI_GL, W_GL, NQ, u2_c, -1.0)
print(f'  done {time.time()-t0:.1f}s')
print(f'  REF A_v: A0={A0_ref:.6e}  A1={A1_ref:.6e}')
print(f'  rel err A0: {abs(A0_v11_2-A0_ref)/max(abs(A0_ref),1e-30):.3e}')
print(f'  rel err A1: {abs(A1_v11_2-A1_ref)/max(abs(A1_ref),1e-30):.3e}')
mu_v11 = (f_signal_(u2_c, VM1, COEF, TAU)*A1_v11_2 /
          (f_signal_(u2_c, VM0, COEF, TAU)*A0_v11_2 + f_signal_(u2_c, VM1, COEF, TAU)*A1_v11_2))
mu_ref = (f_signal_(u2_c, VM1, COEF, TAU)*A1_ref /
          (f_signal_(u2_c, VM0, COEF, TAU)*A0_ref + f_signal_(u2_c, VM1, COEF, TAU)*A1_ref))
print(f'  agent-2 posterior μ: V11={mu_v11:.6f}  REF={mu_ref:.6f}  diff={mu_ref-mu_v11:+.6f}')
