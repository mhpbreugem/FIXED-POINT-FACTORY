"""Step-by-step audit of V11 σ-δ cubic-spline + GL co-area construction.

For each construction stage, compare V11's @njit output to an independent
SciPy reference implementation. The cell we audit is the WORST cell from
verify_pointwise (i=1, j=4, k=3, residual = -0.308).

Stages tested:
  1. Natural cubic spline coefficients M[i] (y'' at knots)
     V11: natural_spline_M  vs  scipy.interpolate.CubicSpline(bc='natural')
  2. Spline value & derivative
     V11: spline_eval_val/pair  vs  scipy CubicSpline.__call__, .derivative()
  3. Root finding spline(xi) = p_target
     V11: spline_roots_fill  vs  scipy.optimize.brentq on dense grid
  4. Agent 1 A_v(p) via 16-point Gauss-Legendre + partition-of-unity
     V11: evidence_agent1_v10  vs  reference Python using scipy spline/quad
  5. Posterior μ from A_v and clearing
     V11: phi_v10 inner block  vs  reference

If any stage disagrees > tolerance, that's a bug.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from v11_strict_h0_hardwired import (
    natural_spline_M, spline_eval_val, spline_eval_pair, spline_roots_fill,
    evidence_agent1_v10, evidence_agent_oblique_v10, phi_v10,
    f_signal_, crra_clear_nb,
    XI_GL, W_GL, NQ, EPSB,
    TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

# Load interpolated u-grid FP
P_sd = np.load(os.path.join(HERE, 'fixed_mapping_P_sd.npy'))
G_FULL = P_sd.shape[0]
INNER_LO, INNER_HI = 1, G_FULL - 1
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1] - xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)
print(f'σ-δ cube G={G_FULL}, dxi={dxi:.6f}, EPSB={EPSB:.1e}, NQ={NQ}')

# JIT warmup
print('JIT warmup...', flush=True); t=time.time()
_ = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  done {time.time()-t:.0f}s', flush=True)

# ---------- Pick the worst cell ----------
i, j, k = 1, 4, 3
print(f'\n=== Auditing cell (i={i}, j={j}, k={k}) ===')
print(f'  ξ = ({xi_arr[i]:+.4f}, {xi_arr[j]:+.4f}, {xi_arr[k]:+.4f})')
print(f'  u = ({u_arr[i]:+.4f}, {u_arr[j]:+.4f}, {u_arr[k]:+.4f})')
print(f'  P_interp at this cell = {P_sd[i,j,k]:.6f}')

# ---------- Stage 1: Natural cubic spline coefficients ----------
print('\n--- STAGE 1: Natural cubic spline M[i] (y'' at knots) ---')
# Test on the line P[i, j, :] (δ-direction at fixed (u₁=ξ_i, Σ=ξ_j))
y = P_sd[i, j, :].copy()
M_v11 = natural_spline_M(y, dxi)
cs = CubicSpline(xi_arr, y, bc_type='natural')
# scipy gives second derivatives at knots via cs(x, 2)
M_scipy = cs(xi_arr, 2)
err_M = np.max(np.abs(M_v11 - M_scipy))
print(f'  V11 M[1..n-2]:    {np.round(M_v11[1:-1], 5)}')
print(f'  scipy M[1..n-2]:  {np.round(M_scipy[1:-1], 5)}')
print(f'  max |M_v11 - M_scipy| = {err_M:.3e}  {"OK" if err_M < 1e-10 else "FAIL"}')

# ---------- Stage 2: spline eval value + derivative ----------
print('\n--- STAGE 2: spline value & derivative ---')
test_ts = np.linspace(-0.95, 0.95, 25)
max_err_val = 0.0; max_err_der = 0.0
for t in test_ts:
    v_v11, d_v11 = spline_eval_pair(y, M_v11, dxi, xi_arr[0], t)
    v_sc = float(cs(t)); d_sc = float(cs(t, 1))
    max_err_val = max(max_err_val, abs(v_v11 - v_sc))
    max_err_der = max(max_err_der, abs(d_v11 - d_sc))
print(f'  max |val_v11 - val_scipy|  over 25 test points: {max_err_val:.3e}')
print(f'  max |der_v11 - der_scipy|  over 25 test points: {max_err_der:.3e}')
print(f'  {"OK" if max(max_err_val, max_err_der) < 1e-10 else "FAIL"}')

# Check for overshoot
overshoots_max = 0.0; overshoots_min = 0.0
ts_dense = np.linspace(xi_arr[0], xi_arr[-1], 1000)
vals_dense = cs(ts_dense)
overshoots_max = vals_dense.max() - max(y)
overshoots_min = min(y) - vals_dense.min()
print(f'  raw y range: [{y.min():.4f}, {y.max():.4f}]')
print(f'  spline overshoot UP:   {overshoots_max:+.4e}')
print(f'  spline overshoot DOWN: {overshoots_min:+.4e}')

# ---------- Stage 3: spline root finding ----------
print('\n--- STAGE 3: spline root finding spline(ξ) = p_target ---')
p_target = float(P_sd[i, j, k])
# V11
out_r = np.empty(4); out_d = np.empty(4)
n_r_v11 = spline_roots_fill(y, M_v11, dxi, xi_arr[0], p_target, 8, out_r, out_d)
roots_v11 = sorted(out_r[:n_r_v11].tolist())
# scipy reference: find sign-changes on dense grid then brentq
sign_dense = np.sign(cs(ts_dense) - p_target)
brackets = []
for idx in range(len(sign_dense)-1):
    if sign_dense[idx] != sign_dense[idx+1] and sign_dense[idx] != 0:
        brackets.append((ts_dense[idx], ts_dense[idx+1]))
roots_scipy = sorted(brentq(lambda x: float(cs(x)) - p_target, a, b, xtol=1e-12) for a, b in brackets)
print(f'  p_target = {p_target:.6f}')
print(f'  V11 roots ({n_r_v11}): {[f"{r:+.6f}" for r in roots_v11]}')
print(f'  scipy roots ({len(roots_scipy)}): {[f"{r:+.6f}" for r in roots_scipy]}')
if len(roots_v11) == len(roots_scipy):
    err_roots = max(abs(rv - rs) for rv, rs in zip(roots_v11, roots_scipy))
    print(f'  max |root error| = {err_roots:.3e}  {"OK" if err_roots < 1e-8 else "FAIL"}')
else:
    print(f'  ROOT COUNT MISMATCH (V11={n_r_v11} vs scipy={len(roots_scipy)})')

# ---------- Stage 4: Agent-1 evidence A_v(p) ----------
print('\n--- STAGE 4: Agent 1 A_v(p) via GL + partition-of-unity ---')
A0_v11, A1_v11 = evidence_agent1_v10(P_sd, i, p_target, xi_arr, dxi, XI_GL, W_GL, NQ,
                                       TOT_S, TOT_d, VM0, VM1, COEF, TAU)
print(f'  V11 A_v: A0={A0_v11:.6e}  A1={A1_v11:.6e}')

# Reference: pure-Python using scipy splines (i.e., the entire pipeline)
def ref_agent1_Av(P, i_u, p_target, xi_arr, dxi, xi_gl, w_gl, nq):
    G = xi_arr.size
    # Build splines along δ for each ka (Σ-row), and along Σ for each kb (δ-col)
    cs_d_arr = [CubicSpline(xi_arr, P[i_u, ka, :], bc_type='natural') for ka in range(G)]
    cs_s_arr = [CubicSpline(xi_arr, P[i_u, :, kb], bc_type='natural') for kb in range(G)]
    A0 = 0.0; A1 = 0.0
    half_count = 0  # will multiply by 0.5 at end to match V11 convention
    # Pass A: fix δ at GL, root-find in Σ
    for q in range(nq):
        xi_b = xi_gl[q]; w_b = w_gl[q]
        delta_b = TOT_d * np.arctanh(xi_b)
        # P_line[ka] = cs_d[ka](xi_b)
        P_line = np.array([float(cs_d_arr[ka](xi_b)) for ka in range(G)])
        cs_line = CubicSpline(xi_arr, P_line, bc_type='natural')
        # roots
        ts_d = np.linspace(xi_arr[0], xi_arr[-1], 500)
        s = np.sign(cs_line(ts_d) - p_target)
        roots = []
        for idx in range(len(s)-1):
            if s[idx] != s[idx+1] and s[idx] != 0:
                try:
                    roots.append(brentq(lambda x: float(cs_line(x)) - p_target, ts_d[idx], ts_d[idx+1], xtol=1e-12))
                except ValueError:
                    pass
        for xi_a_star in roots:
            if abs(xi_a_star) >= 1 - 1e-12: continue
            Sigma_a = TOT_S * np.arctanh(xi_a_star)
            dP_dxi_a = float(cs_line(xi_a_star, 1))
            dP_dSigma = dP_dxi_a * (1 - xi_a_star**2) / TOT_S
            if abs(dP_dSigma) < 1e-14: continue
            # dP/dδ at (xi_a_star, xi_b): build vertical spline at xi_a_star, deriv at xi_b
            P_vert = np.array([float(cs_s_arr[kb](xi_a_star)) for kb in range(G)])
            cs_v = CubicSpline(xi_arr, P_vert, bc_type='natural')
            dP_dxi_b = float(cs_v(xi_b, 1))
            dP_ddelta = dP_dxi_b * (1 - xi_b**2) / TOT_d
            w_a_part = dP_dSigma**2 / max(dP_dSigma**2 + dP_ddelta**2, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b); u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal_(u_2, VM0, COEF, TAU) * f_signal_(u_3, VM0, COEF, TAU)
            f1 = f_signal_(u_2, VM1, COEF, TAU) * f_signal_(u_3, VM1, COEF, TAU)
            contrib = w_b * w_a_part / abs(dP_dSigma)
            A0 += contrib * f0; A1 += contrib * f1
    # Pass B: fix Σ at GL, root-find in δ
    for q in range(nq):
        xi_a = xi_gl[q]; w_a_gl = w_gl[q]
        Sigma_a = TOT_S * np.arctanh(xi_a)
        P_line = np.array([float(cs_s_arr[kb](xi_a)) for kb in range(G)])
        cs_line = CubicSpline(xi_arr, P_line, bc_type='natural')
        ts_d = np.linspace(xi_arr[0], xi_arr[-1], 500)
        s = np.sign(cs_line(ts_d) - p_target)
        roots = []
        for idx in range(len(s)-1):
            if s[idx] != s[idx+1] and s[idx] != 0:
                try:
                    roots.append(brentq(lambda x: float(cs_line(x)) - p_target, ts_d[idx], ts_d[idx+1], xtol=1e-12))
                except ValueError: pass
        for xi_b_star in roots:
            if abs(xi_b_star) >= 1 - 1e-12: continue
            delta_b = TOT_d * np.arctanh(xi_b_star)
            dP_dxi_b = float(cs_line(xi_b_star, 1))
            dP_ddelta = dP_dxi_b * (1 - xi_b_star**2) / TOT_d
            if abs(dP_ddelta) < 1e-14: continue
            P_horiz = np.array([float(cs_d_arr[ka](xi_b_star)) for ka in range(G)])
            cs_h = CubicSpline(xi_arr, P_horiz, bc_type='natural')
            dP_dxi_a = float(cs_h(xi_a, 1))
            dP_dSigma_loc = dP_dxi_a * (1 - xi_a**2) / TOT_S
            w_b_part = dP_ddelta**2 / max(dP_dSigma_loc**2 + dP_ddelta**2, 1e-30)
            u_2 = 0.5*(Sigma_a + delta_b); u_3 = 0.5*(Sigma_a - delta_b)
            f0 = f_signal_(u_2, VM0, COEF, TAU) * f_signal_(u_3, VM0, COEF, TAU)
            f1 = f_signal_(u_2, VM1, COEF, TAU) * f_signal_(u_3, VM1, COEF, TAU)
            contrib = w_a_gl * w_b_part / abs(dP_ddelta)
            A0 += contrib * f0; A1 += contrib * f1
    return 0.5*A0, 0.5*A1

A0_ref, A1_ref = ref_agent1_Av(P_sd, i, p_target, xi_arr, dxi, XI_GL, W_GL, NQ)
print(f'  REF A_v: A0={A0_ref:.6e}  A1={A1_ref:.6e}')
print(f'  rel err A0: {abs(A0_v11-A0_ref)/max(abs(A0_ref),1e-30):.3e}')
print(f'  rel err A1: {abs(A1_v11-A1_ref)/max(abs(A1_ref),1e-30):.3e}')
ratio_v11 = A1_v11 / (A0_v11 + A1_v11)
ratio_ref = A1_ref / (A0_ref + A1_ref)
print(f'  agent-1 posterior μ from A_v: V11={ratio_v11:.6f}  REF={ratio_ref:.6f}')

# ---------- Stage 5: Full Phi(P_interp) at this cell ----------
print('\n--- STAGE 5: full Φ at this cell ---')
P_phi = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                  XI_GL, W_GL, NQ, 0.1, False,
                  TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  P_sd  [{i},{j},{k}] = {P_sd[i,j,k]:.6f}')
print(f'  Phi   [{i},{j},{k}] = {P_phi[i,j,k]:.6f}')
print(f'  resid             = {P_phi[i,j,k] - P_sd[i,j,k]:+.6f}')

# Manual posterior using V11 A_v and CRRA clearing
u1_c = u_arr[i]; S_c = S_arr[j]; d_c = d_arr[k]
u2_c = 0.5*(S_c + d_c); u3_c = 0.5*(S_c - d_c)
A0o2_v11, A1o2_v11 = evidence_agent_oblique_v10(P_sd, p_target, xi_arr, dxi, XI_GL, W_GL, NQ,
                                                  u2_c, -1.0, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU)
A0o3_v11, A1o3_v11 = evidence_agent_oblique_v10(P_sd, p_target, xi_arr, dxi, XI_GL, W_GL, NQ,
                                                  u3_c, +1.0, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU)
f0_u1 = f_signal_(u1_c, VM0, COEF, TAU); f1_u1 = f_signal_(u1_c, VM1, COEF, TAU)
f0_u2 = f_signal_(u2_c, VM0, COEF, TAU); f1_u2 = f_signal_(u2_c, VM1, COEF, TAU)
f0_u3 = f_signal_(u3_c, VM0, COEF, TAU); f1_u3 = f_signal_(u3_c, VM1, COEF, TAU)
mu0 = f1_u1*A1_v11 / (f0_u1*A0_v11 + f1_u1*A1_v11)
mu1 = f1_u2*A1o2_v11 / (f0_u2*A0o2_v11 + f1_u2*A1o2_v11)
mu2 = f1_u3*A1o3_v11 / (f0_u3*A0o3_v11 + f1_u3*A1o3_v11)
print(f'  μ from V11 A_v: μ0={mu0:.6f}  μ1={mu1:.6f}  μ2={mu2:.6f}')
P_recompute = crra_clear_nb(mu0, mu1, mu2, 0.1, 120)
print(f'  CRRA clear at γ=0.1, μ=({mu0:.4f},{mu1:.4f},{mu2:.4f}) → P = {P_recompute:.6f}')
print(f'  matches Phi[i,j,k]? expected {P_phi[i,j,k]:.6f}  recomputed {P_recompute:.6f}')

print('\n=== AUDIT COMPLETE ===')
