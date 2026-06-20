"""Workflow:
  STEP 1: Solve CRRA gamma=0.1 on the LINEAR u-grid kernel co-area (the proven
          operator, nails ferr ~ 1e-12). Use Newton-Krylov on phi_K3_halo_smooth.
  STEP 2: Take the solved P*(u_1, u_2, u_3), interpolate to sigma-delta
          (u_1, Sigma, delta) xi-cube via trilinear interp.
  STEP 3: Apply V10 (numba h-free smooth GL co-area) Phi on the xi-cube ONCE.
          Measure max|Phi_sigdelta(P_interp) - P_interp|.
          If small -> both operators agree on the equilibrium.
          If large -> operators disagree (already established earlier with V7).
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep")
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from reznsrc.contour_K3_halo import phi_K3_halo_smooth, init_no_learning_K3

# ===== STEP 1: solve on u-grid =====
TAU = 2.0; GAMMA = 0.1; W_F = 1.0; UMAX = 4.0; PAD = 2
G_INNER_U = 17
du = 2*UMAX/(G_INNER_U-1)
G_FULL_U = G_INNER_U + 2*PAD
uf = np.array([-UMAX + (q-PAD)*du for q in range(G_FULL_U)])
lo_u, hi_u = PAD, PAD + G_INNER_U
slc_u = (slice(lo_u, hi_u),)*3
tv = np.full(3, TAU); gv = np.full(3, GAMMA); Wv = np.full(3, W_F)
C_H = 0.45; h_kernel = C_H * (du**0.5)

print('STEP 1: u-grid kernel co-area solve at γ=0.1, τ=2, G_inner=17, UMAX=4', flush=True)
# Check if already saved
saved = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy'
if os.path.exists(saved):
    P_inner_FP = np.load(saved)
    print(f'  loaded saved FP: shape {P_inner_FP.shape}', flush=True)
else:
    halo = init_no_learning_K3(uf, tv, gv, Wv)
    def resid(x):
        P = halo.copy(); P[slc_u] = x.reshape((G_INNER_U,)*3)
        return (phi_K3_halo_smooth(P, uf, lo_u, hi_u, tv, gv, Wv, h_kernel) - P)[slc_u].ravel()
    x0 = halo[slc_u].ravel()
    t = time.time()
    try:
        sol = newton_krylov(resid, x0, f_tol=1e-9, maxiter=120, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel()
    print(f'  NK done {time.time()-t:.0f}s  ferr={float(np.max(np.abs(resid(sol)))):.3e}', flush=True)
    P_inner_FP = sol.reshape((G_INNER_U,)*3)
    np.save(saved, P_inner_FP)
# Metrics on u-grid
ui = uf[lo_u:hi_u]
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T_u = TAU*(U1+U2+U3)
def metrics_u(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_u.ravel(), y, 1); pr = a[0]*T_u.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/np.sum((y-y.mean())**2))
    return float(a[0]), float(a[1]), defi, float(np.sqrt(np.mean((P-1/(1+np.exp(-T_u)))**2)))
s, ic, df, dFR = metrics_u(P_inner_FP)
print(f'  P_FP (u-grid): slope={s:.5f} intc={ic:+.5f} deficit={df:.4f} d_FR={dFR:.3e}', flush=True)

# ===== STEP 2: interpolate to sigma-delta xi-cube =====
print('\nSTEP 2: interpolate P*(u_1, u_2, u_3) -> P_sd(u_1, Sigma, delta) on xi-cube', flush=True)
from v10_hfree_gl_numba import (phi_v10, set_boundary, XI_GL, W_GL, NQ,
                                  TOT_u as TOT_u_sd, TOT_S as TOT_S_sd, TOT_d as TOT_d_sd,
                                  VM0, VM1, COEF, EPS_PRICE)
G_FULL_SD = 13; INNER_LO_SD, INNER_HI_SD = 1, G_FULL_SD - 1; G_INNER_SD = INNER_HI_SD - INNER_LO_SD
xi_arr = np.linspace(-1.0, 1.0, G_FULL_SD); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr_sd = TOT_u_sd * np.arctanh(safe)
S_arr_sd = TOT_S_sd * np.arctanh(safe)
d_arr_sd = TOT_d_sd * np.arctanh(safe)

interp_u = RegularGridInterpolator((ui, ui, ui), P_inner_FP, bounds_error=False, fill_value=None)
P_sd = np.zeros((G_FULL_SD,)*3)
for i in range(G_FULL_SD):
    if abs(xi_arr[i]) >= 1 - 1e-12:
        P_sd[i, :, :] = 0 if xi_arr[i] < 0 else 1
        continue
    u_1 = u_arr_sd[i]
    for j in range(G_FULL_SD):
        if abs(xi_arr[j]) >= 1 - 1e-12:
            P_sd[i, j, :] = 0 if xi_arr[j] < 0 else 1
            continue
        Sig = S_arr_sd[j]
        for k in range(G_FULL_SD):
            if abs(xi_arr[k]) >= 1 - 1e-12: continue
            dlt = d_arr_sd[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            u1c = max(ui[0], min(ui[-1], u_1))
            u2c = max(ui[0], min(ui[-1], u_2))
            u3c = max(ui[0], min(ui[-1], u_3))
            P_sd[i, j, k] = float(interp_u(np.array([u1c, u2c, u3c]))[0])
P_sd = set_boundary(P_sd)
inner_sd = P_sd[INNER_LO_SD:INNER_HI_SD, INNER_LO_SD:INNER_HI_SD, INNER_LO_SD:INNER_HI_SD]
print(f'  interp to G_inner={G_INNER_SD} sigma-delta xi-cube done. range [{inner_sd.min():.4f}, {inner_sd.max():.4f}]', flush=True)

# σ-δ metrics
u_in = u_arr_sd[INNER_LO_SD:INNER_HI_SD]
S_in = S_arr_sd[INNER_LO_SD:INNER_HI_SD]
d_in = d_arr_sd[INNER_LO_SD:INNER_HI_SD]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm); Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u,vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5)+
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics_sd(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred=slope*fl_x+intc
    m=float(np.average(fl_lp,weights=fl_w)); vt=float(np.average((fl_lp-m)**2,weights=fl_w))
    vr=float(np.average((fl_lp-pred)**2,weights=fl_w))
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))
s, ic, omr, dfr = metrics_sd(inner_sd)
print(f'  P_sd (interpolated): slope={s:.5f} intc={ic:+.5f} 1-R²={omr:.4f} d_FR_w={dfr:.4f}', flush=True)

# ===== STEP 3: Apply V10 Phi once =====
print('\nSTEP 3: Apply V10 σ-δ Phi (smooth h=0 cubic spline + GL) ONCE to P_sd', flush=True)
t = time.time()
# JIT warmup
_ = phi_v10(P_sd, INNER_LO_SD, INNER_HI_SD, xi_arr, dxi, u_arr_sd, S_arr_sd, d_arr_sd,
              XI_GL, W_GL, NQ, GAMMA, False,
              TOT_u_sd, TOT_S_sd, TOT_d_sd, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  JIT warmup: {time.time()-t:.0f}s', flush=True)
t = time.time()
P_phi = phi_v10(P_sd, INNER_LO_SD, INNER_HI_SD, xi_arr, dxi, u_arr_sd, S_arr_sd, d_arr_sd,
                  XI_GL, W_GL, NQ, GAMMA, False,
                  TOT_u_sd, TOT_S_sd, TOT_d_sd, VM0, VM1, COEF, TAU, EPS_PRICE)
P_phi = set_boundary(P_phi)
inner_phi = P_phi[INNER_LO_SD:INNER_HI_SD, INNER_LO_SD:INNER_HI_SD, INNER_LO_SD:INNER_HI_SD]
print(f'  Phi done {time.time()-t:.1f}s', flush=True)

ferr = float(np.max(np.abs(inner_phi - inner_sd)))
s_p, ic_p, omr_p, dfr_p = metrics_sd(inner_phi)
print(f'\n>>> ferr = max|Φ_σδ(P_interp) - P_interp| = {ferr:.4e} <<<')
print(f'    BEFORE: slope={s:.5f}  1-R²={omr:.4f}  d_FR_w={dfr:.4f}')
print(f'    AFTER:  slope={s_p:.5f}  1-R²={omr_p:.4f}  d_FR_w={dfr_p:.4f}')
print(f'    delta slope: {s_p-s:+.5f}')
print(f'    delta 1-R²: {omr_p-omr:+.5f}')

# locate worst cell
diff = inner_phi - inner_sd
i_m, j_m, k_m = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
print(f'\n    worst cell (i,j,k)=({i_m},{j_m},{k_m})  u_1={u_in[i_m]:+.2f}  Σ={S_in[j_m]:+.2f}  δ={d_in[k_m]:+.2f}')
print(f'      P_interp = {inner_sd[i_m, j_m, k_m]:.5f}   Phi = {inner_phi[i_m, j_m, k_m]:.5f}')

# Verdict
print('\n=== VERDICT ===')
if ferr < 1e-2:
    print(f'AGREEMENT: σ-δ V10 sees u-grid FP as approximately a fixed point (ferr={ferr:.3e}).')
else:
    print(f'DISAGREEMENT: σ-δ V10 does NOT see u-grid FP as a FP (ferr={ferr:.3e}).')
    print('  Operators in different basins -- σ-δ V10 will drift to different attractor.')

np.save(os.path.join(HERE, 'verify_P_sd_from_ugrid.npy'), inner_sd)
np.save(os.path.join(HERE, 'verify_P_sd_after_phi.npy'), inner_phi)
json.dump({'ferr':ferr, 'slope_before':s, 'slope_after':s_p, '1mR2_before':omr, '1mR2_after':omr_p,
           'd_FR_w_before':dfr, 'd_FR_w_after':dfr_p, 'worst_cell':(int(i_m), int(j_m), int(k_m))},
          open(os.path.join(HERE, 'verify_ugrid_to_sigdelta.json'), 'w'), indent=2)
print('saved')
