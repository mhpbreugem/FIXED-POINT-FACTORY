"""Iterate V11 σ-δ Phi holding the HALO PINNED to the u-grid FP interpolation
(Dirichlet BC = u-grid values at ξ=±1 ring, NOT FR limits).

Test: does σ-δ Phi converge to a FP that matches the u-grid FP when the
boundary supports the PR solution? If yes -- the discrepancy was that
σ-δ was being asked to support a PR interior with FR boundaries (an
inconsistent BC); under u-grid BCs the operator is self-consistent on PR.

Strategy:
  1. Interpolate u-grid FP onto σ-δ cube at ALL cells (halo included).
  2. Save halo = P_sd outside inner block.
  3. Iterate: P_new = phi_v10(P); P_new[halo] = halo_saved; (damped Picard).
  4. Report ferr trajectory and final agreement with the u-grid interp.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from v11_strict_h0_hardwired import (phi_v10, XI_GL, W_GL, NQ, EPSB,
                                       TOT_u, TOT_S, TOT_d,
                                       TAU, VM0, VM1, COEF, EPS_PRICE)

# --- Load u-grid FP ---
P_ugrid = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
G_INNER_U = P_ugrid.shape[0]; UMAX = 4.0
ui = np.linspace(-UMAX, UMAX, G_INNER_U)
print(f'u-grid FP: G={G_INNER_U}, u∈[±{UMAX}], P∈[{P_ugrid.min():.4f},{P_ugrid.max():.4f}]', flush=True)

# --- σ-δ ξ-cube ---
G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)

# --- Interpolate u-grid FP onto σ-δ cube at ALL cells ---
interp = RegularGridInterpolator((ui, ui, ui), P_ugrid, bounds_error=False, fill_value=None, method='linear')
P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1 - 1e-12: continue
    u_1 = u_arr[i]
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1 - 1e-12: continue
        Sig = S_arr[j]
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1 - 1e-12: continue
            dlt = d_arr[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            u1c = max(ui[0], min(ui[-1], u_1))
            u2c = max(ui[0], min(ui[-1], u_2))
            u3c = max(ui[0], min(ui[-1], u_3))
            P_sd[i,j,k] = float(interp(np.array([u1c, u2c, u3c]))[0])
# absolute boundary (xi=±1): use FR-limit for u_1, Σ; zero-order for δ
for i in (0, G_FULL-1):
    P_sd[i,:,:] = 0 if xi_arr[i] < 0 else 1
for j in (0, G_FULL-1):
    P_sd[:,j,:] = 0 if xi_arr[j] < 0 else 1
for k in (0, G_FULL-1):
    inn = 1 if k == 0 else G_FULL - 2
    P_sd[:,:,k] = P_sd[:,:,inn]

# Halo mask (everything outside the inner block)
halo_mask = np.ones((G_FULL,)*3, dtype=bool)
halo_mask[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = False
P_halo_pinned = P_sd.copy()  # store the halo we want to PIN
inner_target = P_sd[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()

# --- Metrics ---
u_in = u_arr[INNER_LO:INNER_HI]; S_in = S_arr[INNER_LO:INNER_HI]; d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm); Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u,vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5)+
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred=slope*fl_x+intc; m=float(np.average(fl_lp,weights=fl_w))
    vt=float(np.average((fl_lp-m)**2,weights=fl_w)); vr=float(np.average((fl_lp-pred)**2,weights=fl_w))
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))

s0, _, omr0, dfr0 = metrics(inner_target)
print(f'\nu-grid FP (interp σ-δ interior): slope={s0:.4f} 1-R²={omr0:.3f} d_FR_w={dfr0:.3f}', flush=True)

# --- JIT warmup ---
print('\nJIT warmup...', flush=True); t=time.time()
_ = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

# --- Damped Picard, halo pinned to u-grid interp at every iter ---
print('\nIterating σ-δ Phi with halo PINNED to u-grid interp (Dirichlet BC = u-grid)...', flush=True)
P = P_sd.copy()
omega = 1.0; res_prev = 1e100
MAX_IT = 60; TOL = 1e-6
history = []
for it in range(1, MAX_IT+1):
    t=time.time()
    P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                      XI_GL, W_GL, NQ, 0.1, False,
                      TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    P_damp = (1-omega)*P + omega*P_phi
    # pin halo back to u-grid interp
    P_damp[halo_mask] = P_halo_pinned[halo_mask]
    inner_new = P_damp[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    inner_cur = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
    res = float(np.max(np.abs(inner_new - inner_cur)))
    dev_from_ugrid = float(np.max(np.abs(inner_new - inner_target)))
    s, _, omr, dfr = metrics(inner_new)
    history.append(dict(it=it, omega=omega, ferr=res, dev_from_ugrid=dev_from_ugrid,
                        slope=s, one_minus_R2=omr, d_FR_w=dfr))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    print(f'  it {it:2d}  ω={omega:.3f}  ferr={res:.3e}  |P-Pugrid|={dev_from_ugrid:.3e}'
          f'  slope={s:.4f}  1-R²={omr:.3f}  d_FR_w={dfr:.3f}  ({time.time()-t:.1f}s)', flush=True)
    if res < TOL:
        print(f'  CONVERGED at it {it}', flush=True); break

inner_final = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
final_dev = float(np.max(np.abs(inner_final - inner_target)))
sf, _, omrf, dfrf = metrics(inner_final)
print(f'\n=== RESULT ===')
print(f'  u-grid FP target:    slope={s0:.4f} 1-R²={omr0:.3f} d_FR_w={dfr0:.3f}')
print(f'  σ-δ converged FP:    slope={sf:.4f} 1-R²={omrf:.3f} d_FR_w={dfrf:.3f}')
print(f'  max|σ-δ FP - u-grid|: {final_dev:.4e}')
if final_dev < 0.05:
    print('  >>> AGREEMENT: σ-δ operator nails the u-grid FP given u-grid BCs')
elif final_dev < 0.15:
    print('  >>> NEAR AGREEMENT: σ-δ converges to a PR FP close to but not equal to u-grid')
else:
    print('  >>> DISAGREEMENT: σ-δ pulls away even with u-grid BCs')

np.save(os.path.join(HERE, 'iter_ugrid_bc_final.npy'), inner_final)
json.dump(dict(target_slope=s0, final_slope=sf, target_d_FR_w=dfr0, final_d_FR_w=dfrf,
               max_dev_from_ugrid=final_dev, history=history),
          open(os.path.join(HERE, 'iter_ugrid_bc_result.json'),'w'), indent=2)
print('saved')
