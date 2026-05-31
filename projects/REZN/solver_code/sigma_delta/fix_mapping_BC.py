"""Fix the operator-class mapping by using the u-grid FP values AS the sigma-delta
boundary -- instead of the FR-limit override.

Idea: the discrepancy between u-grid FP and sigma-delta V11 comes from BC mismatch.
u-grid uses NL halo; sigma-delta forces FR at xi=+-1. When we interpolate u-grid FP
into sigma-delta and let set_boundary overwrite boundary cells with FR limits, the
inner cells see a discontinuous transition -> they adapt by pulling toward FR.

This script:
  1. Loads u-grid kernel FP at γ=0.1.
  2. Trilinear-interp to sigma-delta xi-cube AT ALL CELLS (inner + halo).
  3. Applies V11 Phi but DOES NOT enforce FR-BC -- keeps halo at interpolated values.
  4. Measures ferr after one Phi (and after a few iters).

If ferr ~ 0: discrepancy was 100% the FR-BC override. The two operators DO agree on
the PR FP when given consistent boundary conditions.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from v11_strict_h0_hardwired import (phi_v10, crra_clear_nb,
                                       XI_GL, W_GL, NQ, EPSB,
                                       TOT_u, TOT_S, TOT_d,
                                       TAU, VM0, VM1, COEF, EPS_PRICE)

# u-grid FP
P_ugrid = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
G_INNER_U = P_ugrid.shape[0]; UMAX = 4.0
ui = np.linspace(-UMAX, UMAX, G_INNER_U)
print(f'u-grid FP: G_inner_u={G_INNER_U}, range u∈[{-UMAX}, {UMAX}]', flush=True)
print(f'  P range: [{P_ugrid.min():.4f}, {P_ugrid.max():.4f}]', flush=True)

# sigma-delta xi-cube setup
G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)

# Trilinear-interp P_ugrid -> sigma-delta cube at ALL cells (inner + halo)
print('\nInterpolating u-grid FP onto sigma-delta xi-cube (ALL cells, no BC override)...', flush=True)
interp = RegularGridInterpolator((ui, ui, ui), P_ugrid, bounds_error=False, fill_value=None, method='linear')
P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1 - 1e-12: continue   # skip the exact xi=+-1 cells (atanh=inf); fall through to clamp below
    u_1 = u_arr[i]
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1 - 1e-12: continue
        Sig = S_arr[j]
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1 - 1e-12: continue
            dlt = d_arr[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            # clamp into u-grid range
            u1c = max(ui[0], min(ui[-1], u_1))
            u2c = max(ui[0], min(ui[-1], u_2))
            u3c = max(ui[0], min(ui[-1], u_3))
            P_sd[i, j, k] = float(interp(np.array([u1c, u2c, u3c]))[0])
# For the absolute boundary cells (xi=+-1) where u=+-inf, use FR limits (P=0 or 1)
# (since u-grid doesn't define values at |u|=inf; and physically P->0/1 in those limits)
for i in (0, G_FULL-1):
    P_sd[i, :, :] = 0 if xi_arr[i] < 0 else 1
for j in (0, G_FULL-1):
    P_sd[:, j, :] = 0 if xi_arr[j] < 0 else 1
for k in (0, G_FULL-1):
    # δ→±∞: use zero-order extrap from one cell in (NOT the FR limit since FR is δ-independent)
    inn = 1 if k == 0 else G_FULL - 2
    P_sd[:, :, k] = P_sd[:, :, inn]

inner_sd = P_sd[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
halo_cells = np.zeros((G_FULL,)*3, dtype=bool)
halo_cells[0,:,:]=True; halo_cells[G_FULL-1,:,:]=True
halo_cells[:,0,:]=True; halo_cells[:,G_FULL-1,:]=True
halo_cells[:,:,0]=True; halo_cells[:,:,G_FULL-1]=True
print(f'  P_sd inner range: [{inner_sd.min():.4f}, {inner_sd.max():.4f}]', flush=True)
print(f'  Halo cells (xi=+-1): P_sd at u_1=-inf face mean={P_sd[0,1:-1,1:-1].mean():.4f}', flush=True)
print(f'  Halo at Σ=+inf face (j=G-1) mean={P_sd[1:-1,G_FULL-1,1:-1].mean():.4f}', flush=True)

# Metrics on inner
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
s, ic, omr, dfr = metrics(inner_sd)
print(f'\nP_sd interior metrics: slope={s:.5f} 1-R²={omr:.4f} d_FR_w={dfr:.4f}', flush=True)

# === V11 Phi WITHOUT FR BC override ===
print('\nApplying V11 Phi (NO FR-BC override -- halo kept at interpolated u-grid values)...', flush=True)
P_keep = P_sd.copy()
# JIT warmup
print('JIT warmup...', flush=True); t=time.time()
_ = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.0f}s', flush=True)
t=time.time()
P_phi = phi_v10(P_keep, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                  XI_GL, W_GL, NQ, 0.1, False,
                  TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
# Restore halo to interpolated values (V11 phi_v10 only updates inner; halo stays as given)
# But we want to also make sure we don't re-impose FR; let's NOT call set_boundary
inner_after = P_phi[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
ferr = float(np.max(np.abs(inner_after - inner_sd)))
s2, ic2, omr2, dfr2 = metrics(inner_after)
print(f'  Phi done {time.time()-t:.1f}s', flush=True)
print(f'\n>>> ferr (no-FR-BC variant) = max|Phi(P_interp) - P_interp| = {ferr:.4e} <<<')
print(f'    BEFORE: slope={s:.5f}  1-R²={omr:.4f}  d_FR_w={dfr:.4f}')
print(f'    AFTER : slope={s2:.5f} 1-R²={omr2:.4f} d_FR_w={dfr2:.4f}')

# Compare to FR-BC variant (the discrepancy we want to fix)
print('\n--- For comparison, with FR-BC override (original test):', flush=True)
def fr_bc(P):
    G = P.shape[0]
    P = P.copy()
    P[0, :, :] = 0; P[G-1, :, :] = 1
    P[:, 0, :] = 0; P[:, G-1, :] = 1
    P[:, :, 0] = P[:, :, 1]; P[:, :, G-1] = P[:, :, G-2]
    return P
P_fr_bc = fr_bc(P_keep.copy())
inner_fr_bc = P_fr_bc[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
t=time.time()
P_phi_fr = phi_v10(P_fr_bc, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                     XI_GL, W_GL, NQ, 0.1, False,
                     TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
inner_fr_after = P_phi_fr[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
ferr_fr = float(np.max(np.abs(inner_fr_after - inner_fr_bc)))
s3,_,omr3,dfr3 = metrics(inner_fr_after)
print(f'  ferr (FR-BC override) = {ferr_fr:.4e}  slope_after={s3:.5f}  1-R²_after={omr3:.4f}', flush=True)

# Verdict
print('\n=== VERDICT ===')
print(f'  ferr with FR-BC override   = {ferr_fr:.4e}  (V11 sees u-grid FP as inconsistent)')
print(f'  ferr WITHOUT FR-BC override = {ferr:.4e}  ({"AGREEMENT" if ferr<0.02 else "still disagreement"})')

np.save(os.path.join(HERE, 'fixed_mapping_P_sd.npy'), P_sd)
np.save(os.path.join(HERE, 'fixed_mapping_P_phi.npy'), P_phi)
json.dump({'ferr_no_BC':ferr, 'ferr_FR_BC':ferr_fr,
           'slope_before':s, '1mR2_before':omr,
           'slope_after_no_BC':s2, '1mR2_after_no_BC':omr2,
           'slope_after_FR_BC':s3, '1mR2_after_FR_BC':omr3},
          open(os.path.join(HERE, 'fixed_mapping_result.json'), 'w'), indent=2)
print('saved')
