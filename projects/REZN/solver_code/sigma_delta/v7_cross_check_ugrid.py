"""Cross-check: u-grid kernel CRRA FP -> sigma-delta coords -> apply V7 once.

If V7 Phi preserves it (small ferr) -> both operators agree, sigma-delta
Picard just doesn't find this attractor from no-learning IC.

If V7 changes it significantly -> ops disagree on the same physical FP -> bug
somewhere.
"""
import os, sys, time, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from v7_strict_h0 import (phi_strict_sigdelta, set_boundary, make_grids,
                           TAU, TOT_u, TOT_S, TOT_d)

# Load u-grid kernel FP
P_ugrid_inner = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
print(f'u-grid FP shape: {P_ugrid_inner.shape}')

# u-grid: G_inner=17, UMAX=4, du=8/16=0.5, inner u from -4 to +4
G_ugrid = P_ugrid_inner.shape[0]
u_ugrid = np.linspace(-4.0, 4.0, G_ugrid)
print(f'u-grid axis: {u_ugrid[0]:.2f} .. {u_ugrid[-1]:.2f}')

# Build trilinear interpolator
interp = RegularGridInterpolator((u_ugrid, u_ugrid, u_ugrid), P_ugrid_inner,
                                   bounds_error=False, fill_value=None)

# sigma-delta grid (G_inner=10 to match V7 test)
G_FULL = 12; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
xi_arr, u_arr, S_arr, d_arr, Ju, JS, Jd = make_grids(G_FULL)
print(f'sigma-delta inner u-range: {u_arr[INNER_LO]:.2f} .. {u_arr[INNER_HI-1]:.2f}')
print(f'sigma-delta inner Sigma: {S_arr[INNER_LO]:.2f} .. {S_arr[INNER_HI-1]:.2f}')
print(f'sigma-delta inner delta: {d_arr[INNER_LO]:.2f} .. {d_arr[INNER_HI-1]:.2f}')

# Build P_sd_inner by trilinear interp from u-grid FP
G = G_FULL
P_sd = np.zeros((G, G, G))
for i in range(G):
    if abs(xi_arr[i]) >= 1 - 1e-12:
        P_sd[i, :, :] = 0 if xi_arr[i] < 0 else 1
        continue
    u_1 = u_arr[i]
    for j in range(G):
        if abs(xi_arr[j]) >= 1 - 1e-12:
            P_sd[i, j, :] = 0 if xi_arr[j] < 0 else 1
            continue
        Sigma = S_arr[j]
        for k in range(G):
            if abs(xi_arr[k]) >= 1 - 1e-12:
                # delta-extrap; just set boundary
                continue
            delta = d_arr[k]
            u_2 = 0.5*(Sigma + delta)
            u_3 = 0.5*(Sigma - delta)
            # clip to ugrid range
            u1c = max(u_ugrid[0], min(u_ugrid[-1], u_1))
            u2c = max(u_ugrid[0], min(u_ugrid[-1], u_2))
            u3c = max(u_ugrid[0], min(u_ugrid[-1], u_3))
            P_sd[i, j, k] = float(interp(np.array([u1c, u2c, u3c]))[0])
# apply BC
P_sd = set_boundary(P_sd)
inner_sd = P_sd[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f'Interpolated P_sd inner: min={inner_sd.min():.4f} max={inner_sd.max():.4f} mean={inner_sd.mean():.4f}')

# Check FR-distance and metrics
u_in = u_arr[INNER_LO:INNER_HI]
S_in = S_arr[INNER_LO:INNER_HI]
d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
S_full_in = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
Tstar = TAU * S_full_in
def fa(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5) +
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x+intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    P_FR = 1.0/(1+np.exp(-Tstar))
    d_FR_w = float(np.sqrt(np.sum((P - P_FR)**2 * Wd)))
    return slope, intc, vr/vt if vt>0 else float('nan'), d_FR_w

slope_in, intc_in, omr2_in, dfr_in = metrics(inner_sd)
print(f'\nINTERPOLATED u-grid FP on sigma-delta inner block:')
print(f'  slope={slope_in:.5f}  intc={intc_in:+.5f}  1-R²={omr2_in:.4f}  d_FR_w={dfr_in:.4f}')

# Apply V7 sigma-delta operator ONCE
print('\nApplying V7 strict h=0 sigma-delta Phi to interpolated P...')
t = time.time()
P_phi = phi_strict_sigdelta(P_sd, INNER_LO, INNER_HI, xi_arr, u_arr, S_arr, d_arr, JS, Jd,
                             gamma=0.1, clearing='crra')
P_phi = set_boundary(P_phi)
inner_phi = P_phi[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
print(f'  done {time.time()-t:.0f}s')

ferr = float(np.max(np.abs(inner_phi - inner_sd)))
print(f'\n>>> ferr = max|Phi(P_ugrid_interp) - P_ugrid_interp| = {ferr:.4e} <<<')
print('  (small = sigma-delta V7 ALSO sees u-grid FP as a FP -> ops agree)')
print('  (large = sigma-delta and u-grid disagree on this equilibrium)')

slope_phi, intc_phi, omr2_phi, dfr_phi = metrics(inner_phi)
print(f'\nAfter one Phi:')
print(f'  slope={slope_phi:.5f}  intc={intc_phi:+.5f}  1-R²={omr2_phi:.4f}  d_FR_w={dfr_phi:.4f}')
print(f'  delta slope: {slope_phi - slope_in:+.5f}')
print(f'  delta 1-R²:  {omr2_phi - omr2_in:+.5f}')

# Show which inner cells deviate most
diff = inner_phi - inner_sd
i_m, j_m, k_m = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
print(f'  worst cell (i,j,k) inner = ({i_m},{j_m},{k_m}): '
      f'u_1={u_in[i_m]:+.2f}, Σ={S_in[j_m]:+.2f}, δ={d_in[k_m]:+.2f}')
print(f'    P_ugrid_interp = {inner_sd[i_m, j_m, k_m]:.5f}  Phi = {inner_phi[i_m, j_m, k_m]:.5f}')

np.save('/tmp/P_sd_from_ugrid.npy', inner_sd)
np.save('/tmp/P_sd_from_ugrid_phi.npy', inner_phi)
