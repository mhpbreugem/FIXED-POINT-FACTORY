"""Definitive test: u-grid kernel FP at γ=0.1, τ=2 vs u-grid RAW SCAN.

If u-grid raw scan also sees this P* as a fixed point (small ferr), then
kernel and raw scan agree on the equilibrium and the σ-δ Picard discrepancy
is just basin-of-attraction.

If u-grid raw scan changes P* significantly, then kernel and raw scan find
DIFFERENT equilibria -- the operator class matters.

Also: apply NK to raw scan from this IC -- if it nails, raw scan supports
the PR FP.
"""
import os, sys
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
import numpy as np
from reznsrc.contour_K3_halo import phi_K3_halo, phi_K3_halo_smooth, init_no_learning_K3
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

# Load u-grid kernel FP
P_inner_FP = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
print(f'u-grid kernel FP P_inner shape: {P_inner_FP.shape}')

G_INNER = P_inner_FP.shape[0]
PAD = 2; UMAX = 4.0; TAU = 2.0; GAMMA = 0.1; W = 1.0
du = 2*UMAX/(G_INNER-1)
G_FULL = G_INNER + 2*PAD
uf = np.array([-UMAX + (q-PAD)*du for q in range(G_FULL)])
lo, hi = PAD, PAD + G_INNER
slc = (slice(lo, hi),)*3
tv = np.full(3, TAU); gv = np.full(3, GAMMA); Wv = np.full(3, W)

# Build NL halo with FP inner
halo_NL = init_no_learning_K3(uf, tv, gv, Wv)
P_NL_inner = halo_NL[slc].copy()
print(f'NL halo built. NL inner range: [{P_NL_inner.min():.4f}, {P_NL_inner.max():.4f}]')
print(f'kernel-FP inner range:        [{P_inner_FP.min():.4f}, {P_inner_FP.max():.4f}]')

# Set P_full = NL halo, inner = kernel FP
P_full = halo_NL.copy()
P_full[slc] = P_inner_FP

# Apply u-grid RAW SCAN (no kernel) once
P_raw = phi_K3_halo(P_full, uf, lo, hi, tv, gv, Wv)
diff_raw = (P_raw - P_full)[slc]
ferr_raw = float(np.max(np.abs(diff_raw)))
print(f'\nu-grid RAW SCAN at kernel FP: ferr = max|Phi_raw(P*) - P*| = {ferr_raw:.4e}')

# Apply u-grid KERNEL once (sanity check -- should give ferr << 1)
C_H = 0.45; h_kernel = C_H * (du**0.5)
P_kern = phi_K3_halo_smooth(P_full, uf, lo, hi, tv, gv, Wv, h_kernel)
diff_kern = (P_kern - P_full)[slc]
ferr_kern = float(np.max(np.abs(diff_kern)))
print(f'u-grid KERNEL CO-AREA at kernel FP: ferr = {ferr_kern:.4e}  (sanity: should be ~1e-12)')

# Slope/R² metrics
ui = uf[lo:hi]
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU*(U1+U2+U3)
def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel() + aa[1]
    deficit = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    return float(aa[0]), float(aa[1]), deficit

slope_FP, intc_FP, defi_FP = metrics(P_inner_FP)
slope_raw, intc_raw, defi_raw = metrics(P_raw[slc])
print(f'\nP_FP                slope={slope_FP:.5f} intc={intc_FP:+.5f} deficit={defi_FP:.4f}')
print(f'Phi_raw(P_FP) inner slope={slope_raw:.5f} intc={intc_raw:+.5f} deficit={defi_raw:.4f}')

# Newton-Krylov on raw scan from kernel FP IC
print('\nNewton-Krylov on RAW SCAN starting from u-grid kernel FP IC...')
def resid_raw(x):
    P = halo_NL.copy(); P[slc] = x.reshape((G_INNER,)*3)
    return (phi_K3_halo(P, uf, lo, hi, tv, gv, Wv) - P)[slc].ravel()
x0 = P_inner_FP.ravel().copy()
try:
    sol = newton_krylov(resid_raw, x0, f_tol=1e-9, maxiter=80, method='lgmres')
    nk_conv = True
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); nk_conv = False
ferr_nk = float(np.max(np.abs(resid_raw(sol))))
Pin_nk = sol.reshape((G_INNER,)*3)
slope_nk, intc_nk, defi_nk = metrics(Pin_nk)
print(f'  NK conv={nk_conv}  ferr={ferr_nk:.4e}  slope={slope_nk:.5f}  deficit={defi_nk:.4f}')
print(f'  drift from kernel FP: max|Pin_nk - P_inner_FP| = {float(np.max(np.abs(Pin_nk - P_inner_FP))):.4e}')
