"""(a) Test whether σ-δ V11 nails a PR FP when warm-started from the
hfree_smooth strict-h=0 PR FP (slope=0.36, d_FR=0.26) AND uses Newton-Krylov
instead of Picard.

The hfree_smooth operator at u-grid G=9 nails PR with strict h=0 + Newton from
kernel warm-start (commit 78c27216). My earlier σ-δ V11 tests failed because:
  (i) Picard, not Newton
  (ii) Started from no-learning IC (wrong basin)
  (iii) FR-BC overrides on the σ-δ halo where the warm-start has finite P
This test fixes (i) and (ii) and is honest about (iii) by both FR-BC and
extrapolation-BC variants.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

from v11_strict_h0_hardwired import (phi_v10, set_boundary, XI_GL, W_GL, NQ,
                                       TOT_u, TOT_S, TOT_d,
                                       TAU, VM0, VM1, COEF, EPS_PRICE)

# Load hfree_smooth PR FP (G=9 on u-grid linear, UMAX=4)
P_hf = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G9.npy')
UMAX_HF = 4.0
G_HF = P_hf.shape[0]
ui_HF = np.linspace(-UMAX_HF, UMAX_HF, G_HF)
interp_hf = RegularGridInterpolator((ui_HF, ui_HF, ui_HF), P_hf,
                                       bounds_error=False, fill_value=None,
                                       method='linear')
print(f'hfree_smooth PR FP: G={G_HF}, u∈[±{UMAX_HF}], P∈[{P_hf.min():.4f},{P_hf.max():.4f}]', flush=True)

# σ-δ ξ-cube
G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL-1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)

# Interpolate hfree FP onto σ-δ ξ-cube ALL cells
print(f'\nInterpolating hfree_smooth FP onto σ-δ ξ-cube G_FULL={G_FULL}...', flush=True)
P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1-1e-12: continue
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1-1e-12: continue
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1-1e-12: continue
            u_1 = u_arr[i]; Sig = S_arr[j]; dlt = d_arr[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            u1c=max(ui_HF[0],min(ui_HF[-1],u_1))
            u2c=max(ui_HF[0],min(ui_HF[-1],u_2))
            u3c=max(ui_HF[0],min(ui_HF[-1],u_3))
            P_sd[i,j,k] = float(interp_hf(np.array([u1c,u2c,u3c]))[0])
# Boundary at ξ=±1
for i in (0, G_FULL-1):
    P_sd[i,:,:] = 0 if xi_arr[i] < 0 else 1
for j in (0, G_FULL-1):
    P_sd[:,j,:] = 0 if xi_arr[j] < 0 else 1
for k in (0, G_FULL-1):
    inn = 1 if k == 0 else G_FULL-2
    P_sd[:,:,k] = P_sd[:,:,inn]
inner_warm = P_sd[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()

# Metrics
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
    return slope, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_in)**2*Wd)))

s0, o0, d0 = metrics(inner_warm)
print(f'WARM-START on σ-δ inner: slope={s0:.4f} 1-R²={o0:.4f} d_FR_w={d0:.4f}', flush=True)

# JIT warmup
print('\nJIT warmup phi_v10...', flush=True); t0=time.time()
halo = set_boundary(P_sd.copy())
_ = phi_v10(halo, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t0:.1f}s', flush=True)

def phi_inner(x_flat, halo_template):
    P = halo_template.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x_flat.reshape((G_INNER,)*3)
    Pn = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                   XI_GL, W_GL, NQ, 0.1, False,
                   TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return Pn[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

# Two boundary variants
results = {}
for bc_name, halo_var in [('FR_BC', set_boundary(P_sd.copy())),
                           ('warm_halo', P_sd.copy())]:
    print(f'\n=========== BC variant: {bc_name} ===========', flush=True)
    F = lambda x: phi_inner(x, halo_var) - x
    F0 = float(np.max(np.abs(F(inner_warm.ravel()))))
    print(f'  warm-start ||F||_∞ = {F0:.4e}', flush=True)
    # Newton-Krylov
    t = time.time()
    iters = {'n': 0, 'ferr_hist': []}
    def cb(x, fx):
        iters['n'] += 1
        iters['ferr_hist'].append(float(np.max(np.abs(fx))))
        if iters['n'] % 5 == 0 or iters['n'] == 1:
            inner = x.reshape((G_INNER,)*3)
            s, o, d = metrics(inner)
            print(f'  NK it {iters["n"]:3d} ||F||={iters["ferr_hist"][-1]:.3e} slope={s:.4f} d_FR_w={d:.4f}', flush=True)
    conv = True
    try:
        sol = newton_krylov(F, inner_warm.ravel(), f_tol=1e-7, maxiter=50,
                              method='lgmres', callback=cb, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    F_final = float(np.max(np.abs(F(sol))))
    inner_final = sol.reshape((G_INNER,)*3)
    sf, of, df = metrics(inner_final)
    print(f'\n  RESULT {bc_name}: conv={conv}, ||F||={F_final:.3e}, iters={iters["n"]}', flush=True)
    print(f'    slope={sf:.4f} 1-R²={of:.4f} d_FR_w={df:.4f}', flush=True)
    print(f'    walltime {time.time()-t:.0f}s', flush=True)
    results[bc_name] = dict(converged=conv, Finf=F_final, iters=iters['n'],
                              slope=sf, omr=of, d_FR_w=df,
                              ferr_history=iters['ferr_hist'],
                              warmstart_slope=s0, warmstart_d_FR_w=d0)
    np.save(os.path.join(HERE, f'sigdelta_nk_from_hfree_{bc_name}.npy'), inner_final)

json.dump({'warmstart_source':'hfree_smooth P_nailed_G9 (slope=0.36, d_FR=0.26)',
           'sigma_delta_G_inner': G_INNER, 'results': results},
          open(os.path.join(HERE, 'sigdelta_nk_from_hfree_result.json'), 'w'),
          indent=2)
print('\nsaved')
