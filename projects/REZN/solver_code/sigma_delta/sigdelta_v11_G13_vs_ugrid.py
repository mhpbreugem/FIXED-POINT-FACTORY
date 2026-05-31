"""σ-δ V11 NK at G_inner=13 warm-started from u-grid hfree_smooth G=13 PR FP,
then POINT-BY-POINT comparison of u-grid (u_1,u_2,u_3) frame result vs
σ-δ (u_1,Σ,δ→ξ) frame result.

Goal: confirm that both frames find the SAME PR FP when both use strict h=0
and the right warm-start. If they match, that resolves the long-standing
disagreement.
"""
import os, sys, time, json, csv
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

# Load u-grid G=13 PR FP from A3 (best PR-basin result)
P_ug = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G13_BEST.npy')
UMAX = 4.0
G_UG = P_ug.shape[0]
ui_UG = np.linspace(-UMAX, UMAX, G_UG)
print(f'u-grid G=13 hfree PR FP (A3): G={G_UG}, P∈[{P_ug.min():.4f},{P_ug.max():.4f}]', flush=True)

# u-grid metrics
U1u, U2u, U3u = np.meshgrid(ui_UG, ui_UG, ui_UG, indexing='ij')
Tu = TAU*(U1u+U2u+U3u); Pcu = np.clip(P_ug, 1e-12, 1-1e-12); yu = np.log(Pcu/(1-Pcu)).ravel()
au = np.polyfit(Tu.ravel(), yu, 1); preu = au[0]*Tu.ravel()+au[1]
defu = float(((yu-preu)**2).mean()/max(((yu-yu.mean())**2).mean(),1e-30))
slope_ug = float(au[0]); P_FRu = 1/(1+np.exp(-Tu)); dfru = float(np.sqrt(np.mean((P_ug-P_FRu)**2)))
print(f'  u-grid metrics: deficit={defu:.4f} slope={slope_ug:.4f} d_FR={dfru:.4f}', flush=True)

interp_ug = RegularGridInterpolator((ui_UG, ui_UG, ui_UG), P_ug, bounds_error=False,
                                       fill_value=None, method='linear')

# σ-δ ξ-cube at G_FULL=15, G_inner=13 (matching u-grid G=13)
G_FULL = 15; INNER_LO, INNER_HI = 1, G_FULL-1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)
print(f'σ-δ ξ-cube G_FULL={G_FULL}, G_inner={G_INNER}', flush=True)

# Interpolate u-grid FP onto σ-δ ξ-cube ALL cells
print(f'Interpolating u-grid FP onto σ-δ cube...', flush=True)
P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1-1e-12: continue
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1-1e-12: continue
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1-1e-12: continue
            u_1 = u_arr[i]; Sig = S_arr[j]; dlt = d_arr[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            u1c=max(ui_UG[0],min(ui_UG[-1],u_1))
            u2c=max(ui_UG[0],min(ui_UG[-1],u_2))
            u3c=max(ui_UG[0],min(ui_UG[-1],u_3))
            P_sd[i,j,k] = float(interp_ug(np.array([u1c,u2c,u3c]))[0])
# Boundary at ξ=±1
for i in (0, G_FULL-1):
    P_sd[i,:,:] = 0 if xi_arr[i] < 0 else 1
for j in (0, G_FULL-1):
    P_sd[:,j,:] = 0 if xi_arr[j] < 0 else 1
for k in (0, G_FULL-1):
    inn = 1 if k == 0 else G_FULL-2
    P_sd[:,:,k] = P_sd[:,:,inn]

inner_warm = P_sd[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()

# σ-δ metrics
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

s_ws, o_ws, d_ws = metrics(inner_warm)
print(f'σ-δ warm-start: slope={s_ws:.4f} 1-R²={o_ws:.4f} d_FR_w={d_ws:.4f}', flush=True)

# JIT warmup
print('\nJIT warmup phi_v10 at G_FULL=15...', flush=True); t=time.time()
halo = set_boundary(P_sd.copy())
_ = phi_v10(halo, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

def phi_inner(x_flat, halo_template):
    P = halo_template.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x_flat.reshape((G_INNER,)*3)
    Pn = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                   XI_GL, W_GL, NQ, 0.1, False,
                   TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return Pn[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

halo_FRBC = set_boundary(P_sd.copy())
F = lambda x: phi_inner(x, halo_FRBC) - x
F0 = float(np.max(np.abs(F(inner_warm.ravel()))))
print(f'\nσ-δ NK from u-grid G=13 warm-start, FR-BC. Initial ||F||={F0:.3e}', flush=True)

iters = {'n': 0, 'hist': []}
def cb(x, fx):
    iters['n'] += 1
    iters['hist'].append(float(np.max(np.abs(fx))))
    if iters['n'] % 3 == 0 or iters['n'] == 1:
        s, o, d = metrics(x.reshape((G_INNER,)*3))
        print(f'  NK it {iters["n"]:3d} ||F||={iters["hist"][-1]:.3e} slope={s:.4f} d_FR_w={d:.4f}', flush=True)
conv = True; t = time.time()
try:
    sol = newton_krylov(F, inner_warm.ravel(), f_tol=1e-7, maxiter=40,
                          method='lgmres', callback=cb, verbose=False)
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); conv = False
F_final = float(np.max(np.abs(F(sol))))
inner_final = sol.reshape((G_INNER,)*3)
sf, of, df = metrics(inner_final)
print(f'\nσ-δ V11 NK RESULT: conv={conv} ||F||={F_final:.3e} iters={iters["n"]}', flush=True)
print(f'  slope={sf:.4f} 1-R²={of:.4f} d_FR_w={df:.4f}', flush=True)
print(f'  walltime {time.time()-t:.0f}s', flush=True)
np.save(os.path.join(HERE, 'sigdelta_v11_G13_NK_FP.npy'), inner_final)

# POINT-BY-POINT COMPARISON
print('\n=== POINT-BY-POINT COMPARISON: σ-δ V11 G_inner=13 vs u-grid G=13 ===', flush=True)
records = []
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            u1 = u_in[i]; S_c = S_in[j]; d_c = d_in[k]
            u2 = 0.5*(S_c + d_c); u3 = 0.5*(S_c - d_c)
            u1c=max(ui_UG[0],min(ui_UG[-1],u1))
            u2c=max(ui_UG[0],min(ui_UG[-1],u2))
            u3c=max(ui_UG[0],min(ui_UG[-1],u3))
            P_sd_cell = float(inner_final[i,j,k])
            P_ug_cell = float(interp_ug(np.array([u1c,u2c,u3c]))[0])
            r = P_sd_cell - P_ug_cell
            records.append(dict(i=i, j=j, k=k, u1=u1, Sigma=S_c, delta=d_c,
                                u2=u2, u3=u3, P_sigdelta=P_sd_cell,
                                P_ugrid=P_ug_cell, resid=r, abs_resid=abs(r)))

# CSV + stats
csv_path = os.path.join(HERE, 'sigdelta_G13_vs_ugrid_G13_pointwise.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader(); w.writerows(records)

arr_a = np.array([r['abs_resid'] for r in records])
arr_r = np.array([r['resid'] for r in records])
arr_sd = np.array([r['P_sigdelta'] for r in records])
arr_ug = np.array([r['P_ugrid'] for r in records])
print(f'  n_cells = {len(records)}', flush=True)
print(f'  max |r| = {arr_a.max():.4f}', flush=True)
print(f'  mean |r| = {arr_a.mean():.4f}', flush=True)
print(f'  rms r = {np.sqrt((arr_r**2).mean()):.4f}', flush=True)
print(f'  signed mean = {arr_r.mean():+.4f}', flush=True)
print(f'  σ-δ range = [{arr_sd.min():.4f}, {arr_sd.max():.4f}]', flush=True)
print(f'  u-grid range = [{arr_ug.min():.4f}, {arr_ug.max():.4f}]', flush=True)
worst = sorted(records, key=lambda r:-r['abs_resid'])[0]
print(f'\n  worst cell: ({worst["i"]},{worst["j"]},{worst["k"]}) '
      f'u=({worst["u1"]:+.2f},{worst["u2"]:+.2f},{worst["u3"]:+.2f}) '
      f'P_sd={worst["P_sigdelta"]:.4f} P_ug={worst["P_ugrid"]:.4f} r={worst["resid"]:+.4f}')
best = sorted(records, key=lambda r:r['abs_resid'])[0]
print(f'  best cell:  ({best["i"]},{best["j"]},{best["k"]}) '
      f'u=({best["u1"]:+.2f},{best["u2"]:+.2f},{best["u3"]:+.2f}) '
      f'P_sd={best["P_sigdelta"]:.4f} P_ug={best["P_ugrid"]:.4f} r={best["resid"]:+.4f}')

summ = dict(
    ugrid_G=13, sigdelta_Ginner=G_INNER,
    ugrid_metrics=dict(deficit=defu, slope=slope_ug, d_FR=dfru, Finf_hfree=0.076),
    sigdelta_warmstart=dict(slope=float(s_ws), omr=float(o_ws), d_FR_w=float(d_ws), Finf_init=F0),
    sigdelta_NK_final=dict(slope=float(sf), omr=float(of), d_FR_w=float(df),
                            Finf=F_final, converged=conv, iters=iters['n']),
    pointwise=dict(n=len(records), max_abs=float(arr_a.max()),
                    mean_abs=float(arr_a.mean()), rms=float(np.sqrt((arr_r**2).mean())),
                    signed_mean=float(arr_r.mean()),
                    worst=worst, best=best),
)
json.dump(summ, open(os.path.join(HERE, 'sigdelta_G13_vs_ugrid_G13_summary.json'),'w'),
          indent=2, default=str)
print('\nsaved comparison CSV + summary JSON')
