"""Nail both converged strict-h=0 PR FPs at machine-precision in float64:
(1) hfree_smooth G=9 (linear u-grid) — already at ||F||=3e-11; push to 1e-15
(2) σ-δ V11 G_inner=11 NK from hfree warm-start — push to 1e-12+
"""
import os, sys, time, json
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/tmp')
sys.path.insert(0, os.path.join(HERE))
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta')
import numpy as np
import hfree_operator as H
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence
from scipy.interpolate import RegularGridInterpolator

UMAX=4.0; TAU=2.0; GAMMA=0.1; NQ=40; SUB=4
tau=np.full(3,TAU); gam=np.full(3,GAMMA); W=np.full(3,1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)

def metrics_u(P, ui):
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing="ij"); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    deficit=float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR=1.0/(1.0+np.exp(-T)); d_FR=float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))

# ---------- (1) hfree_smooth G=9 to machine precision ----------
def F_hfree(x, G, ui):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gnodes,gweights,tau,gam,W,SUB)-P).ravel()

print('='*60)
print('(1) hfree_smooth G=9 to machine precision')
print('='*60, flush=True)
G9=9; ui9=np.linspace(-UMAX,UMAX,G9)
P9_load = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G9.npy')
print(f'Loaded P_nailed_G9.npy: {metrics_u(P9_load, ui9)}', flush=True)
F0 = np.max(np.abs(F_hfree(P9_load.ravel(), G9, ui9)))
print(f'  ||F||_0 = {F0:.3e}')

t=time.time()
print(f'  JIT warmup hfree...', flush=True)
_ = H.phi_hfree(P9_load, ui9, gnodes, gweights, tau, gam, W, SUB)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

cnt = {'n':0, 'hist':[]}
def cb1(x, fx):
    cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
    if cnt['n'] <= 30 and (cnt['n'] % 3 == 0 or cnt['n'] == 1):
        print(f'    hfree NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e}', flush=True)
print(f'  NK on hfree G=9 with f_tol=1e-15...', flush=True)
ts = time.time()
try:
    sol = newton_krylov(lambda x: F_hfree(x, G9, ui9), P9_load.ravel(),
                          f_tol=1e-15, maxiter=60, method='lgmres', callback=cb1)
    conv = True
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); conv = False
F1 = float(np.max(np.abs(F_hfree(sol, G9, ui9))))
m9 = metrics_u(sol.reshape((G9,)*3), ui9)
print(f'  RESULT: conv={conv}, ||F||={F1:.3e}, iters={cnt["n"]}, walltime={time.time()-ts:.0f}s')
print(f'    metrics: deficit={m9["deficit"]:.6f}, slope_T={m9["slope_T"]:.6f}, d_FR={m9["d_FR"]:.6f}')
np.save(os.path.join(HERE, 'hfree_G9_machine_prec.npy'), sol.reshape((G9,)*3))
hfree_summary = dict(G=G9, conv=conv, Finf=F1, iters=cnt['n'], hist=cnt['hist'], **m9)

# ---------- (2) σ-δ V11 to tighter tolerance ----------
print('\n' + '='*60)
print('(2) σ-δ V11 G_inner=11 NK from hfree warm-start, f_tol=1e-12')
print('='*60, flush=True)
from v11_strict_h0_hardwired import (phi_v10, set_boundary, XI_GL, W_GL, NQ as NQ_SD,
                                       TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

# σ-δ ξ-cube
G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL-1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)

# Interp hfree FP onto σ-δ cube
interp_hf = RegularGridInterpolator((ui9, ui9, ui9), sol.reshape((G9,)*3),
                                       bounds_error=False, fill_value=None, method='linear')
P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1-1e-12: continue
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1-1e-12: continue
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1-1e-12: continue
            u1c = max(ui9[0], min(ui9[-1], u_arr[i]))
            Sig = S_arr[j]; dlt = d_arr[k]
            u2 = 0.5*(Sig+dlt); u3 = 0.5*(Sig-dlt)
            u2c = max(ui9[0], min(ui9[-1], u2))
            u3c = max(ui9[0], min(ui9[-1], u3))
            P_sd[i,j,k] = float(interp_hf(np.array([u1c, u2c, u3c]))[0])

halo = set_boundary(P_sd.copy())
print(f'JIT warmup σ-δ V11...', flush=True); t=time.time()
_ = phi_v10(halo, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ_SD, 0.1, False,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

def phi_inner(x_flat):
    P = halo.copy()
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = x_flat.reshape((G_INNER,)*3)
    Pn = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                   XI_GL, W_GL, NQ_SD, 0.1, False,
                   TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    return Pn[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].ravel()

def F_sd(x): return phi_inner(x) - x
inner_warm = halo[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()
F0_sd = float(np.max(np.abs(F_sd(inner_warm.ravel()))))
print(f'σ-δ warm-start ||F||={F0_sd:.3e}', flush=True)

cnt2 = {'n':0, 'hist':[]}
def cb2(x, fx):
    cnt2['n'] += 1; cnt2['hist'].append(float(np.max(np.abs(fx))))
    if cnt2['n'] <= 30 and (cnt2['n'] % 3 == 0 or cnt2['n'] == 1):
        print(f'    σ-δ NK it {cnt2["n"]:3d} ||F||={cnt2["hist"][-1]:.3e}', flush=True)
print(f'  NK on σ-δ V11 with f_tol=1e-12...', flush=True)
ts = time.time()
try:
    sol_sd = newton_krylov(F_sd, inner_warm.ravel(), f_tol=1e-12, maxiter=80,
                              method='lgmres', callback=cb2)
    conv2 = True
except NoConvergence as e:
    sol_sd = np.asarray(e.args[0]).ravel(); conv2 = False
F1_sd = float(np.max(np.abs(F_sd(sol_sd))))
inner_final = sol_sd.reshape((G_INNER,)*3)
# Compute metrics on σ-δ inner block
u_in = u_arr[INNER_LO:INNER_HI]; S_in = S_arr[INNER_LO:INNER_HI]; d_in = d_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)
T = TAU*Sfull
Pc = np.clip(inner_final, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
a = np.polyfit(T.ravel(), y, 1)
slope_sd = float(a[0]); pred = a[0]*T.ravel()+a[1]
deficit_sd = float(((y-pred)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
P_FR_in = 1/(1+np.exp(-T))
dFR_sd = float(np.sqrt(np.mean((inner_final - P_FR_in)**2)))
print(f'  RESULT: conv={conv2}, ||F||={F1_sd:.3e}, iters={cnt2["n"]}, walltime={time.time()-ts:.0f}s')
print(f'    σ-δ inner metrics: deficit={deficit_sd:.6f}, slope_T={slope_sd:.6f}, d_FR={dFR_sd:.6f}')
np.save(os.path.join(HERE, 'sigdelta_G11_machine_prec.npy'), inner_final)
sd_summary = dict(G_inner=G_INNER, conv=conv2, Finf=F1_sd, iters=cnt2['n'],
                    hist=cnt2['hist'], deficit=deficit_sd, slope_T=slope_sd, d_FR=dFR_sd)

# Save combined summary
json.dump({'hfree_G9_machine_prec':hfree_summary,
            'sigdelta_G11_machine_prec':sd_summary},
          open(os.path.join(HERE,'machine_prec_PR_summary.json'),'w'), indent=2, default=str)
print('\nsaved machine_prec_PR_summary.json')
