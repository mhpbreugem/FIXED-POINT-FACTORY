"""Thick-skin Dirichlet: pin BOTH the halo AND the first interior ring to
u-grid interp values. Only the deep 9³ core is allowed to evolve. If σ-δ
agrees with u-grid given this very thick supporting boundary -> the issue
is just that σ-δ's effective domain is too coarse. If it still pulls away
-> the operator-class disagreement is structural in the interior.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from v11_strict_h0_hardwired import (phi_v10, XI_GL, W_GL, NQ,
                                       TOT_u, TOT_S, TOT_d,
                                       TAU, VM0, VM1, COEF, EPS_PRICE)

P_ugrid = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
UMAX = 4.0; G_U = P_ugrid.shape[0]
ui = np.linspace(-UMAX, UMAX, G_U)
interp = RegularGridInterpolator((ui, ui, ui), P_ugrid, bounds_error=False, fill_value=None, method='linear')

G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1
# THICK skin: only cells with min distance from any face >= 2 are free
CORE_LO, CORE_HI = 2, G_FULL - 2
G_INNER = INNER_HI - INNER_LO; G_CORE = CORE_HI - CORE_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe); S_arr = TOT_S * np.arctanh(safe); d_arr = TOT_d * np.arctanh(safe)

P_sd = np.zeros((G_FULL,)*3)
for i in range(G_FULL):
    if abs(xi_arr[i]) >= 1 - 1e-12: continue
    for j in range(G_FULL):
        if abs(xi_arr[j]) >= 1 - 1e-12: continue
        for k in range(G_FULL):
            if abs(xi_arr[k]) >= 1 - 1e-12: continue
            u_1 = u_arr[i]; Sig = S_arr[j]; dlt = d_arr[k]
            u_2 = 0.5*(Sig + dlt); u_3 = 0.5*(Sig - dlt)
            u1c=max(ui[0],min(ui[-1],u_1)); u2c=max(ui[0],min(ui[-1],u_2)); u3c=max(ui[0],min(ui[-1],u_3))
            P_sd[i,j,k] = float(interp(np.array([u1c,u2c,u3c]))[0])
for i in (0, G_FULL-1):
    P_sd[i,:,:] = 0 if xi_arr[i] < 0 else 1
for j in (0, G_FULL-1):
    P_sd[:,j,:] = 0 if xi_arr[j] < 0 else 1
for k in (0, G_FULL-1):
    inn = 1 if k == 0 else G_FULL - 2
    P_sd[:,:,k] = P_sd[:,:,inn]

P_pinned = P_sd.copy()  # values to pin everywhere except deep core
# Thick-skin mask: True = pin, False = free (deep core)
free_mask = np.zeros((G_FULL,)*3, dtype=bool)
free_mask[CORE_LO:CORE_HI, CORE_LO:CORE_HI, CORE_LO:CORE_HI] = True
pin_mask = ~free_mask
print(f'Thick-skin: full {G_FULL}^3, free core {G_CORE}^3, {int(pin_mask.sum())} pinned cells', flush=True)

# Metrics on the FREE core only
u_co=u_arr[CORE_LO:CORE_HI]; S_co=S_arr[CORE_LO:CORE_HI]; d_co=d_arr[CORE_LO:CORE_HI]
U1m,SIm,DEm = np.meshgrid(u_co,S_co,d_co,indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm)+0.5*(SIm-DEm); Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_co = sg(Tstar)
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
    return slope, intc, vr/vt if vt>0 else float('nan'), float(np.sqrt(np.sum((P-P_FR_co)**2*Wd)))

core_target = P_pinned[CORE_LO:CORE_HI, CORE_LO:CORE_HI, CORE_LO:CORE_HI].copy()
s0,_,o0,d0 = metrics(core_target)
print(f'\nu-grid FP target (deep core {G_CORE}^3): slope={s0:.4f} 1-R²={o0:.3f} d_FR_w={d0:.3f}', flush=True)

# JIT warmup
print('\nJIT warmup...', flush=True); t=time.time()
_ = phi_v10(P_sd.copy(), INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, False, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

print(f'\nIterating σ-δ Phi with THICK skin (2 rings) pinned to u-grid...', flush=True)
P = P_sd.copy(); omega = 1.0; res_prev = 1e100
MAX_IT = 60; TOL = 1e-6
history = []
for it in range(1, MAX_IT+1):
    t=time.time()
    P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                      XI_GL, W_GL, NQ, 0.1, False,
                      TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
    P_damp = (1-omega)*P + omega*P_phi
    P_damp[pin_mask] = P_pinned[pin_mask]
    core_new = P_damp[CORE_LO:CORE_HI, CORE_LO:CORE_HI, CORE_LO:CORE_HI]
    core_cur = P[CORE_LO:CORE_HI, CORE_LO:CORE_HI, CORE_LO:CORE_HI]
    res = float(np.max(np.abs(core_new - core_cur)))
    dev = float(np.max(np.abs(core_new - core_target)))
    s,_,o,d = metrics(core_new)
    history.append(dict(it=it, omega=omega, ferr=res, dev=dev, slope=s, omr=o, dfr=d))
    if it > 3:
        if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_damp; res_prev = res
    print(f'  it {it:2d}  ω={omega:.3f}  ferr={res:.3e}  dev={dev:.3e}'
          f'  slope={s:.4f}  1-R²={o:.3f}  d_FR_w={d:.3f}  ({time.time()-t:.1f}s)', flush=True)
    if res < TOL:
        print(f'  CONVERGED at it {it}', flush=True); break

core_final = P[CORE_LO:CORE_HI, CORE_LO:CORE_HI, CORE_LO:CORE_HI]
final_dev = float(np.max(np.abs(core_final - core_target)))
sf,_,of,df = metrics(core_final)
print(f'\n=== RESULT (thick-skin Dirichlet) ===')
print(f'  u-grid target (core):  slope={s0:.4f} 1-R²={o0:.3f} d_FR_w={d0:.3f}')
print(f'  σ-δ converged (core):  slope={sf:.4f} 1-R²={of:.3f} d_FR_w={df:.3f}')
print(f'  max|core - u-grid|:    {final_dev:.4e}')
if final_dev < 0.05:
    print('  >>> AGREEMENT in deep core')
else:
    print('  >>> STRUCTURAL operator-class disagreement; not a BC issue')

np.save(os.path.join(HERE, 'thick_skin_final.npy'), core_final)
json.dump(dict(target_slope=s0, final_slope=sf, max_dev=final_dev, history=history),
          open(os.path.join(HERE, 'thick_skin_result.json'),'w'), indent=2)
print('saved')
