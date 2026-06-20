"""Test 3 (proper): PCHIP CDF-cube + Anderson + NK at γ=0.1, G=13, NQ=64.

Compares to cubic-spline-with-clip variant which got ||F||≈0.04 at γ=0.1.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

from cdf_cube_pchip import (phi_cdf_pchip, build_zeta_grid, gauss_legendre,
                              crra_clear_nb, metrics, TAU, GAMMA,
                              TAB_Z, TAB_U, TAB_DUDZ)

G = 13; NQ = 64
zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])

def sg(x): return 1.0/(1.0+np.exp(-x))
U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
P_NL = np.empty_like(U1)
for i in range(G):
    for j in range(G):
        for k in range(G):
            mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
            P_NL[i,j,k] = crra_clear_nb(mu1, mu2, mu3, GAMMA, 120)
m_ic = metrics(P_NL, u_arr)
print(f'IC: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}', flush=True)

print('JIT warmup...', flush=True); t=time.time()
_ = phi_cdf_pchip(P_NL.copy(), zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                    TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

def F_of(x):
    P = x.reshape((G,G,G))
    return (phi_cdf_pchip(P, zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                            TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU) - P).ravel()

# Anderson(m=8)
x = P_NL.ravel().copy()
G_h=[]; F_h=[]
best = dict(ferr=np.inf, x=x.copy())
print('\n=== Anderson(m=8) PCHIP ===', flush=True)
for it in range(1, 31):
    ts = time.time()
    g = (F_of(x) + x); f = g - x
    ferr = float(np.max(np.abs(f)))
    if ferr < best['ferr']: best = dict(ferr=ferr, x=g.copy(), it=it)
    if it % 5 == 0 or it == 1 or it < 5:
        inner = x.reshape((G,)*3)
        m = metrics(inner, u_arr)
        print(f'  A it {it:3d} ferr={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
    if ferr < 1e-7: break
    G_h.append(g.copy()); F_h.append(f.copy())
    if len(F_h) > 9: G_h.pop(0); F_h.pop(0)
    mk = len(F_h) - 1
    if mk == 0: x_new = g
    else:
        dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
        dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
        try:
            gc, *_ = np.linalg.lstsq(dF, f, rcond=None)
            x_new = g - dG @ gc
        except np.linalg.LinAlgError:
            x_new = g
    x = np.clip(x_new, 1e-12, 1-1e-12)

print(f'\nAnderson best: ||F||={best["ferr"]:.3e} at it {best["it"]}', flush=True)

# NK from best
print('\n=== NK PCHIP from Anderson best ===', flush=True)
cnt = {'n':0, 'hist':[]}
def cb(x, fx):
    cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
    if cnt['n'] % 5 == 0 or cnt['n'] == 1:
        m = metrics(x.reshape((G,)*3), u_arr)
        print(f'  NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}', flush=True)
try:
    sol = newton_krylov(F_of, best['x'], f_tol=1e-7, maxiter=30, method='lgmres', callback=cb)
    conv = True
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); conv = False
Finf = float(np.max(np.abs(F_of(sol))))
m = metrics(sol.reshape((G,)*3), u_arr)
print(f'\nPCHIP+NK RESULT γ=0.1: conv={conv}, ||F||={Finf:.3e}, slope={m["slope_T"]:.4f} '
      f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f}')
print(f'Compare to cubic-spline-clip same setup: ||F||≈0.04, slope=0.66, deficit=0.12')

np.save(os.path.join(HERE, 'pchip_NK_g0.1_G13.npy'), sol.reshape((G,)*3))
json.dump({'G':G,'NQ':NQ,'gamma':float(GAMMA),'final_Finf':Finf,'slope':m['slope_T'],
            'deficit':m['deficit'],'d_FR':m['d_FR'],'NK_iters':cnt['n'],
            'NK_hist':cnt['hist'],'anderson_best_ferr':best['ferr']},
          open(os.path.join(HERE,'pchip_NK_g0.1_G13.json'),'w'), indent=2, default=str)
print('saved')
