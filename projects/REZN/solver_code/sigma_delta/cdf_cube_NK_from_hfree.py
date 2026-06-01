"""CDF-cube G=13 NK from the hfree G=9 machine-precision PR FP warm-start.
NO boundary conditions, NO pinning. Pure operator iteration.
"""
import os, sys, time, json
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

from cdf_cube_numba import (phi_cdf_numba, build_zeta_grid, gauss_legendre,
                              crra_clear_nb, metrics, TAU, GAMMA, TAB_Z, TAB_U,
                              TAB_DUDZ)

G = 13; NQ = 64
zeta_arr, u_arr, h_z, z0 = build_zeta_grid(G)
gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
print(f'CDF-cube G={G}, NQ={NQ}, NO BC')

# Warm-start: interpolate hfree G=9 machine-prec FP onto CDF u-grid
P_hf = np.load(os.path.join(HERE, 'hfree_G9_machine_prec.npy'))
u_hf = np.linspace(-4.0, 4.0, 9)
interp = RegularGridInterpolator((u_hf, u_hf, u_hf), P_hf, bounds_error=False,
                                    fill_value=None, method='linear')
U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
# Clamp to u_hf range
U1c = np.clip(U1, u_hf[0], u_hf[-1])
U2c = np.clip(U2, u_hf[0], u_hf[-1])
U3c = np.clip(U3, u_hf[0], u_hf[-1])
pts = np.stack([U1c.ravel(), U2c.ravel(), U3c.ravel()], axis=1)
P_warm = interp(pts).reshape((G, G, G))
m_w = metrics(P_warm, u_arr)
print(f'Warm-start (hfree interp): slope={m_w["slope_T"]:.4f} deficit={m_w["deficit"]:.4f} d_FR={m_w["d_FR"]:.4f}', flush=True)

print('JIT warmup...', flush=True); t=time.time()
_ = phi_cdf_numba(P_warm.copy(), zeta_arr, h_z, z0, gl_z_nodes, gl_z_weights,
                    TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

def F_of(x):
    P = x.reshape((G,G,G))
    return (phi_cdf_numba(P, zeta_arr, h_z, z0,
                            gl_z_nodes, gl_z_weights,
                            TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU) - P).ravel()

F0 = float(np.max(np.abs(F_of(P_warm.ravel()))))
print(f'\nWarm-start ||F||={F0:.3e}', flush=True)

cnt = {'n':0, 'hist':[]}
def cb(x, fx):
    cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
    if cnt['n'] % 3 == 0 or cnt['n'] == 1:
        m = metrics(x.reshape((G,)*3), u_arr)
        print(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}', flush=True)
print(f'NK on CDF-cube G={G}, no BC, f_tol=1e-10...', flush=True)
ts = time.time()
try:
    sol = newton_krylov(F_of, P_warm.ravel(), f_tol=1e-10, maxiter=80,
                          method='lgmres', callback=cb)
    conv = True
except NoConvergence as e:
    sol = np.asarray(e.args[0]).ravel(); conv = False
Finf = float(np.max(np.abs(F_of(sol))))
m = metrics(sol.reshape((G,)*3), u_arr)
print(f'\nRESULT γ=0.1: conv={conv}, ||F||={Finf:.3e}, iters={cnt["n"]} ({time.time()-ts:.0f}s)')
print(f'  slope={m["slope_T"]:.6f}, deficit={m["deficit"]:.6f}, d_FR={m["d_FR"]:.6f}')

np.save(os.path.join(HERE, 'cdf_cube_NK_from_hfree_G13.npy'), sol.reshape((G,)*3))
json.dump({'G':G, 'NQ':NQ, 'gamma':float(GAMMA), 'tau':float(TAU),
            'warm_source':'hfree_G9_machine_prec',
            'Finf':Finf, 'iters':cnt['n'], 'NK_hist':cnt['hist'],
            **m},
          open(os.path.join(HERE,'cdf_cube_NK_from_hfree_G13.json'),'w'), indent=2, default=str)
print('saved')
