"""γ-sweep using the numba CDF-cube operator + Newton-Krylov + continuation.

γ ∈ {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0} with each γ warm-started from
the previous FP. Anderson + NK combined: first Anderson to get into basin,
then NK to nail.

Saves per γ:
  cdf_cube_NB_g{γ}.npy
  cdf_cube_NB_sweep.json
Auto-commits + pushes after each γ.
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

from cdf_cube_numba import (phi_cdf_numba, build_zeta_grid, gauss_legendre,
                             crra_clear_nb, metrics, TAU,
                             TAB_Z, TAB_U, TAB_DUDZ)

G = 9
NQ = 24
GAMMAS = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
TOL = 1e-7
NK_MAXIT = 30

zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
print(f'CDF-cube (numba) G={G}, NQ={NQ}, ζ∈[{zeta_arr[0]:.4f},{zeta_arr[-1]:.4f}]', flush=True)

def F_of(x, gamma):
    P = x.reshape((G,G,G))
    return (phi_cdf_numba(P, zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                           TAB_Z, TAB_U, TAB_DUDZ, gamma, TAU) - P).ravel()

def sigmoid(x): return 1/(1+np.exp(-x))

def NL_IC(gamma_val):
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sigmoid(TAU*u_arr[i]); mu2 = sigmoid(TAU*u_arr[j]); mu3 = sigmoid(TAU*u_arr[k])
                P[i,j,k] = crra_clear_nb(mu1, mu2, mu3, gamma_val, 120)
    return P

def anderson(x0, gamma, m_max=8, max_iter=40, tol=1e-7):
    x = x0.copy()
    G_h=[]; F_h=[]; hist=[]
    best = dict(ferr=np.inf, x=x.copy(), it=0)
    for it in range(1, max_iter+1):
        g_x = (F_of(x, gamma) + x)
        f_x = g_x - x
        ferr = float(np.max(np.abs(f_x)))
        hist.append(ferr)
        if ferr < best['ferr']: best = dict(ferr=ferr, x=g_x.copy(), it=it)
        if ferr < tol: break
        G_h.append(g_x.copy()); F_h.append(f_x.copy())
        if len(F_h) > m_max+1: G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0:
            x_new = g_x
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f_x, rcond=None)
                x_new = g_x - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g_x
        x = np.clip(x_new, 1e-12, 1-1e-12)
    return best['x'], hist, best

LOG = os.path.join(HERE, 'cdf_cube_NB_sweep.log'); open(LOG,'w').close()
def lg(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True); open(LOG,'a').write(line+'\n')

# Warmup numba
lg('JIT warmup...')
_ = phi_cdf_numba(np.full((G,G,G), 0.5), zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                   TAB_Z, TAB_U, TAB_DUDZ, 0.1, TAU)
lg('warmup done')

all_results = []
x_warm = None
t_total = time.time()

for gv in GAMMAS:
    lg(f'\n=== γ={gv} ===')
    if x_warm is None:
        P_ic = NL_IC(gv); x_warm = P_ic.ravel()
        lg(f'  IC (NL proper-CRRA): {metrics(P_ic, u_arr)}')
    else:
        lg(f'  IC (warm from prev γ): {metrics(x_warm.reshape((G,)*3), u_arr)}')

    F0 = float(np.max(np.abs(F_of(x_warm, gv))))
    lg(f'  initial ||F||={F0:.3e}')

    # Phase 1: Anderson to get into basin (fast)
    lg('  Phase 1: Anderson(m=8)...')
    x_a, hist_a, best_a = anderson(x_warm, gv, m_max=8, max_iter=30, tol=TOL)
    m_a = metrics(x_a.reshape((G,)*3), u_arr)
    lg(f'  Anderson best: ||F||={best_a["ferr"]:.3e} at it {best_a["it"]} '
       f'slope={m_a["slope_T"]:.4f} d_FR={m_a["d_FR"]:.4f}')

    if best_a['ferr'] < TOL:
        x_final = x_a; Finf = best_a['ferr']; conv = True; nk_iters = 0
        lg('  Anderson converged, skipping NK')
    else:
        # Phase 2: NK
        lg('  Phase 2: NK...')
        cnt = {'n':0, 'hist':[]}
        def cb(x, fx):
            cnt['n'] += 1
            cnt['hist'].append(float(np.max(np.abs(fx))))
            if cnt['n'] % 5 == 0 or cnt['n'] == 1:
                m = metrics(x.reshape((G,)*3), u_arr)
                lg(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}')
        conv = True; ts = time.time()
        try:
            sol = newton_krylov(lambda x: F_of(x, gv), x_a,
                                  f_tol=TOL, maxiter=NK_MAXIT, method='lgmres',
                                  callback=cb, verbose=False)
        except NoConvergence as e:
            sol = np.asarray(e.args[0]).ravel(); conv = False
        Finf = float(np.max(np.abs(F_of(sol, gv))))
        nk_iters = cnt['n']
        x_final = sol
        lg(f'  NK done: conv={conv} ||F||={Finf:.3e} iters={nk_iters} ({time.time()-ts:.1f}s)')

    P_fp = x_final.reshape((G,)*3)
    m = metrics(P_fp, u_arr)
    dt = time.time() - t_total
    lg(f'  RESULT γ={gv}: conv={conv} ||F||={Finf:.3e} '
       f'deficit={m["deficit"]:.4f} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} '
       f'(cum {dt/60:.1f}m)')

    np.save(os.path.join(HERE, f'cdf_cube_NB_g{gv:g}.npy'), P_fp)
    rec = dict(gamma=gv, conv=conv, Finf=Finf, anderson_iters=len(hist_a),
                anderson_best_ferr=best_a['ferr'], nk_iters=nk_iters, dt=dt,
                **m)
    all_results.append(rec)
    json.dump({'G':G,'NQ':NQ,'gammas':GAMMAS,'tol':TOL,'results':all_results,
                'u_arr':u_arr.tolist(), 'zeta_arr':zeta_arr.tolist()},
              open(os.path.join(HERE,'cdf_cube_NB_sweep.json'),'w'), indent=2, default=str)

    # auto push
    cmd = (f'cd /home/user/FIXED-POINT-FACTORY && '
           f'git add projects/REZN/solver_code/sigma_delta/cdf_cube_NB_g{gv:g}.npy '
           f'projects/REZN/solver_code/sigma_delta/cdf_cube_NB_sweep.json '
           f'projects/REZN/solver_code/sigma_delta/cdf_cube_NB_sweep.log && '
           f'git commit -m "CDF-cube (numba) γ={gv}: ||F||={Finf:.2e} slope={m["slope_T"]:.3f} '
           f'deficit={m["deficit"]:.3f} d_FR={m["d_FR"]:.3f}" 2>&1 | tail -2 && '
           f'git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')
    os.system(cmd)

    x_warm = x_final  # warm-start next γ

lg(f'\nSWEEP DONE in {(time.time()-t_total)/60:.1f} min  '
   f'converged {sum(1 for r in all_results if r["conv"])}/{len(all_results)}')
