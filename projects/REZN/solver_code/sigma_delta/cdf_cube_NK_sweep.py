"""γ-SWEEP of CDF-cube CRRA operator using Newton-Krylov.

Sweeps γ ∈ {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0} (PR-deep → CARA limit),
warm-starting each γ from the previous FP (continuation). Uses
scipy.optimize.newton_krylov(LGMRES, f_tol=1e-7).

CDF-cube axes: ζ_k = F̄(u_k), u_k = F̄⁻¹(ζ_k), where F̄ is the unconditional
Gaussian-mixture signal CDF.

Strict h=0, NO kernel, NO bandwidth.

Saves per γ:
  cdf_cube_NK_g{γ}.npy   — FP on CDF cube
  cdf_cube_NK_sweep.json — aggregate (γ, ferr, slope, deficit, d_FR, iters)
Auto-commits + pushes after each γ.

LATER (if needed): port to double-double (mp/flint or dd_ops) for tighter
nail at small γ (e.g., γ<0.01 where Jensen tail is delicate).
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

from cdf_cube_operator import (
    phi_cdf_cube, build_zeta_u_grid, gauss_legendre, crra_clear,
    metrics, F_bar_inv, TAU, EPS_PRICE
)

G = 9
NQ = 24
GAMMAS = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
TOL = 1e-7
MAXIT = 30

# Build CDF cube once
zeta_arr, u_arr = build_zeta_u_grid(G)
print(f'CDF-cube G={G}, ζ∈[{zeta_arr[0]:.4f},{zeta_arr[-1]:.4f}], u∈[{u_arr[0]:.3f},{u_arr[-1]:.3f}]', flush=True)
gl_u_a, gl_w_a = gauss_legendre(NQ, u_arr[0], u_arr[-1])

def F_of(x, gamma):
    P = x.reshape((G,G,G))
    return (phi_cdf_cube(P, u_arr, gl_u_a, gl_w_a, gamma) - P).ravel()

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

# IC for first γ: proper-CRRA no-learning
def initial_IC(gamma_val):
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sigmoid(TAU*u_arr[i])
                mu2 = sigmoid(TAU*u_arr[j])
                mu3 = sigmoid(TAU*u_arr[k])
                P[i,j,k] = crra_clear(mu1, mu2, mu3, gamma_val)
    return P

LOG = os.path.join(HERE, 'cdf_cube_NK_sweep.log')
open(LOG, 'w').close()
def lg(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True); open(LOG,'a').write(line+'\n')

lg(f'CDF-cube NK γ-sweep G={G}, NQ={NQ}, γs={GAMMAS}, tol={TOL}')

all_results = []
x_warm = None  # warm-start
t_total = time.time()

for gi, gamma in enumerate(GAMMAS):
    lg(f'\n=== γ={gamma} ===')
    if x_warm is None:
        P_ic = initial_IC(gamma)
        x_warm = P_ic.ravel()
        m_ic = metrics(P_ic, u_arr)
        lg(f'  IC (NL proper-CRRA): deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}')
    else:
        m_ic = metrics(x_warm.reshape((G,)*3), u_arr)
        lg(f'  IC (warm from prev γ): deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}')

    F0 = float(np.max(np.abs(F_of(x_warm, gamma))))
    lg(f'  initial ||F||={F0:.3e}')

    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1
        cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 3 == 0 or cnt['n'] == 1:
            m = metrics(x.reshape((G,)*3), u_arr)
            lg(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}')

    conv = True; ts = time.time()
    try:
        sol = newton_krylov(lambda x: F_of(x, gamma), x_warm,
                              f_tol=TOL, maxiter=MAXIT, method='lgmres',
                              callback=cb, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, gamma))))
    P_fp = sol.reshape((G,)*3)
    m = metrics(P_fp, u_arr)
    dt = time.time() - ts
    lg(f'  RESULT: conv={conv} ||F||={Finf:.3e} iters={cnt["n"]} '
       f'deficit={m["deficit"]:.4f} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({dt:.0f}s)')

    np.save(os.path.join(HERE, f'cdf_cube_NK_g{gamma:g}.npy'), P_fp)
    rec = dict(gamma=gamma, conv=conv, Finf=Finf, iters=cnt['n'], dt=dt,
                ic_metrics=m_ic, **m)
    all_results.append(rec)
    json.dump({'G':G,'NQ':NQ,'gammas':GAMMAS,'tol':TOL,'results':all_results,
                'u_arr':u_arr.tolist(), 'zeta_arr':zeta_arr.tolist()},
              open(os.path.join(HERE,'cdf_cube_NK_sweep.json'),'w'), indent=2, default=str)

    # auto push
    cmd = (f'cd /home/user/FIXED-POINT-FACTORY && '
           f'git add projects/REZN/solver_code/sigma_delta/cdf_cube_NK_g{gamma:g}.npy '
           f'projects/REZN/solver_code/sigma_delta/cdf_cube_NK_sweep.json '
           f'projects/REZN/solver_code/sigma_delta/cdf_cube_NK_sweep.log && '
           f'git commit -m "CDF-cube NK γ={gamma}: ||F||={Finf:.2e} slope={m["slope_T"]:.3f} '
           f'deficit={m["deficit"]:.3f} d_FR={m["d_FR"]:.3f} ({cnt["n"]} NK iters)" 2>&1 | tail -2 && '
           f'git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')
    os.system(cmd)

    # warm-start next γ from this FP
    x_warm = sol

lg(f'\nSWEEP DONE in {(time.time()-t_total)/60:.1f} min  '
   f'converged {sum(1 for r in all_results if r["conv"])}/{len(all_results)}')
