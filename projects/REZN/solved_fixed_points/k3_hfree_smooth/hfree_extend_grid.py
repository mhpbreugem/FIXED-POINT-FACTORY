"""(b) Extend the hfree_smooth strict-h=0 PR FP nail to higher G to verify
grid-convergence of the PR FP (commit 78c27216 nailed G=9; this tries G=13, 17).

Strategy: Newton-Krylov from interpolated-and-Picard-preconditioned warm-start.
G=9 warm-start: load P_nailed_G9.npy; interp to G=13, 13->17.
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp")
import numpy as np
import hfree_operator as H
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from scipy.interpolate import RegularGridInterpolator

UMAX=4.0; TAU=2.0; GAMMA=0.1; NQ=40; SUB=4
tau=np.full(3,TAU); gam=np.full(3,GAMMA); W=np.full(3,1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)

def metrics(P, ui):
    G=ui.size
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing="ij"); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    deficit=float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR=1.0/(1.0+np.exp(-T)); d_FR=float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))

def F_of(x, G, ui):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gnodes,gweights,tau,gam,W,SUB)-P).ravel()

def nail_NK(G, x0, ui, label='', f_tol=1e-7, maxiter=80):
    print(f'\n=== {label} G={G} NK ===', flush=True)
    print(f'  warm-start metrics: {metrics(x0.reshape((G,)*3), ui)}', flush=True)
    cnt = {'n': 0, 'hist': []}
    def cb(x, fx):
        cnt['n'] += 1
        fn = float(np.max(np.abs(fx)))
        cnt['hist'].append(fn)
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            print(f'    NK it {cnt["n"]:3d} ||F||={fn:.3e}', flush=True)
    conv = True
    t = time.time()
    try:
        sol = newton_krylov(lambda x: F_of(x, G, ui), x0,
                              f_tol=f_tol, maxiter=maxiter, method='lgmres',
                              callback=cb, verbose=False)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, G, ui))))
    m = metrics(sol.reshape((G,)*3), ui)
    print(f'  RESULT: conv={conv} ||F||={Finf:.3e} iters={cnt["n"]} '
          f'deficit={m["deficit"]:.4f} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}'
          f' ({time.time()-t:.0f}s)', flush=True)
    return sol.reshape((G,)*3), dict(conv=conv, Finf=Finf, iters=cnt['n'], hist=cnt['hist'], **m)

# JIT warmup
print('JIT warmup hfree_smooth...', flush=True); t=time.time()
ui9 = np.linspace(-UMAX, UMAX, 9)
P0 = np.load(os.path.join(HERE, 'P_nailed_G9.npy'))
_ = H.phi_hfree(P0, ui9, gnodes, gweights, tau, gam, W, SUB)
print(f'  warmup {time.time()-t:.0f}s', flush=True)

results = {}
# G=9 baseline (re-verify)
P9, r9 = nail_NK(9, P0.ravel(), ui9, label='re-verify G=9', f_tol=1e-9, maxiter=20)
np.save(os.path.join(HERE, 'P_nailed_G9_NK.npy'), P9)
results[9] = r9

# G=13 from interp of G=9
ui13 = np.linspace(-UMAX, UMAX, 13)
rgi9 = RegularGridInterpolator((ui9, ui9, ui9), P9, bounds_error=False, fill_value=None)
A,B,C = np.meshgrid(ui13, ui13, ui13, indexing='ij')
P13_init = rgi9(np.column_stack([A.ravel(), B.ravel(), C.ravel()])).reshape((13,)*3)
# Picard preconditioning
for _ in range(20):
    P13_init = 0.7*P13_init + 0.3*H.phi_hfree(P13_init, ui13, gnodes, gweights, tau, gam, W, SUB)
P13, r13 = nail_NK(13, P13_init.ravel(), ui13, label='G=13 from interp+Picard', f_tol=1e-7, maxiter=60)
np.save(os.path.join(HERE, 'P_nailed_G13_extend.npy'), P13)
results[13] = r13
json.dump(results, open(os.path.join(HERE, 'hfree_extend_grid.json'),'w'), indent=2, default=str)
print('checkpoint after G=13', flush=True)

# G=17 from interp of G=13
ui17 = np.linspace(-UMAX, UMAX, 17)
rgi13 = RegularGridInterpolator((ui13, ui13, ui13), P13, bounds_error=False, fill_value=None)
A,B,C = np.meshgrid(ui17, ui17, ui17, indexing='ij')
P17_init = rgi13(np.column_stack([A.ravel(), B.ravel(), C.ravel()])).reshape((17,)*3)
for _ in range(20):
    P17_init = 0.7*P17_init + 0.3*H.phi_hfree(P17_init, ui17, gnodes, gweights, tau, gam, W, SUB)
P17, r17 = nail_NK(17, P17_init.ravel(), ui17, label='G=17 from interp+Picard', f_tol=1e-7, maxiter=80)
np.save(os.path.join(HERE, 'P_nailed_G17_extend.npy'), P17)
results[17] = r17
json.dump(results, open(os.path.join(HERE, 'hfree_extend_grid.json'),'w'), indent=2, default=str)

print('\n=== GRID-CONVERGENCE SUMMARY ===')
print(f'  G  | deficit   | slope    | d_FR     | ||F||')
for G in (9, 13, 17):
    r = results[G]
    print(f'  {G:2d} | {r["deficit"]:.4f}   | {r["slope_T"]:.4f}  | {r["d_FR"]:.4f}  | {r["Finf"]:.2e}')
print('DONE')
