"""Descend from CARA-limit (γ large, FR ansatz) toward γ=0.3 region.

Hellwig: CARA noiseless K=3 → FR (slope=1, deficit=0). So at large γ the
FR ansatz P = sigmoid(τ·Σu) should be a strict FP. Descend in γ:
  100 → 30 → 10 → 5 → 3 → 1 → 0.7 → 0.5 → 0.4 → 0.35 → 0.3 → 0.27

If γ=0.3 nails from this side, the apparent fold at γ≈0.26-0.30 is a
one-sided warm-start artifact (UP from γ=0.1 hits a basin discontinuity).
If γ=0.3 fails from this side too, the fold is genuine.
"""
import os, sys, time, json
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/tmp')
import numpy as np
import hfree_operator as H
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError: from scipy.optimize._nonlin import NoConvergence

UMAX=4.0; TAU=2.0; NQ=40; SUB=4; G=9
ui = np.linspace(-UMAX, UMAX, G)
W = np.full(3, 1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU*(U1+U2+U3)
P_FR = 1.0/(1.0+np.exp(-T))

GAMMAS = [100.0, 30.0, 10.0, 5.0, 3.0, 1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.27, 0.25, 0.22, 0.20, 0.18]
TOL = 1e-11

def metrics(P):
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    defi=float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
    dFR=float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=defi, d_FR=dFR, slope_T=float(a[0]))

def F_of(x, gv):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gnodes,gweights,np.full(3,TAU),np.full(3,gv),W,SUB)-P).ravel()

LOG = os.path.join(HERE,'cara_side_descent.log'); open(LOG,'w').close()
def lg(m):
    line=f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True); open(LOG,'a').write(line+'\n')

# warmup
lg('JIT warmup'); t=time.time()
_ = H.phi_hfree(P_FR.copy(), ui, gnodes, gweights, np.full(3,TAU), np.full(3,100.0), W, SUB)
lg(f'  warmup {time.time()-t:.0f}s')

# IC = FR ansatz (CARA limit)
P_warm = P_FR.copy()
F0_at_FR_100 = float(np.max(np.abs(F_of(P_warm.ravel(), 100.0))))
lg(f'IC: FR ansatz, ||F||@γ=100 = {F0_at_FR_100:.3e}')

results = []
OUT = os.path.join(HERE, 'cara_descent')
os.makedirs(OUT, exist_ok=True)

for gv in GAMMAS:
    lg(f'\n--- γ={gv} ---')
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            lg(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e}')
    ts = time.time()
    try:
        sol = newton_krylov(lambda x: F_of(x, gv), P_warm.ravel(),
                              f_tol=TOL, maxiter=80, method='lgmres', callback=cb)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, gv))))
    m = metrics(sol.reshape((G,)*3))
    acc = (Finf <= TOL)
    lg(f'  {"ACC" if acc else "REJ"} γ={gv}: conv={conv} ||F||={Finf:.3e} '
       f'slope={m["slope_T"]:.5f} deficit={m["deficit"]:.5f} d_FR={m["d_FR"]:.4f} '
       f'({cnt["n"]} iters, {time.time()-ts:.0f}s)')
    if acc:
        np.save(os.path.join(OUT, f'P_g{gv:g}.npy'), sol.reshape((G,)*3))
        P_warm = sol.reshape((G,)*3)  # warm next step
    results.append(dict(gamma=gv, conv=conv, Finf=Finf, iters=cnt['n'], acc=acc, **m))
    json.dump(results, open(os.path.join(HERE,'cara_side_descent.json'),'w'), indent=2, default=str)
    # auto-push
    if acc:
        os.system(f'cd /home/user/FIXED-POINT-FACTORY && '
                  f'git add projects/REZN/solved_fixed_points/k3_hfree_smooth/cara_descent/P_g{gv:g}.npy '
                  f'projects/REZN/solved_fixed_points/k3_hfree_smooth/cara_side_descent.json && '
                  f'git commit -m "CARA-side γ={gv:g}: ||F||={Finf:.2e} '
                  f'slope={m["slope_T"]:.5f} deficit={m["deficit"]:.5f}" 2>&1 | tail -2 && '
                  f'git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')

lg(f'\nDONE: {sum(1 for r in results if r["acc"])}/{len(results)} accepted')
