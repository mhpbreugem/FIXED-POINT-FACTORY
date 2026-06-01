"""γ-sweep on hfree_smooth G=9 PR FP via small-step continuation.

Start from the machine-precision FP at γ=0.1 (slope=0.364, deficit=0.172).
Sweep:
  DOWN: γ=0.1 → 0.05 → 0.02 → 0.01 → 0.005 → 0.002 → 0.001  (toward strong PR)
  UP:   γ=0.1 → 0.15 → 0.2 → 0.3 → 0.5 → 1 → 3 → 10 → 30 → 100  (toward CARA/FR)

Each step: NK from previous γ's FP with f_tol=1e-12, maxiter=40.

Auto-commits + pushes per γ.
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

UMAX = 4.0; TAU = 2.0; NQ = 40; SUB = 4
G = 9
ui = np.linspace(-UMAX, UMAX, G)
W = np.full(3, 1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)

GAMMAS_DOWN = [0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
GAMMAS_UP = [0.12, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]

def metrics(P):
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing='ij'); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    deficit=float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR=1.0/(1.0+np.exp(-T)); d_FR=float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))

def F_of(x, gamma_val):
    P=x.reshape((G,G,G))
    tau=np.full(3,TAU); gam=np.full(3,gamma_val)
    return (H.phi_hfree(P,ui,gnodes,gweights,tau,gam,W,SUB)-P).ravel()

LOG = os.path.join(HERE, 'gamma_sweep_warmstart.log')
open(LOG, 'w').close()
def lg(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True); open(LOG,'a').write(line+'\n')

# Load machine-precision γ=0.1 FP as anchor
P_anchor = np.load(os.path.join(HERE, '..', '..', 'solver_code/sigma_delta/hfree_G9_machine_prec.npy'))
m0 = metrics(P_anchor)
lg(f'Anchor γ=0.1: ||F||=verified machine-prec, slope={m0["slope_T"]:.4f}, '
   f'deficit={m0["deficit"]:.4f}, d_FR={m0["d_FR"]:.4f}')

# JIT warmup
lg('JIT warmup...')
t=time.time()
_ = H.phi_hfree(P_anchor, ui, gnodes, gweights, np.full(3,TAU), np.full(3,0.1), W, SUB)
lg(f'  warmup {time.time()-t:.0f}s')

def nail(gamma_val, P_warm, f_tol=1e-12, maxiter=40):
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 3 == 0 or cnt['n'] == 1:
            lg(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e}')
    conv = True
    try:
        sol = newton_krylov(lambda x: F_of(x, gamma_val), P_warm.ravel(),
                              f_tol=f_tol, maxiter=maxiter, method='lgmres', callback=cb)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, gamma_val))))
    m = metrics(sol.reshape((G,)*3))
    return sol.reshape((G,)*3), conv, Finf, m, cnt['n'], cnt['hist']

def push_step(gamma_val, conv, Finf, m, iters):
    fname = f'P_g{gamma_val:g}.npy'
    cmd = (f'cd /home/user/FIXED-POINT-FACTORY && '
           f'git add projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep/{fname} '
           f'projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep.json '
           f'projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_warmstart.log && '
           f'git commit -m "γ={gamma_val:g} hfree G=9: ||F||={Finf:.2e} slope={m["slope_T"]:.4f} '
           f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f} (NK {iters} iters {"conv" if conv else "soft"})" '
           f'2>&1 | tail -2 && git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')
    os.system(cmd)

OUT = os.path.join(HERE, 'gamma_sweep')
os.makedirs(OUT, exist_ok=True)

# Save anchor
np.save(os.path.join(OUT, 'P_g0.1.npy'), P_anchor)
m_anchor = metrics(P_anchor)
all_results = [dict(gamma=0.1, conv=True, Finf=9.4e-16, iters=0, **m_anchor)]

# Sweep DOWN
lg('\n=== SWEEP DOWN ===')
P_warm = P_anchor.copy()
for gv in GAMMAS_DOWN:
    lg(f'\n--- γ={gv} (down) ---')
    ts = time.time()
    P_fp, conv, Finf, m, iters, hist = nail(gv, P_warm)
    lg(f'  RESULT γ={gv}: conv={conv} ||F||={Finf:.3e} iters={iters} '
       f'slope={m["slope_T"]:.4f} deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.0f}s)')
    np.save(os.path.join(OUT, f'P_g{gv:g}.npy'), P_fp)
    all_results.append(dict(gamma=gv, conv=conv, Finf=Finf, iters=iters, **m))
    json.dump(all_results, open(os.path.join(HERE,'gamma_sweep.json'),'w'), indent=2, default=str)
    push_step(gv, conv, Finf, m, iters)
    P_warm = P_fp  # warm for next-lower γ

# Sweep UP
lg('\n=== SWEEP UP ===')
P_warm = P_anchor.copy()
for gv in GAMMAS_UP:
    lg(f'\n--- γ={gv} (up) ---')
    ts = time.time()
    P_fp, conv, Finf, m, iters, hist = nail(gv, P_warm)
    lg(f'  RESULT γ={gv}: conv={conv} ||F||={Finf:.3e} iters={iters} '
       f'slope={m["slope_T"]:.4f} deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.0f}s)')
    np.save(os.path.join(OUT, f'P_g{gv:g}.npy'), P_fp)
    all_results.append(dict(gamma=gv, conv=conv, Finf=Finf, iters=iters, **m))
    json.dump(all_results, open(os.path.join(HERE,'gamma_sweep.json'),'w'), indent=2, default=str)
    push_step(gv, conv, Finf, m, iters)
    P_warm = P_fp

lg('\nDONE')
