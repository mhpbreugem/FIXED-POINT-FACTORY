"""Adaptive-step γ-sweep on hfree G=9 PR FP. Reject any step with ||F||>1e-11
and halve the step size from the previous accepted γ.

Down: γ=0.1 → ... → 0.001 (multiplicative)
Up:   γ=0.1 → ... → 100 (multiplicative)

Per step:
  γ_try = γ_prev * factor
  NK from P[γ_prev]; if ||F||<=1e-11 accept and continue with same factor
  else: halve factor, retry from γ_prev. Stop if factor < 1.01 (no progress).

Each accepted step is auto-committed + pushed.
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

TOL = 1e-11
MAXIT_NK = 60

def metrics(P):
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing='ij'); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    defi=float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
    PFR=1/(1+np.exp(-T)); dFR=float(np.sqrt(np.mean((P-PFR)**2)))
    return dict(deficit=defi, d_FR=dFR, slope_T=float(a[0]))

def F_of(x, gv):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gnodes,gweights,np.full(3,TAU),np.full(3,gv),W,SUB)-P).ravel()

LOG = os.path.join(HERE, 'gamma_sweep_adaptive.log')
open(LOG,'w').close()
def lg(m):
    line = f'[{time.strftime("%H:%M:%S")}] {m}'
    print(line, flush=True); open(LOG,'a').write(line+'\n')

# Anchor
P_anchor = np.load(os.path.join(HERE,'..','..','solver_code/sigma_delta/hfree_G9_machine_prec.npy'))
m0 = metrics(P_anchor)
lg(f'Anchor γ=0.1: slope={m0["slope_T"]:.4f} deficit={m0["deficit"]:.4f} d_FR={m0["d_FR"]:.4f}')

lg('JIT warmup...')
t=time.time()
_ = H.phi_hfree(P_anchor, ui, gnodes, gweights, np.full(3,TAU), np.full(3,0.1), W, SUB)
lg(f'  warmup {time.time()-t:.0f}s')

OUT = os.path.join(HERE,'gamma_sweep')
os.makedirs(OUT, exist_ok=True)
all_records = [dict(gamma=0.1, conv=True, Finf=9.4e-16, iters=0, **m0)]

def try_step(gv, P_warm):
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
    try:
        sol = newton_krylov(lambda x: F_of(x,gv), P_warm.ravel(),
                              f_tol=TOL, maxiter=MAXIT_NK, method='lgmres', callback=cb)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, gv))))
    m = metrics(sol.reshape((G,)*3))
    return sol.reshape((G,)*3), conv, Finf, m, cnt['n']

def push_commit(gv, conv, Finf, m, iters):
    cmd = (f'cd /home/user/FIXED-POINT-FACTORY && '
           f'git add projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep/P_g{gv:g}.npy '
           f'projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_adaptive.json && '
           f'git commit -m "adaptive γ={gv:g}: ||F||={Finf:.2e} '
           f'slope={m["slope_T"]:.5f} deficit={m["deficit"]:.5f} ({iters} NK iters {"ACC" if conv else "REJ"})" '
           f'2>&1 | tail -2 && git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')
    os.system(cmd)

def sweep_direction(direction):
    """direction: 'down' (factor<1) or 'up' (factor>1)."""
    if direction == 'down':
        gv_target_lo = 1e-3; init_factor = 0.7  # 30% step
        cmp = lambda g: g >= gv_target_lo
    else:
        gv_target_hi = 100.0; init_factor = 1.5  # 50% step
        cmp = lambda g: g <= gv_target_hi
    lg(f'\n=== SWEEP {direction.upper()} ===')
    gv = 0.1
    P_warm = P_anchor.copy()
    factor = init_factor
    while True:
        gv_try = gv * factor
        if (direction=='down' and gv_try < gv_target_lo*0.5) or (direction=='up' and gv_try > gv_target_hi*1.5):
            lg(f'  reached limit, stop')
            break
        ts = time.time()
        P_fp, conv, Finf, m, iters = try_step(gv_try, P_warm)
        if conv and Finf <= TOL:
            lg(f'  ACC  γ={gv_try:g}: ||F||={Finf:.3e} slope={m["slope_T"]:.5f} '
               f'deficit={m["deficit"]:.5f} d_FR={m["d_FR"]:.4f} ({iters} iters, {time.time()-ts:.0f}s, factor={factor:.3f})')
            np.save(os.path.join(OUT, f'P_g{gv_try:g}.npy'), P_fp)
            all_records.append(dict(gamma=gv_try, conv=conv, Finf=Finf, iters=iters, factor=factor, **m))
            json.dump(all_records, open(os.path.join(HERE,'gamma_sweep_adaptive.json'),'w'), indent=2, default=str)
            push_commit(gv_try, conv, Finf, m, iters)
            P_warm = P_fp; gv = gv_try
            # try to grow factor slightly back toward initial
            if direction == 'down':
                factor = min(init_factor, factor*1.05)
            else:
                factor = min(init_factor, factor*1.05)  # cap at init
        else:
            # REJECT, halve step
            if direction == 'down':
                factor_new = 1.0 - 0.5*(1.0 - factor)  # closer to 1
            else:
                factor_new = 1.0 + 0.5*(factor - 1.0)
            min_factor_gap = 0.005  # 0.5% minimum step
            if abs(factor_new - 1.0) < min_factor_gap:
                lg(f'  REJ  γ={gv_try:g}: ||F||={Finf:.3e}, halving below min step. STOP.')
                break
            lg(f'  REJ  γ={gv_try:g}: ||F||={Finf:.3e} ({iters} iters), halve {factor:.3f}->{factor_new:.3f}, retry')
            factor = factor_new

# Save anchor first
np.save(os.path.join(OUT, 'P_g0.1.npy'), P_anchor)

sweep_direction('down')
sweep_direction('up')

lg(f'\nDONE total {len(all_records)} accepted γ-points')
