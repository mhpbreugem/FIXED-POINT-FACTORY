"""G=15 hfree_smooth: nail anchor at γ=0.1 from G=9 warm-start, then adaptive
γ-sweep with strict tol=1e-11. Same architecture as G=9 sweep but at finer
discretization (3375 unknowns vs 729). Tests whether the fold at γ≈0.27 at
G=9 shrinks/disappears at G=15.
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
from scipy.interpolate import RegularGridInterpolator

UMAX=4.0; TAU=2.0; NQ=40; SUB=4; G=15
ui = np.linspace(-UMAX, UMAX, G)
W = np.full(3, 1.0)
gnodes, gweights = H.gauss_legendre(NQ, -UMAX, UMAX)
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU*(U1+U2+U3)
P_FR_arr = 1.0/(1.0+np.exp(-T))

TOL = 1e-11
MAXIT = 80

def metrics(P):
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    defi=float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
    dFR=float(np.sqrt(np.mean((P-P_FR_arr)**2)))
    return dict(deficit=defi, d_FR=dFR, slope_T=float(a[0]))

def F_of(x, gv):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gnodes,gweights,np.full(3,TAU),np.full(3,gv),W,SUB)-P).ravel()

LOG = os.path.join(HERE, 'gamma_sweep_G15.log'); open(LOG,'w').close()
def lg(m):
    line=f'[{time.strftime("%H:%M:%S")}] {m}'; print(line, flush=True); open(LOG,'a').write(line+'\n')

lg('JIT warmup...'); t=time.time()
_ = H.phi_hfree(P_FR_arr.copy(), ui, gnodes, gweights, np.full(3,TAU), np.full(3,0.1), W, SUB)
lg(f'  warmup {time.time()-t:.0f}s')

# ANCHOR: interpolate G=9 hfree machine-prec FP to G=15
P9 = np.load(os.path.join(HERE,'..','..','solver_code/sigma_delta/hfree_G9_machine_prec.npy'))
ui9 = np.linspace(-UMAX, UMAX, 9)
rgi = RegularGridInterpolator((ui9, ui9, ui9), P9, bounds_error=False, fill_value=None)
A, B, C = np.meshgrid(ui, ui, ui, indexing='ij')
P_anchor = rgi(np.column_stack([A.ravel(), B.ravel(), C.ravel()])).reshape((G,G,G))
F0 = float(np.max(np.abs(F_of(P_anchor.ravel(), 0.1))))
m0 = metrics(P_anchor)
lg(f'G=15 anchor (interp from G=9): ||F||={F0:.3e}, slope={m0["slope_T"]:.4f}, '
   f'deficit={m0["deficit"]:.4f}, d_FR={m0["d_FR"]:.4f}')

def nail(gv, P_warm):
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            lg(f'    NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e}')
    try:
        sol = newton_krylov(lambda x: F_of(x, gv), P_warm.ravel(),
                              f_tol=TOL, maxiter=MAXIT, method='lgmres', callback=cb)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol, gv))))
    m = metrics(sol.reshape((G,)*3))
    return sol.reshape((G,)*3), conv, Finf, m, cnt['n']

# Nail anchor
lg('\n=== ANCHOR γ=0.1 G=15 ===')
ts = time.time()
P_anchor_n, conv, Finf, m_a, iters = nail(0.1, P_anchor)
lg(f'  RESULT γ=0.1 anchor: conv={conv} ||F||={Finf:.3e} '
   f'slope={m_a["slope_T"]:.5f} deficit={m_a["deficit"]:.5f} d_FR={m_a["d_FR"]:.4f} '
   f'({iters} iters, {time.time()-ts:.0f}s)')

OUT = os.path.join(HERE, 'gamma_sweep_G15')
os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, 'P_g0.1.npy'), P_anchor_n)
all_records = [dict(gamma=0.1, conv=conv, Finf=Finf, iters=iters, **m_a)]

# Push anchor
os.system(f'cd /home/user/FIXED-POINT-FACTORY && git add projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_G15/P_g0.1.npy '
          f'projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_G15.json && '
          f'git commit -m "G=15 anchor γ=0.1: ||F||={Finf:.2e} slope={m_a["slope_T"]:.5f} '
          f'deficit={m_a["deficit"]:.5f}" 2>&1 | tail -1 && '
          f'git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -1')

def push_step(gv, conv, Finf, m, iters):
    cmd = (f'cd /home/user/FIXED-POINT-FACTORY && '
           f'git add projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_G15/P_g{gv:g}.npy '
           f'projects/REZN/solved_fixed_points/k3_hfree_smooth/gamma_sweep_G15.json && '
           f'git commit -m "G=15 γ={gv:g}: ||F||={Finf:.2e} slope={m["slope_T"]:.5f} '
           f'deficit={m["deficit"]:.5f} ({iters} NK iters)" 2>&1 | tail -1 && '
           f'git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -1')
    os.system(cmd)

def sweep(direction, init_factor):
    lg(f'\n=== SWEEP {direction.upper()} G=15 ===')
    gv = 0.1; P_warm = P_anchor_n.copy(); factor = init_factor
    while True:
        gv_try = gv * factor
        # Limits
        if (direction=='down' and gv_try < 5e-4) or (direction=='up' and gv_try > 100.0):
            lg(f'  reached limit, stop'); break
        ts = time.time()
        P_fp, conv, Finf, m, iters = nail(gv_try, P_warm)
        if conv and Finf <= TOL:
            lg(f'  ACC  γ={gv_try:g}: ||F||={Finf:.3e} slope={m["slope_T"]:.5f} '
               f'deficit={m["deficit"]:.5f} d_FR={m["d_FR"]:.4f} '
               f'({iters} iters, {time.time()-ts:.0f}s, factor={factor:.3f})')
            np.save(os.path.join(OUT, f'P_g{gv_try:g}.npy'), P_fp)
            all_records.append(dict(gamma=gv_try, conv=conv, Finf=Finf, iters=iters, factor=factor, **m))
            json.dump(all_records, open(os.path.join(HERE,'gamma_sweep_G15.json'),'w'), indent=2, default=str)
            push_step(gv_try, conv, Finf, m, iters)
            P_warm = P_fp; gv = gv_try
            factor = min(init_factor, factor*1.05)
        else:
            # halve step
            if direction == 'down':
                factor_new = 1.0 - 0.5*(1.0 - factor)
            else:
                factor_new = 1.0 + 0.5*(factor - 1.0)
            if abs(factor_new - 1.0) < 0.005:
                lg(f'  REJ  γ={gv_try:g}: ||F||={Finf:.3e}, min step. STOP.')
                break
            lg(f'  REJ  γ={gv_try:g}: ||F||={Finf:.3e} ({iters} iters), halve {factor:.3f}->{factor_new:.3f}')
            factor = factor_new

sweep('down', 0.7)
sweep('up', 1.5)
lg(f'\nDONE: {sum(1 for r in all_records if r.get("Finf",1) <= TOL)}/{len(all_records)} accepted')
