"""Solve Hermite operator at gamma=1, tau=1."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from hermite_op import phi_hermite, setup

G = 7
u_nodes, gh_weights, V, V_inv = setup(G)
U1, U2, U3 = np.meshgrid(u_nodes, u_nodes, u_nodes, indexing='ij')
TAU = 1.0; GAMMA = 1.0
def sg(x): return 1/(1+np.exp(-x))
P0 = sg(0.5*TAU*(U1+U2+U3))
T_full = TAU*(U1+U2+U3)

def F(x):
    return (phi_hermite(x.reshape(G,G,G), u_nodes, gh_weights, V, V_inv,
                          gamma=GAMMA, tau=TAU) - x.reshape(G,G,G)).ravel()

print(f'=== Hermite POU strict h=0, G={G} ===')
print(f'Initial F: {float(np.max(np.abs(F(P0.ravel())))):.3e}')

# Anderson
print('\nAnderson:')
x = P0.ravel().copy()
Xh, Gh = [], []; Fs = []; m = 10
x_best = x.copy(); f_best = float('inf')
t0 = time.time()
for it in range(50):
    t1 = time.time()
    Fv = F(x); gx = Fv + x
    f = float(np.max(np.abs(Fv))); Fs.append(f)
    if f < f_best: f_best = f; x_best = x.copy()
    print(f'  it {it+1:3d}: |F|={f:.3e}  ({time.time()-t1:.1f}s)')
    if f < 1e-15: break
    Xh.append(x.copy()); Gh.append(gx.copy())
    if len(Xh) > m: Xh.pop(0); Gh.pop(0)
    k = len(Xh)
    if k <= 1: x = gx
    else:
        DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
        R_k = Gh[k-1] - Xh[k-1]
        try:
            A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
            ga = np.linalg.solve(A, -DR.T @ R_k)
            DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
            x = Gh[k-1] + DG @ ga
        except: x = gx
print(f'Anderson best: {min(Fs):.3e} ({time.time()-t0:.1f}s)')

print('\nNK from Anderson best:')
nk_it = [0]
def cb(x, f):
    nk_it[0] += 1
    Fx = F(x)
    print(f'  NK it {nk_it[0]:3d}: |F|={float(np.max(np.abs(Fx))):.3e}', flush=True)
try:
    x_sol = newton_krylov(F, x_best, f_tol=1e-15, maxiter=15, callback=cb, verbose=False)
    f_nk = float(np.max(np.abs(F(x_sol))))
    print(f'NK final: {f_nk:.3e}')
except NoConvergence as e:
    x_sol = e.args[0]
    f_nk = float(np.max(np.abs(F(x_sol))))
    print(f'NK NoConv: {f_nk:.3e}')

# Statistics
P_fp = x_sol.reshape(G, G, G)
Pc = np.clip(P_fp, 1e-15, 1-1e-15)
L = np.log(Pc/(1-Pc)).ravel()
Tf = T_full.ravel()
slope = float(np.sum(L*Tf)/np.sum(Tf**2))
uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
ss = float(np.sum((L-L.mean())**2)); w = 0.0
for g in range(len(uT)):
    mask = (inv==g); w += float(np.sum((L[mask]-L[mask].mean())**2))
def_1to1 = w/ss
print(f'\n=== Hermite FP ===')
print(f'  F final: {min(min(Fs), f_nk):.3e}')
print(f'  slope alpha*: {slope:.4f}')
print(f'  def_1to1: {def_1to1:.4f}')
print(f'  P range: [{P_fp.min():.4f}, {P_fp.max():.4f}]')
np.save('/tmp/cheby_h0/fps_hermite/P_FP_hermite_g1.npy', P_fp) if False else None
import os
os.makedirs('/tmp/cheby_h0/fps_hermite', exist_ok=True)
np.save('/tmp/cheby_h0/fps_hermite/P_FP_hermite_g1.npy', P_fp)
print('saved')
