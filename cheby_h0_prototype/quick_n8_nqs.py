"""Quick test: N=8 POU+chebroots, try several NQs. Print as we go."""
import sys, time, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import C_STRETCH
from cheby_numba_kern_tab_N import make_grid_N

TAU = 1.0; GAMMA = 1.0
def sg(x): return 1/(1+np.exp(-x))
def fit(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    return s, float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))

N = 8
G, lob, u_n, V_inv = make_grid_N(N)
U1, U2, U3 = np.meshgrid(u_n, u_n, u_n, indexing='ij')
T = TAU*(U1+U2+U3)
P0 = sg(0.5*T)
print(f'N={N}, G={G}, total cells = {G**3}, sym DOFs = ?', flush=True)
G_p = 121
p_grid = make_p_grid(G_p)

def phi(P, NQ):
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_inv, lob, u_n, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)

print('JIT warmup at NQ=16...', flush=True)
t0 = time.time(); _ = phi(P0, 16)
print(f'  done {time.time()-t0:.1f}s', flush=True)

def anderson(F_func, x0, n_iter=60):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    x_best = x.copy(); f_best = float('inf')
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        f = float(np.max(np.abs(F))); Fs.append(f)
        if f < f_best: f_best = f; x_best = x.copy()
        if f < 1e-15: break
        Xh.append(x.copy()); Gh.append(gx.copy())
        if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
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
    return Fs, x_best

OUT = '/tmp/cheby_h0/quick_n8_nqs.json'
results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

print(f'\n{"NQ":>4} {"And":>14} {"NK":>14} {"slope":>8} {"def":>8} {"t":>6}',
      flush=True)
for NQ in [16, 14, 18, 20, 22, 12, 24]:  # try sweet candidates first
    key = f'NQ={NQ}'
    if key in results: continue
    print(f'NQ={NQ}: ', end='', flush=True)
    t0 = time.time()
    F_func = lambda x_flat, NQ=NQ: (phi(x_flat.reshape(G,G,G), NQ)
                                       - x_flat.reshape(G,G,G)).ravel()
    if NQ != 16:
        _ = phi(P0, NQ)  # JIT for this NQ
    Fs, x_a = anderson(F_func, P0.ravel(), n_iter=60)
    print(f'And={min(Fs):.3e} ', end='', flush=True)
    try:
        x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=15, verbose=False)
        f_nk = float(np.max(np.abs(F_func(x_nk))))
        if f_nk < min(Fs): x_a = x_nk
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
    print(f'NK={f_nk:.3e}', flush=True)
    P_fp = x_a.reshape(G, G, G)
    s, d = fit(P_fp, T)
    results[key] = dict(NQ=NQ, anderson=float(min(Fs)), nk=f_nk,
                          slope=s, deficit=d, t=time.time()-t0)
    json.dump(results, open(OUT, 'w'), indent=2, default=str)

print('\n=== Results at N=8 ===')
for nq in sorted(int(k.split('=')[1]) for k in results.keys()):
    r = results[f'NQ={nq}']
    print(f'  NQ={nq:>3d}: And={r["anderson"]:.3e}, NK={r["nk"]:.3e}, '
          f'slope={r["slope"]:.4f}, def={r["deficit"]:.4f}')
