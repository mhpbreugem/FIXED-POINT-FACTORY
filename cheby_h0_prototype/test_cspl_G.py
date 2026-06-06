"""Test if C^2 cubic spline reaches machine eps at higher G."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from lin_cdf_cspl import phi_lin_cspl, make_cdf_uniform_grid, make_p_grid

def sg(x): return 1/(1+np.exp(-x))

def anderson(F_func, x0, n_iter=80):
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

print(f'{"G":>3} {"NQ":>4} {"And":>12} {"+NK":>12} {"slope":>9} {"def_1to1":>10} {"t":>7}',
      flush=True)
results = {}
for G in [7, 11, 15, 21]:
    u_grid = make_cdf_uniform_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing='ij')
    T_full = U1+U2+U3
    P_init = sg(0.5*T_full)
    # Warmup
    for nq in [16, 24, 32]:
        if nq <= 2*G:
            _ = phi_lin_cspl(P_init, u_grid, NQ=nq, G_p=121, gamma=1.0)
    for NQ in [16, 24, 32]:
        if NQ > 2*G: continue
        F_func = lambda x_flat, NQ=NQ, G=G: \
            (phi_lin_cspl(x_flat.reshape(G,G,G), u_grid, NQ=NQ, G_p=121,
                              tau=1.0, gamma=1.0)
              - x_flat.reshape(G,G,G)).ravel()
        x0 = P_init.ravel()
        t0 = time.time()
        Fs, x_a = anderson(F_func, x0, n_iter=60)
        try:
            x_nk = newton_krylov(F_func, x_a, f_tol=1e-15, maxiter=20, verbose=False)
            f_nk = float(np.max(np.abs(F_func(x_nk))))
            if f_nk < min(Fs): x_a = x_nk
        except NoConvergence as e:
            x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
        P_fp = x_a.reshape(G, G, G)
        Pc = np.clip(P_fp, 1e-15, 1-1e-15)
        L = np.log(Pc/(1-Pc)).ravel(); Tf = T_full.ravel()
        s = float(np.sum(L*Tf)/np.sum(Tf**2))
        uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
        ss_tot = float(np.sum((L - L.mean())**2)); within = 0.0
        for g in range(len(uT)):
            mask = (inv == g)
            within += float(np.sum((L[mask] - L[mask].mean())**2))
        def_1to1 = within/ss_tot
        results[f'G={G},NQ={NQ}'] = dict(G=G, NQ=NQ, anderson=float(min(Fs)),
                                            nk=f_nk, slope=s, def_1to1=def_1to1,
                                            t=time.time()-t0)
        print(f'{G:>3d} {NQ:>4d} {min(Fs):>12.3e} {f_nk:>12.3e} {s:>9.4f} '
              f'{def_1to1:>10.4f} {time.time()-t0:>6.1f}s', flush=True)

import json
json.dump(results, open('/tmp/cheby_h0/cspl_G_sweep.json', 'w'),
            indent=2, default=str)
print('saved')
