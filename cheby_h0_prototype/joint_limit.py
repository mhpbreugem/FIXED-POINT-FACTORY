"""Joint (G, h) → (∞, 0) limit:
For each N in {6, 8, 10, 12}, find the smallest h that converges to
machine eps and record slope α* and deficit. As h shrinks, slope rises
toward the true h=0 REE slope.
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

from cheby_numba_kern_tab_N import phi_kern_tab_N, make_grid_N, make_p_grid

TAU = 1.0; GAMMA = 1.0
FIGS = '/tmp/cheby_h0/figs'

def sg(x): return 1/(1+np.exp(-x))

def fit_metrics(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    slope = float(np.sum(L*T.ravel()) / np.sum(T.ravel()**2))
    pred = slope*T.ravel() + np.mean(L - slope*T.ravel())
    res = np.sum((L - pred)**2); tot = np.sum((L - L.mean())**2)
    deficit = float(res/max(tot, 1e-30))
    return slope, deficit

def anderson_track(F_func, x0, n_iter=80, m=12, tol=1e-15):
    x = x0.copy(); Xh, Gh = [], []; Fs = []
    for it in range(n_iter):
        F = F_func(x); gx = F + x
        Ferr = float(np.max(np.abs(F))); Fs.append(Ferr)
        if Ferr < tol: break
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
    return Fs, x

# Sweep
N_LIST = [6, 8, 10, 12]
H_LIST = [0.5, 0.3, 0.2, 0.15, 0.10, 0.07]
G_p = 121

results = {}
print(f'{"N":>3} {"G":>3} {"h":>6} {"Ferr":>14} {"slope":>10} {"def":>10} {"t":>8}')
print('-'*70)
for N in N_LIST:
    G, lob, u, V_inv = make_grid_N(N)
    U1, U2, U3 = np.meshgrid(u, u, u, indexing='ij')
    T = TAU*(U1+U2+U3)
    P0 = sg(0.5*T)
    # Warmup
    _ = phi_kern_tab_N(P0, N, kernel_h=0.5, G_p=G_p, tau=TAU, gamma=GAMMA)
    for h in H_LIST:
        # Solve from cold
        F_func = lambda x_full: (phi_kern_tab_N(x_full.reshape(G,G,G), N,
                                                   kernel_h=h, G_p=G_p,
                                                   tau=TAU, gamma=GAMMA)
                                  - x_full.reshape(G,G,G)).ravel()
        x0 = P0.ravel().copy()
        t0 = time.time()
        Fs, x_and = anderson_track(F_func, x0, n_iter=80, tol=1e-15)
        dt = time.time() - t0
        Ferr_final = float(min(Fs))
        # Maybe finish with Newton-Krylov if Anderson stalled above 1e-10
        if Ferr_final > 1e-10 and Ferr_final < 1e-3:
            try:
                x_nk = newton_krylov(F_func, x_and, f_tol=1e-14,
                                       maxiter=30, verbose=False)
                Ferr_nk = float(np.max(np.abs(F_func(x_nk))))
                if Ferr_nk < Ferr_final:
                    Ferr_final = Ferr_nk; x_and = x_nk
            except NoConvergence as e:
                x_and = e.args[0]
                Ferr_final = float(np.max(np.abs(F_func(x_and))))
        P_fp = x_and.reshape(G, G, G)
        slope, deficit = fit_metrics(P_fp, T)
        key = f'N={N},h={h}'
        results[key] = dict(N=N, G=G, h=h, Ferr=Ferr_final, slope=slope,
                              deficit=deficit, t_solve=dt)
        print(f'{N:>3} {G:>3} {h:>6.2f} {Ferr_final:>14.3e} {slope:>10.4f} '
              f'{deficit:>10.4e} {dt:>7.1f}s')

# Save raw
json.dump(results, open('/tmp/cheby_h0/joint_limit.json', 'w'),
            indent=2, default=str)

# Plot the joint limit
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for N in N_LIST:
    hs = [r['h'] for r in results.values() if r['N'] == N and r['Ferr'] < 1e-10]
    ss = [r['slope'] for r in results.values() if r['N'] == N and r['Ferr'] < 1e-10]
    if hs:
        ax.plot(hs, ss, '-o', markersize=8, label=f'N={N} (G={N+1})')
ax.set_xlabel(r'kernel bandwidth $h$')
ax.set_ylabel(r'slope $\alpha^* = \langle \mathrm{logit}\,P, T\rangle / \langle T,T\rangle$')
ax.set_title(r'Slope $\alpha^*$ vs $h$ at each $N$ (only machine-$\varepsilon$-converged FPs)')
ax.grid(alpha=0.3); ax.legend(); ax.invert_xaxis()

ax = axes[1]
for N in N_LIST:
    hs = [r['h'] for r in results.values() if r['N'] == N and r['Ferr'] < 1e-10]
    ds = [r['deficit'] for r in results.values() if r['N'] == N and r['Ferr'] < 1e-10]
    if hs:
        ax.semilogy(hs, ds, '-s', markersize=8, label=f'N={N}')
ax.set_xlabel(r'$h$')
ax.set_ylabel('information deficit $1-R^2$')
ax.set_title(r'Deficit vs $h$ at each $N$')
ax.grid(alpha=0.3, which='both'); ax.legend(); ax.invert_xaxis()

ax = axes[2]
# Convergence floor (Ferr) heatmap
N_arr = np.array(N_LIST)
H_arr = np.array(H_LIST)
ferr_grid = np.full((len(N_LIST), len(H_LIST)), np.nan)
for i, N in enumerate(N_LIST):
    for j, h in enumerate(H_LIST):
        key = f'N={N},h={h}'
        if key in results:
            ferr_grid[i, j] = max(results[key]['Ferr'], 1e-18)
im = ax.pcolormesh(H_arr, N_arr, np.log10(ferr_grid),
                     shading='auto', cmap='RdYlGn_r', vmin=-17, vmax=-1)
ax.set_xlabel('h'); ax.set_ylabel('N')
ax.set_title(r'$\log_{10}\|F\|_\infty$ achieved')
plt.colorbar(im, ax=ax)
ax.invert_xaxis()

plt.suptitle('Joint $(N, h) \\to (\\infty, 0)$ limit for kernel-band Cheb-tab',
              fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGS}/joint_limit.png', dpi=140, bbox_inches='tight')
plt.close()
print('saved joint_limit.png')
