"""Gold test: reproduce prior session's machine-precision PR FP at τ=2, γ=0.1.

Uses numba operator + sigmoid lift + S₃×Z₂ symmetry, N=6.

Prior result (from analysis.tex, table at γ=0.1): slope=0.3641.
Goal here: confirm Chebyshev h=0 with lift+numba lands in same basin.
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

# Override params before importing operator
TAU_GOLD = 2.0
GAMMA_GOLD = 0.1

from cheby_numba import phi, U_NODES, LOBATTO, C_STRETCH, N, N_GRID, crra_clear
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU_GOLD * (U1 + U2 + U3)
print(f'Gold test: τ={TAU_GOLD}, γ={GAMMA_GOLD}, N={N}, c={C_STRETCH}')
print(f'  T field range: [{T_FIELD.min():.2f}, {T_FIELD.max():.2f}]')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

N_DOF = 1 + len(FREE_REPS)
REP_TO_IDX = {rep: i for i, rep in enumerate(FREE_REPS)}
ORBIT_POS, ORBIT_NEG = [], []
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3)); ORBIT_NEG.append(list(z2_s3))

def expand(x):
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx, rep in enumerate(FREE_REPS):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(FREE_REPS))
    for idx, rep in enumerate(FREE_REPS):
        vals_pos = [h_full[c] for c in ORBIT_POS[idx]]
        vals_neg = [-h_full[c] for c in ORBIT_NEG[idx]]
        h_small[idx] = float(np.mean(vals_pos + vals_neg))
    return np.concatenate([[alpha], h_small])

def phi_lifted(x):
    P = expand(x)
    P_new = phi(P, gamma=GAMMA_GOLD, tau=TAU_GOLD)
    return contract(P_new)

def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_FIELD.ravel(), y, 1); pr = a[0]*T_FIELD.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T_FIELD); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(x, n_iter=10, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton (gold lifted+numba): {Nu} unknowns', flush=True)
    F = phi_lifted(x) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_inf={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; mlist = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_lifted(xp) - xp - F) / eps_fd
        t_build = time.time() - ts
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100, None, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
            Fn = phi_lifted(xn) - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(expand(x)); ferrs.append(F_norm); mlist.append(m)
        print(f'  NewtIter {it+1:2d}  ||F||={F_norm:.3e}  aLS={alpha:.3g}  a_slope={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)
        if F_norm < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, mlist


if __name__ == '__main__':
    print('\n=== GOLD TEST: τ=2, γ=0.1 ===')
    print('Triggering JIT compile...', flush=True)
    t0 = time.time()
    _ = phi(np.full((G, G, G), 0.5), gamma=GAMMA_GOLD, tau=TAU_GOLD)
    print(f'  JIT: {time.time()-t0:.1f}s')

    # No-learning Bayes IC: each agent uses only own signal.
    # Matches the IC used in cheby_sym2 unlifted run (found PR basin).
    def sg_np(x): return 1.0/(1.0+np.exp(-x))
    P_IC_full = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu0 = sg_np(TAU_GOLD*U_NODES[i])
                mu1 = sg_np(TAU_GOLD*U_NODES[j])
                mu2 = sg_np(TAU_GOLD*U_NODES[k])
                P_IC_full[i,j,k] = crra_clear(mu0, mu1, mu2, GAMMA_GOLD)
    x0 = contract(P_IC_full)
    P_IC = expand(x0)
    print(f'\nIC: no-learning Bayes (each agent uses own signal only)')
    print(f'  α from contract: {x0[0]:.4f}')
    print(f'  P range: [{P_IC.min():.4f}, {P_IC.max():.4f}]')
    print(f'  IC metrics: {metrics(P_IC)}')

    print('\n=== Picard preconditioning (8 iters, ω=0.5) ===')
    x = x0.copy(); pferrs = []; pmetrics = []
    for it in range(8):
        ts = time.time()
        x_new = phi_lifted(x)
        ferr = float(np.max(np.abs(x_new - x))); pferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); pmetrics.append(m)
        print(f'  Picard it {it+1:2d}  ||F||={ferr:.3e}  a={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)

    print('\n=== Pure dense Newton ===')
    x_final, nferrs, nmetrics = dense_newton(x, n_iter=10, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal: α={x_final[0]:.4f}, metrics={metrics(P_final)}')
    print(f'Prior session result at τ=2, γ=0.1: slope ≈ 0.3641')

    np.save('/tmp/cheby_h0/P_final_gold.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_gold.npy', x_final)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU_GOLD, gamma=GAMMA_GOLD, c=C_STRETCH, n_dof=N_DOF),
        x_final=x_final.tolist(),
        picard=dict(ferrs=pferrs, metrics=pmetrics),
        newton=dict(ferrs=nferrs, metrics=nmetrics),
        final_metrics=metrics(P_final),
        final_alpha=float(x_final[0]),
        prior_session_slope=0.3641,
    ), open('/tmp/cheby_h0/results_gold.json','w'), indent=2, default=str)
    print('saved')
