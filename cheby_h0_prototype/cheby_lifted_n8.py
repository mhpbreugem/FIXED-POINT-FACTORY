"""High-resolution N=8 run: lifted+numba+symmetric Newton at G=9.

Reuses the numba phi_jit kernel (which is parametric in n_grid). Builds an
independent symmetric-orbit enumeration for G=9 since cheby_sym2 is tied to G=7.

τ=1, γ=1 — same parameters as the N=6 headline run. Goal: smaller P-cell
residual via more spectral modes.
"""
import os, sys, time, math, json, itertools
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_numba import phi_jit, crra_clear_jit

# N=8 setup
N8 = 8
G8 = N8 + 1
C_STRETCH = 2.0
TAU = 1.0
GAMMA = 1.0
NQ = 12

LOBATTO8 = -np.cos(np.pi * np.arange(G8) / N8)
U_NODES8 = C_STRETCH * np.arctanh(np.clip(LOBATTO8, -0.9999, 0.9999))
# Vandermonde at N=8
V8 = np.empty((G8, G8))
for j in range(G8):
    x = LOBATTO8[j]
    V8[j, 0] = 1.0; V8[j, 1] = x
    for k in range(1, G8-1):
        V8[j, k+1] = 2*x*V8[j, k] - V8[j, k-1]
V_INV8 = np.linalg.inv(V8)
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(NQ)

# S₃×Z₂ symmetric enumeration at G=9
def orbit_Z2(i, j, k): return (G8-1-i, G8-1-j, G8-1-k)
def full_orbit(i, j, k):
    s3 = set(itertools.permutations((i, j, k)))
    z2 = orbit_Z2(i, j, k)
    z2_s3 = set(itertools.permutations(z2))
    return s3, z2_s3
def is_z2_fixed(i, j, k):
    s3 = set(itertools.permutations((i, j, k)))
    z2 = orbit_Z2(i, j, k)
    return z2 in s3

seen = set(); free_reps, fixed_reps = [], []
for i in range(G8):
    for j in range(i, G8):
        for k in range(j, G8):
            if (i, j, k) in seen: continue
            s3, z2_s3 = full_orbit(i, j, k)
            combined = s3 | z2_s3
            for cell in combined:
                seen.add(tuple(sorted(cell)))
            if is_z2_fixed(i, j, k):
                fixed_reps.append((i, j, k))
            else:
                free_reps.append((i, j, k))

print(f'N=8 symmetric enumeration at G={G8}:')
print(f'  Free DOFs: {len(free_reps)}')
print(f'  Fixed (forced to 0.5): {len(fixed_reps)}: {fixed_reps}')
print(f'  Total reps: {len(free_reps)+len(fixed_reps)} (full cube: {G8**3})')

ORBIT_POS, ORBIT_NEG = [], []
for rep in free_reps:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3))
    ORBIT_NEG.append(list(z2_s3))

# T-field on N=8 grid
U1, U2, U3 = np.meshgrid(U_NODES8, U_NODES8, U_NODES8, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)
print(f'  T-field range: [{T_FIELD.min():.2f}, {T_FIELD.max():.2f}]')

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

N_DOF = 1 + len(free_reps)
print(f'  DOFs: 1 (α) + {len(free_reps)} (h) = {N_DOF}', flush=True)

def expand(x):
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G8, G8, G8))
    for idx in range(len(free_reps)):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(free_reps))
    for idx in range(len(free_reps)):
        vals = [h_full[c] for c in ORBIT_POS[idx]] + [-h_full[c] for c in ORBIT_NEG[idx]]
        h_small[idx] = float(np.mean(vals))
    return np.concatenate([[alpha], h_small])

def phi_call(P):
    return phi_jit(P, V_INV8, LOBATTO8, GL_NODES, GL_WEIGHTS, TAU, GAMMA, C_STRETCH, G8, NQ)

def phi_lift(x):
    return contract(phi_call(expand(x)))

def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_FIELD.ravel(), y, 1); pr = a[0]*T_FIELD.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T_FIELD); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton(x, n_iter=6, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton N=8: {Nu} unknowns', flush=True)
    F = phi_lift(x) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_inf={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; mlist = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((Nu, Nu))
        for j in range(Nu):
            xp = x.copy(); xp[j] += eps_fd
            J[:, j] = (phi_lift(xp) - xp - F) / eps_fd
        t_build = time.time() - ts
        try: dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (x, F_norm, F, 0.0)
        for _ in range(20):
            xn = x + alpha*dx
            Fn = phi_lift(xn) - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(expand(x)); ferrs.append(F_norm); mlist.append(m)
        # P-cell residual
        P_now = expand(x); P_resid = float(np.max(np.abs(phi_call(P_now) - P_now)))
        print(f'  NewtIter {it+1:2d}  ||F||={F_norm:.3e}  P-resid={P_resid:.3e}  aLS={alpha:.3g}  a_slope={x[0]:.4f}  slope_T={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  (J build {t_build:.0f}s, total {time.time()-ts:.0f}s)', flush=True)
        if P_resid < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, mlist


if __name__ == '__main__':
    print('\n=== N=8 lifted+numba (τ=1, γ=1) ===')
    print('JIT warmup...', flush=True)
    t0 = time.time()
    _ = phi_call(np.full((G8, G8, G8), 0.5))
    print(f'  JIT: {time.time()-t0:.1f}s')

    # NL Bayes IC
    P_IC_full = np.empty((G8, G8, G8))
    for i in range(G8):
        for j in range(G8):
            for k in range(G8):
                mu0 = sigmoid(TAU*U_NODES8[i])
                mu1 = sigmoid(TAU*U_NODES8[j])
                mu2 = sigmoid(TAU*U_NODES8[k])
                P_IC_full[i,j,k] = crra_clear_jit(mu0, mu1, mu2, GAMMA)
    x = contract(P_IC_full)
    print(f'\nIC: NL Bayes; α={x[0]:.4f}, metrics={metrics(expand(x))}', flush=True)

    # Picard preconditioning
    print(f'\n=== Picard (5 iters, ω=0.5) ===')
    pferrs = []; pmetrics = []
    for it in range(5):
        ts = time.time()
        x_new = phi_lift(x)
        ferr = float(np.max(np.abs(x_new - x))); pferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); pmetrics.append(m)
        print(f'  Picard {it+1}  ||F||={ferr:.3e}  α={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)

    # Newton
    print(f'\n=== Dense Newton ===')
    x_final, nferrs, nmetrics = dense_newton(x, n_iter=6, tol=1e-9)

    P_final = expand(x_final)
    P_resid_final = float(np.max(np.abs(phi_call(P_final) - P_final)))
    print(f'\nFinal N=8: α={x_final[0]:.4f}, metrics={metrics(P_final)}')
    print(f'Final P-cell residual: {P_resid_final:.3e}')
    print(f'  N=6 baseline (lifted+numba): 5.3e-03')

    np.save('/tmp/cheby_h0/P_final_n8.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_n8.npy', x_final)
    json.dump(dict(
        config=dict(N=N8, NQ=NQ, tau=TAU, gamma=GAMMA, c=C_STRETCH,
                     lobatto=LOBATTO8.tolist(), u_nodes=U_NODES8.tolist(),
                     n_dof=N_DOF, n_fixed=len(fixed_reps), n_free=len(free_reps)),
        x_final=x_final.tolist(),
        picard=dict(ferrs=pferrs, metrics=pmetrics),
        newton=dict(ferrs=nferrs, metrics=nmetrics),
        final_metrics=metrics(P_final),
        final_p_residual=P_resid_final,
        final_alpha=float(x_final[0]),
    ), open('/tmp/cheby_h0/results_n8.json', 'w'), indent=2, default=str)
    print('saved')
