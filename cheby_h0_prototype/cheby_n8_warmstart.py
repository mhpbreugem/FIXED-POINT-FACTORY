"""N=8 Newton WARM-STARTED from the converged N=6 lifted FP.

Strategy:
  1. Load P^{N=6} (Lobatto values, 7×7×7) and its converged α=0.344.
  2. Compute the N=6 Chebyshev coefficient tensor C^{N=6} via Vandermonde inverse.
  3. Evaluate C^{N=6} at the N=8 Lobatto nodes (zero-pad to degree 8 implicit).
  4. This gives P^{N=8}_init with the same equilibrium structure to N=6 accuracy.
  5. Re-contract under the N=8 sigmoid lift, then Newton.

Expected outcome: P-resid at N=8 starts ~10^-2 (N=6 truncation error), Newton
brings it to ~10^-4 or better.
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_numba import phi_jit

# N=8 setup
N8 = 8
G8 = N8 + 1
N6 = 6
G6 = N6 + 1
C_STRETCH = 2.0
TAU = 1.0
GAMMA = 1.0
NQ = 12

# Lobatto nodes
LOBATTO6 = -np.cos(np.pi * np.arange(G6) / N6)
LOBATTO8 = -np.cos(np.pi * np.arange(G8) / N8)
U_NODES6 = C_STRETCH * np.arctanh(np.clip(LOBATTO6, -0.9999, 0.9999))
U_NODES8 = C_STRETCH * np.arctanh(np.clip(LOBATTO8, -0.9999, 0.9999))

# Vandermonde matrices
def build_V(L, G):
    V = np.empty((G, G))
    for j in range(G):
        x = L[j]
        V[j, 0] = 1.0
        if G > 1:
            V[j, 1] = x
            for k in range(1, G-1):
                V[j, k+1] = 2*x*V[j, k] - V[j, k-1]
    return V

V6 = build_V(LOBATTO6, G6)
V_INV6 = np.linalg.inv(V6)
V8 = build_V(LOBATTO8, G8)
V_INV8 = np.linalg.inv(V8)
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(NQ)

def chebval_at(x, c):
    """Clenshaw eval of Cheb series c at x."""
    n = len(c)
    if n == 0: return 0.0
    if n == 1: return c[0]
    bk1 = 0.0; bk2 = 0.0
    for k in range(n-1, 0, -1):
        bk = c[k] + 2*x*bk1 - bk2
        bk2 = bk1; bk1 = bk
    return c[0] + x*bk1 - bk2

# === Step 1-3: Resample N=6 FP onto N=8 grid via spectral interpolation ===
def vals_to_coeffs_3d(P_vals, V_inv):
    """Convert (G,G,G) Lobatto values → Chebyshev coefficients (G,G,G)."""
    G = P_vals.shape[0]
    tmp = np.empty_like(P_vals)
    for j in range(G):
        for k in range(G):
            tmp[:, j, k] = V_inv @ P_vals[:, j, k]
    tmp2 = np.empty_like(P_vals)
    for i in range(G):
        for k in range(G):
            tmp2[i, :, k] = V_inv @ tmp[i, :, k]
    out = np.empty_like(P_vals)
    for i in range(G):
        for j in range(G):
            out[i, j, :] = V_inv @ tmp2[i, j, :]
    return out

def resample_to_n8(P_n6):
    """Spectral interpolation: N=6 Lobatto values → N=8 Lobatto values."""
    C6 = vals_to_coeffs_3d(P_n6, V_INV6)  # (7,7,7) Chebyshev coefficients
    # Evaluate at each N=8 Lobatto node
    P_n8 = np.empty((G8, G8, G8))
    # Precompute T_a(ξ_n8[i]) for a=0..N6 (G6 values), per axis
    T_n8_per_node = np.empty((G8, G6))
    for i in range(G8):
        for a in range(G6):
            basis = np.zeros(G6); basis[a] = 1
            T_n8_per_node[i, a] = chebval_at(LOBATTO8[i], basis)
    # Sum_{a,b,c} C6[a,b,c] * T_a(ξ_i) * T_b(ξ_j) * T_c(ξ_k)
    # Reduce axis by axis:
    # Step 1: contract axis 0 → tensor (G8, G6, G6)
    R1 = np.einsum('ia,abc->ibc', T_n8_per_node, C6)  # (G8, G6, G6)
    R2 = np.einsum('jb,ibc->ijc', T_n8_per_node, R1)  # (G8, G8, G6)
    R3 = np.einsum('kc,ijc->ijk', T_n8_per_node, R2)  # (G8, G8, G8)
    return R3

# === Step 4-5: Lifted Newton at N=8 ===
import itertools
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

ORBIT_POS, ORBIT_NEG = [], []
for rep in free_reps:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3)); ORBIT_NEG.append(list(z2_s3))

U1, U2, U3 = np.meshgrid(U_NODES8, U_NODES8, U_NODES8, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

N_DOF = 1 + len(free_reps)
print(f'N=8 warm-start setup: {N_DOF} DOFs (1 α + {len(free_reps)} h-orbits)', flush=True)

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

def dense_newton(x, n_iter=8, eps_fd=1e-5, tol=1e-9):
    Nu = x.size
    print(f'\n  Dense Newton N=8 (warm-start): {Nu} unknowns', flush=True)
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
        P_now = expand(x); P_resid = float(np.max(np.abs(phi_call(P_now) - P_now)))
        print(f'  NewtIter {it+1:2d}  ||F||={F_norm:.3e}  P-resid={P_resid:.3e}  aLS={alpha:.3g}  a_slope={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.0f}s)', flush=True)
        if P_resid < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, mlist


if __name__ == '__main__':
    print('\n=== N=8 WARM-START from N=6 lifted (α, h) ===\n')
    print('Step 1: Load N=6 converged lifted FP (α, h_orbit) vector', flush=True)
    x_n6 = np.load('/tmp/cheby_h0/x_final_lifted_numba.npy')
    alpha_n6 = float(x_n6[0])
    h_n6 = x_n6[1:]  # 40 orbital values at N=6
    print(f'  α_n6 = {alpha_n6:.4f}  (N=6 converged slope)')
    print(f'  h_n6: 40 orbital h-values, range [{h_n6.min():+.4f}, {h_n6.max():+.4f}]')

    # Build the h-correction at N=6 cells (7×7×7)
    from cheby_sym2 import FREE_REPS as FR6, full_orbit as fo6, FIXED_REPS as FX6
    h_full_n6 = np.zeros((G6, G6, G6))
    for idx, rep in enumerate(FR6):
        h_val = float(h_n6[idx])
        s3, z2_s3 = fo6(*rep)
        for c in s3: h_full_n6[c] = h_val
        for c in z2_s3: h_full_n6[c] = -h_val

    print('Step 2-3: Spectral interpolate h_full_n6 → h_full_n8 via Chebyshev resample', flush=True)
    h_full_n8 = resample_to_n8(h_full_n6)
    print(f'  h_full_n8 range: [{h_full_n8.min():+.4f}, {h_full_n8.max():+.4f}]')

    # Symmetric-project h_full_n8 onto the N=8 free orbits
    h_n8 = np.zeros(len(free_reps))
    for idx, rep in enumerate(free_reps):
        s3, z2_s3 = full_orbit(*rep)
        vals = [h_full_n8[c] for c in s3] + [-h_full_n8[c] for c in z2_s3]
        h_n8[idx] = float(np.mean(vals))
    x_init = np.concatenate([[alpha_n6], h_n8])
    print(f'  After projection: α={x_init[0]:.4f}, h_n8 range [{h_n8.min():+.4f}, {h_n8.max():+.4f}]')

    P_init_check = expand(x_init)
    print(f'  IC metrics: {metrics(P_init_check)}', flush=True)

    print('\nStep 5: JIT warmup', flush=True)
    t = time.time()
    _ = phi_call(np.full((G8, G8, G8), 0.5))
    print(f'  JIT: {time.time()-t:.1f}s')

    # Check initial P-residual
    print('\nStep 6: Initial P-residual (before Newton)', flush=True)
    P_resid_init = float(np.max(np.abs(phi_call(P_init_check) - P_init_check)))
    print(f'  ||phi(P_warm) - P_warm||_inf = {P_resid_init:.3e}', flush=True)

    print('\nStep 7: Newton refinement at N=8', flush=True)
    x_final, ferrs, mlist = dense_newton(x_init, n_iter=8, tol=1e-9)

    P_final = expand(x_final)
    P_resid_final = float(np.max(np.abs(phi_call(P_final) - P_final)))
    print(f'\nFinal: α={x_final[0]:.4f}, metrics={metrics(P_final)}', flush=True)
    print(f'Final P-cell residual: {P_resid_final:.3e}')
    print(f'  N=6 baseline: 5.3e-03  | N=8 cold-start (prior): 1.0e-01  | N=8 warm-start: {P_resid_final:.3e}')

    np.save('/tmp/cheby_h0/P_final_n8_warm.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_n8_warm.npy', x_final)
    json.dump(dict(
        config=dict(N=N8, NQ=NQ, tau=TAU, gamma=GAMMA, c=C_STRETCH, n_dof=N_DOF,
                     warm_start_from='N=6 lifted_numba FP'),
        x_final=x_final.tolist(),
        newton_ferrs=ferrs,
        newton_metrics=mlist,
        final_metrics=metrics(P_final),
        final_p_residual=P_resid_final,
        initial_p_residual=P_resid_init,
        final_alpha=float(x_final[0]),
    ), open('/tmp/cheby_h0/results_n8_warm.json', 'w'), indent=2, default=str)
    print('saved')
