"""Chebyshev h=0 solver, S_3 × Z_2 symmetry, FIXED enumeration.

Counts at G=7 (N=6):
- Total S_3 orbits (multisets): 84
- Z_2-fixed S_3 orbits: those with {i,j,k} = {G-1-i, G-1-j, G-1-k} as multiset.
  At G=7: requires j=3 (middle) and i+k=6: (0,3,6), (1,3,5), (2,3,4), (3,3,3)
  → 4 Z_2-fixed orbits, each forced to value 0.5 (since P_z2 = 1-P implies P=0.5).
- Z_2-paired S_3 orbits: 84 - 4 = 80 paired into 40 combined orbits → 40 free DOFs.

Total: 40 free + 4 fixed-at-0.5 = 44 representatives (343 cube cells).
"""
import os, sys, time, math, json, itertools
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')
from cheby_h0_solver import phi, crra_clear, N_GRID, U_NODES, LOBATTO, TAU, GAMMA, C_STRETCH, N

G = N_GRID  # 7

def orbit_Z2(i, j, k):
    return (G-1-i, G-1-j, G-1-k)

def full_orbit(i, j, k):
    """All cells equivalent to (i,j,k) under S_3 × Z_2 (the cells with value=x).
    Returns (S3_orbit, Z2_S3_orbit_of_z2_image)."""
    s3 = set(itertools.permutations((i, j, k)))
    z2 = orbit_Z2(i, j, k)
    z2_s3 = set(itertools.permutations(z2))
    return s3, z2_s3

def is_z2_fixed(i, j, k):
    s3 = set(itertools.permutations((i, j, k)))
    z2 = orbit_Z2(i, j, k)
    return z2 in s3  # i.e. Z_2 image is in the S_3 orbit

def enumerate_reps():
    """Return (free_reps, fixed_reps).
    free_reps: list of canonical (sorted) reps with 1 free DOF each.
    fixed_reps: list of canonical reps forced to value 0.5."""
    seen = set()
    free, fixed = [], []
    for i in range(G):
        for j in range(i, G):
            for k in range(j, G):
                if (i, j, k) in seen: continue
                s3, z2_s3 = full_orbit(i, j, k)
                combined = s3 | z2_s3
                # Add all S_3-sorted reps in this combined orbit to seen
                for cell in combined:
                    seen.add(tuple(sorted(cell)))
                if is_z2_fixed(i, j, k):
                    fixed.append((i, j, k))
                else:
                    free.append((i, j, k))
    return free, fixed

FREE_REPS, FIXED_REPS = enumerate_reps()
print(f'Sym enumeration at G={G}:')
print(f'  Free DOFs: {len(FREE_REPS)}')
print(f'  Fixed at 0.5: {len(FIXED_REPS)}: {FIXED_REPS}')
print(f'  Total representatives: {len(FREE_REPS)+len(FIXED_REPS)} (full cube: {G**3})', flush=True)

# Build lookup: for each cell (i,j,k), which (rep, sign) does it map to?
# sign = +1 means P[cell] = x_rep, sign = -1 means P[cell] = 1 - x_rep
CELL_TO_REP = {}
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    for cell in s3: CELL_TO_REP[cell] = (rep, +1)
    for cell in z2_s3: CELL_TO_REP[cell] = (rep, -1)
for rep in FIXED_REPS:
    s3, z2_s3 = full_orbit(*rep)
    for cell in s3 | z2_s3:
        CELL_TO_REP[cell] = ('FIXED', 0.5)

# ===== Map small ↔ full =====
def expand(x_small):
    """x_small is array of size len(FREE_REPS)."""
    P = np.empty((G, G, G))
    rep_to_idx = {rep: i for i, rep in enumerate(FREE_REPS)}
    for cell, (key, sign) in CELL_TO_REP.items():
        if key == 'FIXED':
            P[cell] = 0.5
        else:
            x = x_small[rep_to_idx[key]]
            P[cell] = x if sign > 0 else (1.0 - x)
    return P

def contract(P):
    """P → x_small (canonical value per orbit)."""
    return np.array([float(P[rep]) for rep in FREE_REPS])

# ===== Symmetrized phi =====
def phi_sym(x_small, gamma):
    P = expand(x_small)
    P_new = phi(P, gamma)
    # Re-project to symmetric (kill any FP noise)
    P_sym = np.empty_like(P_new)
    rep_to_idx = {rep: i for i, rep in enumerate(FREE_REPS)}
    # For each free rep, average over all cells equivalent to it
    for ridx, rep in enumerate(FREE_REPS):
        s3, z2_s3 = full_orbit(*rep)
        vals = [P_new[c] for c in s3] + [1.0 - P_new[c] for c in z2_s3]
        avg = float(np.mean(vals))
        for c in s3: P_sym[c] = avg
        for c in z2_s3: P_sym[c] = 1.0 - avg
    for rep in FIXED_REPS:
        s3, z2_s3 = full_orbit(*rep)
        for c in s3 | z2_s3:
            P_sym[c] = 0.5
    return contract(P_sym)

def metrics(P):
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0]*T.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = 1.0/(1.0+np.exp(-T)); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def dense_newton_sym(x_small, gamma, n_iter=8, eps_fd=1e-6, tol=1e-9):
    N_unk = x_small.size
    print(f'\n  Dense Newton SYM: {N_unk} unknowns', flush=True)
    x = x_small.copy()
    F = phi_sym(x, gamma) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Initial ||F||_∞={F_norm:.3e}', flush=True)
    ferrs = [F_norm]; metrics_list = [metrics(expand(x))]
    for it in range(n_iter):
        ts = time.time()
        J = np.empty((N_unk, N_unk))
        for j in range(N_unk):
            xp = x.copy(); xp[j] += eps_fd
            Fp = phi_sym(xp, gamma) - xp
            J[:, j] = (Fp - F) / eps_fd
        t_build = time.time() - ts
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        alpha = 1.0; best = (None, 1e100)
        for _ in range(20):
            xn = np.clip(x + alpha*dx, 1e-12, 1-1e-12)
            Fn = phi_sym(xn, gamma) - xn; nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, alpha)
            if nn < (1 - 0.5*alpha)*F_norm: break
            alpha *= 0.5
            if alpha < 1e-12: break
        x, F_norm, F, alpha = best
        m = metrics(expand(x)); ferrs.append(F_norm); metrics_list.append(m)
        print(f'  NewtIter {it+1:2d}  ||F||_∞={F_norm:.3e}  α={alpha:.3g}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  (J build {t_build:.1f}s, total {time.time()-ts:.1f}s)', flush=True)
        if F_norm < tol:
            print('  CONVERGED', flush=True); break
    return x, ferrs, metrics_list

if __name__ == '__main__':
    def sg(x): return 1/(1+np.exp(-x))
    print(f'CHEBYSHEV h=0 SOLVER (S_3 × Z_2 symmetric)')
    print(f'  N={N}  G={G}  N_unk_full={G**3}  N_unk_sym={len(FREE_REPS)}  τ={TAU}  γ={GAMMA}')

    # NL IC
    P_IC_full = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu0 = sg(TAU*U_NODES[i]); mu1 = sg(TAU*U_NODES[j]); mu2 = sg(TAU*U_NODES[k])
                P_IC_full[i,j,k] = crra_clear(mu0, mu1, mu2, GAMMA)
    x_IC = contract(P_IC_full)
    P_IC = expand(x_IC)
    diff = float(np.max(np.abs(P_IC - P_IC_full)))
    print(f'\nIC round-trip diff: {diff:.3e} (should be small — NL IC is already symmetric)')
    print(f'IC metrics: {metrics(P_IC)}')

    print(f'\n=== PHASE 1: Picard on sym subspace ===')
    x = x_IC.copy(); picard_ferrs = []; picard_metrics = []
    for it in range(5):
        ts = time.time()
        x_new = phi_sym(x, GAMMA)
        ferr = float(np.max(np.abs(x_new - x))); picard_ferrs.append(ferr)
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x)); picard_metrics.append(m)
        print(f'  Picard it {it+1:2d}  ||F||={ferr:.3e}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.1f}s)', flush=True)

    print(f'\n=== PHASE 2: Pure dense Newton on sym subspace ===')
    x_final, newton_ferrs, newton_metrics = dense_newton_sym(x, GAMMA, n_iter=6, tol=1e-9)

    P_final = expand(x_final)
    print(f'\nFinal metrics: {metrics(P_final)}')

    np.save('/tmp/cheby_h0/P_final_sym2.npy', P_final)
    np.save('/tmp/cheby_h0/P_IC_sym2.npy', P_IC)
    json.dump(dict(
        config=dict(N=N, NQ=12, tau=TAU, gamma=GAMMA, c=C_STRETCH,
                     lobatto=LOBATTO.tolist(), u_nodes=U_NODES.tolist(),
                     n_unk_full=G**3, n_unk_sym=len(FREE_REPS),
                     n_fixed=len(FIXED_REPS)),
        picard=dict(ferrs=picard_ferrs, metrics=picard_metrics),
        newton=dict(ferrs=newton_ferrs, metrics=newton_metrics),
        final_metrics=metrics(P_final),
    ), open('/tmp/cheby_h0/results_sym2.json','w'), indent=2, default=str)
    print('saved')
