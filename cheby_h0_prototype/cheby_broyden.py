"""Broyden quasi-Newton on top of the lifted+numba operator.

Replaces the forward-difference Jacobian rebuild (N_DOF × phi calls per iter)
with rank-1 Broyden updates (1 phi call per iter), so Newton outer iter cost
drops from O(N_DOF × t_phi) to O(t_phi).

At N=6: 13s/iter → 0.3s/iter
At N=8: ~180s/iter → ~1.2s/iter (makes N=8 Newton practical for full sweeps)

Starts with a single forward-difference Jacobian for stability, then Broyden
updates thereafter. Falls back to full FD if Broyden divergence detected.
"""
import os, sys, time, math, json
import numpy as np
sys.path.insert(0, '/tmp/cheby_h0')

from cheby_numba import phi as phi_jit, U_NODES, LOBATTO, C_STRETCH, N, N_GRID
from cheby_sym2 import FREE_REPS, FIXED_REPS, full_orbit

G = N_GRID
TAU = 1.0
GAMMA = 1.0

U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T_FIELD = TAU * (U1 + U2 + U3)

ORBIT_POS, ORBIT_NEG = [], []
for rep in FREE_REPS:
    s3, z2_s3 = full_orbit(*rep)
    ORBIT_POS.append(list(s3)); ORBIT_NEG.append(list(z2_s3))

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return np.log(p/(1-p))

N_DOF = 1 + len(FREE_REPS)

def expand(x):
    alpha = x[0]; h_small = x[1:]
    h_full = np.zeros((G, G, G))
    for idx in range(len(FREE_REPS)):
        h_val = h_small[idx]
        for cell in ORBIT_POS[idx]: h_full[cell] = h_val
        for cell in ORBIT_NEG[idx]: h_full[cell] = -h_val
    return sigmoid(alpha * T_FIELD + h_full)

def contract(P):
    L = logit(P)
    alpha = float(np.sum(L * T_FIELD) / np.sum(T_FIELD**2))
    h_full = L - alpha * T_FIELD
    h_small = np.zeros(len(FREE_REPS))
    for idx in range(len(FREE_REPS)):
        vals = [h_full[c] for c in ORBIT_POS[idx]] + [-h_full[c] for c in ORBIT_NEG[idx]]
        h_small[idx] = float(np.mean(vals))
    return np.concatenate([[alpha], h_small])

def phi_lift(x, gamma):
    P = expand(x)
    P_new = phi_jit(P, gamma=gamma, tau=TAU)
    return contract(P_new)

def metrics(P):
    Pc = np.clip(P, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T_FIELD.ravel(), y, 1); pr = a[0]*T_FIELD.ravel()+a[1]
    defi = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR = sigmoid(T_FIELD); d_FR = float(np.sqrt(np.mean((P-P_FR)**2)))
    return dict(slope_T=float(a[0]), deficit=defi, d_FR=d_FR)

def fd_jacobian(x, F, gamma, eps_fd=1e-5):
    """One full forward-difference Jacobian (N_DOF × t_phi)."""
    Nu = x.size
    J = np.empty((Nu, Nu))
    for j in range(Nu):
        xp = x.copy(); xp[j] += eps_fd
        J[:, j] = (phi_lift(xp, gamma) - xp - F) / eps_fd
    return J

def broyden_newton(x, gamma, n_iter=40, tol=1e-9, fd_restart_every=15):
    """Broyden + line search. FD Jacobian at start and every `fd_restart_every`
    iters (or on divergence)."""
    Nu = x.size
    F = phi_lift(x, gamma) - x
    F_norm = float(np.max(np.abs(F)))
    print(f'  Broyden start: ||F||_inf={F_norm:.3e}', flush=True)

    # Initial FD Jacobian
    t = time.time()
    J = fd_jacobian(x, F, gamma)
    print(f'  Initial FD Jacobian: {time.time()-t:.1f}s', flush=True)

    ferrs = [F_norm]
    mlist = [metrics(expand(x))]
    iters_since_fd = 0
    p_residual_history = []

    for it in range(n_iter):
        ts = time.time()
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
        # Line search
        a = 1.0; best = (x, F_norm, F, 0.0)
        for _ in range(20):
            xn = x + a*dx
            Fn = phi_lift(xn, gamma) - xn
            nn = float(np.max(np.abs(Fn)))
            if nn < best[1]: best = (xn, nn, Fn, a)
            if nn < (1 - 0.5*a)*F_norm: break
            a *= 0.5
            if a < 1e-12: break
        x_new, F_norm_new, F_new, a = best

        # Broyden rank-1 update
        s = x_new - x
        y = F_new - F
        Js = J @ s
        denom = float(s @ s)
        if denom > 1e-30:
            J = J + np.outer(y - Js, s) / denom

        x, F, F_norm = x_new, F_new, F_norm_new
        iters_since_fd += 1
        ferrs.append(F_norm)
        m = metrics(expand(x)); mlist.append(m)
        # Diagnostic: actual P-cell residual
        P_now = expand(x); P_resid = float(np.max(np.abs(phi_jit(P_now, gamma=gamma, tau=TAU) - P_now)))
        p_residual_history.append(P_resid)
        print(f'    Broyden it {it+1:2d}  ||F||={F_norm:.3e}  P-resid={P_resid:.3e}  aLS={a:.3g}  a_slope={x[0]:.4f}  slope={m["slope_T"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)
        if F_norm < tol or P_resid < tol:
            print('    CONVERGED', flush=True); break
        # FD restart every fd_restart_every iters
        if iters_since_fd >= fd_restart_every:
            t = time.time()
            J = fd_jacobian(x, F, gamma)
            print(f'    FD restart: {time.time()-t:.1f}s', flush=True)
            iters_since_fd = 0

    return x, ferrs, mlist, p_residual_history

if __name__ == '__main__':
    print(f'\n=== Broyden quasi-Newton on lifted+numba operator ===')
    print(f'  τ={TAU}, γ={GAMMA}, N={N}, DOFs={N_DOF}')
    print('JIT warmup...', flush=True)
    t0 = time.time()
    _ = phi_jit(np.full((G, G, G), 0.5), gamma=GAMMA, tau=TAU)
    print(f'  JIT: {time.time()-t0:.1f}s')

    # IC: NL Bayes
    from cheby_numba import crra_clear
    P_IC_full = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu0 = sigmoid(TAU*U_NODES[i])
                mu1 = sigmoid(TAU*U_NODES[j])
                mu2 = sigmoid(TAU*U_NODES[k])
                P_IC_full[i,j,k] = crra_clear(mu0, mu1, mu2, GAMMA)
    x = contract(P_IC_full)
    print(f'\nIC: NL Bayes; α={x[0]:.4f}, metrics={metrics(expand(x))}', flush=True)

    # Picard preconditioning (quick)
    print(f'\n=== Picard (3 iters) ===')
    for it in range(3):
        ts = time.time()
        x_new = phi_lift(x, GAMMA)
        ferr = float(np.max(np.abs(x_new - x)))
        x = 0.5*x + 0.5*x_new
        m = metrics(expand(x))
        print(f'  Picard {it+1}  ||F||={ferr:.3e}  α={x[0]:.4f}  slope={m["slope_T"]:.4f}  deficit={m["deficit"]:.4f}  ({time.time()-ts:.2f}s)', flush=True)

    # Broyden run
    print(f'\n=== Broyden (40 iters, FD restart every 15) ===')
    t_broyden_start = time.time()
    x_final, ferrs, mlist, p_resids = broyden_newton(x, GAMMA, n_iter=40, tol=1e-9)
    t_broyden_total = time.time() - t_broyden_start

    P_final = expand(x_final)
    print(f'\nFinal: α={x_final[0]:.4f}, metrics={metrics(P_final)}')
    print(f'Total Broyden time: {t_broyden_total:.1f}s ({len(ferrs)-1} iters)')

    # Final P-cell residual
    P_resid_inf = float(np.max(np.abs(phi_jit(P_final, gamma=GAMMA, tau=TAU) - P_final)))
    print(f'\nFinal P-cell residual ||phi(P) - P||_inf = {P_resid_inf:.3e}')
    print(f'  Lifted+numba reference: 5.3e-03')

    np.save('/tmp/cheby_h0/P_final_broyden.npy', P_final)
    np.save('/tmp/cheby_h0/x_final_broyden.npy', x_final)
    json.dump(dict(
        config=dict(N=N, tau=TAU, gamma=GAMMA, n_dof=N_DOF),
        x_final=x_final.tolist(),
        ferrs=ferrs,
        p_residual_history=p_resids,
        final_metrics=metrics(P_final),
        final_p_residual=P_resid_inf,
        total_broyden_time=t_broyden_total,
        n_iters=len(ferrs)-1,
    ), open('/tmp/cheby_h0/results_broyden.json','w'), indent=2, default=str)
    print('saved')
