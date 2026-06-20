"""Comprehensive timing + characterization of NQ=16 strict-h=0 FP.

(1) Per-Phi wall time (median of 20 runs)
(2) Full cold-start solve wall time (Anderson + NK)
(3) Full FP description:
    - slope alpha*, deficit, intercept
    - logit regression diagnostics
    - max|P|, min|P|, mean|P|
    - symmetry checks (S3, Z2)
    - boundary values P(±U_max, ±U_max, ±U_max)
    - mu(p, u_k) table values
    - posterior odds at key cells
    - economic interpretation: revealed precision, information ratio
"""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import (phi_pou_cr_jit, make_p_grid,
                                  build_mu_table_pou_cr)
from cheby_numba import (V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH,
                            N_GRID, vals_to_coeffs_3d_jit, f_signal_jit,
                            crra_clear_jit)
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

NQ = 16
G_p = 121
p_grid = make_p_grid(G_p)
gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)

def phi(P): return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                                       TAU, GAMMA, C_STRETCH, G, NQ)
def F(x): return contract(phi(expand(x))) - x

# Warmup
P_init = sg(0.5*T); _ = phi(P_init); _ = F(contract(P_init))

# ======= (1) PER-PHI TIMING =======
print('=== (1) Per-Phi timing (median of 20 runs) ===\n')
ts = []
for _ in range(20):
    t0 = time.time(); _ = phi(P_init); ts.append(time.time() - t0)
t_phi = float(np.median(ts))
print(f'  median: {t_phi*1000:.2f} ms')
print(f'  min:    {min(ts)*1000:.2f} ms')
print(f'  max:    {max(ts)*1000:.2f} ms')

# ======= (2) FULL SOLVE FROM COLD =======
print('\n=== (2) Full cold-start solve to machine eps ===\n')
x0 = contract(P_init)
print(f'  Initial F: {float(np.max(np.abs(F(x0)))):.3e}')

# Anderson
t0 = time.time()
x = x0.copy(); Xh, Gh = [], []; Fs_and = []
x_best = x.copy(); f_best = float('inf')
for it in range(80):
    Fv = F(x); gx = Fv + x
    f = float(np.max(np.abs(Fv))); Fs_and.append(f)
    if f < f_best: f_best = f; x_best = x.copy()
    if f < 1e-15: break
    Xh.append(x.copy()); Gh.append(gx.copy())
    if len(Xh) > 10: Xh.pop(0); Gh.pop(0)
    k = len(Xh)
    if k <= 1: x = gx
    else:
        DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
        R_k = Gh[k-1] - Xh[k-1]
        A = DR.T @ DR + 1e-12*np.eye(DR.shape[1])
        ga = np.linalg.solve(A, -DR.T @ R_k)
        DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
        x = Gh[k-1] + DG @ ga
t_and = time.time() - t0
print(f'  Anderson:        {len(Fs_and)} iters, {t_and:.2f}s -> F = {min(Fs_and):.3e}')

# Newton-Krylov
t0 = time.time()
x_nk = newton_krylov(F, x_best, f_tol=1e-15, maxiter=30, verbose=False)
f_nk = float(np.max(np.abs(F(x_nk))))
t_nk = time.time() - t0
print(f'  Newton-Krylov:   {t_nk:.2f}s -> F = {f_nk:.3e}')
print(f'  TOTAL:           {t_and + t_nk:.2f}s')

# ======= (3) FP CHARACTERIZATION =======
print('\n=== (3) Equilibrium description ===\n')
P_fp = expand(x_nk)

print(f'  ||F||_inf      = {f_nk:.3e}')
print(f'  P range        = [{P_fp.min():.6f}, {P_fp.max():.6f}]')
print(f'  P mean         = {P_fp.mean():.6f}')
print(f'  P at center    = {P_fp[G//2, G//2, G//2]:.6f}')

# Symmetry verification
P_S3 = (P_fp + np.transpose(P_fp, (1,0,2)) + np.transpose(P_fp, (2,1,0)) +
         np.transpose(P_fp, (0,2,1)) + np.transpose(P_fp, (1,2,0)) +
         np.transpose(P_fp, (2,0,1))) / 6
err_S3 = float(np.max(np.abs(P_fp - P_S3)))
print(f'  S_3 symmetry violation = {err_S3:.3e}')

P_Z2 = 1 - P_fp[::-1, ::-1, ::-1]  # P(u) + P(-u) = 1
err_Z2 = float(np.max(np.abs(P_fp - P_Z2)))
print(f'  Z_2 symmetry violation = {err_Z2:.3e}')

# Logit regression
Pc = np.clip(P_fp, 1e-15, 1-1e-15)
L = np.log(Pc/(1-Pc)).ravel()
slope = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
intercept = float(np.mean(L - slope*T.ravel()))
pred = slope*T.ravel() + intercept
ss_res = float(np.sum((L - pred)**2))
ss_tot = float(np.sum((L - L.mean())**2))
r2 = 1 - ss_res/ss_tot
deficit = ss_res/ss_tot
print(f'\nLogit regression:  logit(P) = {slope:.6f}*T + {intercept:.6e}')
print(f'                   R^2 = {r2:.6f}')
print(f'                   deficit (1-R^2) = {deficit:.6f}')

# Boundary values
print(f'\nBoundary values:')
print(f'  P(u_min,u_min,u_min) = P({U_NODES[0]:.2f},...,...) = {P_fp[0,0,0]:.3e}')
print(f'  P(u_max,u_max,u_max) = P({U_NODES[-1]:.2f},...,...) = {P_fp[-1,-1,-1]:.6f}')
print(f'  P(0, 0, 0)           = {P_fp[G//2, G//2, G//2]:.6f}')
print(f'  P(u_max, u_min, 0)   = {P_fp[-1, 0, G//2]:.6f}')

# mu(p, u_k) table
coeffs = vals_to_coeffs_3d_jit(P_fp, V_INV)
mu_table = build_mu_table_pou_cr(coeffs, LOBATTO, U_NODES, p_grid,
                                     gl_n, gl_w, TAU, C_STRETCH, G, NQ)
print(f'\nmu(p, u_k) table summary:')
print(f'  mu range = [{mu_table.min():.4f}, {mu_table.max():.4f}]')
print(f'  At p=0.5, u_k=0: mu = {mu_table[G_p//2, G//2]:.6f}  (expected: 0.5)')
print(f'  At p=0.1, u_k=u_max:   mu = {mu_table[20, -1]:.6f}')
print(f'  At p=0.9, u_k=u_max:   mu = {mu_table[100, -1]:.6f}')
print(f'  At p=0.5, u_k=u_max:   mu = {mu_table[G_p//2, -1]:.6f}')

# Economic interpretation: revealed precision = sqrt(slope * tau)
# In the standard Hellwig framework, alpha* relates to revealed signal precision
print(f'\nEconomic interpretation:')
print(f'  Slope alpha* = {slope:.4f}')
print(f'  Implied revealed precision tau_p = slope*tau = {slope*TAU:.4f}')
print(f'  Information deficit (1-R^2) = {deficit:.4f}')
print(f'    = fraction of price variance NOT explained by T')
print(f'    = how partially-revealing the equilibrium is')
print(f'  R^2 = {r2:.4f} = price reveals {r2*100:.1f}% of total signal variance')

# Per-cell posterior odds spread
mu_at_cells = np.empty((G, G, G))
for i in range(G):
    for j in range(G):
        for k in range(G):
            p_cell = max(min(P_fp[i,j,k], 1-1e-9), 1e-9)
            # Find table index
            i_p = np.searchsorted(p_grid, p_cell) - 1
            i_p = max(0, min(G_p-2, i_p))
            w = (p_cell - p_grid[i_p]) / (p_grid[i_p+1] - p_grid[i_p])
            mu_at_cells[i,j,k] = ((1-w)*mu_table[i_p, i] + w*mu_table[i_p+1, i])
print(f'\n  Posterior mu range across cells: [{mu_at_cells.min():.4f}, {mu_at_cells.max():.4f}]')

# Save complete description
desc = dict(
    NQ=NQ, G=G, N=G-1, G_p=G_p, tau=TAU, gamma=GAMMA,
    timing=dict(per_phi_ms=t_phi*1000, anderson_s=t_and,
                  newton_krylov_s=t_nk,
                  total_solve_s=t_and+t_nk),
    fixed_point=dict(F_inf=f_nk, slope_alpha_star=slope,
                       intercept=intercept, R_squared=r2, deficit=deficit,
                       symmetry_violation_S3=err_S3,
                       symmetry_violation_Z2=err_Z2,
                       P_range=[float(P_fp.min()), float(P_fp.max())],
                       P_at_center=float(P_fp[G//2, G//2, G//2]),
                       P_at_corner_max=float(P_fp[-1,-1,-1]),
                       P_at_corner_min=float(P_fp[0,0,0]),
                       mu_at_p0p5_uk0=float(mu_table[G_p//2, G//2]),
                       mu_range=[float(mu_at_cells.min()),
                                 float(mu_at_cells.max())]),
)
json.dump(desc, open('/tmp/cheby_h0/FP_description.json', 'w'),
            indent=2, default=str)
print('\nsaved FP_description.json')
