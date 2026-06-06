"""Warm-start at multiple NQ values from the NQ=16 saved FP.
If they all converge to machine eps, the FP is FP-of-many-NQ; NQ=16
is just easier to cold-start from sigmoid(0.5T)."""
import sys, time, json
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
from cheby_pou_cr_jit import phi_pou_cr_jit, make_p_grid
from cheby_numba import V_INV, LOBATTO, U_NODES, TAU, GAMMA, C_STRETCH, N_GRID
from cheby_sym2 import expand, contract

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)
def sg(x): return 1/(1+np.exp(-x))

p_grid = make_p_grid(121)
P_warm = np.load('/tmp/cheby_h0/P_FP_pou_nq16_n6.npy')
x_warm = contract(P_warm)

def phi(P, NQ):
    gl_n, gl_w = np.polynomial.legendre.leggauss(NQ)
    return phi_pou_cr_jit(P, V_INV, LOBATTO, U_NODES, p_grid, gl_n, gl_w,
                             TAU, GAMMA, C_STRETCH, G, NQ)
def F_sym(x, NQ):
    return contract(phi(expand(x), NQ)) - x

# Warmup
for nq in [12, 14, 16, 18, 20]:
    _ = phi(sg(0.5*T), nq)

print('=== Warm-start each NQ from the NQ=16 machine-eps FP ===')
print(f'{"NQ":>4} {"F warm":>12} {"After NK":>12} {"slope":>9} {"def":>9}')
results = {}
for NQ in [12, 14, 16, 18, 20]:
    f_warm = float(np.max(np.abs(F_sym(x_warm, NQ))))
    F_func = lambda x, NQ=NQ: F_sym(x, NQ)
    try:
        x_nk = newton_krylov(F_func, x_warm, f_tol=1e-15, maxiter=30, verbose=False)
        f_nk = float(np.max(np.abs(F_func(x_nk))))
    except NoConvergence as e:
        x_nk = e.args[0]; f_nk = float(np.max(np.abs(F_func(x_nk))))
    P = expand(x_nk)
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    s = float(np.sum(L*T.ravel())/np.sum(T.ravel()**2))
    pred = s*T.ravel() + np.mean(L - s*T.ravel())
    d = float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))
    print(f'{NQ:>4d} {f_warm:>12.3e} {f_nk:>12.3e} {s:>9.4f} {d:>9.4f}')
    results[NQ] = dict(f_warm=f_warm, nk=f_nk, slope=s, deficit=d)

json.dump(results, open('/tmp/cheby_h0/warm_nq_check.json', 'w'),
            indent=2, default=str)

# Verify NQ=16 FP is also a FP of other NQs
print('\n=== Conclusion ===')
print(f'NQ=16 FP residual under each NQ operator:')
for NQ, r in results.items():
    print(f'  NQ={NQ}: |F(P_FP_16)| = {r["f_warm"]:.3e}')
print('If all are ~machine eps, the FP is INDEPENDENT of NQ — NQ=16 was')
print('just the easiest cold-start. Otherwise the FP varies with NQ.')
