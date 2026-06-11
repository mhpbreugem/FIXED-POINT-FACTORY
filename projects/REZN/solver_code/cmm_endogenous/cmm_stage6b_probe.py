"""Probe: is the m=3 (p=0.417) cost stationary at the warm start, and is
the FD Jacobian consistent with direct cost differences?"""
import sys
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from cmm_stage6b_solver import (Problem, load_P_full, make_levels,
                                 build_initial_H, OUT)

tau, gamma = 2.0, 0.098
du, uf, lo, hi, P_inner, P_full = load_P_full(21, tau, gamma)
M, Na, margin = 8, 15, 1
p_levels = make_levels(P_inner, M)
pb = Problem(M, Na, half_width=4.5, n_vert_margin=margin, tau=tau, gamma=gamma)
H0 = build_initial_H(P_full, uf, p_levels, pb.a_grid, pb.b_grid)

m = 3
Hm = np.ascontiguousarray(H0[m]); p_m = p_levels[m]
r0, c0, nf = pb.residual(Hm, p_m)
cost0 = 0.5*r0 @ r0
print(f"m={m} p={p_m:.3f}: cost0={cost0:.6e} max|r|={np.max(np.abs(r0)):.3e} "
      f"med={np.median(np.abs(r0)):.3e} nfail={nf}")

# 1) rigid t-translations
for d in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10):
    r, _, _ = pb.residual(np.ascontiguousarray(Hm + d), p_m)
    print(f"  translate dt={d:+.2f}: cost={0.5*r@r:.6e} "
          f"med|r|={np.median(np.abs(r)):.3e}")

# 2) FD-vs-Jacobian consistency on random directions
J = pb.jacobian(Hm, p_m, c0.copy(), r0)
g = J.T @ r0
print(f"||J^T r|| = {np.linalg.norm(g):.3e}   ||r|| = {np.linalg.norm(r0):.3e}")
print(f"||J|| fro = {np.linalg.norm(J):.3e}")
rng = np.random.default_rng(1)
for trial in range(4):
    v = rng.standard_normal(Na*Na); v /= np.linalg.norm(v)
    eps = 1e-4
    rp, _, _ = pb.residual(np.ascontiguousarray(Hm + eps*v.reshape(Na, Na)),
                            p_m)
    dcost_fd = (0.5*rp@rp - cost0)/eps
    dcost_J = g @ v
    print(f"  dir {trial}: dcost FD={dcost_fd:+.6e}  J-pred={dcost_J:+.6e}")

# 3) the GN step itself, undamped least-norm: does it reduce cost at small scale?
U, S, Vt = np.linalg.svd(J, full_matrices=False)
print(f"  J singular values: max={S[0]:.3e} 10th={S[9]:.3e} 50th={S[49]:.3e} "
      f"100th={S[99]:.3e} min={S[-1]:.3e}")
d_gn = -Vt.T @ ((U.T @ r0)/np.maximum(S, 1e-6*S[0]))
for sc in (1.0, 0.3, 0.1, 0.03):
    rp, _, _ = pb.residual(np.ascontiguousarray(Hm + sc*d_gn.reshape(Na, Na)), p_m)
    print(f"  GN step x{sc}: cost={0.5*rp@rp:.6e} max|r|={np.max(np.abs(rp)):.3e}")
# residual decomposition: component of r in range(J)
r_range = U @ (U.T @ r0)
print(f"  ||P_range r||/||r|| = {np.linalg.norm(r_range)/np.linalg.norm(r0):.4f}")
