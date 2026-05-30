"""HELLWIG TEST for CARA in our model: does the analytic FR price p = lam(tau*Sigma u)
satisfy the CARA operator? + spectral radius of Phi'_CARA at FR.

Three possible outcomes:
  (i)  Phi_CARA(P_FR) ~ P_FR (small residual) AND rho < 1 -> FR is stable; Picard converging
       elsewhere means a bug.
  (ii) Phi_CARA(P_FR) ~ P_FR AND rho >= 1 -> FR exists as a discrete fixed point but is a
       Picard-UNSTABLE saddle -> Picard escapes; the deficit~0.09 non-FR plateau is the stable
       attractor of the discrete operator. (Hellwig's existence stands; uniqueness/stability fails.)
  (iii) Phi_CARA(P_FR) NOT close to P_FR -> the discrete op does not even admit FR; the operator
        implementation has a subtle issue (or our model differs from Hellwig's setup more than expected).
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from cara_operator import phi_cara, _grid
from scipy.sparse.linalg import LinearOperator, eigs

TAU = 2.0; a = 1.0; W = 1.0
results = []
for G in [9, 13, 17]:
    du, h, uf, lo, hi, T = _grid(G, TAU)
    slc = (slice(lo, hi),) * 3
    U = np.meshgrid(uf, uf, uf, indexing='ij')
    P_FR_full = np.clip(1.0 / (1.0 + np.exp(-TAU * (U[0] + U[1] + U[2]))), 1e-9, 1 - 1e-9)
    tv = np.full(3, TAU); av = np.full(3, a); Wv = np.full(3, W)

    # (1) Hellwig test: apply Phi_CARA to P_FR
    P_new = phi_cara(P_FR_full, uf, lo, hi, tv, av, Wv, h, model='logodds')
    F_inf = float(np.max(np.abs((P_new - P_FR_full)[slc])))
    Pc = np.clip(P_FR_full[slc], 1e-12, 1 - 1e-12); y = np.log(Pc / (1 - Pc)).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0] * T.ravel() + aa[1]
    defi_FR = float(np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))
    rec = dict(G=G, h=float(h), FR_residual_Finf=F_inf, deficit_at_FR=defi_FR)
    print(f"G={G:2d}: ||Phi_CARA(P_FR) - P_FR||_inf = {F_inf:.3e}   deficit(P_FR) = {defi_FR:.2e}", flush=True)

    # (2) Spectral radius of Phi'_CARA at FR (matrix-free Arnoldi)
    if G <= 17:
        N = G ** 3; Pstar = P_FR_full[slc].copy().ravel()
        halo = P_FR_full.copy(); eps_fd = 1e-6
        def Jv(v):
            Pp = halo.copy(); Pp[slc] = (Pstar + eps_fd * v).reshape((G,) * 3)
            Pm = halo.copy(); Pm[slc] = (Pstar - eps_fd * v).reshape((G,) * 3)
            return ((phi_cara(Pp, uf, lo, hi, tv, av, Wv, h, model='logodds')
                     - phi_cara(Pm, uf, lo, hi, tv, av, Wv, h, model='logodds'))[slc].ravel()) / (2 * eps_fd)
        t = time.time()
        L = LinearOperator((N, N), matvec=Jv, dtype=np.float64)
        try:
            vals, _ = eigs(L, k=6, which='LM', tol=1e-4, maxiter=200)
            rho = float(np.max(np.abs(vals)))
            rec['rho_at_FR'] = rho
            rec['top_eig_abs'] = sorted([float(abs(v)) for v in vals], reverse=True)
            print(f"  rho(Phi'_CARA at P_FR) = {rho:.4f}   top|lambda|: {rec['top_eig_abs']}   ({time.time()-t:.0f}s)", flush=True)
        except Exception as e:
            print(f"  eigs failed: {e}")
    results.append(rec)
    json.dump({'tau': TAU, 'a': a, 'rows': results}, open(os.path.join(HERE, 'FR_test.json'), 'w'), indent=2)

# Verdict
print("\n=== HELLWIG VERDICT ===")
for r in results:
    fr_is_fp = r['FR_residual_Finf'] < 1e-3
    msg = f"G={r['G']}: "
    if not fr_is_fp:
        msg += f"FR NOT a discrete FP (resid={r['FR_residual_Finf']:.1e}) -> (iii) operator/model issue"
    else:
        if 'rho_at_FR' in r:
            if r['rho_at_FR'] >= 1.0 - 0.05:
                msg += f"FR IS a discrete FP (resid {r['FR_residual_Finf']:.1e}) but UNSTABLE (rho={r['rho_at_FR']:.3f}) -> (ii) Picard escapes; non-FR plateau is the Picard attractor. Hellwig existence holds; stability fails on the discrete operator."
            else:
                msg += f"FR IS a stable FP (resid {r['FR_residual_Finf']:.1e}, rho={r['rho_at_FR']:.3f}<1) -> (i) Picard should reach FR; non-FR plateau is a bug or non-eq."
        else:
            msg += f"FR IS a discrete FP (resid {r['FR_residual_Finf']:.1e}); rho not computed."
    print(msg)
print("DONE", flush=True)
