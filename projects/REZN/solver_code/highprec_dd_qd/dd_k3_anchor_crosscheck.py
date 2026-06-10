"""Cross-check multiple K=3 methods against the k3_coarea_limit immortal anchor
at (tau=2, gamma=0.1).

The anchor: joint-limit kernel co-area solver nailed F~1e-11..1e-14 across
G in {9,13,17,21,25,31}, deficit converging to ~0.282-0.291.

Methods evaluated here (all on the same cell):
  M1  legacy R4 DD nail at G=7  (hs=0.5..0.2)
  M2  scaled R4 DD nail at G=9  (hs scaled to local P-range)
  M3  strict-h=0 best-iterate (Anderson) at G=9
  M4  log-odds Picard (no bounded cube) at G=11
  M5  log-odds Newton-Krylov (IFT) at G=11

Reported per method: F (own), F under STRICT-h=0 (cross-operator), deficit
(1-R^2), slope, and absolute deviation from the anchor deficit (0.282).
"""
import os, sys, time, json
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
import numpy as np
from scipy.stats import norm
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, phi_lin_strict_jit
from lin_cdf_richardson import phi_lin_richardson
from dd_k3_solver import solve_dd_k3, solve_warm_f64

TAU, GAMMA = 2.0, 0.1
TARGET_DEFICIT = 0.282  # k3_coarea_limit consensus

def stats(P, u, p_grid, gl_u, gl_du, G, nqk, gamma=GAMMA, tau=TAU):
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    T = (U1+U2+U3).ravel()
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    # slope = sum LT / sum TT
    slope = float(np.sum(L*T)/np.sum(T*T))
    # deficit = 1 - R^2 with regressor "which T-bin"
    pred = slope*T + np.mean(L - slope*T)
    R2 = 1 - float(np.sum((L-pred)**2)/np.sum((L-L.mean())**2))
    # group by unique T (deficit_oneToOne)
    uT, inv = np.unique(np.round(T, 10), return_inverse=True)
    ss = float(np.sum((L - L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m] - L[m].mean())**2))
    deficit = w/ss
    # strict cross-residual (Gaussian-weighted)
    sd = 1/np.sqrt(tau)
    f = 0.5*norm.pdf(u,-0.5,sd) + 0.5*norm.pdf(u,0.5,sd)
    W3 = f[:,None,None]*f[None,:,None]*f[None,None,:]; W3 /= W3.max()
    Pst = phi_lin_strict_jit(P, u, p_grid, gl_u, gl_du, tau, gamma, G, nqk)
    F_strict = float(np.max(np.abs(Pst - P) * W3))
    return slope, deficit, F_strict, float(P.min()), float(P.max())

def method1_legacy_R4_G7():
    G, G_p, NQK = 7, 121, 16
    hs = (0.5, 0.4, 0.3, 0.2)
    u = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)
    t0 = time.time()
    P, F, _, _, _, _ = solve_dd_k3(GAMMA, TAU, u_grid=u, p_grid=p_grid,
                                       gl_u=gl_u, gl_du=gl_du, hs=hs,
                                       target_dd=1e-22, verbose=False)
    s, d, Fst, pmin, pmax = stats(P, u, p_grid, gl_u, gl_du, G, NQK)
    return dict(method="M1 legacy R4 DD G=7", wall=time.time()-t0,
                F_own=float(F), F_strict_w=Fst, slope=s, deficit=d,
                pmin=pmin, pmax=pmax)

def method2_scaled_R4_G9():
    G, G_p, NQK = 9, 361, 24
    # P range at tau=2 ~ 0.2..0.8; use hs that scale with that
    hs = (0.16, 0.13, 0.10, 0.07)
    u = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)
    t0 = time.time()
    P, F, _, _, _, _ = solve_dd_k3(GAMMA, TAU, u_grid=u, p_grid=p_grid,
                                       gl_u=gl_u, gl_du=gl_du, hs=hs,
                                       target_dd=1e-22, verbose=False)
    s, d, Fst, pmin, pmax = stats(P, u, p_grid, gl_u, gl_du, G, NQK)
    return dict(method="M2 scaled R4 DD G=9", wall=time.time()-t0,
                F_own=float(F), F_strict_w=Fst, slope=s, deficit=d,
                pmin=pmin, pmax=pmax)

def method3_strict_best_iter_G9():
    G, G_p, NQK = 9, 361, 24
    u = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)
    # warm with scaled R4
    hs = (0.16, 0.13, 0.10, 0.07)
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    P0 = 1/(1+np.exp(-0.5*(U1+U2+U3)))
    Pw, _ = solve_warm_f64(P0, u, p_grid, hs, GAMMA, TAU, G_p, NQK, target=1e-12)
    t0 = time.time()
    P = Pw.copy(); sd = 1/np.sqrt(TAU)
    f = 0.5*norm.pdf(u,-0.5,sd) + 0.5*norm.pdf(u,0.5,sd)
    W3 = f[:,None,None]*f[None,:,None]*f[None,None,:]; W3 /= W3.max()
    best = (1e9, None)
    for _ in range(80):
        Pn = phi_lin_strict_jit(P, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK)
        F = float(np.max(np.abs(Pn - P) * W3))
        if F < best[0]: best = (F, P.copy())
        P = 0.5*P + 0.5*Pn
    P = best[1]
    s, d, Fst, pmin, pmax = stats(P, u, p_grid, gl_u, gl_du, G, NQK)
    return dict(method="M3 strict-h=0 best-iter G=9", wall=time.time()-t0,
                F_own=float(best[0]), F_strict_w=Fst, slope=s, deficit=d,
                pmin=pmin, pmax=pmax)


def main():
    print(f"Cross-check vs k3_coarea_limit anchor: tau={TAU}, gamma={GAMMA}, "
          f"deficit_anchor = {TARGET_DEFICIT}", flush=True)
    print(f"{'method':<32} {'F_own':>9} {'F_str':>9} {'slope':>8} {'deficit':>8} "
          f"{'|d - anchor|':>13}", flush=True)
    out = []
    for fn in [method1_legacy_R4_G7, method2_scaled_R4_G9, method3_strict_best_iter_G9]:
        r = fn()
        r['deficit_dev'] = abs(r['deficit'] - TARGET_DEFICIT)
        print(f"{r['method']:<32} {r['F_own']:>9.1e} {r['F_strict_w']:>9.1e} "
              f"{r['slope']:>8.5f} {r['deficit']:>8.5f} {r['deficit_dev']:>13.4f}"
              f"  P=[{r['pmin']:.4f},{r['pmax']:.4f}] ({r['wall']:.0f}s)", flush=True)
        out.append(r)
    OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/anchor_crosscheck"
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()
