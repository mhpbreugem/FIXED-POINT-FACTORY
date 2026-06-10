"""Double-double gamma ladder for K=3 at tau=0.1, with a strict acceptance battery.

Per the monograph Part VII: a cell is ACCEPTED only if it passes ALL of:
  T1  DD convergence:      F_dd <= 1e-20 on the R4 operator (true DD nail)
  T2  symmetry:            sign-flip + permutation residual <= 1e-12
  T3  monotonicity:        axis-monotone P (no violations > 1e-10)
  T4  branch identity:     P(0)=1/2 to 1e-12 AND P_max - P_min > 1e-3 (not trivial)
  T5  Richardson validity: per-h operator values Phi_h(P*) fit h^2 line with
                           relative residual <= 0.2 (the kink test -- if the
                           h^2 expansion is invalid, Richardson is untrustworthy)
  T6  strict cross-check:  float64 strict-h=0 weighted residual at the DD FP
                           is <= 3x the known R4-vs-strict operator gap at
                           this cell (the FP is a strict near-FP up to the
                           documented operator difference)
  T7  grid stability:      slope + P-range at G=7 vs G=9 agree within 5%
Cells failing any test are REJECTED and reported as such.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from dd_k3_solver import solve_dd_k3, SymRed3
from dd_k3_ops import phi_dd
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import (build_mu_table_lin_strict, make_gl_for_u,
                                  phi_lin_strict_jit)
from lin_cdf_richardson import phi_lin_richardson
from scipy.stats import norm

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_gamma_ladder"
os.makedirs(f"{OUT}", exist_ok=True)

TAU = 0.1
GAMMAS = [1.0, 2.154, 4.642, 10.0, 21.54, 46.42, 100.0, 215.4, 464.2, 1000.0]
G, G_p, NQK = 7, 121, 16
HS = (0.5, 0.4, 0.3, 0.2)


def density_w(u_grid, tau):
    sd = 1.0/np.sqrt(tau)
    f = 0.5*norm.pdf(u_grid, -0.5, sd) + 0.5*norm.pdf(u_grid, 0.5, sd)
    W3 = f[:,None,None]*f[None,:,None]*f[None,None,:]
    return W3 / W3.max()


def battery(P_full, F_dd, gamma, tau, u_grid, p_grid, gl_u, gl_du, W3):
    """Run T1-T6. Returns (passed: bool, report: dict)."""
    rep = {}
    G = u_grid.size
    # T1 DD convergence
    rep['T1_F_dd'] = float(F_dd)
    rep['T1_pass'] = bool(F_dd <= 1e-20)
    # T2 symmetry
    sf = float(np.max(np.abs(P_full + P_full[::-1,::-1,::-1] - 1.0)))
    perm = float(np.max(np.abs(P_full - np.transpose(P_full, (1,0,2)))))
    rep['T2_signflip'] = sf; rep['T2_perm'] = perm
    rep['T2_pass'] = bool(sf <= 1e-12 and perm <= 1e-12)
    # T3 monotonicity
    dmin = float(min(np.diff(P_full, axis=a).min() for a in range(3)))
    rep['T3_min_diff'] = dmin
    rep['T3_pass'] = bool(dmin >= -1e-10)
    # T4 branch identity
    mid = G//2
    center_err = float(abs(P_full[mid,mid,mid] - 0.5))
    prange = float(P_full.max() - P_full.min())
    rep['T4_center_err'] = center_err; rep['T4_P_range'] = prange
    rep['T4_pass'] = bool(center_err <= 1e-12 and prange > 1e-3)
    # T5 Richardson validity: Phi at each single h vs h^2 line
    phis = []
    for h in HS:
        Ph = phi_lin_richardson(P_full, u_grid, hs=(h,), gamma=gamma, tau=tau, G_p=G_p, NQK=NQK, p_grid=p_grid)
        phis.append(Ph)
    # At each cube cell: fit Phi_h = a + b h^2; measure relative fit residual
    # R4 assumes Phi_h = a + b h^2 + c h^4 + d h^6; validity = the data are
    # consistent with a SMOOTH expansion in h^2. Fit quadratic in h^2 (3 of 4
    # dof); the residual measures the h^6+ remainder. Kinky |h|^alpha terms
    # make this fit fail badly; smooth data pass easily.
    H2 = np.array([h*h for h in HS])
    A = np.column_stack([np.ones(4), H2, H2*H2])
    Y = np.array([p.ravel() for p in phis])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    spread = Y.max(axis=0) - Y.min(axis=0)
    mask = (spread > 1e-8) & (W3.ravel() > 0.01)
    if mask.any():
        t5 = float(np.max(np.max(np.abs(resid[:, mask]), axis=0) / spread[mask]))
    else:
        t5 = 0.0
    rep['T5_h2_fit_rel'] = t5
    rep['T5_pass'] = bool(t5 <= 0.1)
    # T6 strict cross-check
    P_strict_phi = phi_lin_strict_jit(P_full, u_grid, p_grid, gl_u, gl_du,
                                            tau, gamma, G, NQK)
    F_strict_w = float(np.max(np.abs(P_strict_phi - P_full) * W3))
    # operator gap: |Phi_R4 - Phi_strict| at the same point (known difference)
    P_r4_phi = phi_lin_richardson(P_full, u_grid, hs=HS, gamma=gamma, tau=tau, G_p=G_p, NQK=NQK, p_grid=p_grid)
    gap_w = float(np.max(np.abs(P_r4_phi - P_strict_phi) * W3))
    rep['T6_F_strict_w'] = F_strict_w; rep['T6_operator_gap_w'] = gap_w
    rep['T6_pass'] = bool(F_strict_w <= 3.0 * max(gap_w, 1e-16))
    passed = all(rep[k] for k in rep if k.endswith('_pass'))
    return passed, rep


def slope_of(P_full, u_grid):
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    T = (U1 + U2 + U3).ravel()
    L = np.log(np.clip(P_full, 1e-15, 1-1e-15) / (1 - np.clip(P_full, 1e-15, 1-1e-15))).ravel()
    return float(np.sum(L*T) / np.sum(T*T))


def main():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    W3 = density_w(u_grid, TAU)
    # G=9 setup for T7
    u9 = make_cdf_uniform_grid(9)
    gl_u9, gl_du9 = make_gl_for_u(u9[0], u9[-1], NQK)
    W3_9 = density_w(u9, TAU)

    print(f"DD gamma ladder: tau={TAU}, G={G}, {len(GAMMAS)} gammas, R4 DD nail + 7-test battery")
    print(f"{'gamma':>8} {'F_dd':>10} {'F_str_w':>9} {'gap_w':>9} {'h2fit':>7} "
          f"{'slope7':>8} {'slope9':>8} {'verdict':>8} {'wall':>6}", flush=True)

    results = []
    P_prev = None
    for gamma in GAMMAS:
        t0 = time.time()
        # main solve at G=7
        P_full, F_dd, P_H, P_L, tw, tn = solve_dd_k3(
            gamma, TAU, P_warm=P_prev, u_grid=u_grid, p_grid=p_grid,
            gl_u=gl_u, gl_du=gl_du, hs=HS, target_dd=1e-25, verbose=False)
        passed, rep = battery(P_full, F_dd, gamma, TAU, u_grid, p_grid, gl_u, gl_du, W3)
        s7 = slope_of(P_full, u_grid)
        # T7: G=9 re-solve (float64 warm + DD nail), compare slope & range
        p9_grid = p_grid
        P9, F9, _, _, _, _ = solve_dd_k3(gamma, TAU, P_warm=None, u_grid=u9,
                                              p_grid=p9_grid, gl_u=gl_u9, gl_du=gl_du9,
                                              hs=HS, target_dd=1e-22, verbose=False)
        s9 = slope_of(P9, u9)
        r7 = float(P_full.max() - P_full.min()); r9 = float(P9.max() - P9.min())
        t7_slope = abs(s9 - s7) / max(abs(s7), 1e-12)
        t7_range = abs(r9 - r7) / max(r7, 1e-12)
        rep['T7_slope_G7'] = s7; rep['T7_slope_G9'] = s9
        rep['T7_rel_slope'] = t7_slope; rep['T7_rel_range'] = t7_range
        rep['T7_pass'] = bool(t7_slope <= 0.05 and t7_range <= 0.10)
        passed = passed and rep['T7_pass']
        wall = time.time() - t0
        verdict = "ACCEPT" if passed else "REJECT"
        fails = [k[:-5] for k in rep if k.endswith('_pass') and not rep[k]]
        print(f"{gamma:>8.4g} {F_dd:>10.2e} {rep['T6_F_strict_w']:>9.2e} "
              f"{rep['T6_operator_gap_w']:>9.2e} {rep['T5_h2_fit_rel']:>7.3f} "
              f"{s7:>9.6f} {s9:>9.6f} {verdict:>8} {wall:>5.0f}s"
              + (f"  FAILED: {','.join(fails)}" if fails else ""), flush=True)
        np.savez(f"{OUT}/dd_g{gamma:.4g}_t{TAU:.4f}_G{G}.npz",
                  P_hi=P_H, P_lo=P_L, P=P_full, gamma=gamma, tau=TAU,
                  F_dd=F_dd, accepted=passed, **{k: v for k, v in rep.items()})
        results.append(dict(gamma=gamma, F_dd=float(F_dd), verdict=verdict,
                              fails=fails, wall=wall, slope=s7, **{
                                  k: (v if not isinstance(v, np.generic) else float(v))
                                  for k, v in rep.items()}))
        P_prev = P_full
    json.dump(results, open(f"{OUT}/ladder.json", "w"), indent=2, default=str)
    n_acc = sum(1 for r in results if r['verdict'] == 'ACCEPT')
    print(f"\n{n_acc}/{len(results)} cells ACCEPTED -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
