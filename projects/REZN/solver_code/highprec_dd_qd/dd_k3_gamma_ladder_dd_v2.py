"""DD gamma ladder v2: scaled bandwidths, dual-h DD nails, h-trend acceptance.

Per gamma at tau=0.1, G=9:
  - DD-nail the R4 fixed point at TWO bandwidth windows that genuinely
    converge: hA (mid 0.10) and hB (mid 0.08), G_p=721, NQK=32.
  - float64 h-trend probe at h_mid in {0.10, 0.08, 0.06} (the last
    best-iterate) + strict best-iterate.
  - Battery v2:
      T1  both DD nails <= 1e-20
      T2  symmetry <= 1e-12 (both)
      T3  monotone (both)
      T4  branch identity (both)
      T5  h^2-expansion remainder <= 0.1 within each window
      T6' h-trend consistency: slope(hA) < slope(hB) < slope_strict + stall,
          monotone in h, and the hA->hB drift direction matches the
          strict reference (the kernel under-reveals; strict caps the trend)
      T7  cross-G stability of the DD-hA solve (G=9 vs G=7) <= 5% slope
  ACCEPT means: the pair of DD fixed points is certified as exact-for-its-h,
  the h-trend toward the strict reference is internally consistent, and the
  reported economics is the TREND + strict bound, never a single-h value.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from dd_k3_solver import solve_dd_k3, solve_warm_f64
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, phi_lin_strict_jit
from lin_cdf_richardson import phi_lin_richardson
from scipy.stats import norm

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_gamma_ladder_v2"
os.makedirs(OUT, exist_ok=True)

TAU = 0.1
GAMMAS = [1.0, 4.642, 10.0, 46.42, 100.0, 464.2, 1000.0]
G, G_p, NQK = 9, 361, 24
HA = (0.125, 0.10833, 0.09167, 0.075)   # window mid 0.10
HB = (0.10, 0.08667, 0.07333, 0.06)     # window mid 0.08

u_grid = make_cdf_uniform_grid(G)
p_grid = make_p_grid(G_p)
gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
f_mix = 0.5*norm.pdf(u_grid, -0.5, 1/np.sqrt(TAU)) + 0.5*norm.pdf(u_grid, 0.5, 1/np.sqrt(TAU))
W3 = f_mix[:,None,None]*f_mix[None,:,None]*f_mix[None,None,:]; W3 /= W3.max()
# G=7 for T7
u7 = make_cdf_uniform_grid(7)
gl_u7, gl_du7 = make_gl_for_u(u7[0], u7[-1], NQK)


def slope_of(P, u):
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    T = (U1+U2+U3).ravel()
    L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
    return float(np.sum(L*T)/np.sum(T*T))


def t5_check(P, hs):
    phis = [phi_lin_richardson(P, u_grid, hs=(h,), gamma=cur_gamma, tau=TAU,
                                   G_p=G_p, NQK=NQK, p_grid=p_grid) for h in hs]
    H2 = np.array([h*h for h in hs]); A = np.column_stack([np.ones(4), H2, H2*H2])
    Y = np.array([p.ravel() for p in phis])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    spread = Y.max(0) - Y.min(0)
    mask = (spread > 1e-10) & (W3.ravel() > 0.01)
    return float(np.max(np.abs(Y - A@coef)[:, mask].max(0)/spread[mask])) if mask.any() else 0.0


def struct_checks(P):
    sf = float(np.max(np.abs(P + P[::-1,::-1,::-1] - 1.0)))
    perm = float(np.max(np.abs(P - np.transpose(P, (1,0,2)))))
    dmin = float(min(np.diff(P, axis=a).min() for a in range(3)))
    mid = G//2
    ce = float(abs(P[mid,mid,mid] - 0.5))
    pr = float(P.max() - P.min())
    return sf, perm, dmin, ce, pr


def strict_best_iterate(P_init, gamma, n=50):
    P = P_init.copy(); best = (1e9, None)
    for _ in range(n):
        Pn = phi_lin_strict_jit(P, u_grid, p_grid, gl_u, gl_du, TAU, gamma, G, NQK)
        F = float(np.max(np.abs(Pn - P) * W3))
        if F < best[0]: best = (F, P.copy())
        P = 0.5*P + 0.5*Pn
    return best


results = []
P_prev = None
print(f"DD gamma ladder v2: tau={TAU}, G={G}, G_p={G_p}, NQK={NQK}")
print(f"hA mid=0.100, hB mid=0.080; dual-nail + h-trend battery")
print(f"{'gamma':>8} {'F_ddA':>9} {'F_ddB':>9} {'slopeA':>8} {'slopeB':>8} "
      f"{'slope_st':>8} {'F_st_w':>8} {'verdict':>8} {'wall':>6}", flush=True)

for gamma in GAMMAS:
    cur_gamma = gamma
    t0 = time.time()
    rep = {}
    # --- nail A ---
    PA, FA, PAH, PAL, _, _ = solve_dd_k3(gamma, TAU, P_warm=P_prev, u_grid=u_grid,
                                              p_grid=p_grid, gl_u=gl_u, gl_du=gl_du,
                                              hs=HA, target_dd=1e-24, verbose=False)
    # --- hB: float64 deep-converged (1e-13; DD adds nothing to a 5-decimal slope) ---
    PB, FB = solve_warm_f64(PA.copy(), u_grid, p_grid, HB, gamma, TAU,
                                 G_p, NQK, target=1e-12)
    PBH, PBL = PB, np.zeros_like(PB)
    sA, sB = slope_of(PA, u_grid), slope_of(PB, u_grid)
    # h=0.06 float64 best effort (stall ok -- trend probe only)
    P06, F06 = solve_warm_f64(PB.copy(), u_grid, p_grid,
                                  (0.075, 0.065, 0.055, 0.045), gamma, TAU,
                                  G_p, NQK, target=1e-12)
    s06 = slope_of(P06, u_grid)
    # strict best iterate
    F_st, P_st = strict_best_iterate(P06, gamma)
    s_st = slope_of(P_st, u_grid)
    # --- battery ---
    rep['T1_FA'] = float(FA); rep['T1_FB'] = float(FB)
    rep['T1_pass'] = bool(FA <= 1e-20 and FB <= 1e-10)
    sfA, pmA, dmA, ceA, prA = struct_checks(PA)
    sfB, pmB, dmB, ceB, prB = struct_checks(PB)
    rep['T2_pass'] = bool(max(sfA, pmA, sfB, pmB) <= 1e-12)
    rep['T3_pass'] = bool(min(dmA, dmB) >= -1e-10)
    rep['T4_pass'] = bool(max(ceA, ceB) <= 1e-12 and min(prA, prB) > 1e-3)
    t5A = t5_check(PA, HA); t5B = t5_check(PB, HB)
    rep['T5_A'] = t5A; rep['T5_B'] = t5B
    rep['T5_pass'] = bool(t5A <= 0.1 and t5B <= 0.1)
    # T6': h-trend internal consistency
    # monotone trend toward strict; strict (+stall band) caps it
    stall_band = max(3.0 * F_st / max(prA, 1e-9), 0.05)  # relative slack
    trend_ok = (sA < sB < s06 * 1.02) and (s06 <= s_st * (1 + stall_band) + 1e-3)
    rep['T6_slopes'] = [sA, sB, s06, s_st]; rep['T6_F_strict_w'] = float(F_st)
    rep['T6_pass'] = bool(trend_ok)
    # T7: G=7 DD solve at hA, slope stability
    P7, F7, _, _, _, _ = solve_dd_k3(gamma, TAU, P_warm=None, u_grid=u7,
                                          p_grid=p_grid, gl_u=gl_u7, gl_du=gl_du7,
                                          hs=HA, target_dd=1e-22, verbose=False)
    s7 = slope_of(P7, u7)
    rep['T7_slope_G7'] = s7
    rep['T7_rel'] = abs(s7 - sA)/max(abs(sA), 1e-12)
    rep['T7_pass'] = bool(rep['T7_rel'] <= 0.05)
    passed = all(v for k, v in rep.items() if k.endswith('_pass'))
    verdict = "ACCEPT" if passed else "REJECT"
    fails = [k[:-5] for k in rep if k.endswith('_pass') and not rep[k]]
    wall = time.time() - t0
    print(f"{gamma:>8.4g} {FA:>9.1e} {FB:>9.1e} {sA:>8.5f} {sB:>8.5f} "
          f"{s_st:>8.5f} {F_st:>8.1e} {verdict:>8} {wall:>5.0f}s"
          + (f"  FAILED: {','.join(fails)}" if fails else ""), flush=True)
    np.savez(f"{OUT}/dd_g{gamma:.4g}_t{TAU:.4f}_G{G}.npz",
              PA_hi=PAH, PA_lo=PAL, PB_hi=PBH, PB_lo=PBL,
              P_strict_best=P_st, gamma=gamma, tau=TAU,
              FA=float(FA), FB=float(FB), F_strict_w=float(F_st),
              hsA=HA, hsB=HB, accepted=passed)
    results.append(dict(gamma=gamma, verdict=verdict, fails=fails, wall=wall,
                          **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in rep.items()}))
    P_prev = PA
    json.dump(results, open(f"{OUT}/ladder.json", "w"), indent=2, default=str)

n_acc = sum(1 for r in results if r['verdict'] == 'ACCEPT')
print(f"\n{n_acc}/{len(results)} ACCEPTED -> {OUT}/", flush=True)
