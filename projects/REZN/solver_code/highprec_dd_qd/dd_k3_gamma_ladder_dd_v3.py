"""DD gamma ladder v3: dual DD nails, F < 1e-20 acceptance on BOTH bandwidths.

Per user instruction "only accept ferr lower than e-20 with DD": both
the hA and hB fixed points must be DOUBLE-DOUBLE nailed below 1e-20.
Acceptance is:
   T1  F_dd(h_A) <= 1e-20   AND   F_dd(h_B) <= 1e-20
   T2  symmetry (sign-flip + permutation) <= 1e-12 at BOTH
   T3  axis-monotonicity at BOTH
   T4  branch identity (P(0)=1/2 to 1e-12, non-trivial range) at BOTH
   T5  smooth h^2-expansion within each window (qudratic-in-h^2 fit
       remainder <= 0.1)
   T6  h-trend monotone: slope(h_A) < slope(h_B) (the residual partial
       revelation shrinks as h is reduced toward h=0, in the direction
       the strict reference points)
   T7  G=7 DD nail (also F < 1e-20) slope agreement with G=9 within 5%

No float64 layer in the acceptance test -- everything that touches the
verdict is DD-nailed below 1e-20.

GAMMAS shortened to 5 for the ~3h budget at ~25 min/cell. tau=0.1, G=9.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
from dd_k3_solver import solve_dd_k3
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, phi_lin_strict_jit
from lin_cdf_richardson import phi_lin_richardson
from scipy.stats import norm

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_gamma_ladder_v3"
os.makedirs(OUT, exist_ok=True)

TAU = 0.1
GAMMAS = [1.0, 10.0, 46.42, 100.0, 1000.0]
G, G_p, NQK = 9, 361, 24
HA = (0.125, 0.10833, 0.09167, 0.075)   # mid 0.100
HB = (0.10, 0.08667, 0.07333, 0.06)     # mid 0.080  -- finest still-DD-nailable
DD_TARGET = 1e-22  # solver target; verdict gate is 1e-20

u_grid = make_cdf_uniform_grid(G)
p_grid = make_p_grid(G_p)
gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
f_mix = 0.5*norm.pdf(u_grid, -0.5, 1/np.sqrt(TAU)) + 0.5*norm.pdf(u_grid, 0.5, 1/np.sqrt(TAU))
W3 = f_mix[:,None,None]*f_mix[None,:,None]*f_mix[None,None,:]; W3 /= W3.max()
u7 = make_cdf_uniform_grid(7)
gl_u7, gl_du7 = make_gl_for_u(u7[0], u7[-1], NQK)


def slope_of(P, u):
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    T = (U1+U2+U3).ravel()
    L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
    return float(np.sum(L*T)/np.sum(T*T))


def struct_checks(P):
    sf = float(np.max(np.abs(P + P[::-1,::-1,::-1] - 1.0)))
    perm = float(np.max(np.abs(P - np.transpose(P, (1,0,2)))))
    dmin = float(min(np.diff(P, axis=a).min() for a in range(3)))
    mid = G//2
    ce = float(abs(P[mid,mid,mid] - 0.5))
    pr = float(P.max() - P.min())
    return sf, perm, dmin, ce, pr


def t5_check(P, hs, gamma):
    phis = [phi_lin_richardson(P, u_grid, hs=(h,), gamma=gamma, tau=TAU,
                                   G_p=G_p, NQK=NQK, p_grid=p_grid) for h in hs]
    H2 = np.array([h*h for h in hs]); A = np.column_stack([np.ones(4), H2, H2*H2])
    Y = np.array([p.ravel() for p in phis])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    spread = Y.max(0) - Y.min(0)
    mask = (spread > 1e-10) & (W3.ravel() > 0.01)
    return float(np.max(np.abs(Y - A@coef)[:, mask].max(0)/spread[mask])) if mask.any() else 0.0


def strict_best_iterate(P_init, gamma, n=40):
    P = P_init.copy(); best = (1e9, None)
    for _ in range(n):
        Pn = phi_lin_strict_jit(P, u_grid, p_grid, gl_u, gl_du, TAU, gamma, G, NQK)
        F = float(np.max(np.abs(Pn - P) * W3))
        if F < best[0]: best = (F, P.copy())
        P = 0.5*P + 0.5*Pn
    return best


results = []
P_prev = None
print(f"DD gamma ladder v3: tau={TAU}, G={G}, G_p={G_p}, NQK={NQK}")
print(f"hA mid=0.100, hB mid=0.080; DUAL DD nails; verdict gate F_dd <= 1e-20")
print(f"{'gamma':>8} {'F_ddA':>9} {'F_ddB':>9} {'slopeA':>8} {'slopeB':>8} "
      f"{'slope_st':>8} {'F_st':>8} {'verdict':>8} {'wall':>6}", flush=True)

for gamma in GAMMAS:
    t0 = time.time()
    rep = {}
    # --- DD nail at hA ---
    PA, FA, PAH, PAL, _, _ = solve_dd_k3(gamma, TAU, P_warm=P_prev, u_grid=u_grid,
                                              p_grid=p_grid, gl_u=gl_u, gl_du=gl_du,
                                              hs=HA, target_dd=DD_TARGET, verbose=False)
    # --- DD nail at hB warm-started from PA ---
    PB, FB, PBH, PBL, _, _ = solve_dd_k3(gamma, TAU, P_warm=PA, u_grid=u_grid,
                                              p_grid=p_grid, gl_u=gl_u, gl_du=gl_du,
                                              hs=HB, target_dd=DD_TARGET, verbose=False)
    sA, sB = slope_of(PA, u_grid), slope_of(PB, u_grid)
    # strict reference (best iterate)
    F_st, P_st = strict_best_iterate(PB, gamma)
    s_st = slope_of(P_st, u_grid)
    # --- battery ---
    rep['T1_FA'] = float(FA); rep['T1_FB'] = float(FB)
    rep['T1_pass'] = bool(FA <= 1e-20 and FB <= 1e-20)
    sfA, pmA, dmA, ceA, prA = struct_checks(PA)
    sfB, pmB, dmB, ceB, prB = struct_checks(PB)
    rep['T2_signflip_max'] = max(sfA, sfB); rep['T2_perm_max'] = max(pmA, pmB)
    rep['T2_pass'] = bool(max(sfA, pmA, sfB, pmB) <= 1e-12)
    rep['T3_pass'] = bool(min(dmA, dmB) >= -1e-10)
    rep['T4_pass'] = bool(max(ceA, ceB) <= 1e-12 and min(prA, prB) > 1e-3)
    t5A = t5_check(PA, HA, gamma); t5B = t5_check(PB, HB, gamma)
    rep['T5_A'] = t5A; rep['T5_B'] = t5B
    rep['T5_pass'] = bool(t5A <= 0.1 and t5B <= 0.1)
    rep['T6_slopes'] = [sA, sB, s_st]; rep['T6_F_strict'] = float(F_st)
    rep['T6_pass'] = bool(sA < sB)
    # T7: G=7 DD nail at hA
    P7, F7, _, _, _, _ = solve_dd_k3(gamma, TAU, P_warm=None, u_grid=u7,
                                          p_grid=p_grid, gl_u=gl_u7, gl_du=gl_du7,
                                          hs=HA, target_dd=DD_TARGET, verbose=False)
    s7 = slope_of(P7, u7)
    rep['T7_F7'] = float(F7); rep['T7_slope_G7'] = s7
    rep['T7_rel'] = abs(s7 - sA) / max(abs(sA), 1e-12)
    rep['T7_pass'] = bool(F7 <= 1e-20 and rep['T7_rel'] <= 0.05)
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
