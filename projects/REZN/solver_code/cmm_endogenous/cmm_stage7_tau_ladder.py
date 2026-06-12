"""CMM Stage 7: strict-h=0 tau-ladder + tilt-audit screening.

HYPOTHESIS (user's): the Stage 6b strict-h=0 ill-posedness (surface tilt
saturating at sqrt(2) on the one-signal-dominance band; CRRA price atoms
near 1/3, 2/3) is MONOTONE IN tau. Low-tau equilibria have flatter price
surfaces -> strict h=0 should be well-posed and convergent below some
tau*(gamma).

T1 (screen): for each certified emin15 kernel FP, build the Stage 6b
initial heights H_m (diagonal root-solve of P(u)=p_m) and audit the
transverse tilt |grad H_m(a,b)| on a fine sample. Classes:
  WELL-POSED  max_tilt < 1.0
  MARGINAL    1.0 <= max_tilt < sqrt(2)
  ILL-POSED   max_tilt >= sqrt(2)   (graph premise violated somewhere)

T2 (ladder): at gamma=0.098, solve the strict-h=0 graph formulation
(Stage 6b machinery, graph-in-s residual + LM, density-masked rows) on a
tau-ladder 0.2 -> 0.3 -> 0.5 -> 0.7 -> 1.0 -> 1.5 -> 2.0.  Anchor taus
(kernel FP available) warm-start from the kernel FP with that cell's own
M=8 quantile p-levels; intermediate taus continue from the previous
converged H (same p-levels).  Record residuals, tilt audit of the
CONVERGED H, and the reconstructed deficit.  tau* = first tau where LM
will not pass max|r| ~ 1e-3 (or supercritical tilt appears on the
converged surfaces).

Reused verbatim from Stage 6b: build_pad/heval (C1 bicubic + clamped
linear extrapolation), Problem (graph-in-s residual + FD Jacobian),
build_initial_H, deficit_from_H, make_levels, signflip_diag.
"""
import os, sys, time, json, argparse
import numpy as np
from numba import njit

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from cmm_stage6b_solver import (build_pad, heval, Problem, load_P_full,
                                 make_levels, build_initial_H,
                                 deficit_from_H, unweighted_deficit,
                                 signflip_diag, E_A, E_B, E_T, SQ3, OUT,
                                 EMIN15)
from cmm_stage1 import build_grid

SQRT2 = float(np.sqrt(2.0))
EMIN15_JSON = f"{EMIN15}/emin15.json"
SCREEN_JSON = f"{OUT}/stage7_tilt_screen.json"
LADDER_JSON = f"{OUT}/stage7_tau_ladder.json"


# ---------------- tilt audit (numba; reuses Stage 6b heval) ----------------

@njit(cache=True)
def _tilt_fields(Hp, a0, da, Na, b0, db, Nb, fine, ea, eb):
    """|grad H| and worst-direction Fc = 2/sqrt3 + Ha e_a[k] + Hb e_b[k]
    over the fine x fine sample. Returns (gmag, fcmin) arrays."""
    n = fine.size
    g = np.empty((n, n))
    fc = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            _, Ha, Hb = heval(Hp, a0, da, Na, b0, db, Nb, fine[i], fine[j])
            g[i, j] = np.sqrt(Ha*Ha + Hb*Hb)
            worst = 1e9
            for k in range(3):
                v = 2.0/SQ3 + Ha*ea[k] + Hb*eb[k]
                if v < worst:
                    worst = v
            fc[i, j] = worst
    return g, fc


def tilt_audit(Hs, p_levels, pb, n_fine=61, core=2.5):
    """Per-surface + aggregate tilt stats on [-4.5,4.5]^2 (full) and
    |a|,|b|<=core (high-density core)."""
    fine = np.linspace(pb.a_grid[0], pb.a_grid[-1], n_fine)
    core_m = (np.abs(fine)[:, None] <= core) & (np.abs(fine)[None, :] <= core)
    ea = np.ascontiguousarray(E_A); eb = np.ascontiguousarray(E_B)
    per = []
    g_all = []
    for m in range(Hs.shape[0]):
        Hp = build_pad(np.ascontiguousarray(Hs[m]))
        g, fc = _tilt_fields(Hp, pb.a0, pb.da, pb.Na, pb.b0, pb.db, pb.Nb,
                              fine, ea, eb)
        g_all.append(g)
        per.append(dict(m=m, p=float(p_levels[m]),
                        max_tilt=float(g.max()),
                        p90_tilt=float(np.percentile(g, 90)),
                        frac_super=float(np.mean(g >= SQRT2)),
                        min_Fc=float(fc.min()),
                        frac_Fc_lt_005=float(np.mean(fc < 0.05)),
                        max_tilt_core=float(g[core_m].max()),
                        min_Fc_core=float(fc[core_m].min())))
    G = np.stack(g_all)
    agg = dict(max_tilt=float(G.max()),
               p90_tilt=float(np.percentile(G, 90)),
               frac_super=float(np.mean(G >= SQRT2)),
               max_tilt_core=float(max(p['max_tilt_core'] for p in per)),
               min_Fc=float(min(p['min_Fc'] for p in per)))
    agg['cls'] = classify(agg['max_tilt'])
    agg['cls_core'] = classify(agg['max_tilt_core'])
    return per, agg


def classify(max_tilt):
    if max_tilt < 1.0:
        return 'WELL-POSED'
    if max_tilt < SQRT2:
        return 'MARGINAL'
    return 'ILL-POSED'


# ---------------- T1: screening over the certified cells ----------------

def run_screen(args):
    em = json.load(open(EMIN15_JSON))
    cells = {k: v for k, v in em.items() if v['verdict'] == 'ACCEPT'}
    print(f"T1 screening: {len(cells)} certified cells, M={args.M}, "
          f"H grid {args.Na}x{args.Na}, audit {args.n_fine}x{args.n_fine}")
    out = {}
    t_start = time.time()
    for ci, (key, rec) in enumerate(sorted(cells.items())):
        tau, gamma = rec['tau'], rec['gamma']
        t0 = time.time()
        du, uf, lo, hi, P_inner, P_full = load_P_full(21, tau, gamma)
        pb = Problem(args.M, args.Na, half_width=4.5, n_vert_margin=1,
                     tau=tau, gamma=gamma)
        p_levels = make_levels(P_inner, args.M)
        H0 = build_initial_H(P_full, uf, p_levels, pb.a_grid, pb.b_grid)
        per, agg = tilt_audit(H0, p_levels, pb, n_fine=args.n_fine)
        out[key] = dict(tau=tau, gamma=gamma,
                        deficit_kernel=rec['deficit'],
                        p_levels=[float(p) for p in p_levels],
                        per_surface=per, **agg)
        print(f"  [{ci+1:2d}/{len(cells)}] {key:16s} max_tilt={agg['max_tilt']:.3f} "
              f"core={agg['max_tilt_core']:.3f} p90={agg['p90_tilt']:.3f} "
              f"frac>sqrt2={agg['frac_super']:.3f} -> {agg['cls']:10s} "
              f"({time.time()-t0:.1f}s)")
    meta = dict(M=args.M, Na=args.Na, n_fine=args.n_fine,
                thresholds=dict(well_posed='max_tilt<1.0',
                                marginal='1.0<=max_tilt<sqrt2',
                                ill_posed='max_tilt>=sqrt2'),
                wall=time.time() - t_start)
    json.dump(dict(meta=meta, cells=out), open(SCREEN_JSON, 'w'), indent=1)
    ncls = {}
    for v in out.values():
        ncls[v['cls']] = ncls.get(v['cls'], 0) + 1
    print(f"classes: {ncls}  -> {SCREEN_JSON} ({meta['wall']:.0f}s)")


# ---------------- masked LM (graph-in-s residual) ----------------

def lm_masked(pb, H0, p_m, act, tag, max_iter=30, tol=1e-6, lam0=1e-3,
              dmax=1.0, time_cap=600.0, jcap=100.0, log=print):
    """LM on one surface, graph-in-s residual, rows restricted to the
    density-active mask. No Tikhonov anchor (well-posed regime target);
    step cap ||d||_inf <= dmax guards against null-space drift; phantom
    FD-Jacobian entries |J|>jcap (tangency singularities of the
    supercritical band, Stage 6b finding: true derivs are O(1-10))
    are zeroed. NO rows are dropped: an unsatisfiable band must show
    up as a residual plateau — that is the tau* diagnostic."""
    H = H0.copy()
    r_full, c_base, nf = pb.residual(H, p_m)
    n = pb.Na*pb.Nb
    def stats(rf):
        ra = np.abs(rf[act])
        return float(ra.max()), float(np.median(ra))
    mx, md = stats(r_full)
    cost = 0.5*float(r_full[act] @ r_full[act])
    lam = lam0
    traj = [dict(it=0, max_r=mx, med_r=md, cost=cost, lam=lam, nfail=int(nf))]
    it = 0
    t_in = time.time()
    while it < max_iter and mx > tol and (time.time() - t_in) < time_cap:
        it += 1
        t0 = time.time()
        J = pb.jacobian(H, p_m, c_base, r_full)
        J[~act, :] = 0.0
        nph = int(np.sum(np.abs(J) > jcap))
        if nph:
            J[np.abs(J) > jcap] = 0.0
        rm = r_full.copy(); rm[~act] = 0.0
        JtJ = J.T @ J
        Jtr = J.T @ rm
        D = np.diag(JtJ).copy()
        Dfl = max(float(D.max())*1e-12, 1e-14)
        accepted = False
        for trial in range(25):
            A = JtJ + lam*np.diag(D + Dfl)
            try:
                d = np.linalg.solve(A, -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            if np.max(np.abs(d)) > dmax:
                lam *= 4.0
                if lam > 1e13: break
                continue
            Hn = H + d.reshape(pb.Na, pb.Nb)
            cw = c_base.copy()
            rn, cw, nf = pb.residual(Hn, p_m, cw, use_warm=True, max_newton=40)
            costn = 0.5*float(rn[act] @ rn[act])
            if costn < cost:
                H = Hn; r_full = rn; c_base = cw; cost = costn
                lam = max(lam/3.0, 1e-12)
                accepted = True
                break
            lam *= 4.0
            if lam > 1e13:
                break
        mx, md = stats(r_full)
        traj.append(dict(it=it, max_r=mx, med_r=md, cost=cost, lam=lam,
                         accepted=accepted, nfail=int(nf), n_phantom=nph,
                         dH_max=float(np.max(np.abs(H - H0))),
                         wall=time.time() - t0))
        log(f"    [{tag}] it={it:3d} max|r|={mx:.3e} med|r|={md:.3e} "
            f"lam={lam:.1e} dH={traj[-1]['dH_max']:.2f} acc={accepted} "
            f"({traj[-1]['wall']:.1f}s)")
        if not accepted:
            log(f"    [{tag}] LM stuck (lam={lam:.1e}) — stop")
            break
    return H, r_full, traj


# ---------------- T2: tau ladder ----------------

def s_grid(tau, s_min=4.0):
    """Evidence-integral s range/step: integrand ~ exp(-tau s^2); cover
    to exp(-36) at the truncation point (Stage 6b's +/-4 is only
    adequate for tau ~ 2). Step widened at low tau (integrand width
    1/sqrt(tau)); keeps n_s ~ 100-160."""
    s_max = max(s_min, 6.0/np.sqrt(tau))
    if tau >= 1.0:
        ds = 0.1
    elif tau >= 0.2:
        ds = 0.15
    else:
        ds = 0.25
    n_s = int(round(2*s_max/ds)) + 1
    if n_s % 2 == 0:
        n_s += 1
    return s_max, n_s


def run_ladder(args):
    gamma = args.gamma
    taus = [float(t) for t in args.taus.split(',')]
    anchor_taus = {0.2, 0.5, 1.0, 1.5, 2.0}
    em = json.load(open(EMIN15_JSON))
    du, uf, lo, hi = build_grid(21)

    ckpt = f"{OUT}/stage7_tau_ladder.json"
    results = {}
    if os.path.exists(ckpt) and not args.fresh:
        results = json.load(open(ckpt))
    results.setdefault('meta', dict(M=args.M, Na=args.Na, margin=1,
                                    tol=args.tol, max_iter=args.max_iter,
                                    d_inf=0.268, d_inf_err=0.010))
    key_col = f"g{gamma}"
    col = results.setdefault(key_col, {})

    def save():
        json.dump(results, open(ckpt, 'w'), indent=1)

    H_prev = None; p_prev = None
    for tau in taus:
        tk = f"t{tau}"
        if tk in col and col[tk].get('done') and not args.redo:
            print(f"== tau={tau} (cached) max|r|={col[tk]['final']['max_r']:.3e}")
            H_prev = np.load(f"{OUT}/stage7_H_t{tau}_g{gamma}.npy")
            p_prev = np.array(col[tk]['p_levels'])
            continue
        t_tau = time.time()
        is_anchor = any(abs(tau - a) < 1e-12 for a in anchor_taus)
        s_max, n_s = s_grid(tau)
        print(f"== tau={tau} gamma={gamma}  anchor={is_anchor}  "
              f"s in [-{s_max:.1f},{s_max:.1f}] n_s={n_s}")
        pb = Problem(args.M, args.Na, half_width=4.5, n_vert_margin=1,
                     tau=tau, gamma=gamma, s_max=s_max, n_s=n_s)
        if is_anchor:
            _, _, _, _, P_inner, P_full = load_P_full(21, tau, gamma)
            p_levels = make_levels(P_inner, args.M)
            H0 = build_initial_H(P_full, uf, p_levels, pb.a_grid, pb.b_grid)
            warm = 'kernel_FP'
            d_kern = em.get(f"t{tau}_g{gamma}", {}).get('deficit')
        else:
            assert H_prev is not None, "intermediate tau needs a predecessor"
            p_levels = p_prev.copy()
            H0 = H_prev.copy()
            warm = 'continuation'
            d_kern = None
        rec = dict(tau=tau, gamma=gamma, warm_start=warm,
                   p_levels=[float(p) for p in p_levels],
                   s_max=float(s_max), n_s=int(n_s),
                   deficit_kernel_G21=d_kern)

        # tilt audit of warm start (screening fidelity)
        _, agg0 = tilt_audit(H0, p_levels, pb)
        rec['tilt_warmstart'] = agg0
        print(f"  warm-start tilt: max={agg0['max_tilt']:.3f} "
              f"core={agg0['max_tilt_core']:.3f} -> {agg0['cls']}")

        # density masks (frozen at warm start)
        masks = [pb.density_mask(np.ascontiguousarray(H0[m]), args.rho_cut)
                 for m in range(args.M)]
        rec['active_counts'] = [int(mk.sum()) for mk in masks]

        Hs = H0.copy()
        per_s = {}
        finals = []
        for m in range(args.M):
            Hm, rm, traj = lm_masked(pb, np.ascontiguousarray(H0[m]),
                                     p_levels[m], masks[m], tag=f"t{tau} m={m}",
                                     max_iter=args.max_iter, tol=args.tol,
                                     time_cap=args.time_cap_surface)
            Hs[m] = Hm
            finals.append(np.abs(rm[masks[m]]))
            per_s[f"m{m}"] = dict(p=float(p_levels[m]),
                                  max_r=float(finals[-1].max()),
                                  med_r=float(np.median(finals[-1])),
                                  n_iter=traj[-1]['it'],
                                  init_max_r=traj[0]['max_r'],
                                  init_med_r=traj[0]['med_r'],
                                  traj_tail=traj[-3:])
            print(f"  m={m} p={p_levels[m]:.3f}: init max|r|={traj[0]['max_r']:.3e} "
                  f"-> final {per_s[f'm{m}']['max_r']:.3e} "
                  f"({traj[-1]['it']} its)")
        r_all = np.concatenate(finals)
        rec['per_surface'] = per_s
        rec['final'] = dict(max_r=float(r_all.max()),
                            med_r=float(np.median(r_all)),
                            p90_r=float(np.percentile(r_all, 90)),
                            converged=bool(r_all.max() < args.tol))
        print(f"  TOTAL: max|r|={rec['final']['max_r']:.3e} "
              f"med={rec['final']['med_r']:.3e} "
              f"converged={rec['final']['converged']}")

        # tilt audit of CONVERGED H
        _, agg1 = tilt_audit(Hs, p_levels, pb)
        rec['tilt_converged'] = agg1
        print(f"  converged tilt: max={agg1['max_tilt']:.3f} "
              f"core={agg1['max_tilt_core']:.3f} -> {agg1['cls']}")
        dHs, dps = signflip_diag(Hs, p_levels)
        rec['signflip'] = dict(dH=dHs, dp=dps)

        # deficits: warm start (reconstruction check) + converged
        d0, nv0, _ = deficit_from_H(H0, p_levels, pb, uf, lo, hi)
        d1, nv1, _ = deficit_from_H(Hs, p_levels, pb, uf, lo, hi)
        rec['deficit_warmstart_reconstr'] = d0
        rec['deficit_strict_h0'] = d1
        rec['ladder_violations'] = dict(warm=int(nv0), converged=int(nv1))
        print(f"  deficit: warm-start reconstr={d0:.4f} "
              f"(kernel={d_kern})  strict-h0={d1:.4f}")

        np.save(f"{OUT}/stage7_H_t{tau}_g{gamma}.npy", Hs)
        if abs(gamma - 0.098) < 1e-12:
            np.save(f"{OUT}/stage7_H_t{tau}.npy", Hs)   # deliverable name
        rec['wall'] = time.time() - t_tau
        rec['done'] = True
        col[tk] = rec
        save()
        print(f"  checkpoint saved ({rec['wall']:.0f}s)")
        H_prev = Hs; p_prev = np.array(p_levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['screen', 'ladder'])
    ap.add_argument('--M', type=int, default=8)
    ap.add_argument('--Na', type=int, default=15)
    ap.add_argument('--n-fine', type=int, default=61)
    ap.add_argument('--gamma', type=float, default=0.098)
    ap.add_argument('--taus', type=str, default='0.2,0.3,0.5,0.7,1.0,1.5,2.0')
    ap.add_argument('--max-iter', type=int, default=30)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--rho-cut', type=float, default=1e-8)
    ap.add_argument('--time-cap-surface', type=float, default=420.0)
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == 'screen':
        run_screen(args)
    else:
        run_ladder(args)


if __name__ == '__main__':
    main()
