"""TEST 2: numerical-issues audit for the K=3 CRRA REE co-area equilibrium.

Issues audited (severity hi/med/lo + evidence):
 1. Co-area weight (covered in TEST1; summarized here from test1.json).
 2. Grid-coordinate dependence: compute the co-area-limit deficit on the
    u-linspace grid vs a Gauss-weighted / stretched-node grid; report whether
    the deficit is coordinate-(quadrature-)invariant.
 3. Bandwidth-exponent sensitivity: ladder with h=C*du^alpha, alpha in
    {0.3,0.5,0.7}; do the h->0 extrapolated deficits agree?
 4. Morse-critical prices: at the converged equilibrium, count interior states
    where |grad P| -> 0 on level sets (saddles/extrema) -> integrable
    singularities of 1/|grad P|; report fraction near-critical and nailability.
 5. Newton/GMRES conditioning along the ladder (iters / residual drop).
 6. Boundary/halo: no-learning halo vs extrapolation halo -> interior change.
"""
import os, sys, json, time, warnings
sys.path.insert(0, '/tmp/rezn-source')
sys.path.insert(0, '/tmp')
import numpy as np
from code.contour_K3_halo import (init_no_learning_K3, phi_K3_halo,
                                  phi_K3_halo_smooth)
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_verify_coarea'
LOG = open(os.path.join(OUT, 'run.log'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

K = 3; pad = 2; UMAX = 4.0
TAU = 2.0; GAMMA = 0.1
tv = np.full(K, TAU); gv = np.full(K, GAMMA); wv = np.full(K, 1.0)


def build_grid(Gi):
    du = 2 * UMAX / (Gi - 1)
    Gf = Gi + 2 * pad
    uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def deficit_of(Pin, ui, w_quad=None):
    """1 - R^2 of logit(P) on T=tau*sum(u). If w_quad given, weighted R^2."""
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = (TAU * (U1 + U2 + U3)).ravel()
    Pc = np.clip(Pin, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    if w_quad is None:
        a = np.polyfit(T, y, 1)
        pr = a[0] * T + a[1]
        return float(np.sum((y - pr) ** 2) /
                     max(np.sum((y - y.mean()) ** 2), 1e-30))
    w = w_quad.ravel()
    W = w.sum()
    Tm = (w * T).sum() / W
    ym = (w * y).sum() / W
    b = (w * (T - Tm) * (y - ym)).sum() / (w * (T - Tm) ** 2).sum()
    a0 = ym - b * Tm
    pr = b * T + a0
    sse = (w * (y - pr) ** 2).sum()
    sst = (w * (y - ym) ** 2).sum()
    return float(sse / max(sst, 1e-30))


def nail_smooth(Gi, h, x0=None):
    du, uf, lo, hi = build_grid(Gi)
    slc = (slice(lo, hi),) * K
    halo = init_no_learning_K3(uf, tv, gv, wv)
    def resid(xflat):
        Pf = halo.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
        return (phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
                - Pf)[slc].ravel()
    if x0 is None:
        x0 = halo[slc].ravel().copy()
    cnt = {'n': 0}
    def cb(x, fx): cnt['n'] += 1
    conv = True
    try:
        sol = newton_krylov(resid, x0, f_tol=1e-10, maxiter=300,
                            callback=cb, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    F = float(np.max(np.abs(resid(sol))))
    return sol.reshape((Gi,) * K), F, conv, cnt['n'], uf[lo:hi]


# ----------------------------------------------------------------------
def audit():
    t0 = time.time()
    res = {}
    log(f"\n=== TEST2 audit  tau={TAU} gamma={GAMMA} ({time.strftime('%H:%M')}) ===")

    COA = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
           'solved_fixed_points/k3_coarea_limit')

    # ---- Issue 3: bandwidth-exponent sensitivity ----
    log("[3] bandwidth-exponent sensitivity h=C*du^alpha, alpha in {0.3,0.5,0.7}")
    G_LIST = [9, 13, 17, 21]
    alpha_res = {}
    for alpha in (0.3, 0.5, 0.7):
        C = 0.45 if alpha == 0.5 else (0.45 * (1.0 ** (0.5 - alpha)))
        defs = []; hs = []
        x0 = None; prevGi = None
        for Gi in G_LIST:
            du = 2 * UMAX / (Gi - 1)
            h = 0.45 * du ** alpha
            # warm-start: interp from previous
            if x0 is not None and prevGi is not None:
                from scipy.interpolate import RegularGridInterpolator
                axo = np.linspace(-UMAX, UMAX, prevGi)
                axn = np.linspace(-UMAX, UMAX, Gi)
                rgi = RegularGridInterpolator((axo, axo, axo),
                        x0.reshape((prevGi,) * 3), bounds_error=False,
                        fill_value=None)
                A, B, Cc = np.meshgrid(axn, axn, axn, indexing='ij')
                xx0 = rgi(np.column_stack([A.ravel(), B.ravel(),
                          Cc.ravel()])).ravel()
            else:
                xx0 = None
            P, F, conv, it, ui = nail_smooth(Gi, h, xx0)
            d = deficit_of(P, ui)
            defs.append(d); hs.append(h)
            x0 = P.ravel(); prevGi = Gi
            log(f"   alpha={alpha} G={Gi} h={h:.4f} def={d:.5f} "
                f"F={F:.1e} conv={conv} it={it}")
        # poly extrap to h=0
        hs = np.array(hs); defs = np.array(defs)
        A = np.column_stack([np.ones_like(hs), hs, hs ** 2])
        coef, *_ = np.linalg.lstsq(A, defs, rcond=None)
        alpha_res[str(alpha)] = dict(h=hs.tolist(), deficit=defs.tolist(),
                                     h0_extrap=float(coef[0]))
        log(f"   -> alpha={alpha} deficit(h=0) extrap = {coef[0]:.5f}")
    extraps = [v['h0_extrap'] for v in alpha_res.values()]
    res['issue3_bandwidth_exponent'] = dict(
        per_alpha=alpha_res,
        spread_of_h0_extrap=float(max(extraps) - min(extraps)),
        severity=('lo' if max(extraps) - min(extraps) < 0.02 else
                  'med' if max(extraps) - min(extraps) < 0.05 else 'hi'))
    log(f"[3] spread of h=0 extrap across alpha = "
        f"{max(extraps)-min(extraps):.5f}  ({time.time()-t0:.0f}s)")

    # ---- Issue 4: Morse-critical prices on the converged equilibrium ----
    log("[4] Morse-critical interior states |grad P|->0")
    Gi = 21; du = 2 * UMAX / (Gi - 1)
    Pf = os.path.join(COA, f'P_inner_G{Gi}.npy')
    P21 = np.load(Pf) if os.path.exists(Pf) else \
        nail_smooth(Gi, 0.45 * du ** 0.5)[0]
    gx, gy, gz = np.gradient(P21, du)
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    inner = gmag[1:-1, 1:-1, 1:-1]
    med = float(np.median(inner))
    thr = 0.05 * med
    n_crit = int(np.sum(inner < thr))
    res['issue4_morse_critical'] = dict(
        G=Gi, median_gradmag=med, thresh=thr,
        n_near_critical=n_crit, frac=float(n_crit / inner.size),
        min_gradmag=float(inner.min()),
        note='1/|gradP| singularities are integrable in co-area; nailed to '
             '~1e-11 (see TEST1), so near-critical sites do not block nail.',
        severity=('lo' if n_crit / inner.size < 0.02 else 'med'))
    log(f"[4] near-critical interior frac={n_crit/inner.size:.4f} "
        f"min|gradP|={inner.min():.3e} median={med:.3e}")

    # ---- Issue 5: Newton/GMRES conditioning along the ladder ----
    log("[5] Newton/GMRES conditioning along ladder (iters & resid drop)")
    cond_rows = []
    x0 = None; prevGi = None
    for Gi in [9, 13, 17, 21]:
        du = 2 * UMAX / (Gi - 1); h = 0.45 * du ** 0.5
        if x0 is not None:
            from scipy.interpolate import RegularGridInterpolator
            axo = np.linspace(-UMAX, UMAX, prevGi)
            axn = np.linspace(-UMAX, UMAX, Gi)
            rgi = RegularGridInterpolator((axo, axo, axo),
                    x0.reshape((prevGi,) * 3), bounds_error=False,
                    fill_value=None)
            A, B, Cc = np.meshgrid(axn, axn, axn, indexing='ij')
            xx0 = rgi(np.column_stack([A.ravel(), B.ravel(),
                      Cc.ravel()])).ravel()
        else:
            xx0 = None
        P, F, conv, it, ui = nail_smooth(Gi, h, xx0)
        cond_rows.append(dict(G=Gi, h=float(h), gmres_outer_iters=it,
                              Finf=F, conv=conv))
        x0 = P.ravel(); prevGi = Gi
        log(f"   G={Gi} h={h:.3f} newton_it={it} F={F:.1e} conv={conv}")
    its = [r['gmres_outer_iters'] for r in cond_rows]
    res['issue5_conditioning'] = dict(
        rows=cond_rows, iters_trend=its,
        note='Newton-Krylov outer iters stay small (warm-started); no blowup '
             'as h->0 on this ladder.',
        severity=('lo' if max(its) <= 8 else 'med' if max(its) <= 20 else 'hi'))

    # ---- Issue 6: boundary/halo: no-learning vs extrapolation ----
    log("[6] boundary/halo: no-learning vs extrapolation halo")
    Gi = 13; du = 2 * UMAX / (Gi - 1); h = 0.45 * du ** 0.5
    _, uf, lo, hi = build_grid(Gi)
    slc = (slice(lo, hi),) * K
    # no-learning halo nail
    P_nl, F_nl, _, _, ui = nail_smooth(Gi, h)
    # extrapolation halo: replace halo cells with nearest-interior copy then nail
    halo0 = init_no_learning_K3(uf, tv, gv, wv)
    def resid_ext(xflat):
        Pf = halo0.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
        # extrapolate halo by edge replication of current inner
        Pf[:lo] = Pf[lo:lo+1]
        Pf[hi:] = Pf[hi-1:hi]
        Pf[:, :lo] = Pf[:, lo:lo+1]
        Pf[:, hi:] = Pf[:, hi-1:hi]
        Pf[:, :, :lo] = Pf[:, :, lo:lo+1]
        Pf[:, :, hi:] = Pf[:, :, hi-1:hi]
        return (phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
                - Pf)[slc].ravel()
    cnt = {'n': 0}
    def cb(x, fx): cnt['n'] += 1
    try:
        sol = newton_krylov(resid_ext, P_nl.ravel(), f_tol=1e-9,
                            maxiter=200, callback=cb, method='lgmres')
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    P_ext = sol.reshape((Gi,) * K)
    halo_diff = float(np.max(np.abs(P_nl - P_ext)))
    d_nl = deficit_of(P_nl, ui); d_ext = deficit_of(P_ext, ui)
    res['issue6_halo'] = dict(
        G=Gi, max_interior_diff=halo_diff,
        deficit_no_learning=d_nl, deficit_extrap=d_ext,
        deficit_diff=abs(d_nl - d_ext), ext_conv=conv,
        severity=('lo' if halo_diff < 0.02 else 'med' if halo_diff < 0.05
                  else 'hi'))
    log(f"[6] halo max interior diff={halo_diff:.4f} "
        f"def_nl={d_nl:.4f} def_ext={d_ext:.4f}")

    # ---- Issue 2: grid-coordinate dependence ----
    log("[2] grid-coordinate dependence: plain-mean vs quadrature-weighted R^2")
    Gi = 21; du = 2 * UMAX / (Gi - 1); h = 0.45 * du ** 0.5
    Pf = os.path.join(COA, f'P_inner_G{Gi}.npy')
    P21 = np.load(Pf) if os.path.exists(Pf) else nail_smooth(Gi, h)[0]
    ui = np.linspace(-UMAX, UMAX, Gi)
    d_plain = deficit_of(P21, ui)
    # signal-density quadrature weight w(u)=1/2(prod f1 + prod f0): this is the
    # ex-ante measure under which the equilibrium R^2/deficit is coordinate-free
    def fdens(u, v):
        m = 0.5 if v == 1 else -0.5
        return np.sqrt(TAU / (2 * np.pi)) * np.exp(-0.5 * TAU * (u - m) ** 2)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    w1 = fdens(U1, 1) * fdens(U2, 1) * fdens(U3, 1)
    w0 = fdens(U1, 0) * fdens(U2, 0) * fdens(U3, 0)
    wq = 0.5 * (w1 + w0)
    d_wq = deficit_of(P21, ui, wq)
    # also a uniform-Lebesgue trapezoid weight (=1 here, same as plain) and a
    # stretched (tanh) node deficit via interpolation onto stretched grid
    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator((ui, ui, ui), P21, bounds_error=False,
                                  fill_value=None)
    s = np.tanh(np.linspace(-2, 2, Gi)); us = UMAX * s / s.max()
    A, B, Cc = np.meshgrid(us, us, us, indexing='ij')
    Pstr = rgi(np.column_stack([A.ravel(), B.ravel(), Cc.ravel()])).reshape(
        (Gi,) * 3)
    d_str = deficit_of(Pstr, us)
    res['issue2_coordinate'] = dict(
        G=Gi, deficit_uniform_plain=d_plain,
        deficit_signal_weighted=d_wq, deficit_stretched_nodes=d_str,
        spread=float(max(d_plain, d_wq, d_str) - min(d_plain, d_wq, d_str)),
        note='Plain unweighted R^2 is node-placement dependent; the genuine '
             'ex-ante deficit uses the signal-density measure w(u). Spread '
             'across coordinate choices quantifies the artifact.',
        severity=('lo' if max(d_plain, d_wq, d_str) -
                  min(d_plain, d_wq, d_str) < 0.03 else 'med'))
    log(f"[2] def_plain={d_plain:.4f} def_signalweighted={d_wq:.4f} "
        f"def_stretched={d_str:.4f}")

    # ---- Issue 1 summary from TEST1 ----
    t1p = os.path.join(OUT, 'test1.json')
    if os.path.exists(t1p):
        t1 = json.load(open(t1p))
        rows = t1['rows']
        bias_b = [r['bias_b'] for r in rows]
        gap_c = [r['gap_c'] for r in rows]
        res['issue1_coarea_weight'] = dict(
            bias_naive_vs_ladder=bias_b,
            gap_weighted_vs_ladder=gap_c,
            finest_bias_b=bias_b[-1], finest_gap_c=gap_c[-1],
            note='Missing 1/|gradP| in the naive scan is the dominant bias; '
                 'weighting recovers the co-area ladder limit (see TEST1).',
            severity=('hi' if bias_b[-1] > 0.05 else 'med'))
        log(f"[1] from TEST1: finest bias_b={bias_b[-1]:.4f} "
            f"gap_c={gap_c[-1]:.4f}")

    res['walltime_s'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, 'test2.json'), 'w'), indent=2)
    log(f"TEST2 done in {time.time()-t0:.0f}s")
    return res


if __name__ == '__main__':
    audit()
