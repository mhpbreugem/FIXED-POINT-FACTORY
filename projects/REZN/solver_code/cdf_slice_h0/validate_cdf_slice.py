"""Validation ladder V1-V4 for the strict h=0 CDF-slice operator.

Usage:  python validate_cdf_slice.py [v1] [v2] [v3] [v4] [all]

Results are appended to
/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cdf_slice_h0/
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cdf_slice_ops import (CDFSliceOperator, build_grid, make_tris,  # noqa
                           tri_weights, hat_eval, slice_evidence,
                           phi_K3_halo_smooth, init_no_learning_K3)

OUT = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
       'solved_fixed_points/cdf_slice_h0')
WARM = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
        'solved_fixed_points/dd_k3_overnight/coarea_limit_repro')
os.makedirs(OUT, exist_ok=True)

TAU, GAMMA = 2.0, 0.1
LOG = open(os.path.join(OUT, 'run.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n')
    LOG.flush()


def save_json(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=2)


# ======================================================================
# V1 density unit test: S = u2 + u3, weights N(+-1/2, 1/tau)
# ======================================================================

def exact_sum_density(p, v, tau):
    """pdf of u2+u3 with u_k ~ N(v-1/2, 1/tau):  N(2v-1, 2/tau)."""
    m = 2.0 * (v - 0.5)
    return np.sqrt(tau / (4.0 * np.pi)) * np.exp(-0.25 * tau * (p - m) ** 2)


def v1(G_cheb=(21, 41, 81, 161, 321), G_hat_extra=(641, 1281, 2561, 5121)):
    log('=== V1 density unit test: S = u2+u3, tau=%.1f ===' % TAU)
    rows = []
    for G in list(G_cheb) + list(G_hat_extra):
        du, u_full, lo, hi = build_grid(G)
        # offset the test points by an irrational fraction of du so they
        # do NOT align with the vertex-value lattice (alignment is
        # superconvergent and would flatter the result).
        p_test = np.linspace(-2.0, 2.0, 41) + du * 0.299792458
        S = u_full[:, None] + u_full[None, :]
        Wt = tri_weights(u_full, TAU, TAU, du)
        tri = make_tris(S)
        t0 = time.time()
        A_hat = hat_eval(tri, Wt, p_test)
        ex = np.stack([exact_sum_density(p_test, v, TAU) for v in (0, 1)],
                      axis=1)
        rel = np.abs(A_hat - ex) / ex
        bulk = np.abs(p_test) <= 1.0
        rel_hat = float(np.max(rel))
        rel_hat_bulk = float(np.max(rel[bulk]))
        # aligned (vertex-lattice) points for the superconvergence note
        p_al = np.linspace(-2.0, 2.0, 21)
        snap = np.round(p_al / du) * du
        A_al = hat_eval(tri, Wt, snap)
        ex_al = np.stack([exact_sum_density(snap, v, TAU)
                          for v in (0, 1)], axis=1)
        rel_aligned = float(np.max(np.abs(A_al - ex_al) / ex_al))
        row = dict(G_equiv=G, du=du, rel_hat=rel_hat,
                   rel_hat_bulk=rel_hat_bulk, rel_aligned=rel_aligned)
        if G in G_cheb:
            A_cheb = slice_evidence(S, p_test, Wt)
            from cdf_slice_ops import boxcar_eval
            A_box = boxcar_eval(tri, Wt, p_test, 1.0e-7)
            row['rel_cheb'] = float(np.max(np.abs(A_cheb - ex) / ex))
            row['cheb_vs_hat'] = float(np.max(np.abs(A_cheb - A_hat) / ex))
            row['box_vs_hat'] = float(np.max(np.abs(A_box - A_hat) / ex))
        row['wall'] = time.time() - t0
        rows.append(row)
        log('  G=%5d du=%.5f  hat-vs-exact %.3e (bulk %.3e, aligned %.1e)'
            % (G, du, rel_hat, rel_hat_bulk, rel_aligned),
            (' cheb-vs-exact %.3e  cheb-vs-hat %.3e'
             % (row['rel_cheb'], row['cheb_vs_hat'])) if 'rel_cheb' in row
            else '')
    # empirical order
    d = np.array([r['du'] for r in rows])
    e = np.array([r['rel_hat'] for r in rows])
    order = float(np.polyfit(np.log(d), np.log(e), 1)[0])
    log('  empirical convergence order (hat vs exact): %.3f' % order)
    save_json('v1_density.json', dict(rows=rows, order=order))
    return rows


# ======================================================================
# V2 operator cross-check at the converged kernel solution, G=9
# ======================================================================

def v2():
    log('=== V2 operator cross-check: G=9 tau=2 gamma=0.1, kernel h=0.45 ===')
    G = 9
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    P_inner = np.load(os.path.join(WARM, 'P_inner_G9.npy'))
    ui = np.linspace(-4, 4, G)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    Tabs = np.abs(U1 + U2 + U3)
    res = {}
    P_ker = None
    for method in ('cheb', 'boxcar'):
        op = CDFSliceOperator(G, tv, gv, wv, method=method)
        Pf = op.embed(P_inner.ravel())
        s = (slice(op.lo, op.hi),) * 3
        if P_ker is None:
            P_ker = phi_K3_halo_smooth(Pf, op.u_full, op.lo, op.hi,
                                       tv, gv, wv, 0.45)
        t0 = time.time()
        P_cdf = op.phi(Pf)
        t_phi = time.time() - t0
        d = (P_cdf - P_ker)[s]
        shells = {}
        for lim in ((0, 2), (2, 5), (5, 8), (8, 12.1)):
            mm = (Tabs >= lim[0]) & (Tabs < lim[1])
            shells['T%g-%g' % lim] = dict(
                max=float(np.max(np.abs(d[mm]))),
                mean=float(np.mean(np.abs(d[mm]))))
        r = dict(max_abs_diff=float(np.max(np.abs(d))),
                 mean_abs_diff=float(np.mean(np.abs(d))),
                 median_abs_diff=float(np.median(np.abs(d))),
                 rms_diff=float(np.sqrt(np.mean(d ** 2))),
                 kernel_selfres=float(np.max(np.abs((P_ker - Pf)[s]))),
                 wall_phi_cdf=t_phi, shells=shells,
                 stats=op.last_stats)
        log('  [%s] wall %.2fs |phi_cdf-phi_ker|: max %.4e mean %.4e '
            'median %.4e' % (method, t_phi, r['max_abs_diff'],
                             r['mean_abs_diff'], r['median_abs_diff']))
        for k, v in shells.items():
            log('      |T| shell %s: max %.3e mean %.3e'
                % (k, v['max'], v['mean']))
        np.save(os.path.join(OUT, 'v2_diff_field_%s.npy' % method), d)
        res[method] = r
    save_json('v2_crosscheck.json', res)
    return res


# ======================================================================
# V3 Newton at strict h=0
# ======================================================================

def _newton(op, x0, f_tol=1e-10, maxiter=50, label=''):
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize._nonlin import NoConvergence
    hist = []
    nfev = [0]

    def resid(x):
        nfev[0] += 1
        return op.residual(x)

    def cb(x, fx):
        fn = float(np.max(np.abs(fx)))
        hist.append(fn)
        log('    [%s] newton it=%d ||F||inf=%.3e (nfev=%d)'
            % (label, len(hist), fn, nfev[0]))

    t0 = time.time()
    conv = True
    try:
        sol = newton_krylov(resid, x0, f_tol=f_tol, maxiter=maxiter,
                            method='lgmres', callback=cb, verbose=0)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel()
        conv = False
    except ValueError as e:
        log('    newton_krylov ValueError:', e)
        sol = x0
        conv = False
    Finf = float(np.max(np.abs(op.residual(sol))))
    return sol, Finf, conv, time.time() - t0, hist, nfev[0]


def interp_inner(P_old, G_old, G_new, UMAX=4.0):
    from scipy.interpolate import RegularGridInterpolator
    ax_o = np.linspace(-UMAX, UMAX, G_old)
    ax_n = np.linspace(-UMAX, UMAX, G_new)
    rgi = RegularGridInterpolator((ax_o, ax_o, ax_o), P_old,
                                  bounds_error=False, fill_value=None)
    A, B, C = np.meshgrid(ax_n, ax_n, ax_n, indexing='ij')
    return rgi(np.column_stack([A.ravel(), B.ravel(), C.ravel()])
               ).reshape((G_new,) * 3)


def deficit_unweighted(P_inner, G, tau, UMAX=4.0):
    """1 - R^2 of logit(P) on T = u1+u2+u3, as in coarea_path.metrics."""
    ui = np.linspace(-UMAX, UMAX, G)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = tau * (U1 + U2 + U3)
    Pc = np.clip(P_inner, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    return (float(np.sum((y - pr) ** 2)
                  / max(np.sum((y - y.mean()) ** 2), 1e-30)),
            float(a[0]))


def v3(G_list=(9, 13, 17, 21), method='boxcar', boxcar_delta=1.0e-7,
       tag=None):
    tag = tag or method
    log('=== V3 Newton at strict h=0 [%s]: tau=%s gamma=%s ==='
        % (tag, TAU, GAMMA))
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    rows = []
    prev = None
    for G in G_list:
        op = CDFSliceOperator(G, tv, gv, wv, method=method,
                              boxcar_delta=boxcar_delta)
        if prev is None:
            x0 = np.load(os.path.join(WARM, 'P_inner_G9.npy')).ravel()
            if G != 9:
                x0 = interp_inner(x0.reshape((9,) * 3), 9, G).ravel()
        else:
            x0 = interp_inner(prev[0], prev[1], G).ravel()
        F0 = float(np.max(np.abs(op.residual(x0))))
        log('  -- G=%d  initial ||F||inf=%.3e' % (G, F0))
        sol, Finf, conv, wall, hist, nfev = _newton(
            op, x0, label='%s-G%d' % (tag, G))
        defi, slope = deficit_unweighted(sol.reshape((G,) * 3), G, TAU)
        log('  G=%d [%s] final ||F||inf=%.3e  conv=%s  wall=%.1fs  '
            'nfev=%d  deficit=%.4f slope=%.4f'
            % (G, tag, Finf, conv, wall, nfev, defi, slope))
        np.save(os.path.join(OUT, 'P_cdf_h0_%s_G%d.npy' % (tag, G)),
                sol.reshape((G,) * 3))
        rows.append(dict(G=G, method=method, boxcar_delta=boxcar_delta,
                         F0=F0, Finf=Finf, converged=conv,
                         wall=wall, nfev=nfev, deficit=defi, slope=slope,
                         hist=hist[:200]))
        prev = (sol.reshape((G,) * 3), G)
        save_json('v3_newton_%s.json' % tag, rows)
    return rows


# ======================================================================
# V4 truth check: revelation deficit vs immortal anchor + second point
# ======================================================================

def v4(tag='boxcar', method='boxcar', boxcar_delta=1.0e-7):
    log('=== V4 truth check [%s] ===' % tag)
    res = {'anchor': [], 'second_point': None}
    # anchors from V3 solutions
    targets = {9: 0.2912, 13: 0.2838, 17: 0.2813, 21: 0.2793, 25: 0.2765}
    for G in (9, 13, 17, 21):
        fn = os.path.join(OUT, 'P_cdf_h0_%s_G%d.npy' % (tag, G))
        if not os.path.exists(fn):
            continue
        P = np.load(fn)
        defi, slope = deficit_unweighted(P, G, TAU)
        tgt = targets.get(G)
        log('  G=%d  deficit=%.4f  kernel anchor=%.4f  |diff|=%.4f'
            % (G, defi, tgt, abs(defi - tgt)))
        res['anchor'].append(dict(G=G, deficit=defi, target=tgt,
                                  diff=abs(defi - tgt)))

    # second parameter point (tau=0.5, gamma=1.0), kernel sweep ~0.0070
    log('  -- second point tau=0.5 gamma=1.0, G=9')
    tv = np.full(3, 0.5)
    gv = np.full(3, 1.0)
    wv = np.full(3, 1.0)
    op = CDFSliceOperator(9, tv, gv, wv, method=method,
                          boxcar_delta=boxcar_delta)
    s = (slice(op.lo, op.hi),) * 3
    x = op.halo[s].ravel().copy()
    for it in range(8):                      # damped Picard warm-up
        Pf = op.embed(x)
        x = 0.5 * x + 0.5 * op.phi(Pf)[s].ravel()
    F0 = float(np.max(np.abs(op.residual(x))))
    log('    after Picard warm-up ||F||inf=%.3e' % F0)
    sol, Finf, conv, wall, hist, nfev = _newton(op, x, label='t0.5g1.0')
    defi, slope = deficit_unweighted(sol.reshape((9,) * 3), 9, 0.5)
    log('  tau=0.5 gamma=1.0 G=9: ||F||inf=%.3e conv=%s deficit=%.5f '
        '(kernel sweep ~0.0070)' % (Finf, conv, defi))
    np.save(os.path.join(OUT, 'P_cdf_h0_%s_t0.5_g1.0_G9.npy' % tag),
            sol.reshape((9,) * 3))
    res['second_point'] = dict(tau=0.5, gamma=1.0, G=9, Finf=Finf,
                               converged=conv, deficit=defi, slope=slope,
                               target=0.0070)
    save_json('v4_truth_%s.json' % tag, res)
    return res


if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    if 'v1' in args or 'all' in args:
        v1()
    if 'v2' in args or 'all' in args:
        v2()
    if 'v3' in args or 'all' in args:
        v3(method='boxcar', boxcar_delta=1.0e-7, tag='boxcar')
    if 'v3cheb' in args:
        v3(G_list=(9, 13), method='cheb', tag='cheb')
    if 'v3delta' in args:        # delta-robustness at G=9
        for d in (1.0e-6, 1.0e-8):
            v3(G_list=(9,), method='boxcar', boxcar_delta=d,
               tag='boxcar%g' % d)
    if 'v4' in args or 'all' in args:
        v4(tag='boxcar')
