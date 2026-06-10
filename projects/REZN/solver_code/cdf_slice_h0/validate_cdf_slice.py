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
    p_test = np.linspace(-2.0, 2.0, 41)
    rows = []
    for G in list(G_cheb) + list(G_hat_extra):
        du, u_full, lo, hi = build_grid(G)
        S = u_full[:, None] + u_full[None, :]
        Wt = tri_weights(u_full, TAU, TAU, du)
        tri = make_tris(S)
        t0 = time.time()
        A_hat = hat_eval(tri, Wt, p_test)
        ex = np.stack([exact_sum_density(p_test, v, TAU) for v in (0, 1)],
                      axis=1)
        rel_hat = float(np.max(np.abs(A_hat - ex) / ex))
        row = dict(G_equiv=G, du=du, rel_hat=rel_hat)
        if G in G_cheb:
            A_cheb = slice_evidence(S, p_test, Wt)
            row['rel_cheb'] = float(np.max(np.abs(A_cheb - ex) / ex))
            row['cheb_vs_hat'] = float(np.max(np.abs(A_cheb - A_hat) / ex))
        row['wall'] = time.time() - t0
        rows.append(row)
        log('  G=%5d du=%.5f  hat-vs-exact %.3e' % (G, du, rel_hat),
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
    op = CDFSliceOperator(G, tv, gv, wv)
    P_inner = np.load(os.path.join(WARM, 'P_inner_G9.npy'))
    Pf = op.embed(P_inner.ravel())
    s = (slice(op.lo, op.hi),) * 3

    t0 = time.time()
    P_cdf = op.phi(Pf)
    t_phi = time.time() - t0
    P_ker = phi_K3_halo_smooth(Pf, op.u_full, op.lo, op.hi, tv, gv, wv, 0.45)

    d = (P_cdf - P_ker)[s]
    d_fp = (P_cdf - Pf)[s]
    res = dict(
        max_abs_diff=float(np.max(np.abs(d))),
        mean_abs_diff=float(np.mean(np.abs(d))),
        rms_diff=float(np.sqrt(np.mean(d ** 2))),
        median_abs_diff=float(np.median(np.abs(d))),
        max_abs_diff_vs_warmstart=float(np.max(np.abs(d_fp))),
        kernel_selfres=float(np.max(np.abs((P_ker - Pf)[s]))),
        wall_phi_cdf=t_phi,
        stats=op.last_stats,
    )
    log('  phi_cdf wall %.2fs  segments/slice avg %.1f max %d  '
        'targets-at-knot %d'
        % (t_phi, op.last_stats['nseg'] / op.last_stats['nslice'],
           op.last_stats['max_nseg'], op.last_stats['n_at_knot']))
    log('  |phi_cdf - phi_kernel| inner: max %.4e  mean %.4e  median %.4e'
        % (res['max_abs_diff'], res['mean_abs_diff'],
           res['median_abs_diff']))
    log('  kernel self-residual at warm start: %.2e' % res['kernel_selfres'])
    np.save(os.path.join(OUT, 'v2_diff_field.npy'), d)
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


def v3(G_list=(9, 13, 17, 21)):
    log('=== V3 Newton at strict h=0: tau=%s gamma=%s ===' % (TAU, GAMMA))
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    rows = []
    prev = None
    for G in G_list:
        op = CDFSliceOperator(G, tv, gv, wv)
        if prev is None:
            x0 = np.load(os.path.join(WARM, 'P_inner_G9.npy')).ravel()
        else:
            x0 = interp_inner(prev[0], prev[1], G).ravel()
        F0 = float(np.max(np.abs(op.residual(x0))))
        log('  -- G=%d  initial ||F||inf=%.3e' % (G, F0))
        sol, Finf, conv, wall, hist, nfev = _newton(op, x0, label='G%d' % G)
        defi, slope = deficit_unweighted(sol.reshape((G,) * 3), G, TAU)
        log('  G=%d  final ||F||inf=%.3e  conv=%s  wall=%.1fs  nfev=%d  '
            'deficit=%.4f slope=%.4f' % (G, Finf, conv, wall, nfev,
                                         defi, slope))
        np.save(os.path.join(OUT, 'P_cdf_h0_G%d.npy' % G),
                sol.reshape((G,) * 3))
        rows.append(dict(G=G, F0=F0, Finf=Finf, converged=conv,
                         wall=wall, nfev=nfev, deficit=defi, slope=slope,
                         hist=hist))
        prev = (sol.reshape((G,) * 3), G)
        save_json('v3_newton.json', rows)
    return rows


# ======================================================================
# V4 truth check: revelation deficit vs immortal anchor + second point
# ======================================================================

def v4():
    log('=== V4 truth check ===')
    res = {'anchor': [], 'second_point': None}
    # anchors from V3 solutions
    targets = {9: 0.2912, 13: 0.2838, 17: 0.2813, 21: 0.2793, 25: 0.2765}
    for G in (9, 13, 17, 21):
        fn = os.path.join(OUT, 'P_cdf_h0_G%d.npy' % G)
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
    op = CDFSliceOperator(9, tv, gv, wv)
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
    np.save(os.path.join(OUT, 'P_cdf_h0_t0.5_g1.0_G9.npy'),
            sol.reshape((9,) * 3))
    res['second_point'] = dict(tau=0.5, gamma=1.0, G=9, Finf=Finf,
                               converged=conv, deficit=defi, slope=slope,
                               target=0.0070)
    save_json('v4_truth.json', res)
    return res


if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    if 'v1' in args or 'all' in args:
        v1()
    if 'v2' in args or 'all' in args:
        v2()
    if 'v3' in args or 'all' in args:
        v3()
    if 'v4' in args or 'all' in args:
        v4()
