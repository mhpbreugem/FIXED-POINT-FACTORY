"""Dense FD-Jacobian damped Newton in the S3-symmetric subspace,
with boxcar-delta continuation.  Far more robust than Krylov-FD for
this C0-ish operator."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdf_slice_ops import CDFSliceOperator
from newton_matrix import sym_maps

WARM = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
        'solved_fixed_points/dd_k3_overnight/coarea_limit_repro')
OUT = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
       'solved_fixed_points/cdf_slice_h0')
TAU, GAMMA = 2.0, 0.1


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(os.path.join(OUT, 'run.log'), 'a') as f:
        f.write(s + '\n')


def dense_newton(resid, y0, fd_step=1e-6, maxit=40, tol=1e-12,
                 label='', endgame_step=1e-8):
    y = y0.copy()
    F = resid(y)
    n = y.size
    nF = float(np.max(np.abs(F)))
    hist = [nF]
    for it in range(maxit):
        if nF < tol:
            break
        step = endgame_step if nF < 1e-6 else fd_step
        J = np.empty((n, n))
        for k in range(n):
            yp = y.copy()
            yp[k] += step
            J[:, k] = (resid(yp) - F) / step
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(J, -F, rcond=None)[0]
        t = 1.0
        ok = False
        while t >= 2.0 ** -22:
            Ft = resid(y + t * dy)
            nFt = float(np.max(np.abs(Ft)))
            if nFt < (1.0 - 1e-4 * t) * nF:
                ok = True
                break
            t *= 0.5
        if not ok:
            log('    [%s] it=%d line search failed at ||F||=%.3e'
                % (label, it, nF))
            break
        y = y + t * dy
        F = Ft
        nF = nFt
        hist.append(nF)
        log('    [%s] it=%d t=%.2e ||F||inf=%.3e' % (label, it, t, nF))
    return y, nF, hist


def dense_newton_broyden(resid, y0, fd_step=1e-6, maxit=80, tol=1e-12,
                         label='', endgame_step=1e-8, rebuild_every=10):
    """Damped Newton with full FD Jacobian + good-Broyden updates
    between rebuilds (for larger G where the full Jacobian is dear)."""
    y = y0.copy()
    F = resid(y)
    n = y.size
    nF = float(np.max(np.abs(F)))
    J = None
    since_rebuild = 0
    it = 0
    fails = 0
    while it < maxit and nF >= tol:
        step = endgame_step if nF < 1e-6 else fd_step
        if J is None or since_rebuild >= rebuild_every:
            t0 = time.time()
            J = np.empty((n, n))
            for k in range(n):
                yp = y.copy()
                yp[k] += step
                J[:, k] = (resid(yp) - F) / step
            since_rebuild = 0
            log('    [%s] it=%d Jacobian rebuilt (%.0fs)'
                % (label, it, time.time() - t0))
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dy = np.linalg.lstsq(J, -F, rcond=None)[0]
        t = 1.0
        ok = False
        while t >= 2.0 ** -22:
            Ft = resid(y + t * dy)
            nFt = float(np.max(np.abs(Ft)))
            if nFt < (1.0 - 1e-4 * t) * nF:
                ok = True
                break
            t *= 0.5
        if not ok:
            if since_rebuild > 0:
                J = None            # stale Broyden Jacobian: rebuild
                continue
            fails += 1
            log('    [%s] it=%d line search failed at ||F||=%.3e'
                % (label, it, nF))
            break
        s = t * dy
        dF = Ft - F
        J += np.outer(dF - J @ s, s) / (s @ s)
        since_rebuild += 1
        y = y + s
        F = Ft
        nF = nFt
        it += 1
        if t < 0.2:
            since_rebuild = rebuild_every   # force rebuild next time
        log('    [%s] it=%d t=%.2e ||F||inf=%.3e' % (label, it, t, nF))
    return y, nF


def run(G=9, deltas=(3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
        x0=None, tag='dense', maxit=40, tol=1e-12):
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    orbit_of, counts, nor = sym_maps(G)
    log('=== dense sym Newton ladder G=%d  reduced dim %d ===' % (G, nor))

    if x0 is None:
        x0 = np.load(os.path.join(WARM, 'P_inner_G9.npy')).ravel()

    def reduce_(F):
        return np.bincount(orbit_of, weights=F, minlength=nor) / counts

    y = reduce_(x0)
    rows = []
    for d in deltas:
        op = CDFSliceOperator(G, tv, gv, wv, method='boxcar',
                              boxcar_delta=d)

        def resid(yv):
            return reduce_(op.residual(yv[orbit_of]))

        t0 = time.time()
        nF0 = float(np.max(np.abs(resid(y))))
        log('-- G=%d delta=%.0e  initial ||F||=%.3e' % (G, d, nF0))
        y_new, nF, hist = dense_newton(resid, y, label='G%d-d%.0e' % (G, d),
                                       maxit=maxit, tol=tol)
        wall = time.time() - t0
        x_full = y_new[orbit_of]
        full_F = float(np.max(np.abs(op.residual(x_full))))
        log('   G=%d delta=%.0e: reduced ||F||=%.3e  FULL ||F||inf=%.3e  '
            'wall=%.0fs' % (G, d, nF, full_F, wall))
        rows.append(dict(G=G, delta=d, F0=nF0, Finf_reduced=nF,
                         Finf_full=full_F, wall=wall, nit=len(hist) - 1))
        with open(os.path.join(OUT, '%s_G%d.json' % (tag, G)), 'w') as f:
            json.dump(rows, f, indent=2)
        if nF < 1e-8:
            y = y_new
            np.save(os.path.join(OUT, 'P_%s_G%d_d%.0e.npy' % (tag, G, d)),
                    x_full.reshape((G,) * 3))
        else:
            log('   (keeping previous warm start)')
    return rows, y[orbit_of]


if __name__ == '__main__':
    run()
