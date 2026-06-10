"""Newton strategy matrix at G=9, boxcar delta=3e-2: rdiff choices and
S3-symmetry reduction."""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdf_slice_ops import CDFSliceOperator
from scipy.optimize import newton_krylov

try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

TAU, GAMMA = 2.0, 0.1
WARM = ('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
        'solved_fixed_points/dd_k3_overnight/coarea_limit_repro')


def sym_maps(G):
    """S3 orbit structure on the G^3 grid."""
    idx = np.arange(G ** 3).reshape(G, G, G)
    orbits = {}
    orbit_of = np.empty(G ** 3, dtype=np.int64)
    for i in range(G):
        for j in range(G):
            for l in range(G):
                key = tuple(sorted((i, j, l)))
                if key not in orbits:
                    orbits[key] = len(orbits)
                orbit_of[idx[i, j, l]] = orbits[key]
    nor = len(orbits)
    counts = np.bincount(orbit_of, minlength=nor).astype(np.float64)
    return orbit_of, counts, nor


def run_one(label, delta, x0, use_sym=False, rdiff=None, maxiter=60,
            f_tol=1e-10, G=9):
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    op = CDFSliceOperator(G, tv, gv, wv, method='boxcar',
                          boxcar_delta=delta)
    if use_sym:
        orbit_of, counts, nor = sym_maps(G)

        def expand(y):
            return y[orbit_of]

        def reduce_(F):
            return np.bincount(orbit_of, weights=F,
                               minlength=nor) / counts

        def resid(y):
            return reduce_(op.residual(expand(y)))

        y0 = reduce_(x0)
    else:
        resid = op.residual
        y0 = x0.copy()

    hist = []

    def cb(x, fx):
        hist.append(float(np.max(np.abs(fx))))
        if len(hist) % 10 == 0 or len(hist) <= 3:
            print('  [%s] it %d ||F|| %.3e' % (label, len(hist),
                                               hist[-1]), flush=True)

    kw = {}
    if rdiff is not None:
        kw['rdiff'] = rdiff
    t0 = time.time()
    try:
        sol = newton_krylov(resid, y0, f_tol=f_tol, maxiter=maxiter,
                            method='lgmres', callback=cb, **kw)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0])
        conv = False
    except Exception as e:
        print('  [%s] ERROR %s' % (label, e))
        return None
    wall = time.time() - t0
    if use_sym:
        x = expand(sol)
    else:
        x = sol
    Finf = float(np.max(np.abs(op.residual(x))))
    print('[%s] conv=%s wall=%.0fs final full ||F||inf=%.3e'
          % (label, conv, wall, Finf), flush=True)
    return x, Finf, conv


if __name__ == '__main__':
    x0 = np.load(os.path.join(WARM, 'P_inner_G9.npy')).ravel()
    d = 3e-2
    results = {}
    for label, kwargs in [
        ('sym', dict(use_sym=True)),
        ('sym_rdiff1e-4', dict(use_sym=True, rdiff=1e-4)),
        ('rdiff1e-4', dict(rdiff=1e-4)),
        ('rdiff1e-5', dict(rdiff=1e-5)),
    ]:
        r = run_one(label, d, x0, **kwargs)
        if r is not None:
            results[label] = (r[1], r[2])
            if r[2]:
                np.save('/tmp/newton_matrix_%s.npy' % label,
                        r[0].reshape(9, 9, 9))
    print(results)
