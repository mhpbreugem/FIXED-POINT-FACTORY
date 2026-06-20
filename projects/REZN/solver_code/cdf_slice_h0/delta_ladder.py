"""Boxcar-delta continuation at G=9: Newton at each delta, walking
delta down from 1e-1 (smooth) to 1e-7 (effectively strict h=0).
Answers: down to what regularization scale does Newton converge to
machine eps, and do the solutions/deficits converge as delta -> 0?"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdf_slice_ops import CDFSliceOperator
from validate_cdf_slice import (_newton, deficit_unweighted, WARM, OUT,
                                log, interp_inner)

TAU, GAMMA = 2.0, 0.1


def run(G=9, deltas=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4,
                     1e-5, 1e-6, 1e-7), x0=None, tag='dladder'):
    tv = np.full(3, TAU)
    gv = np.full(3, GAMMA)
    wv = np.full(3, 1.0)
    if x0 is None:
        x0 = np.load(os.path.join(WARM, 'P_inner_G9.npy')).ravel()
        if G != 9:
            x0 = interp_inner(x0.reshape((9,) * 3), 9, G).ravel()
    rows = []
    x = x0.copy()
    log('=== boxcar-delta continuation ladder G=%d [%s] ===' % (G, tag))
    for d in deltas:
        op = CDFSliceOperator(G, tv, gv, wv, method='boxcar',
                              boxcar_delta=d)
        F0 = float(np.max(np.abs(op.residual(x))))
        log('-- delta=%.0e  initial ||F||inf=%.3e' % (d, F0))
        sol, Finf, conv, wall, hist, nfev = _newton(
            op, x, label='%s-d%.0e' % (tag, d))
        defi, slope = deficit_unweighted(sol.reshape((G,) * 3), G, TAU)
        log('   delta=%.0e final ||F||inf=%.3e conv=%s wall=%.0fs '
            'nfev=%d deficit=%.5f slope=%.4f Prange=[%.2e, %.8f]'
            % (d, Finf, conv, wall, nfev, defi, slope,
               sol.min(), sol.max()))
        rows.append(dict(G=G, delta=d, F0=F0, Finf=Finf, converged=conv,
                         wall=wall, nfev=nfev, deficit=defi,
                         slope=slope))
        with open(os.path.join(OUT, '%s_G%d.json' % (tag, G)), 'w') as f:
            json.dump(rows, f, indent=2)
        if conv:
            x = sol.copy()          # continue from converged solution
            np.save(os.path.join(OUT, 'P_%s_G%d_d%.0e.npy' % (tag, G, d)),
                    sol.reshape((G,) * 3))
        else:
            log('   (not converged: keeping previous warm start)')
    return rows


if __name__ == '__main__':
    run()
