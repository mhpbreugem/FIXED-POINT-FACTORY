"""Phase 2: step-size scaling of the maximal reachable gamma at tau=2.0, G=13.

If a fold exists, gamma_max(step) saturates as step -> 0; if Newton-Krylov
is merely failing, smaller steps keep inching forward.
"""
import json, time, sys, numpy as np
from dd_k3_stall_diag_lib import *

TAU = 2.0
F_OK = 1e-10
STEPS = [0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
MAX_SOLVES_PER_STEP = 300


def march(x0, g0, step, gmax_cap=2.0):
    """March gamma upward by fixed step until first failure.
    Returns (gamma_max, x_at_gamma_max, n_solves)."""
    x = x0.copy(); g = g0; n = 0
    while g + step <= gmax_cap and n < MAX_SOLVES_PER_STEP:
        gt = round(g + step, 10)
        prob = Problem(TAU, gt, 13)
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        n += 1
        if Fn < F_OK:
            g, x = gt, xs
        else:
            print(f"    step={step}: fail at gamma={gt:.6f} (F={Fn:.2e})",
                  flush=True)
            break
    return g, x, n


def main():
    ph1 = json.load(open(f"{OUT}/phase1_tau2.0.json"))
    ok_pts = [p for p in ph1['points'] if p.get('ok')]
    g_base = ok_pts[-2]['gamma'] if len(ok_pts) >= 2 else ok_pts[-1]['gamma']
    x_base = np.load(f"{OUT}/P13_t{TAU}_g{g_base}.npy").ravel()
    print(f"base: gamma={g_base} (last-1 converged in phase 1)", flush=True)
    res = dict(tau=TAU, G=13, g_base=g_base, runs=[])
    best_g, best_x = g_base, x_base
    for step in STEPS:
        t0 = time.time()
        gmax, xg, n = march(x_base, g_base, step)
        wall = time.time() - t0
        print(f"  step={step:<9} gamma_max={gmax:.6f}  ({n} solves, "
              f"{wall:.0f}s)", flush=True)
        res['runs'].append(dict(step=step, gamma_max=gmax, n_solves=n,
                                wall=wall))
        checkpoint(res, 'phase2_stepscaling.json')
        if gmax > best_g:
            best_g, best_x = gmax, xg
    np.save(f"{OUT}/P13_t{TAU}_gdeepest.npy", best_x.reshape(13, 13, 13))
    res['deepest_gamma'] = best_g
    print(f"deepest reachable gamma = {best_g:.6f}; computing spectrum there",
          flush=True)
    prob = Problem(TAU, best_g, 13)
    rep = spectrum_report(prob, best_x)
    rep['gamma'] = best_g
    res['deepest_spectrum'] = rep
    checkpoint(res, 'phase2_stepscaling.json')
    e = rep['lead_eigs'][0]
    print(f"  deepest: rho={rep['rho']:.8f} sig_min(I-J)={rep['sigma_min_ImJ']:.4e} "
          f"symB={e['sym_break']:.3f}", flush=True)


if __name__ == '__main__':
    main()
