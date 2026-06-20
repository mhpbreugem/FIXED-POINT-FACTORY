"""Phase 3: fold curve gamma*(tau) at G=13 by adaptive continuation
bracketing at tau = 1.5, 1.75, 2.0; plus a transversal tau-march at
fixed gamma to confirm the fold is crossed in the tau direction too.
"""
import json, time, numpy as np
from dd_k3_stall_diag_lib import *

F_OK = 1e-10


def gamma_march_adaptive(tau, g0, x0, step0, min_step, gcap=40.0):
    """March gamma upward, halving the step on failure, until
    step < min_step. Returns (g_last_ok, x, history)."""
    g, x, step = g0, x0.copy(), step0
    hist = []
    while step >= min_step and g < gcap:
        gt = round(g + step, 10)
        prob = Problem(tau, gt, 13)
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        ok = Fn < F_OK
        hist.append(dict(gamma=gt, step=step, F=Fn, ok=ok))
        if ok:
            g, x = gt, xs
        else:
            step *= 0.5
    return g, x, hist


def tau_march(gamma, t0, x0, step, tcap, min_step=0.0025):
    """March tau upward at fixed gamma, halving step on failure."""
    t, x, st = t0, x0.copy(), step
    hist = []
    while st >= min_step and t < tcap:
        tt = round(t + st, 10)
        prob = Problem(tt, gamma, 13)
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        ok = Fn < F_OK
        hist.append(dict(tau=tt, step=st, F=Fn, ok=ok))
        if ok:
            t, x = tt, xs
        else:
            st *= 0.5
    return t, x, hist


def main():
    res = dict(G=13, fold_points=[], transversal=None)

    # --- tau = 2.0: refine from phase-2 deepest ---
    ph2 = json.load(open(f"{OUT}/phase2_stepscaling.json"))
    g0 = ph2['deepest_gamma']
    x0 = np.load(f"{OUT}/P13_t2.0_gdeepest.npy").ravel()
    t0 = time.time()
    g, x, hist = gamma_march_adaptive(2.0, g0, x0, 0.0005, 1e-5)
    print(f"tau=2.00  gamma* = {g:.6f}  (gamma*tau = {2.0*g:.4f}) "
          f"[{time.time()-t0:.0f}s, {len(hist)} solves]", flush=True)
    res['fold_points'].append(dict(tau=2.0, gamma_star=g, gt=2.0*g,
                                   hist=hist))
    checkpoint(res, 'phase3_foldcurve.json')
    np.save(f"{OUT}/P13_fold_t2.0.npy", x.reshape(13, 13, 13))

    # --- tau = 1.5: start from last certified g=7.8027 ---
    x0 = load_warm(1.5, '7.8027', 13)
    prob = Problem(1.5, 7.8027, 13)
    x0, Fn = prob.solve(x0, f_tol=1e-12, maxiter=60)
    print(f"tau=1.5 re-converge g=7.8027 at G=13: F={Fn:.2e}", flush=True)
    t0 = time.time()
    g, x, hist = gamma_march_adaptive(1.5, 7.8027, x0, 1.0, 1e-4)
    print(f"tau=1.50  gamma* = {g:.6f}  (gamma*tau = {1.5*g:.4f}) "
          f"[{time.time()-t0:.0f}s, {len(hist)} solves]", flush=True)
    res['fold_points'].append(dict(tau=1.5, gamma_star=g, gt=1.5*g,
                                   hist=hist))
    checkpoint(res, 'phase3_foldcurve.json')
    np.save(f"{OUT}/P13_fold_t1.5.npy", x.reshape(13, 13, 13))

    # --- tau = 1.75: build a start by tau-march at fixed gamma=2.0294 ---
    x0 = load_warm(1.5, '2.0294', 13)
    prob = Problem(1.5, 2.0294, 13)
    x0, Fn = prob.solve(x0, f_tol=1e-12, maxiter=60)
    print(f"tau=1.5 re-converge g=2.0294: F={Fn:.2e}", flush=True)
    t, x0, thist = tau_march(2.0294, 1.5, x0, 0.05, 1.75)
    print(f"  tau-march at g=2.0294 reached tau={t}", flush=True)
    if abs(t - 1.75) < 1e-9:
        tg0 = time.time()
        g, x, hist = gamma_march_adaptive(1.75, 2.0294, x0, 0.5, 1e-4)
        print(f"tau=1.75  gamma* = {g:.6f}  (gamma*tau = {1.75*g:.4f}) "
              f"[{time.time()-tg0:.0f}s, {len(hist)} solves]", flush=True)
        res['fold_points'].append(dict(tau=1.75, gamma_star=g, gt=1.75*g,
                                       hist=hist))
        np.save(f"{OUT}/P13_fold_t1.75.npy", x.reshape(13, 13, 13))
    else:
        res['fold_points'].append(dict(tau=1.75, note=f"tau-march stalled at {t}",
                                       hist=thist))
    checkpoint(res, 'phase3_foldcurve.json')

    # --- transversal: fix gamma past the tau=2.0 fold, march tau UP ---
    g_fix = round(res['fold_points'][0]['gamma_star'] + 0.05, 4)
    x0 = load_warm(1.5, '0.7391', 13)
    prob = Problem(1.5, 0.7391, 13)
    x0, Fn = prob.solve(x0, f_tol=1e-12, maxiter=60)
    prob = Problem(1.5, g_fix, 13)
    x0, Fn = prob.solve(x0, f_tol=1e-12, maxiter=60)
    print(f"transversal start (tau=1.5, g={g_fix}): F={Fn:.2e}", flush=True)
    t0 = time.time()
    t, x, hist = tau_march(g_fix, 1.5, x0, 0.02, 2.5, min_step=1e-4)
    print(f"transversal: gamma={g_fix} fixed, tau* = {t:.6f} "
          f"(gamma*tau = {g_fix*t:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    res['transversal'] = dict(gamma=g_fix, tau_star=t, gt=g_fix*t, hist=hist)
    checkpoint(res, 'phase3_foldcurve.json')
    np.save(f"{OUT}/P13_fold_transversal_g{g_fix}.npy", x.reshape(13, 13, 13))
    print("phase 3 done")


if __name__ == '__main__':
    main()
