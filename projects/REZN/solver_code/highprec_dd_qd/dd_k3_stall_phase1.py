"""Phase 1: gamma march at tau=2.0, G=13, spectra along the approach."""
import time, numpy as np
from dd_k3_stall_diag_lib import *

TAU = 2.0
GAMMAS = [0.3769, 0.42, 0.46, 0.50, 0.5278, 0.55, 0.58, 0.61, 0.64,
          0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85, 0.90, 1.00]
F_OK = 1e-10

def main():
    results = dict(tau=TAU, G=13, points=[], fail=None)
    x = load_warm(TAU, '0.3769', 13)
    n_fail = 0
    for g in GAMMAS:
        prob = Problem(TAU, g, 13)
        t0 = time.time()
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        wall = time.time() - t0
        ok = Fn < F_OK
        print(f"gamma={g:.4f}  F={Fn:.3e}  {'ok' if ok else 'STALL'}  "
              f"({wall:.1f}s)", flush=True)
        if not ok:
            n_fail += 1
            results['points'].append(dict(gamma=g, F=Fn, ok=False))
            results['fail'] = g
            checkpoint(results, 'phase1_tau2.0.json')
            if n_fail >= 2:
                break
            continue
        n_fail = 0
        x = xs
        np.save(f"{OUT}/P13_t{TAU}_g{g}.npy", xs.reshape(13, 13, 13))
        rep = spectrum_report(prob, xs)
        rep.update(gamma=g, F=Fn, ok=True,
                   fp_sym_defect=symmetry_defect_fp(xs, 13))
        results['points'].append(rep)
        e = rep['lead_eigs'][0]
        print(f"   rho={rep['rho']:.6f} ({e['re']:+.6f}{e['im']:+.6f}i, "
              f"symB={e['sym_break']:.3f})  sig_min(I-J)={rep['sigma_min_ImJ']:.4e}  "
              f"dist(eig,1)={rep['eig_closest_1']['dist']:.4e}", flush=True)
        checkpoint(results, 'phase1_tau2.0.json')
    print("phase 1 done")

if __name__ == '__main__':
    main()
