"""Phase 2b: dense spectra at fine gamma points hugging the fold
(tau=2.0, G=13), warm from the phase-2 deepest solution, marching DOWN.
Appends to phase1 json point list (kept sorted by gamma) for the figure,
and fits 1-rho ~ A*sqrt(gamma*-gamma) to test the fold normal form.
"""
import json, numpy as np
from dd_k3_stall_diag_lib import *

TAU = 2.0

def main():
    ph2 = json.load(open(f"{OUT}/phase2_stepscaling.json"))
    gstar = ph2['deepest_gamma']
    x = np.load(f"{OUT}/P13_t{TAU}_gdeepest.npy").ravel()
    extras = []
    for dg in [0.0005, 0.001, 0.002, 0.004, 0.008]:
        g = round(gstar - dg, 10)
        prob = Problem(TAU, g, 13)
        xs, Fn = prob.solve(x, f_tol=1e-12, maxiter=60)
        if Fn > 1e-10:
            print(f"g={g}: failed to converge downward?! F={Fn:.2e}", flush=True)
            continue
        rep = spectrum_report(prob, xs)
        rep.update(gamma=g, F=Fn, ok=True,
                   fp_sym_defect=symmetry_defect_fp(xs, 13))
        extras.append(rep)
        e = rep['lead_eigs'][0]
        print(f"g={g:.6f} rho={rep['rho']:.8f} sig_min={rep['sigma_min_ImJ']:.4e} "
              f"symB={e['sym_break']:.3f}", flush=True)
    ph1 = json.load(open(f"{OUT}/phase1_tau2.0.json"))
    pts = [p for p in ph1['points'] if p.get('ok')] + extras
    if 'deepest_spectrum' in ph2:
        ds = dict(ph2['deepest_spectrum']); ds.update(ok=True, F=0.0)
        pts.append(ds)
    pts.sort(key=lambda p: p['gamma'])
    fails = [p for p in ph1['points'] if not p.get('ok')]
    ph1['points'] = pts + fails
    checkpoint(ph1, 'phase1_tau2.0.json')
    # normal-form fit: (1-rho)^2 vs gamma should be linear near fold
    gs = np.array([p['gamma'] for p in pts if p['rho'] > 0.9])
    r2 = np.array([(1.0 - p['rho'])**2 for p in pts if p['rho'] > 0.9])
    if gs.size >= 3:
        a, b = np.polyfit(gs, r2, 1)
        gfold = -b/a
        print(f"normal-form fit: (1-rho)^2 = {a:.4g}*(gamma - {gfold:.6f}); "
              f"predicted fold gamma* = {gfold:.6f}", flush=True)
        json.dump(dict(gamma_fold_fit=float(gfold), slope=float(a)),
                  open(f"{OUT}/phase2b_normalform.json", 'w'), indent=1)

if __name__ == '__main__':
    main()
