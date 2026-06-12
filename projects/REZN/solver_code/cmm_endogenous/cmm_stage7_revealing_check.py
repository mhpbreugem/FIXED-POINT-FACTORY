"""Stage 7 post-hoc: distance of warm-start and converged H to the exact
fully-revealing strict-h=0 solution H_m == logit(p_m)/(tau*sqrt(3)),
plus deficit of the revealing solution (analytically 0; reconstruction
bias of the M=8 logit-linear ladder reported for honesty)."""
import json, sys
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from cmm_stage6b_solver import (Problem, load_P_full, make_levels,
                                 build_initial_H, deficit_from_H, OUT)
from cmm_stage1 import build_grid

SQ3 = np.sqrt(3.0)


def dist_stats(Hs, p_levels, tau):
    T0 = np.log(p_levels/(1 - p_levels))/(tau*SQ3)
    d = np.abs(Hs - T0[:, None, None])
    return float(np.mean(d)), float(np.max(d))


def main():
    lad = json.load(open(f"{OUT}/stage7_tau_ladder.json"))
    du, uf, lo, hi = build_grid(21)
    out = {}
    for key in [k for k in lad if k.startswith('g')]:
        gamma = float(key[1:])
        for tk, rec in lad[key].items():
            if not isinstance(rec, dict) or not rec.get('done'):
                continue
            tau = rec['tau']
            p_levels = np.array(rec['p_levels'])
            Hs = np.load(f"{OUT}/stage7_H_t{tau}_g{gamma}.npy")
            pb = Problem(8, 15, half_width=4.5, n_vert_margin=1,
                         tau=tau, gamma=gamma)
            mean_f, max_f = dist_stats(Hs, p_levels, tau)
            # warm start distance
            if rec['warm_start'] == 'kernel_FP':
                _, _, _, _, P_inner, P_full = load_P_full(21, tau, gamma)
                H0 = build_initial_H(P_full, uf, p_levels, pb.a_grid, pb.b_grid)
                mean_0, max_0 = dist_stats(H0, p_levels, tau)
            else:
                mean_0, max_0 = None, None
            # deficit of exact revealing ladder (reconstruction bias check)
            T0 = np.log(p_levels/(1 - p_levels))/(tau*SQ3)
            Hrev = np.repeat(T0[:, None, None], 15, 1).repeat(15, 2)
            d_rev, nv, _ = deficit_from_H(np.ascontiguousarray(Hrev),
                                          p_levels, pb, uf, lo, hi)
            # residual of the analytic revealing solution at this cell's
            # own p-levels and quadrature (certification of an exact
            # strict-h=0 fixed point)
            pbq = Problem(8, 15, half_width=4.5, n_vert_margin=1,
                          tau=tau, gamma=gamma, s_max=rec['s_max'],
                          n_s=rec['n_s'])
            rmax = 0.0
            for m in range(len(p_levels)):
                r, _, nf = pbq.residual(
                    np.ascontiguousarray(Hrev[m]), p_levels[m])
                rmax = max(rmax, float(np.max(np.abs(r))))
            out[f"{key}_{tk}"] = dict(
                tau=tau, gamma=gamma,
                dist_warm_mean=mean_0, dist_warm_max=max_0,
                dist_final_mean=mean_f, dist_final_max=max_f,
                deficit_revealing_reconstr=d_rev,
                revealing_max_r=rmax)
            w = f"{mean_0:.3f}" if mean_0 is not None else "  -  "
            print(f"{key} {tk}: mean|H-Hrev| warm={w} final={mean_f:.3f} "
                  f"  d(revealing,reconstr)={d_rev:.2e}  "
                  f"max|r|(revealing)={rmax:.2e}")
    json.dump(out, open(f"{OUT}/stage7_revealing_check.json", 'w'), indent=1)
    print(f"-> {OUT}/stage7_revealing_check.json")


if __name__ == '__main__':
    main()
