"""High-tau / low-gamma certification: extend lowtau.json to corners of
max revelation deficit.

Plan A: new tau rows {0.80, 1.20} on the standard 20-gamma list.
Plan B: low-gamma extension {0.01, 0.02, 0.03} at tau in {0.10, 0.20, 0.40,
0.60, 0.80, 1.00, 1.50}.
Plan C: extend tau=2.0 (which truncated at gamma=1.45) with a few more cells.

Uses the same machinery as lowtau_polish.py (chain64 from ld_polish + ld
polish) but adds tau-continuation rescue using the existing emin15+lowtau
solutions as warm-start sources at the same gamma.
"""
import os, sys, json, time, warnings
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/tmp')
import numpy as np
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
warnings.filterwarnings('ignore')
import shutil
shutil.copy('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd/ld_ops.py', '/tmp/ld_ops.py')
import ld_polish as P  # imports ld_ops, defines chain64, polish_ld, build_grid, solve64, metrics

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
os.makedirs(OUT, exist_ok=True)

# Standard 20-gamma list
GAMMAS_STD = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 20), 4))
GAMMAS_LOW = [0.01, 0.02, 0.03]   # low-gamma extension

# Plan A: new tau rows with full 20-gamma list (also includes low-gamma corner via prefix)
PLAN_A = [(tau, GAMMAS_LOW + GAMMAS_STD) for tau in [0.80, 1.20]]
# Plan B: low-gamma extension at existing taus (cells already certified at standard list)
PLAN_B = [(tau, GAMMAS_LOW) for tau in [0.10, 0.20, 0.40, 0.60, 1.00, 1.50]]
# Plan C: top-up at tau=2.0 with a few more gamma values (low-gamma + a couple in [2,3])
PLAN_C = [(2.0, GAMMAS_LOW + [2.0294, 2.8418])]


def find_warm_source(target_tau, gamma, accept_table):
    """Return P_inner (np.float64, 21x21x21) from the nearest lower tau at the
    same gamma, by searching lowtau dir then emin15 dir.  None if missing."""
    # candidate sources by tau distance below target
    cands = sorted([t for (t, g) in accept_table if abs(g - gamma) < 1e-5
                    and t < target_tau], reverse=True)
    for t in cands:
        for d in (OUT, EMIN15):
            f = f"{d}/P_ld_t{t}_g{gamma}.npy"
            if os.path.exists(f):
                return t, np.load(f).astype(np.float64)
    return None, None


def tau_march(P_warm, tau_src, tau_dst, gamma, step=0.1, maxiter=30):
    """March from tau_src to tau_dst on G=21 in steps of `step`."""
    P_cur = P_warm
    tt = tau_src
    while tt < tau_dst - 1e-9:
        nxt = min(round(tt + step, 4), tau_dst)
        P_cur, F_cur = P.solve64(round(nxt, 4), gamma, 21, P_cur, maxiter=maxiter)
        if P_cur is None or F_cur > 1e-6:
            return None, np.inf, nxt
        tt = nxt
    return P_cur, F_cur, tt


def collect_certified():
    """Return list of (tau, gamma) with certified saved P_ld files."""
    out = []
    try:
        d = json.load(open(f"{OUT}/lowtau.json"))
        for v in d.values():
            if v.get('verdict') == 'ACCEPT':
                out.append((float(v['tau']), float(v['gamma'])))
    except FileNotFoundError: pass
    try:
        d = json.load(open(f"{EMIN15}/emin15.json"))
        for v in d.values():
            if v.get('verdict') == 'ACCEPT':
                out.append((float(v['tau']), float(v['gamma'])))
    except FileNotFoundError: pass
    return out


def run_cell(tau, gamma, P_warm_cross, accept_table):
    """Try chain64 with optional warm start; fall back to tau-rescue."""
    t0 = time.time()
    # Attempt 1: plain chain64 with cross-cell warm start
    P64, F64 = P.chain64(tau, gamma, P_warm_cross)
    rescued = False
    if P64 is None or F64 > 1e-8:
        # Rescue: tau-continuation from lower-tau at same gamma
        tsrc, Pwarm = find_warm_source(tau, gamma, accept_table)
        if Pwarm is not None:
            P_march, F_march, t_reached = tau_march(Pwarm, tsrc, tau, gamma, step=0.1)
            if P_march is not None:
                P2, F2 = P.solve64(tau, gamma, 21, P_march, maxiter=40)
                if P2 is not None and F2 < 1e-8:
                    P64, F64, rescued = P2, F2, True
        if (P64 is None or F64 > 1e-8) and Pwarm is None:
            # Last resort: try seeding chain64 with cross-gamma warm if available
            pass
    if P64 is None or F64 > 1e-8:
        return dict(tau=tau, gamma=gamma, F64=float(F64) if np.isfinite(F64) else None,
                    verdict='chain_fail', wall=time.time()-t0, rescued=rescued), None

    du, uf, lo, hi = P.build_grid(21)
    U1,U2,U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
    T = U1+U2+U3
    P_ld, hist = P.polish_ld(tau, gamma, P64)
    F_ld = hist[-1]
    s, d = P.metrics(P_ld, T)
    wall = time.time() - t0
    verdict = 'ACCEPT' if F_ld <= 1e-15 else 'PROVISIONAL'
    return dict(tau=tau, gamma=gamma, F64=float(F64), F_ld=float(F_ld),
                slope=float(s), deficit=float(d), verdict=verdict,
                rescued=rescued, wall=float(wall)), (P_ld if verdict == 'ACCEPT' else None)


def main():
    t_start = time.time()
    results_path = f"{OUT}/hightau.json"
    if os.path.exists(results_path):
        results = json.load(open(results_path))
        print(f"Loaded {len(results)} pre-existing cells from hightau.json", flush=True)
    else:
        results = {}

    accept_table = set(collect_certified())
    print(f"Have {len(accept_table)} pre-existing certified (tau,gamma) cells "
          f"for warm-start sourcing", flush=True)

    all_plans = [('A_t0.80', [(0.80, GAMMAS_LOW + GAMMAS_STD)]),
                 ('A_t1.20', [(1.20, GAMMAS_LOW + GAMMAS_STD)]),
                 ('B_lowg', PLAN_B),
                 ('C_t2.0', PLAN_C)]

    for plan_name, plan_cells in all_plans:
        print(f"\n=== Plan {plan_name} ===", flush=True)
        for tau, gammas in plan_cells:
            print(f"\n--- TAU = {tau} ---", flush=True)
            # Walk gammas in increasing order with cross-cell warm start
            P_cross = None
            for gamma in sorted(gammas):
                key = f"t{tau}_g{gamma}"
                if key in results and results[key].get('verdict') == 'ACCEPT':
                    print(f"  {key}: already ACCEPT, skip", flush=True)
                    # Still try to use as warm-start for next gamma
                    pf = f"{OUT}/P_ld_t{tau}_g{gamma}.npy"
                    if not os.path.exists(pf):
                        pf = f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy"
                    if os.path.exists(pf):
                        P_cross = np.load(pf).astype(np.float64)
                    continue
                # Skip if existing emin15 ACCEPT (don't re-do work)
                em_p = f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy"
                if os.path.exists(em_p) and (tau, gamma) in accept_table:
                    print(f"  {key}: already certified in emin15, copying record", flush=True)
                    em_json = json.load(open(f"{EMIN15}/emin15.json"))
                    if key in em_json and em_json[key].get('verdict') == 'ACCEPT':
                        v = em_json[key]
                        results[key] = dict(tau=float(v['tau']), gamma=float(v['gamma']),
                                            F64=float(v['F64']),
                                            F_ld=float(v['F_ld']),
                                            slope=float(v['slope']),
                                            deficit=float(v['deficit']),
                                            verdict='ACCEPT',
                                            src='emin15')
                        P_cross = np.load(em_p).astype(np.float64)
                        json.dump(results, open(results_path, 'w'), indent=2, default=str)
                    continue
                rec, P_ld = run_cell(tau, gamma, P_cross, accept_table)
                results[key] = rec
                if rec.get('verdict') == 'ACCEPT':
                    np.save(f"{OUT}/P_ld_t{tau}_g{gamma}.npy",
                            np.asarray(P_ld, np.float64))
                    accept_table.add((tau, gamma))
                    # update warm-start chain
                    P_cross = np.asarray(P_ld, np.float64)
                tag = ''
                if rec.get('rescued'): tag += ' [rescued]'
                if 'F_ld' in rec:
                    print(f"  tau={tau:5.2f} gamma={gamma:8.4f}  F64={rec['F64']:.2e}  "
                          f"F_ld={rec['F_ld']:.2e}  d={rec.get('deficit',float('nan')):.3e}  "
                          f"{rec['verdict']}{tag}  ({rec['wall']:.0f}s)", flush=True)
                else:
                    print(f"  tau={tau:5.2f} gamma={gamma:8.4f}  FAIL {rec['verdict']}{tag}  "
                          f"({rec['wall']:.0f}s)", flush=True)
                json.dump(results, open(results_path, 'w'), indent=2, default=str)

    n_acc = sum(1 for v in results.values() if v.get('verdict') == 'ACCEPT')
    print(f"\n{n_acc}/{len(results)} ACCEPTED in hightau.json; "
          f"total {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
