"""Compute VoI metrics for newly certified hightau cells.

Reuses lowtau_metrics.cell_metrics. Appends to LOWTAU/metrics.json without
overwriting existing entries.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from lowtau_metrics import cell_metrics, build_grid

LOWTAU = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)

    # Load existing metrics
    metrics_path = f"{LOWTAU}/metrics.json"
    if os.path.exists(metrics_path):
        results = json.load(open(metrics_path))
    else:
        results = {}
    print(f"Existing metrics: {len(results)} cells", flush=True)

    # Discover all certified cells across the three JSONs
    cells = set()
    for path, name in [(f"{LOWTAU}/hightau.json", 'hightau'),
                       (f"{LOWTAU}/lowtau.json", 'lowtau'),
                       (f"{EMIN15}/emin15.json", 'emin15')]:
        if not os.path.exists(path): continue
        d = json.load(open(path))
        for v in d.values():
            if v.get('verdict') == 'ACCEPT':
                cells.add((float(v['tau']), float(v['gamma']), name))

    # For each, compute metrics if not already done
    to_do = []
    for tau, gamma, src in sorted(cells):
        key = f"t{tau}_g{gamma}"
        if key in results: continue
        # find P file
        for d in (LOWTAU, EMIN15):
            f = f"{d}/P_ld_t{tau}_g{gamma}.npy"
            if os.path.exists(f):
                to_do.append((tau, gamma, src, f)); break

    print(f"To compute: {len(to_do)} new cells", flush=True)
    t0_all = time.time()
    for i, (tau, gamma, src, p_src) in enumerate(to_do):
        key = f"t{tau}_g{gamma}"
        P_inner = np.load(p_src)
        t0 = time.time()
        m = cell_metrics(P_inner, uf, lo, hi, tau, gamma)
        wall = time.time() - t0
        if m is None: continue
        m['tau'] = tau; m['gamma'] = gamma; m['src'] = src; m['wall'] = wall
        results[key] = m
        if (i+1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_do)}] tau={tau:.2f} g={gamma:.4f}  "
                  f"Vi_priv={m['Vi_private']:.3e}  Vi_pub={m['Vi_public']:.3e}  "
                  f"Vi_FRgap={m['Vi_FR_gap']:.3e}  ({wall:.1f}s)", flush=True)
        json.dump(results, open(metrics_path, 'w'), indent=2)
    print(f"\nFinal: {len(results)} cells in metrics.json "
          f"({(time.time()-t0_all)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
