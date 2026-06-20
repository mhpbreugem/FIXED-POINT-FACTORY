"""Deep G-ladder continuum extrapolation study (referee response).

For 5 representative (tau, gamma) cells, extend the G-ladder past the
certified G=21 solutions to G = 25, 29, 33 (and 37 if wall-clock allows),
warm-starting each rung by index-resampling the previous one.
h = 0.45*sqrt(du) joint-limit schedule, identical to the 20x5 sweep.

Checkpoints results.json after EVERY solve.
"""
import os, sys, json, time, warnings
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/deep_ladder'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
GRID20x5 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/anchor_sweep_20x5/grid.json'
os.makedirs(OUT, exist_ok=True)

K = 3; pad = 2; UMAX = 4.0
C = 0.45
DEEP_G = [25, 29, 33, 37]
WALL_CAP = 25 * 60.0  # stop deepening a cell if a single solve exceeds this
CELLS = [(0.2, 1.035), (0.5, 1.035), (1.0, 1.035), (1.0, 0.098), (2.0, 0.098)]


def build_grid(Gi):
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def resample(P_warm, Gi):
    Gj = P_warm.shape[0]
    idx = np.linspace(0, Gj-1, Gi).astype(int)
    return P_warm[idx][:, idx][:, :, idx].copy()


def metrics(Pin, T):
    Pc = np.clip(Pin, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    slope = a[0]
    pr = a[0]*T.ravel() + a[1]
    deficit = np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30)
    return float(slope), float(deficit)


def solve_one(tau, gamma, Gi, P_warm_inner, maxiter=40, f_tol=1e-11):
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    U1, U2, U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
    T = U1 + U2 + U3
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    if P_warm_inner is None:
        P_inner0 = P_full[lo:hi, lo:hi, lo:hi].copy()
    else:
        P_inner0 = resample(P_warm_inner, Gi)
    P_full[lo:hi, lo:hi, lo:hi] = P_inner0

    def F(x):
        Pi = x.reshape(Gi, Gi, Gi)
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = Pi
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - Pi).ravel()

    try:
        x = newton_krylov(F, P_inner0.ravel(), f_tol=f_tol, maxiter=maxiter,
                          verbose=False)
        Pn = x.reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(x))))
    except NoConvergence as e:
        Pn = e.args[0].reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(e.args[0]))))
    s, d = metrics(Pn, T)
    return Pn, Fn, s, d, float(du), float(h)


def main():
    res_path = f"{OUT}/results.json"
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    grid20 = json.load(open(GRID20x5))

    for tau, gamma in CELLS:
        key = f"t{tau}_g{gamma}"
        cell = results.get(key, {})
        cell['tau'] = tau; cell['gamma'] = gamma
        # rungs G=9..21 from the certified 20x5 sweep (free)
        prior = grid20[key]['ladder']
        ladder = cell.get('ladder', [])
        have_G = {r['G'] for r in ladder}
        for r in prior:
            if r['G'] not in have_G:
                du = 2*UMAX/(r['G'] - 1)
                ladder.append(dict(G=r['G'], du=du, h=C*np.sqrt(du),
                                   F=r['F'], slope=r['slope'],
                                   deficit=r['deficit'], wall=None,
                                   source='anchor_sweep_20x5'))
                have_G.add(r['G'])
        ladder.sort(key=lambda r: r['G'])
        cell['ladder'] = ladder
        results[key] = cell
        json.dump(results, open(res_path, 'w'), indent=2)

        # warm start: certified G=21 long-double-refined solution
        P = np.load(f"{EMIN15}/P_ld_{key}.npy").astype(np.float64)
        print(f"\n=== cell {key} (warm start {P.shape}) ===", flush=True)
        stopped = cell.get('stopped_at')
        for Gi in DEEP_G:
            pfile = f"{OUT}/P_{key}_G{Gi}.npy"
            if Gi in have_G and os.path.exists(pfile):
                P = np.load(pfile)
                print(f"  G={Gi}: already done, reloaded", flush=True)
                continue
            if stopped:
                break
            t0 = time.time()
            Pn, Fn, s, d, du, h = solve_one(tau, gamma, Gi, P)
            wall = time.time() - t0
            np.save(pfile, Pn)
            ladder.append(dict(G=Gi, du=du, h=h, F=Fn, slope=s, deficit=d,
                               wall=wall, source='deep_ladder'))
            ladder.sort(key=lambda r: r['G'])
            have_G.add(Gi)
            ok = Fn < 1e-8
            print(f"  G={Gi}: F={Fn:.2e} slope={s:.5f} deficit={d:.6f} "
                  f"wall={wall:.0f}s" + ("" if ok else " [STALL]"), flush=True)
            P = Pn
            if wall > WALL_CAP:
                cell['stopped_at'] = Gi
                stopped = Gi
                print(f"  -> wall {wall:.0f}s > cap; stop deepening {key}",
                      flush=True)
            json.dump(results, open(res_path, 'w'), indent=2)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
