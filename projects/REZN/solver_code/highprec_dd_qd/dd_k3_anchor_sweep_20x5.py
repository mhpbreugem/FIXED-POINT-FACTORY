"""20x5 (gamma, tau) joint-limit anchor sweep with continuation warm-start.

Per (gamma, tau) cell: G-ladder {9, 13, 17, 21}, h = 0.45*sqrt(du), kernel-
smooth Phi + Newton-Krylov. Chained warm-start scheme:
  1. Within a cell: walk G=9 -> 21 warm-starting each from the previous.
  2. Across cells at fixed tau: walk gamma low -> high, warm-starting each
     gamma from the previous gamma's converged P_G21. If the no-learning
     init is closer to a different basin (low-PR vs high-PR), continuation
     keeps us in the right one.
  3. If a cell stalls (F > 1e-8): try the no-learning init as a fallback.
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

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/anchor_sweep_20x5'
os.makedirs(OUT, exist_ok=True)

K = 3; pad = 2; UMAX = 4.0
C = 0.45
G_LIST = [9, 13, 17, 21]
TAUS = [0.2, 0.5, 1.0, 1.5, 2.0]
GAMMAS = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 20), 4))


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


def solve_one(tau, gamma, Gi, P_warm_inner, maxiter=30, f_tol=1e-12):
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    U1,U2,U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
    T = U1+U2+U3
    if P_warm_inner is None:
        P_full = init_no_learning_K3(uf, tv, gv, wv)
        P_inner0 = P_full[lo:hi, lo:hi, lo:hi].copy()
    else:
        P_inner0 = resample(P_warm_inner, Gi)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P_full[lo:hi, lo:hi, lo:hi] = P_inner0
    Pinner = P_full[lo:hi, lo:hi, lo:hi].copy()
    def F(x):
        Pi = x.reshape(Gi, Gi, Gi)
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = Pi
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - Pi).ravel()
    try:
        x = newton_krylov(F, Pinner.ravel(), f_tol=f_tol, maxiter=maxiter, verbose=False)
        Pn = x.reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(x))))
    except NoConvergence as e:
        Pn = e.args[0].reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(e.args[0]))))
    s, d = metrics(Pn, T)
    return Pn, Fn, s, d


def solve_cell(tau, gamma, P_warm_cross_cell):
    """Solve a cell with G-ladder. Returns dict + final P_inner at G=21."""
    P = None  # within-cell warm chain
    if P_warm_cross_cell is not None:
        P = P_warm_cross_cell  # this is at G=21 from previous cell
    ladder = []
    P_final = None
    for Gi in G_LIST:
        Pn, Fn, s, d = solve_one(tau, gamma, Gi, P)
        ladder.append(dict(G=Gi, F=Fn, slope=s, deficit=d))
        P = Pn
        P_final = Pn
    last = ladder[-1]
    # Fallback if stalled: try the no-learning init from scratch
    if last['F'] > 1e-8:
        ladder2 = []
        P = None
        for Gi in G_LIST:
            Pn, Fn, s, d = solve_one(tau, gamma, Gi, P)
            ladder2.append(dict(G=Gi, F=Fn, slope=s, deficit=d))
            P = Pn
        if ladder2[-1]['F'] < last['F']:
            ladder = ladder2; last = ladder2[-1]; P_final = P
    return dict(tau=tau, gamma=gamma, ladder=ladder,
                F_final=last['F'], slope=last['slope'], deficit=last['deficit'],
                ok=(last['F'] < 1e-8)), P_final


def main():
    t_global = time.time()
    print(f"20x5 anchor sweep: {len(GAMMAS)} gammas in [{GAMMAS[0]}, {GAMMAS[-1]}], "
          f"{len(TAUS)} taus in {TAUS}")
    print(f"{'tau':>5} {'gamma':>8} {'F':>10} {'slope':>8} {'deficit':>8} {'wall':>6}",
          flush=True)
    grid = {}
    for tau in TAUS:
        P_cross = None
        for gamma in GAMMAS:
            t0 = time.time()
            res, P_cross = solve_cell(tau, gamma, P_cross)
            wall = time.time() - t0
            res['wall'] = wall
            print(f"{tau:>5.2f} {gamma:>8.4f} {res['F_final']:>10.2e} "
                  f"{res['slope']:>8.4f} {res['deficit']:>8.4f} {wall:>5.0f}s"
                  + ("" if res['ok'] else "  [stall]"), flush=True)
            grid[f"t{tau}_g{gamma}"] = res
            json.dump(grid, open(f"{OUT}/grid.json", "w"), indent=2, default=str)
    total = time.time() - t_global
    nok = sum(1 for v in grid.values() if v['ok'])
    print(f"\n{nok}/{len(grid)} cells nailed (F < 1e-8); total wall {total/60:.1f} min")


if __name__ == "__main__":
    main()
