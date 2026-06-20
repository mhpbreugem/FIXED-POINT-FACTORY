"""Joint-limit anchor on a (gamma, tau) grid: extend the immortal anchor
from a single (2, 0.1) cell to a small map of nailed PR equilibria.

Pipeline per cell: G ladder {9,13,17,21}, h = 0.45*sqrt(du), kernel-smooth
phi + newton_krylov, chained warm-start. Report deficit at the finest G as
the cell's certified value.
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

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/anchor_grid'
os.makedirs(OUT, exist_ok=True)

K = 3; pad = 2; UMAX = 4.0
C = 0.45
G_LIST = [9, 13, 17, 21]
CELLS = [(tau, gamma) for tau in [0.5, 1.0, 2.0]
                       for gamma in [0.1, 1.0, 10.0]]


def build_grid(Gi):
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def metrics(Pin, T):
    Pc = np.clip(Pin, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    slope = a[0]
    pr = a[0]*T.ravel() + a[1]
    deficit = np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30)
    return float(slope), float(deficit)


def solve_cell(tau, gamma):
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_warm = None
    out = []
    for Gi in G_LIST:
        du, uf, lo, hi = build_grid(Gi)
        h = C*np.sqrt(du)
        U1,U2,U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
        T = U1+U2+U3
        if P_warm is None:
            P_full = init_no_learning_K3(uf, tv, gv, wv)
            P_inner0 = P_full[lo:hi, lo:hi, lo:hi].copy()
        else:
            Gj = P_warm.shape[0]
            idx = np.linspace(0, Gj-1, Gi).astype(int)
            P_inner0 = P_warm[idx][:, idx][:, :, idx].copy()
        P_full = init_no_learning_K3(uf, tv, gv, wv)
        P_full[lo:hi, lo:hi, lo:hi] = P_inner0
        Pinner = P_full[lo:hi, lo:hi, lo:hi].copy()
        def F(x):
            Pi = x.reshape(Gi, Gi, Gi)
            Pf = P_full.copy()
            Pf[lo:hi, lo:hi, lo:hi] = Pi
            Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
            return (Pn[lo:hi, lo:hi, lo:hi] - Pi).ravel()
        t0 = time.time()
        try:
            x = newton_krylov(F, Pinner.ravel(), f_tol=1e-12, maxiter=30, verbose=False)
            Pn = x.reshape(Gi, Gi, Gi)
            Fn = float(np.max(np.abs(F(x))))
            ok = Fn < 1e-8
        except NoConvergence as e:
            Pn = e.args[0].reshape(Gi, Gi, Gi)
            Fn = float(np.max(np.abs(F(e.args[0]))))
            ok = False
        except Exception as e:
            print(f"    (tau={tau} gamma={gamma} G={Gi}) ERROR {e}", flush=True)
            return None
        wall = time.time() - t0
        slope, deficit = metrics(Pn, T)
        out.append(dict(G=Gi, du=du, h=h, F=Fn, slope=slope, deficit=deficit,
                          wall=wall, ok=ok))
        P_warm = Pn
    return out


def main():
    print(f"Joint-limit anchor grid over {len(CELLS)} (tau, gamma) cells", flush=True)
    print(f"{'tau':>5} {'gamma':>7} {'F(G=21)':>10} {'slope':>8} "
          f"{'deficit':>8} {'wall':>6}", flush=True)
    grid = {}
    for tau, gamma in CELLS:
        t0 = time.time()
        seq = solve_cell(tau, gamma)
        if seq is None: continue
        last = seq[-1]
        wall = time.time() - t0
        print(f"{tau:>5.2f} {gamma:>7.2f} {last['F']:>10.1e} "
              f"{last['slope']:>8.4f} {last['deficit']:>8.4f} {wall:>5.0f}s", flush=True)
        grid[f"t{tau}_g{gamma}"] = dict(tau=tau, gamma=gamma, ladder=seq, wall=wall)
    json.dump(grid, open(f"{OUT}/grid.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {OUT}/grid.json", flush=True)


if __name__ == "__main__":
    main()
