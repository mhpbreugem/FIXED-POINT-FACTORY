"""Reproduce the joint-limit K=3 PR equilibrium anchor.

The k3_coarea_limit recipe: at fixed (tau, gamma) = (2, 0.1), take
G in {9, 13, 17, 21, 25, 31} with h = C * du^0.5 where C = 0.45 and
du = 2*UMAX/(G-1). Solve at each G in turn, warm-starting from
init_no_learning_K3 and using newton_krylov; verify F < 1e-11 and
deficit converges to ~0.282-0.291.

If this reproduces, we have a working reproduction of the immortal anchor
inside the current session repository -- useful as the truth reference
for all the cross-check experiments and as the basis for future work.
"""
import os, sys, json, time, warnings
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth, phi_K3_halo
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/coarea_limit_repro'
os.makedirs(OUT, exist_ok=True)

K = 3; pad = 2; UMAX = 4.0
TAU = 2.0; GAMMA = 0.1
C = 0.45
G_LIST = [9, 13, 17, 21, 25]  # G=31 takes too long
tv = np.full(K, TAU); gv = np.full(K, GAMMA); wv = np.full(K, 1.0)


def build_grid(Gi):
    du = 2 * UMAX / (Gi - 1)
    Gf = Gi + 2 * pad
    uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def metrics(Pin, T):
    Pc = np.clip(Pin, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    slope = a[0]
    pr = a[0] * T.ravel() + a[1]
    deficit = np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(slope), float(deficit)


def run_one_G(Gi, P_warm_inner=None):
    du, uf, lo, hi = build_grid(Gi)
    h = C * np.sqrt(du)
    print(f"--- G={Gi} du={du:.4f} h={h:.4f} h/du={h/du:.2f} ---", flush=True)
    # Inner indexing
    U1, U2, U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
    T = U1 + U2 + U3
    # Initial guess (no-learning) or warm-start
    if P_warm_inner is None:
        P_full = init_no_learning_K3(uf, tv, gv, wv)
        P_inner0 = P_full[lo:hi, lo:hi, lo:hi].copy()
    else:
        # interpolate from coarser grid to current G (simple resample by indexing)
        Gj = P_warm_inner.shape[0]
        idx = np.linspace(0, Gj-1, Gi).astype(int)
        P_inner0 = P_warm_inner[idx][:, idx][:, :, idx].copy()
    # Build full P with halo
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
        P_sol = x.reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(x))))
        ok = Fn < 1e-8
    except NoConvergence as e:
        x = e.args[0]
        P_sol = x.reshape(Gi, Gi, Gi)
        Fn = float(np.max(np.abs(F(x))))
        ok = False
    except Exception as e:
        print(f"  error: {e}", flush=True)
        return None
    wall = time.time() - t0
    slope, deficit = metrics(P_sol, T)
    print(f"  Fn={Fn:.3e} slope={slope:.4f} deficit={deficit:.4f} ({wall:.0f}s, ok={ok})", flush=True)
    np.save(f"{OUT}/P_inner_G{Gi}.npy", P_sol)
    return dict(G=Gi, du=du, h=h, F=Fn, slope=slope, deficit=deficit,
                wall=wall, ok=ok, P_inner=P_sol)


def main():
    print(f"Coarea-limit reproduction: tau={TAU}, gamma={GAMMA}, h=C*du^0.5 (C={C})")
    print(f"Target: deficit converges to ~0.282-0.291 across G={G_LIST}\n", flush=True)
    results = []
    P_warm = None
    for G in G_LIST:
        r = run_one_G(G, P_warm)
        if r is None: continue
        results.append({k: v for k, v in r.items() if k != 'P_inner'})
        P_warm = r['P_inner']
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print("\nSummary:")
    print(f"{'G':>3} {'F':>10} {'slope':>8} {'deficit':>8}")
    for r in results:
        print(f"{r['G']:>3} {r['F']:>10.2e} {r['slope']:>8.4f} {r['deficit']:>8.4f}")


if __name__ == "__main__":
    main()
