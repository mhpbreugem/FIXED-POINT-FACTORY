"""Extend h-ladder to G=37, 41, 45 at (tau=2, gamma=0.01) and rebuild the
extrapolation with 6-7 rungs for a much tighter continuum bar."""
import os, sys, json, time, warnings
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov, curve_fit
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass
warnings.filterwarnings('ignore')

K = 3; pad = 2; UMAX = 4.0; C = 0.45
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/deep_max_deficit'


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
    Pc = np.clip(np.asarray(Pin, np.float64), 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0]*T.ravel() + a[1]
    deficit = np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30)
    return float(a[0]), float(deficit)


def solve_one(Gi, P_warm_inner, tau, gamma, maxiter=50, f_tol=1e-11):
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    if P_warm_inner is not None:
        P_full[lo:hi, lo:hi, lo:hi] = resample(P_warm_inner, Gi)
    Pinner = P_full[lo:hi, lo:hi, lo:hi].copy()
    U1, U2, U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
    T = U1 + U2 + U3
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - x.reshape(Gi, Gi, Gi)).ravel()
    t0 = time.time()
    try:
        x = newton_krylov(F, Pinner.ravel(), f_tol=f_tol, maxiter=maxiter, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    Pn = x.reshape(Gi, Gi, Gi)
    Fn = float(np.max(np.abs(F(x))))
    wall = time.time() - t0
    s, d = metrics(Pn, T)
    return Pn, Fn, s, d, wall


def main():
    tau, gamma = 2.0, 0.01
    # Start from the deepest existing solution
    P33 = np.load(f"{OUT}/P_t{tau}_g{gamma}_G33.npy")
    print(f"Resuming from G=33; extending to G in {{37, 41, 45}}", flush=True)
    # Load existing results
    d = json.load(open(f"{OUT}/results.json"))
    rungs = d['rungs']
    P_warm = P33
    for Gi in [37, 41, 45]:
        Pn, Fn, s, dval, wall = solve_one(Gi, P_warm, tau, gamma)
        h_ = C*np.sqrt(2*UMAX/(Gi-1))
        nailed = Fn < 1e-8
        rungs.append(dict(G=Gi, h=h_, F=Fn, slope=s, deficit=dval, wall=wall,
                          src=f'extended warm from G={P_warm.shape[0]}'))
        print(f"  G={Gi}: F={Fn:.3e}  deficit={dval:.6f}  slope={s:.6f}  h={h_:.4f}  "
              f"({wall:.0f}s)  {'NAILED' if nailed else 'STALL'}", flush=True)
        np.save(f"{OUT}/P_t{tau}_g{gamma}_G{Gi}.npy", Pn)
        if nailed:
            P_warm = Pn
        else:
            print(f"  Stop deepening: G={Gi} did not nail.", flush=True)
            break
    # Final extrapolation over ALL nailed rungs
    nailed_rows = [r for r in rungs if r.get('F') is None or r['F'] < 1e-8]
    h_arr = np.array([r['h'] for r in nailed_rows])
    d_arr = np.array([r['deficit'] for r in nailed_rows])
    print(f"\n{len(nailed_rows)} nailed rungs; deficits: {d_arr}", flush=True)
    a1, b1 = np.polyfit(h_arr, d_arr, 1)
    a2, b2 = np.polyfit(h_arr**2, d_arr, 1)
    def model(h, m, a, q): return m + a*h**q
    try:
        p, pcov = curve_fit(model, h_arr, d_arr, p0=[0.27, 0.05, 1.5], maxfev=5000,
                            bounds=([0.1, 0.001, 0.5], [0.5, 1, 4]))
        qfree_inf = float(p[0]); qfree_q = float(p[2])
        qfree_unc = float(np.sqrt(pcov[0,0])) if pcov is not None else 0.0
    except Exception as e:
        qfree_inf = float('nan'); qfree_q = float('nan'); qfree_unc = 0.0
    print(f"q=1 fit:    d_inf = {b1:.4f}", flush=True)
    print(f"q=2 fit:    d_inf = {b2:.4f}", flush=True)
    print(f"q-free fit: d_inf = {qfree_inf:.4f}  (q={qfree_q:.3f}, unc {qfree_unc:.4f})",
          flush=True)
    # Honest central + spread
    candidates = [b1, b2, qfree_inf]
    candidates = [c for c in candidates if 0.1 < c < 0.5]
    central = b1
    err = max(abs(central - c) for c in candidates) if candidates else 0.005
    print(f"\nCENTRAL: d_inf = {central:.4f} +/- {err:.4f}", flush=True)
    d['rungs'] = rungs
    d['fits_extended'] = dict(q1=dict(d_inf=float(b1)),
                              q2=dict(d_inf=float(b2)),
                              qfree=dict(d_inf=qfree_inf, q=qfree_q, unc=qfree_unc))
    d['d_inf_extended'] = float(central)
    d['d_err_extended'] = float(err)
    json.dump(d, open(f"{OUT}/results.json", 'w'), indent=2, default=str)
    print(f"\nsaved {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
