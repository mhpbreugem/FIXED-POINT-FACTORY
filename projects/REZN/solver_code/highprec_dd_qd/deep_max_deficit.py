"""Deep G-ladder at the max-deficit corner (tau=2, gamma=0.01) to give
an honest continuum bar on the certified G=21 max deficit 0.2871.
Extends G in {25, 29, 33} warm-started from the certified G=21 solution,
runs Newton-Krylov on the kernel-smoothed operator, computes deficit, and
fits a continuum extrapolation."""
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

K = 3; pad = 2; UMAX = 4.0; C = 0.45
LOWTAU = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/deep_max_deficit'
os.makedirs(OUT, exist_ok=True)


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
    slope = a[0]
    pr = a[0]*T.ravel() + a[1]
    deficit = np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30)
    return float(slope), float(deficit)


def solve_one(Gi, P_warm_inner, tau, gamma, maxiter=40, f_tol=1e-11):
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
    P21 = np.load(f"{LOWTAU}/P_ld_t{tau}_g{gamma}.npy")
    print(f"Loaded G=21 fixed point at (tau={tau}, gamma={gamma}); shape {P21.shape}",
          flush=True)
    results = []
    # Include G=21 as reference
    du21, uf21, lo21, hi21 = build_grid(21)
    U1, U2, U3 = np.meshgrid(uf21[lo21:hi21], uf21[lo21:hi21], uf21[lo21:hi21], indexing='ij')
    T21 = U1 + U2 + U3
    s21, d21 = metrics(P21, T21)
    h21 = C*np.sqrt(du21)
    results.append(dict(G=21, h=h21, F=None, slope=s21, deficit=d21,
                         wall=None, src='certified G=21'))
    print(f"  G=21 (certified): deficit={d21:.6f} slope={s21:.6f} h={h21:.4f}",
          flush=True)
    P_warm = P21
    for Gi in [25, 29, 33]:
        Pn, Fn, s, d, wall = solve_one(Gi, P_warm, tau, gamma)
        h_ = C*np.sqrt(2*UMAX/(Gi-1))
        nailed = Fn < 1e-8
        results.append(dict(G=Gi, h=h_, F=Fn, slope=s, deficit=d, wall=wall,
                            src=f'deep-ladder warm from G={P_warm.shape[0]}'))
        print(f"  G={Gi}: F={Fn:.3e} deficit={d:.6f} slope={s:.6f} h={h_:.4f} "
              f"({wall:.0f}s) {'NAILED' if nailed else 'STALL'}",
              flush=True)
        np.save(f"{OUT}/P_t{tau}_g{gamma}_G{Gi}.npy", Pn)
        if nailed: P_warm = Pn
    # Continuum extrapolation: only nailed rungs (incl G=21)
    nailed_rows = [r for r in results if r['F'] is None or r['F'] < 1e-8]
    h_arr = np.array([r['h'] for r in nailed_rows])
    d_arr = np.array([r['deficit'] for r in nailed_rows])
    s_arr = np.array([r['slope'] for r in nailed_rows])
    # 3 fits
    fits = {}
    # q-free
    from scipy.optimize import curve_fit
    def model(h, m, a, q): return m + a*h**q
    try:
        p, _ = curve_fit(model, h_arr, d_arr, p0=[d_arr[-1] - 0.05, 0.05, 1.5],
                         maxfev=5000, bounds=([0, -2, 0.3], [1, 2, 5]))
        fits['q_free'] = dict(d_inf=float(p[0]), a=float(p[1]), q=float(p[2]))
    except Exception as e:
        fits['q_free'] = dict(error=str(e))
    a1, b1 = np.polyfit(h_arr, d_arr, 1); fits['q1'] = dict(d_inf=float(b1), a=float(a1))
    a2, b2 = np.polyfit(h_arr**2, d_arr, 1); fits['q2'] = dict(d_inf=float(b2), a=float(a2))
    d_inf_vals = [v.get('d_inf') for v in fits.values() if 'd_inf' in v]
    d_center = fits['q_free'].get('d_inf', fits['q1']['d_inf'])
    err = max(abs(d_center - x) for x in d_inf_vals) if len(d_inf_vals) > 1 else 0.0
    print(f"\nContinuum extrapolation:", flush=True)
    print(f"  q-free fit: d_inf = {fits['q_free'].get('d_inf', 'NA')}", flush=True)
    print(f"  q=1 fit:    d_inf = {fits['q1']['d_inf']:.4f}", flush=True)
    print(f"  q=2 fit:    d_inf = {fits['q2']['d_inf']:.4f}", flush=True)
    print(f"\nCENTRAL ESTIMATE: d_inf = {d_center:.4f} +/- {err:.4f}", flush=True)
    json.dump(dict(tau=tau, gamma=gamma, rungs=results, fits=fits,
                    d_inf=d_center, d_err=err),
              open(f"{OUT}/results.json", 'w'), indent=2, default=str)
    print(f"\nsaved {OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
