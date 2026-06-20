"""emin15 certification pass over the 20x5 sweep.

Policy: a cell is ACCEPTED only if max|Phi(P)-P| <= 1e-15 under the
longdouble (80-bit) operator. The float64 solver floors at ~4e-15
(bisection break + rounding), so every cell gets:

  1. float64 G-ladder {9,13,17,21} with gamma-continuation (as in the
     20x5 sweep) -> warm start.
  2. longdouble Newton-GMRES polish at G=21 (finite-difference matvec,
     step 3e-10; GMRES tol 1e-3, restart 40; up to 6 Newton steps).
  3. Accept iff F_ld <= 1e-15.

Stalled cells (float64 F > 1e-8) additionally get a tau-continuation
rescue: warm-start from the highest already-solved lower-tau row at the
same gamma, marching tau in 0.1 steps at G=21. If the march lands with
F < 1e-8 the cell rejoins step 2.

h stays on the strict joint-limit schedule h = 0.45*sqrt(du) -> 0.
"""
import os, sys, json, time, warnings
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
sys.path.insert(0, '/tmp')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from ld_ops import phi_ld, init_no_learning_ld, LD
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    try: from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
os.makedirs(OUT, exist_ok=True)

K = 3; pad = 2; UMAX = 4.0; C = 0.45
G_LIST = [9, 13, 17, 21]
TAUS = [0.2, 0.5, 1.0, 1.5, 2.0]
GAMMAS = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 20), 4))
ACCEPT = 1e-15


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


def solve64(tau, gamma, Gi, P_warm_inner, maxiter=30):
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    if P_warm_inner is not None:
        P_full[lo:hi, lo:hi, lo:hi] = resample(P_warm_inner, Gi)
    Pinner = P_full[lo:hi, lo:hi, lo:hi].copy()
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - x.reshape(Gi, Gi, Gi)).ravel()
    try:
        x = newton_krylov(F, Pinner.ravel(), f_tol=1e-12, maxiter=maxiter, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    except Exception:
        return None, np.inf
    return x.reshape(Gi, Gi, Gi), float(np.max(np.abs(F(x))))


def chain64(tau, gamma, P_cross):
    """float64 G-ladder with optional cross-cell warm start."""
    P = P_cross
    for Gi in G_LIST:
        P, Fn = solve64(tau, gamma, Gi, P)
        if P is None: return None, np.inf
    return P, Fn


def gmres_ld(matvec, b, tol=1e-3, m=40):
    """Plain GMRES(m), longdouble, no restarts (single cycle)."""
    n = b.size
    beta = np.sqrt(np.dot(b, b))
    if beta == 0: return np.zeros_like(b)
    V = np.zeros((m+1, n), LD); V[0] = b / beta
    H = np.zeros((m+1, m), LD)
    for j in range(m):
        w = matvec(V[j])
        for i in range(j+1):
            H[i, j] = np.dot(w, V[i]); w -= H[i, j]*V[i]
        H[j+1, j] = np.sqrt(np.dot(w, w))
        if H[j+1, j] > 0: V[j+1] = w / H[j+1, j]
        # solve least squares in float64 (coefficients only; residual vector stays LD)
        Hf = np.asarray(H[:j+2, :j+1], np.float64)
        e1 = np.zeros(j+2); e1[0] = float(beta)
        y, res, *_ = np.linalg.lstsq(Hf, e1, rcond=None)
        rn = np.linalg.norm(Hf @ y - e1)
        if rn / float(beta) < tol or H[j+1, j] == 0:
            return (np.asarray(y, LD)[None, :] @ V[:j+1]).ravel()
    return (np.asarray(y, LD)[None, :] @ V[:m]).ravel()


def polish_ld(tau, gamma, P64_inner, max_newton=6, gmres_m=40):
    Gi = P64_inner.shape[0]
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    uf_ld = np.asarray(uf, LD)
    tau_ld = LD(tau); gam_ld = LD(gamma); h_ld = LD(float(h))
    P_full = init_no_learning_ld(uf_ld, tau_ld, gam_ld)
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_ld(Pf, uf_ld, lo, hi, tau_ld, gam_ld, h_ld)
        return (Pn - x.reshape(Gi, Gi, Gi)).ravel()
    x = np.asarray(P64_inner, LD).ravel()
    r = F(x); F0 = float(np.max(np.abs(r)))
    hist = [F0]
    eps_fd = LD('3e-10')
    for it in range(max_newton):
        Fn = float(np.max(np.abs(r)))
        if Fn <= 0.3e-15: break
        xn = np.sqrt(np.dot(x, x))
        def matvec(v, x=x, r=r, xn=xn):
            vn = np.sqrt(np.dot(v, v))
            if vn == 0: return np.zeros_like(v)
            s = eps_fd * max(xn, LD(1)) / vn
            return (F(x + s*v) - r) / s     # J v where F = Phi - I residual
        dx = gmres_ld(lambda v: -matvec(v), r, tol=1e-3, m=gmres_m)
        x = x + dx
        r = F(x)
        hist.append(float(np.max(np.abs(r))))
        if hist[-1] > 0.9*hist[-2]:  # stagnation
            break
    return x.reshape(Gi, Gi, Gi), hist


def main():
    t_start = time.time()
    print(f"emin15 pass: accept iff F_ld <= {ACCEPT:.0e}", flush=True)
    print(f"{'tau':>5} {'gamma':>8} {'F64':>9} {'Fld0':>9} {'Fld':>10} "
          f"{'deficit':>8} {'verdict':>8} {'wall':>6}", flush=True)
    results = {}
    P21 = {}            # (tau, gamma) -> float64 G=21 solution
    for tau in TAUS:
        P_cross = None
        for gamma in GAMMAS:
            t0 = time.time()
            key = f"t{tau}_g{gamma}"
            P64, F64 = chain64(tau, gamma, P_cross)
            if P64 is not None and F64 < 1e-8:
                P_cross = P64
            rescued = False
            if P64 is None or F64 > 1e-8:
                # tau-continuation rescue from lower-tau row, same gamma
                src = None
                for tlow in sorted([t for t in TAUS if t < tau], reverse=True):
                    if (tlow, gamma) in P21 and results[f"t{tlow}_g{gamma}"]['F64'] < 1e-8:
                        src = tlow; break
                if src is not None:
                    P = P21[(src, gamma)]
                    ok = True
                    for tmid in np.arange(src + 0.1, tau + 1e-9, 0.1):
                        P, Fm = solve64(round(float(tmid), 2), gamma, 21, P, maxiter=30)
                        if P is None or Fm > 1e-6:
                            ok = False; break
                    if ok and P is not None:
                        P2, F2 = solve64(tau, gamma, 21, P, maxiter=40)
                        if P2 is not None and F2 < min(F64, 1e-8):
                            P64, F64, rescued = P2, F2, True
            if P64 is None:
                results[key] = dict(tau=tau, gamma=gamma, F64=np.inf, verdict='REJECT')
                print(f"{tau:>5.2f} {gamma:>8.4f} {'err':>9} {'-':>9} {'-':>10} "
                      f"{'-':>8} {'REJECT':>8}", flush=True)
                continue
            P21[(tau, gamma)] = P64
            du, uf, lo, hi = build_grid(21)
            U1,U2,U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
            T = U1+U2+U3
            if F64 < 1e-8:
                P_ld, hist = polish_ld(tau, gamma, P64)
                F_ld = hist[-1]
                s, d = metrics(P_ld, T)
                verdict = 'ACCEPT' if F_ld <= ACCEPT else 'REJECT'
                if verdict == 'ACCEPT':
                    np.save(f"{OUT}/P_ld_t{tau}_g{gamma}.npy",
                            np.asarray(P_ld, np.float64))
            else:
                F_ld = None; hist = []
                s, d = metrics(P64, T)
                verdict = 'REJECT'
            wall = time.time() - t0
            results[key] = dict(tau=tau, gamma=gamma, F64=F64,
                                F_ld_hist=hist, F_ld=F_ld, slope=s, deficit=d,
                                rescued=rescued, verdict=verdict, wall=wall)
            print(f"{tau:>5.2f} {gamma:>8.4f} {F64:>9.2e} "
                  f"{(hist[0] if hist else float('nan')):>9.2e} "
                  f"{(F_ld if F_ld is not None else float('nan')):>10.2e} "
                  f"{d:>8.4f} {verdict:>8} {wall:>5.0f}s"
                  + (" [rescued]" if rescued else ""), flush=True)
            json.dump(results, open(f"{OUT}/emin15.json", "w"), indent=2, default=str)
    n_acc = sum(1 for v in results.values() if v.get('verdict') == 'ACCEPT')
    print(f"\n{n_acc}/{len(results)} ACCEPTED at F_ld <= 1e-15; "
          f"total {(time.time()-t_start)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
