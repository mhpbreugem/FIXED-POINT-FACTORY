"""FASTER float64 h-free solver: operator computes ONLY symmetric-representative
cells (i<=j<=l), not all G^3 -> ~4-5x fewer slice_evidence calls, same fixed point.
Lives in the repo (NOT /tmp, which gets wiped). Imports the repo hfree_operator.py."""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
from itertools import combinations_with_replacement
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hfree_operator as H
from numba import njit, prange
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

slice_evidence = H.slice_evidence
bayes = H.bayes
clear_crra = H.clear_crra


@njit(cache=True, parallel=True, fastmath=False)
def phi_rep(P, ui, gn, gw, tau, gam, W, sub, ri, rj, rl):
    n = ri.size; h = ui[1] - ui[0]; u0 = ui[0]; G = ui.size
    out = np.empty(n)
    for c in prange(n):
        i = ri[c]; j = rj[c]; l = rl[c]; p = P[i, j, l]
        Mc = np.empty((G, G)); Mr = np.empty((G, G))
        a0, a1 = slice_evidence(P[i, :, :], h, u0, p, gn, gw, tau[1], tau[2], sub, Mc, Mr); mu0 = bayes(ui[i], tau[0], a0, a1)
        b0, b1 = slice_evidence(P[:, j, :], h, u0, p, gn, gw, tau[0], tau[2], sub, Mc, Mr); mu1 = bayes(ui[j], tau[1], b0, b1)
        c0, c1 = slice_evidence(P[:, :, l], h, u0, p, gn, gw, tau[0], tau[1], sub, Mc, Mr); mu2 = bayes(ui[l], tau[2], c0, c1)
        out[c] = clear_crra(mu0, mu1, mu2, gam[0], gam[1], gam[2], W[0], W[1], W[2])
    return out


class Red:
    def __init__(self, G):
        self.G = G; self.ms = list(combinations_with_replacement(range(G), 3)); self.n = len(self.ms)
        self.ri = np.array([m[0] for m in self.ms]); self.rj = np.array([m[1] for m in self.ms]); self.rl = np.array([m[2] for m in self.ms])
        idx = {m: r for r, m in enumerate(self.ms)}; self.roc = np.empty((G, G, G), np.int64)
        for i in range(G):
            for j in range(G):
                for l in range(G):
                    self.roc[i, j, l] = idx[tuple(sorted((i, j, l)))]

    def expand(self, x):
        return x[self.roc]


UMAX = 4.0; NQ = 40; SUB = 4


def metrics(P, ui, TAU):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1 - 1e-12); y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1); pr = a[0] * T.ravel() + a[1]
    return dict(deficit=float(np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)),
                d_FR=float(np.sqrt(np.mean((P - 1 / (1 + np.exp(-T))) ** 2))), slope=float(a[0]))


def solve(G, TAU, GAM, P0_full=None, f_tol=1e-9):
    ui = np.linspace(-UMAX, UMAX, G); gn, gw = H.gauss_legendre(NQ, -UMAX, UMAX)
    tau = np.full(3, TAU); gam = np.full(3, GAM); W = np.full(3, 1.0); R = Red(G)
    if P0_full is None:
        U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU * (U1 + U2 + U3); P0_full = np.clip(1 / (1 + np.exp(-T)), 1e-9, 1 - 1e-9)
    x0 = np.array([P0_full[m] for m in R.ms]); cnt = {'n': 0}

    def F(x):
        cnt['n'] += 1; P = R.expand(x); return phi_rep(P, ui, gn, gw, tau, gam, W, SUB, R.ri, R.rj, R.rl) - x
    conv = True
    try:
        sol = newton_krylov(F, x0, f_tol=f_tol, maxiter=120, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F(sol)))); P = R.expand(sol); m = metrics(P, ui, TAU)
    return P, ui, Finf, cnt['n'], conv, m, R.n


if __name__ == '__main__':
    OUT = os.path.join(os.path.dirname(HERE), 'k3_hfree_ginf'); os.makedirs(OUT, exist_ok=True)
    t = time.time(); P, ui, Finf, it, conv, m, nred = solve(9, 2.0, 0.1); dt = time.time() - t
    print(f"[FAST-REP] G=9: deficit={m['deficit']:.4f} ||F||={Finf:.1e} iters={it} n_rep={nred} time={dt:.1f}s (old full-grid solver ~21s)", flush=True)
    from scipy.interpolate import RegularGridInterpolator
    rows = []; prevP = None; prevui = None
    print("G->inf (FAST reduced-output operator), tau=2 gamma=0.1:", flush=True)
    for G in [9, 13, 17, 21, 25, 29]:
        t = time.time(); P0 = None
        if prevP is not None:
            itp = RegularGridInterpolator((prevui,) * 3, prevP, bounds_error=False, fill_value=None)
            ug = np.linspace(-UMAX, UMAX, G); X1, X2, X3 = np.meshgrid(ug, ug, ug, indexing='ij')
            P0 = np.clip(itp(np.stack([X1.ravel(), X2.ravel(), X3.ravel()], 1)).reshape(G, G, G), 1e-9, 1 - 1e-9)
        try:
            P, ui, Finf, it, conv, m, nred = solve(G, 2.0, 0.1, P0); prevP, prevui = P, ui
            r = dict(G=G, deficit=m['deficit'], d_FR=m['d_FR'], Finf=Finf, iters=it, conv=conv, sec=round(time.time() - t, 1), n_rep=nred)
            rows.append(r); print(f"  G={G:2d}: deficit={r['deficit']:.4f} d_FR={r['d_FR']:.4f} ||F||={Finf:.1e} it={it} conv={conv} ({r['sec']}s)", flush=True)
            json.dump({'rows': rows, 'operator': 'fast reduced-output float64'}, open(os.path.join(OUT, 'ginf_fast.json'), 'w'), indent=2)
        except Exception as e:
            print(f"  G={G}: ERR {e}", flush=True)
    if len(rows) >= 3:
        Gs = np.array([r['G'] for r in rows]); D = np.array([r['deficit'] for r in rows])
        for p, lab in [(1, '1/G'), (2, '1/G^2')]:
            A = np.column_stack([np.ones_like(Gs, float), 1.0 / Gs ** p]); c, *_ = np.linalg.lstsq(A, D, rcond=None)
            print(f"  extrap deficit(G->inf)[{lab}]={c[0]:.4f}  (kernel limit ~0.27)", flush=True); rows.append({f'extrap_{lab}': float(c[0])})
        json.dump({'rows': rows}, open(os.path.join(OUT, 'ginf_fast.json'), 'w'), indent=2)
    print("DONE", flush=True)
