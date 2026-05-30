"""PROPER K=3 CARA on the bounded xi-grid:
explicit co-area contour integral via PCHIP root-find. NO kernel smoothing.

Architecture (analogous to production DD K=2 solver, ported to K=3):
  - xi-grid uniform on [-(1-EPSB), +(1-EPSB)] (padded for halo)
  - For each agent k and each price target p, the contour {P=p} in the
    OFF-axis 2D slice is a 1D curve. We parametrize it by xi_a (one
    off-axis var) and root-find xi_b(xi_a) using PCHIP interpolation of
    P along the b-axis. The co-area integral is then
        A_v(p, u_own) = sum_{a} f_v(u_a) f_v(u_b(xi_a)) / |dP/du_b| du_a
    with du_a = du/dxi(xi_a) * dxi.
  - Averaged over symmetric passes (vary xi_a, root-find xi_b) and
    (vary xi_b, root-find xi_a) to handle both vertical and horizontal
    tangents.
  - Bayes: mu_k = f_1(u_own) A_1 / (f_0(u_own) A_0 + f_1(u_own) A_1)
  - CARA clear: pi = mean_k logit(mu_k), p_new = sigmoid(pi).

NO kernel band -- the contour is found EXACTLY (up to PCHIP error), so
there is no bandwidth-vs-saturation tradeoff. CARA's analytic FR
p* = sigma(tau Sigma u) should be a discrete fixed point to PCHIP order.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.interpolate import PchipInterpolator

EPSB = 0.10  # xi-grid in [-(1-EPSB), +(1-EPSB)] = [-0.9, +0.9]
EPS_PRICE = 1e-9
TAU = 2.0

def f_signal(u, v, tau):
    vm = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2*np.pi)) * np.exp(-tau * (u - vm)**2 / 2)

def logit(p): return np.log(p / (1 - p))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def contour_pchip(xi, vals_minus_p):
    """All roots of PCHIP(xi -> vals_minus_p) = 0, with their slopes (dP/dxi)."""
    try:
        interp = PchipInterpolator(xi, vals_minus_p, extrapolate=False)
        roots = interp.roots(discontinuity=False, extrapolate=False)
    except Exception:
        return []
    if len(roots) == 0:
        return []
    deriv = interp.derivative()
    return [(float(r), float(deriv(r))) for r in roots if not np.isnan(r)]

def integrate_contour(P_slice, xi_full, u_full, dudxi, p_target, tau, v):
    """A_v(p) = sum over contour points of f_v(u_a) f_v(u_b) / |dP/du_b| * (xi_a weight) * dudxi_a.
    Averaged over the two passes (vary axis 0, root-find axis 1) and (swap).
    P_slice has shape (G, G) with axes (a, b)."""
    G = xi_full.size
    dxi = xi_full[1] - xi_full[0]
    A_total = 0.0
    n_total = 0
    f_v = lambda u: f_signal(u, v, tau)
    # PASS 0: iterate over axis 0 (index ia), root-find xi on axis 1
    for ia in range(G):
        slc_1d = P_slice[ia, :] - p_target
        u_a = u_full[ia]
        for xi_b, dPdxi_b in contour_pchip(xi_full, slc_1d):
            if abs(dPdxi_b) < 1e-30:
                continue
            u_b = (2.0 / tau) * np.arctanh(xi_b)
            dPdu_b = dPdxi_b * (1.0 - xi_b * xi_b) * tau / 2.0
            wt = f_v(u_a) * f_v(u_b) * dudxi[ia] * dxi / abs(dPdu_b)
            A_total += wt
            n_total += 1
    # PASS 1: iterate over axis 1, root-find xi on axis 0
    for ib in range(G):
        slc_1d = P_slice[:, ib] - p_target
        u_b = u_full[ib]
        for xi_a, dPdxi_a in contour_pchip(xi_full, slc_1d):
            if abs(dPdxi_a) < 1e-30:
                continue
            u_a = (2.0 / tau) * np.arctanh(xi_a)
            dPdu_a = dPdxi_a * (1.0 - xi_a * xi_a) * tau / 2.0
            wt = f_v(u_b) * f_v(u_a) * dudxi[ib] * dxi / abs(dPdu_a)
            A_total += wt
            n_total += 1
    return A_total / 2.0  # average two passes

def phi_cara_xi(P_full, xi_full, u_full, dudxi, lo, hi, tau):
    G_full = P_full.shape[0]
    P_new = P_full.copy()
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                # Agent 0: own u_i; contour in slice P[i, :, :]
                A0 = integrate_contour(P_full[i, :, :], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour(P_full[i, :, :], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[i], 0, tau); f1o = f_signal(u_full[i], 1, tau)
                den = f0o*A0 + f1o*A1
                mu0 = (f1o*A1) / den if den > 1e-30 else 0.5
                # Agent 1: own u_j; contour in slice P[:, j, :]
                A0 = integrate_contour(P_full[:, j, :], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour(P_full[:, j, :], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[j], 0, tau); f1o = f_signal(u_full[j], 1, tau)
                den = f0o*A0 + f1o*A1
                mu1 = (f1o*A1) / den if den > 1e-30 else 0.5
                # Agent 2: own u_l; contour in slice P[:, :, l]
                A0 = integrate_contour(P_full[:, :, l], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour(P_full[:, :, l], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[l], 0, tau); f1o = f_signal(u_full[l], 1, tau)
                den = f0o*A0 + f1o*A1
                mu2 = (f1o*A1) / den if den > 1e-30 else 0.5
                # Clip mu, CARA log-odds clear
                mus = np.clip(np.array([mu0, mu1, mu2]), EPS_PRICE, 1 - EPS_PRICE)
                m_bar = np.mean(np.log(mus / (1 - mus)))
                P_new[i, j, l] = np.clip(sigmoid(m_bar), EPS_PRICE, 1 - EPS_PRICE)
    return P_new

def nail(G_inner, epsb=EPSB, max_iter=200, tol=1e-12, verbose=False):
    # No padded halo: inner xi-grid is the FULL grid; PCHIP root-find inside.
    G_full = G_inner
    dxi = 2*(1 - epsb) / (G_inner - 1)
    xi_full = np.linspace(-(1 - epsb), +(1 - epsb), G_inner)
    u_full = (2.0 / TAU) * np.arctanh(xi_full)
    dudxi = (2.0 / TAU) / (1.0 - xi_full**2)
    # Analytic FR over the full grid (= inner block)
    U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')
    P_FR = np.clip(sigmoid(TAU * (U1 + U2 + U3)), EPS_PRICE, 1 - EPS_PRICE)
    halo = P_FR.copy()
    Pf = P_FR.copy()
    lo, hi = 0, G_inner
    slc = (slice(lo, hi),)*3
    Finf = np.inf
    for it in range(max_iter):
        Pn = phi_cara_xi(Pf, xi_full, u_full, dudxi, lo, hi, TAU)
        Pn_h = halo.copy(); Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h
        if verbose: print(f"   iter {it+1}: F={Finf:.3e}", flush=True)
        if Finf < tol:
            break
    Pin = Pf[slc]
    ui = u_full[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU*(U1+U2+U3)
    y = np.log(np.clip(Pin, 1e-12, 1-1e-12) / (1 - np.clip(Pin, 1e-12, 1-1e-12))).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel() + aa[1]
    defi = float(np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30))
    d_FR = float(np.sqrt(np.mean((Pin - sigmoid(T))**2)))
    return dict(G=G_inner, epsb=epsb, dxi=float(dxi),
                u_max=float(u_full[hi-1]), deficit=defi, d_FR=d_FR,
                slope=float(aa[0]), Finf=Finf, iters=it+1)

if __name__ == "__main__":
    print(f"K=3 CARA PROPER xi-grid (PCHIP co-area, no kernel), tau={TAU}.", flush=True)
    print("Expected: deficit ~ machine zero (CARA = analytic FR, Hellwig).", flush=True)
    rows = []
    for epsb, label in [(0.10, 'xi=[-0.90,+0.90]'), (0.30, 'xi=[-0.70,+0.70]')]:
        print(f"\n{label}", flush=True)
        for G in [5, 7, 9, 11]:
            t = time.time()
            r = nail(G, epsb=epsb)
            r['sec'] = round(time.time() - t, 1)
            rows.append(r)
            print(f"  G={G:2d}  dxi={r['dxi']:.3f}  u_max={r['u_max']:.2f}: "
                  f"deficit={r['deficit']:.3e}  d_FR={r['d_FR']:.1e}  "
                  f"slope={r['slope']:.3f}  ||F||={r['Finf']:.1e}  "
                  f"iters={r['iters']}  ({r['sec']}s)", flush=True)
            json.dump({'tau': TAU, 'rows': rows,
                       'theory': 'PROPER xi-grid co-area: CARA -> FR with deficit ~ 0 (Hellwig).'},
                      open(os.path.join(HERE, 'cara_xi_explicit.json'), 'w'), indent=2)
    print('\nDONE.', flush=True)
