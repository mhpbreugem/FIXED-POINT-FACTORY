"""K=3 CARA fixed-point operator on the BOUNDED xi-grid.

Why this exists: on the u-grid (|u| up to UMAX) the analytic FR price
p* = sigma(tau*Sigma u) is clip-saturated at the corners whenever
3*tau*UMAX > log(1/eps_clip). That's exactly the artifact the UMAX-collapse
diagnostic exposed. The CLEAN fix is the bounded xi-grid:

    xi = tanh(tau*u/2)  in  (-1, 1)
    u(xi) = (2/tau) * atanh(xi)
    du/dxi = (2/tau) / (1 - xi^2)

With xi_max ~ 0.95, u_max = (1/tau)*atanh(0.95) ~ 1.83/tau and
3*tau*u_max ~ 5.5 << log(1e9) ~ 20.7, so the FR price never gets clipped.

This is a self-contained K=3 CARA operator (no numba, no shared imports
from contour_K3_halo) using kernel-smoothed co-area in xi-coordinates. The
signal density in xi-space is g_v(xi) = f_v(u(xi)) * du/dxi: this absorbs
the Jacobian so the uniform-xi Riemann sum is correct.

Market clearing: CARA log-odds, pi = mean_i(logit(mu_i)), p = sigmoid(pi).

Verification protocol: nail by Picard from FR IC + FR halo, check deficit
on the inner block (1-R^2 of logit(p) vs tau*Sigma u). Expected: deficit
~ machine zero for all G, because no clip activates and the analytic FR
is now self-consistent on the grid.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np

EPSB = 0.05  # xi-grid extends to +-(1-EPSB) = +-0.95
EPS_PRICE = 1e-9  # logit clip on mu (not on p)
TAU = 2.0; K = 3

def f_signal(u, v, tau):
    vm = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2*np.pi)) * np.exp(-tau * (u - vm)**2 / 2)

def logit(p): return np.log(p / (1 - p))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def cara_xi_operator(P_full, g0, g1, g0_own, g1_own, lo, hi, h_band):
    """One pass of Phi_CARA on the xi-grid (kernel co-area inference + CARA log-odds clear).

    g0[i] = f_0(u(xi_i)) * du/dxi(xi_i)   (other-agent xi-density, v=0)
    g1[i] = f_1(u(xi_i)) * du/dxi(xi_i)   (other-agent xi-density, v=1)
    g0_own/g1_own: same densities, used for the own-signal Bayes (kept
    separate for clarity; in equal-tau-W case they are the same arrays).
    """
    G_full = P_full.shape[0]
    P_new = P_full.copy()
    inv_2h2 = 0.5 / (h_band * h_band)
    # Precompute outer products g_v(xi_a)*g_v(xi_b) for the 2D slice
    g0_outer = np.outer(g0, g0); g1_outer = np.outer(g1, g1)
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                # Agent 0: slice P[i, :, :], indexed (axis_j, axis_l)
                slc = P_full[i, :, :]
                W = np.exp(-(slc - p)**2 * inv_2h2)
                A0 = (W * g0_outer).sum(); A1 = (W * g1_outer).sum()
                num = g1_own[i] * A1; den = g0_own[i] * A0 + num
                mu0 = num / max(den, 1e-30)
                # Agent 1: slice P[:, j, :], indexed (axis_i, axis_l)
                slc = P_full[:, j, :]
                W = np.exp(-(slc - p)**2 * inv_2h2)
                A0 = (W * g0_outer).sum(); A1 = (W * g1_outer).sum()
                num = g1_own[j] * A1; den = g0_own[j] * A0 + num
                mu1 = num / max(den, 1e-30)
                # Agent 2: slice P[:, :, l], indexed (axis_i, axis_j)
                slc = P_full[:, :, l]
                W = np.exp(-(slc - p)**2 * inv_2h2)
                A0 = (W * g0_outer).sum(); A1 = (W * g1_outer).sum()
                num = g1_own[l] * A1; den = g0_own[l] * A0 + num
                mu2 = num / max(den, 1e-30)
                # Clip mu (not p)
                mu_vec = np.clip(np.array([mu0, mu1, mu2]), EPS_PRICE, 1 - EPS_PRICE)
                m_bar = np.mean(np.log(mu_vec / (1 - mu_vec)))
                P_new[i, j, l] = np.clip(sigmoid(m_bar), EPS_PRICE, 1 - EPS_PRICE)
    return P_new

def nail_xi(G_inner, pad=2, C_H=0.45, max_iter=400, tol=1e-12):
    """Nail the K=3 CARA xi-grid fixed point starting from analytic FR."""
    G_full = G_inner + 2*pad
    # xi_grid: uniform over [-(1-EPSB), +(1-EPSB)] including halo on both sides
    dxi = 2*(1 - EPSB) / (G_inner - 1)
    xi_full = np.array([-(1 - EPSB) + (q - pad) * dxi for q in range(G_full)])
    # Halo cells outside [-1+EPSB, 1-EPSB] are kept but their xi values
    # might exceed 1 if pad*dxi > EPSB. Clip to safety:
    xi_full = np.clip(xi_full, -0.999, 0.999)
    u_full = (2.0 / TAU) * np.arctanh(xi_full)
    dudxi = (2.0 / TAU) / (1.0 - xi_full**2)
    g0 = f_signal(u_full, 0, TAU) * dudxi
    g1 = f_signal(u_full, 1, TAU) * dudxi
    # kernel band: same heuristic as u-grid op (C * sqrt(dxi))
    h_band = C_H * np.sqrt(dxi)
    # Analytic FR over the full padded grid
    U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')
    P_FR = sigmoid(TAU * (U1 + U2 + U3))
    P_FR = np.clip(P_FR, EPS_PRICE, 1 - EPS_PRICE)
    halo = P_FR.copy()
    Pf = P_FR.copy()
    lo, hi = pad, pad + G_inner
    slc = (slice(lo, hi),)*3
    Finf = np.inf
    for it in range(max_iter):
        Pn = cara_xi_operator(Pf, g0, g1, g0, g1, lo, hi, h_band)
        Pn_h = halo.copy(); Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h
        if Finf < tol:
            break
    # deficit on inner block
    ui = u_full[lo:hi]
    Ui1, Ui2, Ui3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = TAU * (Ui1 + Ui2 + Ui3)
    Pin = Pf[slc]
    y = np.log(np.clip(Pin, 1e-12, 1-1e-12) / (1 - np.clip(Pin, 1e-12, 1-1e-12))).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel() + aa[1]
    defi = float(np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30))
    d_FR = float(np.sqrt(np.mean((Pin - sigmoid(T))**2)))
    return dict(G=G_inner, dxi=float(dxi), h_band=float(h_band),
                xi_max=float(xi_full[lo:hi].max()),
                u_max=float(u_full[lo:hi].max()),
                deficit=defi, d_FR=d_FR, slope=float(aa[0]),
                Finf=Finf, iters=it+1)

def nail_xi_at(G_inner, epsb, pad=2, C_H=0.45, max_iter=400, tol=1e-12):
    """As nail_xi but with adjustable xi-bound = 1 - epsb."""
    global EPSB
    epsb_saved = EPSB
    EPSB = epsb
    try:
        out = nail_xi(G_inner, pad=pad, C_H=C_H, max_iter=max_iter, tol=tol)
    finally:
        EPSB = epsb_saved
    out['epsb'] = epsb
    return out

if __name__ == "__main__":
    print(f"K=3 CARA on BOUNDED xi-grid, tau={TAU}.", flush=True)
    print("Two protocols: (A) wide xi (saturated prices, kernel band too wide -> bias)")
    print("              (B) narrow xi (off saturation -> kernel localizes -> FR)\n", flush=True)
    rows = []

    print("(A) xi in [-0.95, +0.95]  (u_max=1.83, tau*Sigma u_max=11, sigma'~2e-5 -> saturated)", flush=True)
    for G in [7, 9, 11, 13, 15]:
        t = time.time(); r = nail_xi_at(G, 0.05); dt = time.time() - t
        r['sec'] = round(dt, 1); r['protocol'] = 'A_wide'
        rows.append(r)
        print(f"  G={G:2d}  dxi={r['dxi']:.3f}  h={r['h_band']:.3f}  u_max={r['u_max']:.2f}: "
              f"deficit={r['deficit']:.3e}  d_FR={r['d_FR']:.1e}  slope={r['slope']:.3f}  "
              f"||F||={r['Finf']:.1e}", flush=True)

    print("\n(B) xi in [-0.7, +0.7]  (u_max~0.87, tau*Sigma u_max~5.2, sigma'~0.007 -> off-saturation)", flush=True)
    for G in [7, 9, 11, 13, 15]:
        t = time.time(); r = nail_xi_at(G, 0.30); dt = time.time() - t
        r['sec'] = round(dt, 1); r['protocol'] = 'B_narrow'
        rows.append(r)
        print(f"  G={G:2d}  dxi={r['dxi']:.3f}  h={r['h_band']:.3f}  u_max={r['u_max']:.2f}: "
              f"deficit={r['deficit']:.3e}  d_FR={r['d_FR']:.1e}  slope={r['slope']:.3f}  "
              f"||F||={r['Finf']:.1e}", flush=True)

    print("\n(C) xi in [-0.5, +0.5]  (u_max~0.55, tau*Sigma u_max~3.3, sigma'~0.035 -> fully off-saturation)", flush=True)
    for G in [7, 9, 11, 13, 15]:
        t = time.time(); r = nail_xi_at(G, 0.50); dt = time.time() - t
        r['sec'] = round(dt, 1); r['protocol'] = 'C_very_narrow'
        rows.append(r)
        print(f"  G={G:2d}  dxi={r['dxi']:.3f}  h={r['h_band']:.3f}  u_max={r['u_max']:.2f}: "
              f"deficit={r['deficit']:.3e}  d_FR={r['d_FR']:.1e}  slope={r['slope']:.3f}  "
              f"||F||={r['Finf']:.1e}", flush=True)

    json.dump({'tau': TAU, 'rows': rows,
               'theory': 'CARA on bounded xi-grid: deficit small only when xi-bound keeps prices off saturation.'},
              open(os.path.join(HERE, 'cara_xi_grid.json'), 'w'), indent=2)
    print('\nDONE.', flush=True)
