"""K=3 CRRA on the BOUNDED ξ-grid -- port of k3_cara/cara_xi_explicit.py
to CRRA market clearing.

The "code in which the line smoothly changes with a change of variables, no h":
- Per-axis change of variables: ξ_k = tanh(τ·u_k/2),  u_k = (2/τ)·atanh(ξ_k)
- Jacobian du/dξ = (2/τ)/(1-ξ²) absorbed into the signal density:
    g_v(ξ) = f_v(u(ξ)) · du/dξ
- Uniform Riemann sum on the ξ-grid is correct (no per-cell weights needed).
- Contour root-find via scipy PCHIP (shape-preserving, NO overshoot, smooth
  in p) — no kernel bandwidth, h is not a parameter at all.
- Two-pass average over (vary ξ_a / root-find ξ_b) and swap.

Test: CRRA γ=0.1, τ=2 — does this h=0 + change-of-vars + PCHIP architecture
nail the u-grid PR FP?
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from scipy.interpolate import PchipInterpolator

# ----- Parameters (match k3_coarea_sweep target) -----
TAU = 2.0
GAMMA = 0.1
EPSB = 0.10               # ξ ∈ [-(1-EPSB), +(1-EPSB)]
EPS_PRICE = 1e-9

def f_signal(u, v, tau):
    vm = 0.5 if v == 1 else -0.5
    return np.sqrt(tau/(2*np.pi)) * np.exp(-tau*(u-vm)**2/2)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def logit(p): return np.log(p/(1-p))

def contour_pchip_roots(xi, vals_minus_p):
    """All roots of PCHIP(ξ → vals - p) = 0 with slopes dP/dξ."""
    try:
        interp = PchipInterpolator(xi, vals_minus_p, extrapolate=False)
        roots = interp.roots(discontinuity=False, extrapolate=False)
    except Exception:
        return []
    if len(roots) == 0:
        return []
    deriv = interp.derivative()
    return [(float(r), float(deriv(r))) for r in roots if not np.isnan(r)]

def integrate_contour_Av(P_slice, xi_full, u_full, dudxi, p_target, tau, v):
    """A_v(p) via PCHIP contour, two-pass average."""
    G = xi_full.size
    dxi = xi_full[1] - xi_full[0]
    f_v = lambda u: f_signal(u, v, tau)
    A_total = 0.0
    # PASS 0: vary ξ_a (axis 0), root-find on ξ_b (axis 1)
    for ia in range(G):
        slc = P_slice[ia, :] - p_target
        u_a = u_full[ia]
        for xi_b, dPdxi_b in contour_pchip_roots(xi_full, slc):
            if abs(dPdxi_b) < 1e-30: continue
            u_b = (2.0/tau) * np.arctanh(xi_b)
            dPdu_b = dPdxi_b * (1.0 - xi_b*xi_b) * tau / 2.0
            A_total += f_v(u_a)*f_v(u_b) * dudxi[ia]*dxi / abs(dPdu_b)
    # PASS 1: vary ξ_b, root-find on ξ_a
    for ib in range(G):
        slc = P_slice[:, ib] - p_target
        u_b = u_full[ib]
        for xi_a, dPdxi_a in contour_pchip_roots(xi_full, slc):
            if abs(dPdxi_a) < 1e-30: continue
            u_a = (2.0/tau) * np.arctanh(xi_a)
            dPdu_a = dPdxi_a * (1.0 - xi_a*xi_a) * tau / 2.0
            A_total += f_v(u_b)*f_v(u_a) * dudxi[ib]*dxi / abs(dPdu_a)
    return A_total / 2.0

def crra_clear(mu0, mu1, mu2, gamma, steps=120):
    """CRRA clearing: solve p such that sum (mu_k - p)/[p(1-p)] g_k = 0
    where g_k = 1/gamma (CARA-like log-utility), Newton iteration on logit.
    """
    p = (mu0 + mu1 + mu2) / 3.0
    p = max(EPS_PRICE, min(1 - EPS_PRICE, p))
    for _ in range(steps):
        f = ((mu0 - p) + (mu1 - p) + (mu2 - p)) / (p*(1-p)) - gamma*np.log(p/(1-p))*0
        # Actually use the same crra_clear logic as in v11: pi = mean logit(mu)
        # (CRRA reduces to CARA log-odds at gamma->0)
        break
    # CARA-like log-odds clearing as base (works for any gamma; full CRRA needs Newton)
    eps = EPS_PRICE
    mus = np.clip(np.array([mu0,mu1,mu2]), eps, 1-eps)
    pi_bar = np.mean(np.log(mus/(1-mus)))
    return sigmoid(pi_bar)  # CARA limit; for proper CRRA we'd Newton-iterate

def crra_clear_proper(mu0, mu1, mu2, gamma, steps=120):
    """Proper CRRA clearing via bisection on the logit-price (matches
    v11_strict_h0_hardwired.crra_clear_nb).

    Each agent's demand at log-price lp:  R_k = exp((lm_k - lp)/gamma)
    Demand_k(m) = (R_k - 1) / ((1-m) + R_k m)  where m = sigmoid(lp).
    Sum_k Demand_k(m) = 0 -> bisect on m in [eps, 1-eps].
    """
    eps = 1e-30
    a = eps; b = 1.0 - eps
    me0 = max(min(mu0, 1-EPS_PRICE), EPS_PRICE)
    me1 = max(min(mu1, 1-EPS_PRICE), EPS_PRICE)
    me2 = max(min(mu2, 1-EPS_PRICE), EPS_PRICE)
    lm0 = np.log(me0/(1-me0))
    lm1 = np.log(me1/(1-me1))
    lm2 = np.log(me2/(1-me2))
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = np.log(m/(1-m))
        R0 = np.exp((lm0-lp)/gamma)
        R1 = np.exp((lm1-lp)/gamma)
        R2 = np.exp((lm2-lp)/gamma)
        e = (R0-1)/((1-m)+R0*m) + (R1-1)/((1-m)+R1*m) + (R2-1)/((1-m)+R2*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)

def phi_crra_xi(P_full, xi_full, u_full, dudxi, lo, hi, tau, gamma):
    G_full = P_full.shape[0]
    P_new = P_full.copy()
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                # Agent 0: contour in slice (u2, u3) = (axis 1, axis 2)
                A0 = integrate_contour_Av(P_full[i, :, :], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour_Av(P_full[i, :, :], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[i], 0, tau); f1o = f_signal(u_full[i], 1, tau)
                den = f0o*A0 + f1o*A1
                mu0 = (f1o*A1)/den if den > 1e-30 else 0.5
                # Agent 1: contour in slice (u1, u3) = (axis 0, axis 2)
                A0 = integrate_contour_Av(P_full[:, j, :], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour_Av(P_full[:, j, :], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[j], 0, tau); f1o = f_signal(u_full[j], 1, tau)
                den = f0o*A0 + f1o*A1
                mu1 = (f1o*A1)/den if den > 1e-30 else 0.5
                # Agent 2: contour in slice (u1, u2) = (axis 0, axis 1)
                A0 = integrate_contour_Av(P_full[:, :, l], xi_full, u_full, dudxi, p, tau, 0)
                A1 = integrate_contour_Av(P_full[:, :, l], xi_full, u_full, dudxi, p, tau, 1)
                f0o = f_signal(u_full[l], 0, tau); f1o = f_signal(u_full[l], 1, tau)
                den = f0o*A0 + f1o*A1
                mu2 = (f1o*A1)/den if den > 1e-30 else 0.5
                P_new[i, j, l] = crra_clear_proper(mu0, mu1, mu2, gamma)
    return P_new

def metrics(Pin, u_full, lo, hi, tau):
    ui = u_full[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = tau*(U1 + U2 + U3)
    P_FR = sigmoid(T)
    Pc = np.clip(Pin, 1e-12, 1-1e-12)
    y = np.log(Pc/(1-Pc)).ravel()
    aa = np.polyfit(T.ravel(), y, 1)
    pred = aa[0]*T.ravel() + aa[1]
    m = y.mean()
    vt = ((y-m)**2).mean(); vr = ((y-pred)**2).mean()
    return float(aa[0]), float(vr/vt if vt>0 else float('nan')), float(np.sqrt(np.mean((Pin-P_FR)**2)))

def nail(G_inner, gamma, tau, epsb=EPSB, max_iter=200, tol=1e-9, verbose=True, ic_from=None):
    G_full = G_inner; lo, hi = 0, G_inner
    dxi = 2*(1-epsb)/(G_inner-1)
    xi_full = np.linspace(-(1-epsb), +(1-epsb), G_inner)
    u_full = (2.0/tau) * np.arctanh(xi_full)
    dudxi = (2.0/tau) / (1.0 - xi_full**2)
    # IC: no-learning ansatz (CARA log-odds of own-signal beliefs)
    U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')
    if ic_from is None:
        mu1 = sigmoid(tau*U1); mu2 = sigmoid(tau*U2); mu3 = sigmoid(tau*U3)
        # IC via proper CRRA clearing (vectorized loop)
        Pf = np.empty_like(U1)
        for ii in range(G_inner):
            for jj in range(G_inner):
                for kk in range(G_inner):
                    Pf[ii,jj,kk] = crra_clear_proper(mu1[ii,jj,kk], mu2[ii,jj,kk], mu3[ii,jj,kk], gamma)
    else:
        Pf = ic_from.copy()
    s0, om0, d0 = metrics(Pf, u_full, lo, hi, tau)
    print(f'IC: slope={s0:.4f} 1-R²={om0:.4f} d_FR={d0:.4f}', flush=True)
    Finf = np.inf; history=[]
    for it in range(max_iter):
        ts = time.time()
        Pn = phi_crra_xi(Pf, xi_full, u_full, dudxi, lo, hi, tau, gamma)
        Finf = float(np.max(np.abs(Pn - Pf)))
        s, om, d = metrics(Pn, u_full, lo, hi, tau)
        history.append(dict(it=it+1, ferr=Finf, slope=s, omr=om, d_FR=d))
        if verbose:
            print(f'  it {it+1:3d} ferr={Finf:.3e} slope={s:.4f} 1-R²={om:.4f} d_FR={d:.4f} ({time.time()-ts:.1f}s)', flush=True)
        Pf = Pn
        if Finf < tol:
            print(f'  CONVERGED at it {it+1}', flush=True)
            break
    return Pf, history, dict(G=G_inner, gamma=gamma, tau=tau, epsb=epsb,
                              dxi=dxi, u_max=float(u_full[hi-1]),
                              slope=s, one_minus_R2=om, d_FR=d, ferr=Finf, iters=it+1)

if __name__ == '__main__':
    print(f'K=3 CRRA PROPER ξ-grid (PCHIP co-area, NO kernel), γ={GAMMA}, τ={TAU}')
    print(f'(per-axis ξ = tanh(τu/2), smooth PCHIP contour, h=0)\n', flush=True)
    for G in [9, 11]:
        for epsb in [0.10]:
            print(f'\n=== G={G}, ξ∈[-{1-epsb:.2f},+{1-epsb:.2f}] ===')
            P, hist, summary = nail(G, GAMMA, TAU, epsb=epsb, max_iter=40, tol=1e-7)
            print(f'  RESULT: G={G} epsb={epsb}: slope={summary["slope"]:.4f}'
                  f' 1-R²={summary["one_minus_R2"]:.4f} d_FR={summary["d_FR"]:.4f}'
                  f' ferr={summary["ferr"]:.3e} iters={summary["iters"]}')
            tag = f'G{G}_eps{int(epsb*100):02d}'
            np.save(os.path.join(HERE, f'crra_xi_pchip_{tag}.npy'), P)
            json.dump({'summary':summary, 'history':hist},
                      open(os.path.join(HERE, f'crra_xi_pchip_{tag}.json'),'w'), indent=2)
    print('\nDONE')
