"""Pure-numpy longdouble (80-bit, ~18.9 digits) port of phi_K3_halo_smooth.

Same math as reznsrc.contour_K3_halo:
  - Gaussian-kernel co-area evidence A_v = sum_ab K_h(P_ab - p) f_v(u_a) f_v(u_b)
  - Bayes posterior mu = f1*A1 / (f0*A0 + f1*A1)
  - CRRA market clearing by vectorized bisection (90 iters, width ~ 1e-27)

Differences vs the float64/numba original (all precision-related):
  - bisection: 90 iterations, no early break at 1e-14 (that break is the
    float64 residual floor ~5e-15)
  - everything in np.longdouble
Assumes symmetric agents (tau, gamma, W identical across k), which holds
for the whole sweep.
"""
import numpy as np

LD = np.longdouble
EPS_PRICE = LD('1e-12')


def f_sig_ld(u, v, tau):
    mean = LD(0.5) if v == 1 else LD(-0.5)
    d = u - mean
    return np.sqrt(tau / (2 * np.pi * np.ones((), LD))) * np.exp(LD(-0.5) * tau * d * d)


def logit_ld(p):
    return np.log(p) - np.log(1 - p)


def lam_ld(z):
    out = np.empty_like(z)
    pos = z >= 0
    e = np.exp(-z[pos]); out[pos] = 1 / (1 + e)
    e = np.exp(z[~pos]); out[~pos] = e / (1 + e)
    return out


def clear_crra_ld(mu, gamma, n_iter=90):
    """Vectorized bisection. mu: (3, N) posteriors. Returns p: (N,)."""
    lmu = logit_ld(mu)                      # (3, N)
    N = mu.shape[1]
    a = np.full(N, EPS_PRICE, LD)
    b = np.full(N, 1 - EPS_PRICE, LD)

    def excess(p):
        lp = logit_ld(p)                    # (N,)
        z = (lmu - lp[None, :]) / gamma     # (3, N)
        x = np.empty_like(z)
        pos = z >= 0
        pB = np.broadcast_to(p, z.shape)
        e = np.exp(-z[pos])
        x[pos] = (1 - e) / ((1 - pB[pos]) * e + pB[pos])
        e = np.exp(z[~pos])
        x[~pos] = (e - 1) / ((1 - pB[~pos]) + pB[~pos] * e)
        return x.sum(axis=0)

    fa = excess(a); fb = excess(b)
    out = np.full(N, LD(0.5))
    lo_clip = fa <= 0; hi_clip = fb >= 0
    for _ in range(n_iter):
        c = (a + b) / 2
        fc = excess(c)
        take_a = fc >= 0
        a = np.where(take_a, c, a)
        b = np.where(take_a, b, c)
    out = (a + b) / 2
    out[lo_clip] = EPS_PRICE
    out[hi_clip] = 1 - EPS_PRICE
    return out


def init_no_learning_ld(uf, tau, gamma):
    Gf = uf.size
    m = lam_ld(tau * uf)
    M1, M2, M3 = np.meshgrid(m, m, m, indexing='ij')
    mu = np.stack([M1.ravel(), M2.ravel(), M3.ravel()])
    return clear_crra_ld(mu, gamma).reshape(Gf, Gf, Gf)


def phi_ld(P_full, uf, lo, hi, tau, gamma, h):
    """Longdouble phi_K3_halo_smooth, symmetric agents."""
    Gi = hi - lo
    f0 = f_sig_ld(uf, 0, tau)
    f1 = f_sig_ld(uf, 1, tau)
    F00 = np.outer(f0, f0).ravel()         # (Gf^2,)
    F11 = np.outer(f1, f1).ravel()
    inv2h2 = LD(0.5) / (h * h)
    fo0 = f0[lo:hi]; fo1 = f1[lo:hi]       # own-signal densities, inner
    p_in = P_full[lo:hi, lo:hi, lo:hi]     # (Gi,Gi,Gi)

    mu = np.empty((3, Gi, Gi, Gi), LD)
    # agent 0: slice over (axis1, axis2) at fixed i; own signal u_i
    for i in range(Gi):
        Ps = P_full[lo + i].ravel()        # (Gf^2,)
        d = Ps[None, :] - p_in[i].reshape(Gi * Gi, 1)
        w = np.exp(-d * d * inv2h2)        # (Gi^2, Gf^2)
        A0 = w @ F00; A1 = w @ F11
        num = fo1[i] * A1; den = fo0[i] * A0 + num
        mu[0, i] = np.where(den > 0, num / np.where(den > 0, den, 1), LD(0.5)).reshape(Gi, Gi)
    # agent 1: slice over (axis0, axis2) at fixed j; own signal u_j
    for j in range(Gi):
        Ps = P_full[:, lo + j, :].ravel()
        d = Ps[None, :] - p_in[:, j].reshape(Gi * Gi, 1)
        w = np.exp(-d * d * inv2h2)
        A0 = w @ F00; A1 = w @ F11
        num = fo1[j] * A1; den = fo0[j] * A0 + num
        mu[1, :, j] = np.where(den > 0, num / np.where(den > 0, den, 1), LD(0.5)).reshape(Gi, Gi)
    # agent 2: slice over (axis0, axis1) at fixed l; own signal u_l
    for l in range(Gi):
        Ps = P_full[:, :, lo + l].ravel()
        d = Ps[None, :] - p_in[:, :, l].reshape(Gi * Gi, 1)
        w = np.exp(-d * d * inv2h2)
        A0 = w @ F00; A1 = w @ F11
        num = fo1[l] * A1; den = fo0[l] * A0 + num
        mu[2, :, :, l] = np.where(den > 0, num / np.where(den > 0, den, 1), LD(0.5)).reshape(Gi, Gi)

    mu = np.clip(mu, EPS_PRICE, 1 - EPS_PRICE)
    return clear_crra_ld(mu.reshape(3, -1), gamma).reshape(Gi, Gi, Gi)
