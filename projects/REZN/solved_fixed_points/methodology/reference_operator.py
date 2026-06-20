"""Reference implementation of the K=3 CRRA rational-expectations fixed-point
operator with co-area (level-set) inference, plus the Newton "nail".

This is a clean, self-contained distillation of two pieces of project code:

  * ``phi_K3_halo_smooth`` from contour_K3_halo.py  -- the Gaussian-band
    ("co-area smoothed") inference + CRRA market-clearing operator Phi;
  * the joint-limit Newton driver from coarea_path.py -- which "nails" the
    smooth fixed point Phi(P)=P to machine precision.

Everything needed (signal densities, CRRA demand, market clearing, the
operator, the solver) lives in this one file. No numba, no project imports;
plain NumPy + SciPy so it is easy to read and runs anywhere.

Run it directly::

    python reference_operator.py

It nails the partially-revealing (PR) equilibrium at gamma=0.1, tau=2 and
prints the information deficit (1 - R^2), which lands near ~0.28.

Symbols (all defined again where they are used):
  v        binary asset value, v in {0,1}
  u_i      agent i's private signal (centred: mean +1/2 if v=1, -1/2 if v=0)
  tau      signal precision (1/variance) -- bigger tau = sharper signals
  gamma    CRRA risk aversion -- bigger gamma = more cautious traders
  W_i      agent i's market mass / wealth weight
  p        a candidate market price (= probability the market assigns to v=1)
  P        the whole price function P(u_1,u_2,u_3) on the grid (the unknown)
  mu_i     agent i's posterior P(v=1 | own signal, evidence from price)
  h        Gaussian band width -- the co-area / interior-point smoothing knob
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import newton_krylov

try:                                    # SciPy moved this class around
    from scipy.optimize import NoConvergence
except ImportError:                     # pragma: no cover
    from scipy.optimize._nonlin import NoConvergence


# ----------------------------------------------------------------------
# 1. Information primitives: the signal density and the logistic helpers
# ----------------------------------------------------------------------

def f_signal(u: np.ndarray, v: int, tau: float) -> np.ndarray:
    """Density of the private signal u under state v.

    The signal is Gaussian, centred at +1/2 if the asset is worth 1 and at
    -1/2 if it is worth 0, with variance 1/tau. So a high u is "evidence
    for v=1" and the strength of that evidence grows with the precision tau.
    """
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-0.5 * tau * (u - mean) ** 2)


def lam(z):
    """Logistic sigmoid 1/(1+e^-z), written so it never overflows."""
    out = np.empty_like(np.asarray(z, dtype=float))
    pos = np.asarray(z) >= 0
    ez = np.exp(-np.abs(z))
    out[pos] = 1.0 / (1.0 + ez[pos])
    out[~pos] = ez[~pos] / (1.0 + ez[~pos])
    return out


def logit(p):
    """ln(p/(1-p)); caller guarantees p in (0,1)."""
    return np.log(p) - np.log(1.0 - p)


EPS = 1e-12   # keep probabilities strictly inside (0,1)


# ----------------------------------------------------------------------
# 2. CRRA demand and market clearing
# ----------------------------------------------------------------------

def x_crra(mu: float, p: float, gamma: float, W: float) -> float:
    """CRRA demand of one agent.

        x = W (R - 1) / ((1 - p) + R p),   R = (odds mu / odds p)^(1/gamma)

    R compares the agent's posterior odds to the price's odds; if R>1 the
    agent thinks the asset is underpriced and buys (x>0). Risk aversion
    gamma damps how aggressively the odds gap translates into a position.
    The expression below is the same formula, just rearranged for numerical
    safety (no overflow when the odds gap is large).
    """
    z = (logit(mu) - logit(p)) / gamma
    if z >= 0.0:
        e = np.exp(-z)
        return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = np.exp(z)
    return W * (e - 1.0) / ((1.0 - p) + p * e)


def clear_crra(mu_vec, gamma_vec, W_vec) -> float:
    """Market-clearing price: the unique p with sum_i W_i x_i(mu_i,p)=0.

    Aggregate excess demand is strictly decreasing in p, so a guarded
    bisection on (EPS, 1-EPS) is bullet-proof and fast (~1e-15 in 60 steps).
    """
    a, b = EPS, 1.0 - EPS

    def excess(p):
        return sum(x_crra(mu_vec[k], p, gamma_vec[k], W_vec[k])
                   for k in range(len(mu_vec)))

    if excess(a) <= 0.0:
        return a
    if excess(b) >= 0.0:
        return b
    for _ in range(60):
        c = 0.5 * (a + b)
        if excess(c) >= 0.0:
            a = c
        else:
            b = c
        if b - a < 1e-14:
            break
    return 0.5 * (a + b)


def bayes(u_own, tau_own, A0, A1) -> float:
    """Posterior P(v=1) from the own signal and the price-evidence (A0,A1).

    A_v is the (co-area weighted) likelihood of the observed price under
    state v, gathered from the OTHER agents' signals -- see section 3.
    This is just Bayes' rule: combine own-signal likelihood with A_v.
    """
    f0 = f_signal(u_own, 0, tau_own)
    f1 = f_signal(u_own, 1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    return min(max(num / den, EPS), 1.0 - EPS)


# ----------------------------------------------------------------------
# 3. Co-area smoothed inference: gather price-evidence on the level set
# ----------------------------------------------------------------------

def agent_evidence_smooth(P_slice, p_target, u_full, tau_a, tau_b, h):
    """Co-area evidence A_v(p) for one agent.

    The two OTHER agents' signals live on the 2-D ``P_slice``. The agent
    sees the price p_target, so the others' signals must lie (roughly) on
    the level set {P = p_target}. Instead of finding that curve exactly, we
    weight EVERY cell by a Gaussian band K_h(P - p_target) of width h and
    sum the joint signal density over the slice:

        A_v = sum_cells  exp(-(P-p)^2 / 2h^2) * f_v(u_a) * f_v(u_b)

    As h->0 (with the grid fine enough that the band spans many cells) this
    band sum converges to the co-area integral  int_{P=p} f_v / |grad P| dsigma.
    The 1/|grad P| weight appears automatically: where P is steep the band
    in u-space is thin, so steep regions contribute less -- exactly the
    co-area measure. (A naive "count the edge crossings" scan misses this
    weight and is biased ~19%.)
    """
    inv_2h2 = 0.5 / (h * h)
    f0a = f_signal(u_full, 0, tau_a)[:, None]
    f1a = f_signal(u_full, 1, tau_a)[:, None]
    f0b = f_signal(u_full, 0, tau_b)[None, :]
    f1b = f_signal(u_full, 1, tau_b)[None, :]
    w = np.exp(-(P_slice - p_target) ** 2 * inv_2h2)
    A0 = np.sum(w * f0a * f0b)
    A1 = np.sum(w * f1a * f1b)
    return A0, A1


# ----------------------------------------------------------------------
# 4. The fixed-point operator Phi (one full grid sweep)
# ----------------------------------------------------------------------

def phi_smooth(P_full, u_full, lo, hi, tau_vec, gamma_vec, W_vec, h):
    """One application of Phi: map the current price function to the next.

    For every interior grid point (i,j,l) with current price p:
      1. each agent reads off the price, forms co-area evidence from the
         relevant 2-D slice, and updates its posterior mu_i via Bayes;
      2. CRRA market clearing on the three posteriors gives the new price.

    A fixed point Phi(P)=P is a rational-expectations equilibrium: the price
    that agents condition on is exactly the price their trades produce.

    ``lo:hi`` are the interior (unknown) indices; cells outside that band
    are a fixed "halo" boundary (the no-learning surface) held constant.
    """
    P_new = P_full.copy()
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                # Agent 0 sees the (u2,u3) slice at fixed u1=i.
                A0, A1 = agent_evidence_smooth(P_full[i, :, :], p, u_full,
                                               tau_vec[1], tau_vec[2], h)
                mu0 = bayes(u_full[i], tau_vec[0], A0, A1)
                # Agent 1 sees the (u1,u3) slice at fixed u2=j.
                A0, A1 = agent_evidence_smooth(P_full[:, j, :], p, u_full,
                                               tau_vec[0], tau_vec[2], h)
                mu1 = bayes(u_full[j], tau_vec[1], A0, A1)
                # Agent 2 sees the (u1,u2) slice at fixed u3=l.
                A0, A1 = agent_evidence_smooth(P_full[:, :, l], p, u_full,
                                               tau_vec[0], tau_vec[1], h)
                mu2 = bayes(u_full[l], tau_vec[2], A0, A1)
                P_new[i, j, l] = clear_crra([mu0, mu1, mu2],
                                            gamma_vec, W_vec)
    return P_new


def init_no_learning(u_full, tau_vec, gamma_vec, W_vec):
    """Starting price surface: agents use only their own signal (no learning
    from price). This is the halo/boundary and the Newton warm start."""
    G = u_full.size
    P = np.empty((G, G, G))
    for i in range(G):
        m0 = float(lam(tau_vec[0] * u_full[i]))
        for j in range(G):
            m1 = float(lam(tau_vec[1] * u_full[j]))
            for l in range(G):
                m2 = float(lam(tau_vec[2] * u_full[l]))
                P[i, j, l] = clear_crra([m0, m1, m2], gamma_vec, W_vec)
    return P


# ----------------------------------------------------------------------
# 5. The "nail": Newton-Krylov on the smooth residual + the deficit metric
# ----------------------------------------------------------------------

def deficit(P_inner, T):
    """Information deficit = 1 - R^2 of logit(P) regressed on the aggregate
    statistic T = tau*(u1+u2+u3).

    Under a FULLY-revealing equilibrium the price would be an exact function
    of T, so R^2 = 1 and deficit = 0. A positive deficit measures how much
    price variation is NOT explained by the aggregate signal -- the size of
    the partially-revealing information gap.
    """
    y = logit(np.clip(P_inner, EPS, 1 - EPS)).ravel()
    t = T.ravel()
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(ss_res / max(ss_tot, 1e-30))


def nail(G_inner=17, tau=2.0, gamma=0.1, C=0.45, umax=4.0, pad=2):
    """Solve Phi(P)=P to machine precision and return (deficit, ||F||, P).

    The band width is tied to the grid: h = C*du^0.5 with du<<... so that
    as the grid refines (du->0) the band still spans many cells -- the joint
    co-area / interior-point limit. Newton-Krylov then converges quadratically
    because Phi is a smooth (Gaussian) function of P.
    """
    K = 3
    tau_vec = np.full(K, tau)
    gamma_vec = np.full(K, gamma)
    W_vec = np.full(K, 1.0)

    du = 2 * umax / (G_inner - 1)
    h = C * du ** 0.5
    G_full = G_inner + 2 * pad
    u_full = np.array([-umax + (q - pad) * du for q in range(G_full)])
    lo, hi = pad, pad + G_inner
    slc = (slice(lo, hi),) * K

    ui = u_full[lo:hi]
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = tau * (U1 + U2 + U3)

    halo = init_no_learning(u_full, tau_vec, gamma_vec, W_vec)

    def resid(xflat):
        Pf = halo.copy()
        Pf[slc] = xflat.reshape((G_inner,) * K)
        return (phi_smooth(Pf, u_full, lo, hi, tau_vec, gamma_vec,
                           W_vec, h) - Pf)[slc].ravel()

    x0 = halo[slc].ravel().copy()
    try:
        sol = newton_krylov(resid, x0, f_tol=1e-10, maxiter=200,
                            method="lgmres")
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel()

    Finf = float(np.max(np.abs(resid(sol))))
    P_inner = sol.reshape((G_inner,) * K)
    return deficit(P_inner, T), Finf, P_inner


# ----------------------------------------------------------------------
# 6. Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Nailing the K=3 partially-revealing equilibrium "
          "(gamma=0.1, tau=2) ...")
    defi, Finf, _ = nail(G_inner=17, tau=2.0, gamma=0.1)
    print(f"  residual ||Phi(P) - P||_inf = {Finf:.2e}   "
          f"(<1e-9 means 'nailed')")
    print(f"  information deficit 1 - R^2 = {defi:.4f}")
    nailed = Finf < 1e-8
    print("  status:", "NAILED -- genuine smooth PR fixed point"
          if nailed else "did not nail")
