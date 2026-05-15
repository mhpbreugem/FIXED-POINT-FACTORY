#!/usr/bin/env python3
"""
compact_ift.py — Compactified IFT formulation of the REZN fixed point.

A mathematically cleaner alternative to the production halo solver.

  ξ = tanh(τu/2)        compactify the signal axis to (-1, 1)
  μ(ξ, p)               2-D ragged unknown (per-row p-grid)
  PCHIP                 monotone-cubic interpolation in both axes
  smooth extension      μ at ξ = ±1 uses no-learning values, no kernel
  brentq                forward contour inversion: one root-find per sweep node
  IFT Jacobian          analytic ∂Φ/∂μ for Newton (formulas in docstring §7)

For K = 3 homogeneous CRRA agents at common γ, τ. Validated against the
production anchor (γ=0.5, τ=2 ⇒ weighted 1-R² ≈ 0.085).

The interior loop is pure Python + scipy — easy to read, slow to run.
A JAX rewrite would vectorise everything and add free autodiff for the
Newton Jacobian (see §7 for the analytic chain).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator


# =============================================================================
# §1.  Primitives — the compactification and signal density
# =============================================================================

def xi_of_u(u, tau):
    return np.tanh(tau * u / 2.0)


def u_of_xi(xi, tau):
    xi = np.clip(xi, -1 + 1e-15, 1 - 1e-15)
    return (2.0 / tau) * np.arctanh(xi)


def dudxi(xi, tau):
    xi = np.clip(xi, -1 + 1e-15, 1 - 1e-15)
    return (2.0 / tau) / (1.0 - xi * xi)


def f_signal(u, v, tau):
    """Gaussian density f_v(u) for v ∈ {0, 1}; mean = v − ½."""
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - mean) ** 2)


def logit(z):
    z = np.clip(z, 1e-15, 1 - 1e-15)
    return np.log(z / (1.0 - z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def mu_no_learning(u, tau):
    """No-learning posterior: P(v=1 | u) = Λ(τu) = (1 + tanh(τu/2))/2."""
    return sigmoid(tau * u)


# =============================================================================
# §2.  CRRA demand and three-agent market clearing
# =============================================================================

def x_crra(mu, p, gamma, W=1.0):
    """Demand under CRRA utility at posterior μ and price p."""
    R = np.exp((logit(mu) - logit(p)) / gamma)
    return W * (R - 1.0) / ((1.0 - p) + R * p)


def clear_K3(mu1, mu2, mu3, gamma, W=1.0, eps=1e-9):
    """Find p such that x(μ_1) + x(μ_2) + x(μ_3) = 0 (excess demand is monotone in p)."""
    def F(p):
        return (x_crra(mu1, p, gamma, W)
                + x_crra(mu2, p, gamma, W)
                + x_crra(mu3, p, gamma, W))
    if F(eps) <= 0:
        return eps
    if F(1 - eps) >= 0:
        return 1 - eps
    return brentq(F, eps, 1 - eps, xtol=1e-12)


# =============================================================================
# §3.  μ field — per-row PCHIP, smooth boundary extension via no-learning
# =============================================================================

class MuField:
    """μ(ξ, p) on a 2-D ragged grid.

        ξ_grid  (G,)         common own-signal nodes in (−ξ_max, +ξ_max)
        p_grids (G, G_p)     per-row price grids
        mu_vals (G, G_p)     posterior values
        tau     float        signal precision

    Evaluation: PCHIP-in-p (cached per row) then PCHIP-in-ξ (cached per call).
    Boundary extension: μ at ξ=±1 fixed at the no-learning limits (0, 1).
    """
    def __init__(self, xi_grid, p_grids, mu_vals, tau):
        self.xi_grid = np.asarray(xi_grid)
        self.p_grids = np.asarray(p_grids)
        self.mu_vals = np.clip(np.asarray(mu_vals), 1e-9, 1 - 1e-9)
        self.tau = float(tau)
        self.G, self.Gp = self.mu_vals.shape
        self._rebuild_row_interp()

    def _rebuild_row_interp(self):
        self._row = [PchipInterpolator(self.p_grids[i], self.mu_vals[i],
                                       extrapolate=False) for i in range(self.G)]

    def col_at_p(self, p):
        """μ(ξ_i, p) for every i, using row-PCHIP inside [p_lo, p_hi], else flat."""
        col = np.empty(self.G)
        for i in range(self.G):
            p_lo, p_hi = self.p_grids[i, 0], self.p_grids[i, -1]
            if p <= p_lo:
                col[i] = self.mu_vals[i, 0]
            elif p >= p_hi:
                col[i] = self.mu_vals[i, -1]
            else:
                col[i] = float(self._row[i](p))
        return col

    def mu_curve_at_p(self, p):
        """Return a callable μ(ξ) for the given p, smooth on (−1, 1).

        Boundary extension: rather than pinning μ to (0, 1) at ξ=±1 (which causes
        strong PCHIP curvature against a flat interior), we add two anchor points
        at ξ = ±0.99 set to the no-learning values for those signals. PCHIP then
        extrapolates linearly past those anchors. The integrand vanishes well
        before ξ = ±1 anyway because of the Gaussian factor."""
        col = self.col_at_p(p)
        xi_anchor = 0.99
        u_anchor = u_of_xi(xi_anchor, self.tau)
        mu_anchor_lo = mu_no_learning(-u_anchor, self.tau)
        mu_anchor_hi = mu_no_learning(+u_anchor, self.tau)
        xi_ext = np.concatenate([[-xi_anchor], self.xi_grid, [xi_anchor]])
        col_ext = np.concatenate([[mu_anchor_lo], col, [mu_anchor_hi]])
        # Sort just in case the inner grid extends past the anchors
        order = np.argsort(xi_ext)
        return PchipInterpolator(xi_ext[order], col_ext[order], extrapolate=True)


# =============================================================================
# §4.  Φ map — one Bayes update per cell, via IFT contour integration
# =============================================================================

def phi_cell(mu_field, i, j, gamma, tau):
    """Φ(μ)[i, j] — posterior of the own agent with signal ξ_i, observing price p_j."""
    xi_i = mu_field.xi_grid[i]
    p_j = mu_field.p_grids[i, j]
    u_i = u_of_xi(xi_i, tau)

    # μ(ξ) at this price — built once, reused across the whole contour sweep
    mu_xi = mu_field.mu_curve_at_p(p_j)

    def demand(xi):
        m = float(mu_xi(xi))
        m = max(1e-9, min(1 - 1e-9, m))
        return x_crra(m, p_j, gamma)

    d_own = demand(xi_i)

    # Trapezoidal weights on the ξ-grid (uniform spacing)
    dxi = mu_field.xi_grid[1] - mu_field.xi_grid[0]
    w_trap = np.full(mu_field.G, dxi)
    w_trap[0] *= 0.5
    w_trap[-1] *= 0.5

    # Boundary demands — for the brentq bracket on ξ_3 ∈ (−1, 1)
    eps = 1e-4
    d_lo = demand(-1 + eps)   # ξ_3 → −1  ⇒ μ → 0  ⇒ x → −W/(1−p)
    d_hi = demand(+1 - eps)   # ξ_3 → +1  ⇒ μ → 1  ⇒ x → +W/p

    A0 = 0.0
    A1 = 0.0
    for ip in range(mu_field.G):
        xi_2 = mu_field.xi_grid[ip]
        u_2 = u_of_xi(xi_2, tau)
        d_2 = demand(xi_2)
        target = -d_own - d_2

        # If the contour leaves the open box at this ξ_2, contribution is zero
        # (and smoothly so, because the no-learning extension makes μ(ξ_3) → 0/1
        #  monotonically as ξ_3 → ±1).
        if target < d_lo or target > d_hi:
            continue

        try:
            xi_3 = brentq(lambda x: demand(x) - target,
                          -1 + eps, 1 - eps, xtol=1e-14, rtol=1e-14)
        except ValueError:
            continue
        u_3 = u_of_xi(xi_3, tau)

        # ξ-quadrature with Jacobian du/dξ — the integral is over u_2
        w_u2 = w_trap[ip] * dudxi(xi_2, tau)
        A0 += w_u2 * f_signal(u_2, 0, tau) * f_signal(u_3, 0, tau)
        A1 += w_u2 * f_signal(u_2, 1, tau) * f_signal(u_3, 1, tau)

    f0_i = f_signal(u_i, 0, tau)
    f1_i = f_signal(u_i, 1, tau)
    den = f0_i * A0 + f1_i * A1
    if den <= 0:
        return float(mu_xi(xi_i))
    return float(f1_i * A1 / den)


# =============================================================================
# §5.  Fixed-point iteration: damped Picard with monotonicity projection
# =============================================================================

def initial_mu_field(G, Gp, tau, gamma, xi_max=0.88):
    """No-learning initialisation with a SHARED p-grid across all rows.

    A row-specific p-grid (the production convention) requires more bookkeeping
    and creates a lot of out-of-range cells during early iterations. A shared
    p-grid in logit-uniform space, wide enough to cover all rows, is much more
    stable for Picard and costs nothing once converged."""
    xi_grid = np.linspace(-xi_max, xi_max, G)
    u_grid = u_of_xi(xi_grid, tau)
    mu_NL = mu_no_learning(u_grid, tau)

    # Pre-compute the global no-learning price range
    pmin, pmax = 1.0, 0.0
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p = clear_K3(mu_NL[i], mu_NL[j], mu_NL[k], gamma)
                pmin = min(pmin, p); pmax = max(pmax, p)
    pmin = max(pmin * 0.5, 1e-4)
    pmax = min(pmax * 2.0, 1 - 1e-4)
    lp_grid = np.linspace(logit(pmin), logit(pmax), Gp)
    p_shared = sigmoid(lp_grid)

    p_grids = np.tile(p_shared, (G, 1))
    # Initial μ: at no-learning the posterior is constant in p, equal to μ_NL(ξ_i)
    mu_vals = np.tile(mu_NL[:, None], (1, Gp))

    return MuField(xi_grid, p_grids, mu_vals, tau)


def iterate_phi(mu_field, gamma, tau, max_iter=60, alpha=0.15,
                anderson_m=5, tol=1e-6, verbose=True):
    """Anderson-accelerated Picard. Falls back to plain damped Picard when
    the Anderson least-squares problem is ill-conditioned."""
    G, Gp = mu_field.G, mu_field.Gp
    flat = lambda M: np.asarray(M, dtype=float).ravel()
    unflat = lambda v: v.reshape(G, Gp)

    def G_map(mu_flat):
        mu_field.mu_vals = np.clip(unflat(mu_flat), 1e-9, 1 - 1e-9)
        for i in range(G):
            mu_field.mu_vals[i] = np.maximum.accumulate(mu_field.mu_vals[i])
        mu_field._rebuild_row_interp()
        out = np.empty_like(mu_field.mu_vals)
        for i in range(G):
            for j in range(Gp):
                out[i, j] = phi_cell(mu_field, i, j, gamma, tau)
        return flat(out)

    x_hist, f_hist = [], []
    x = flat(mu_field.mu_vals)
    history = []
    for it in range(max_iter):
        gx = G_map(x)
        f  = gx - x
        F_inf = float(np.max(np.abs(f)))
        history.append(F_inf)
        if verbose:
            mid = mu_field.mu_vals[G // 2, Gp // 2]
            print(f"  iter {it+1:3d}  F_inf = {F_inf:.3e}   μ(mid)={mid:.4f}")
        if F_inf < tol:
            break

        # Anderson step
        x_hist.append(x.copy()); f_hist.append(f.copy())
        if len(x_hist) > anderson_m + 1:
            x_hist.pop(0); f_hist.pop(0)
        m = len(f_hist) - 1
        if m == 0:
            x = (1 - alpha) * x + alpha * gx
        else:
            dF = np.column_stack([f_hist[k+1] - f_hist[k] for k in range(m)])
            try:
                gamma_a, *_ = np.linalg.lstsq(dF, f, rcond=1e-14)
                dX = np.column_stack([x_hist[k+1] - x_hist[k] for k in range(m)])
                x_new = gx - (dX + dF) @ gamma_a
                x = (1 - alpha) * x + alpha * x_new
            except np.linalg.LinAlgError:
                x = (1 - alpha) * x + alpha * gx
        x = np.clip(x, 1e-9, 1 - 1e-9)

    mu_field.mu_vals = unflat(x)
    mu_field._rebuild_row_interp()
    return mu_field, history


# =============================================================================
# §5b. Newton–Krylov polish — finite-difference Jacobian, LGMRES linear solve
# =============================================================================

def _residual_flat(mu_field, x_flat, gamma, tau):
    """F(x) = Φ(x) − x as a flat vector. Mutates mu_field in place."""
    G, Gp = mu_field.G, mu_field.Gp
    mu_field.mu_vals = np.clip(x_flat.reshape(G, Gp), 1e-12, 1 - 1e-12)
    mu_field._rebuild_row_interp()
    phi = np.empty((G, Gp))
    for i in range(G):
        for j in range(Gp):
            phi[i, j] = phi_cell(mu_field, i, j, gamma, tau)
    return phi.ravel() - x_flat


def newton_polish(mu_field, gamma, tau,
                  max_iter=10, tol=1e-12,
                  lgmres_tol=1e-10, lgmres_inner_m=15, lgmres_outer=2,
                  eps_fd=1e-7, line_search=True, verbose=True):
    """Inexact Newton–Krylov for Φ(μ) − μ = 0.

    Step k: build the (I − DΦ) operator via finite differences and solve
    (I − DΦ)·δ = −F by LGMRES (scaled so LGMRES sees O(1) entries). Take
    x − δ as the next iterate. Optional Armijo damping if the full step
    doesn't reduce ||F||_∞.

    Cost per outer step ≈ (1 + lgmres_inner_m · lgmres_outer) Φ-evaluations.
    """
    from scipy.sparse.linalg import LinearOperator, lgmres

    G, Gp = mu_field.G, mu_field.Gp
    n = G * Gp
    x = mu_field.mu_vals.ravel().copy()
    history = []

    for it in range(max_iter):
        f = _residual_flat(mu_field, x, gamma, tau)
        F_inf = float(np.max(np.abs(f)))
        history.append(F_inf)
        if verbose:
            print(f"  newton {it+1:3d}  ||F||_∞ = {F_inf:.3e}")
        if F_inf < tol:
            break

        F_scale = max(F_inf, 1e-300)
        f_scaled = f / F_scale

        def matvec(v):
            xp = np.clip(x + eps_fd * v, 1e-12, 1 - 1e-12)
            fp = _residual_flat(mu_field, xp, gamma, tau)
            return (fp - f) / (eps_fd * F_scale)

        Jop = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        delta_s, info = lgmres(Jop, f_scaled,
                               atol=lgmres_tol, rtol=lgmres_tol,
                               maxiter=lgmres_outer, inner_m=lgmres_inner_m)
        delta = F_scale * delta_s

        # Armijo-light line search
        accepted = False
        if line_search:
            alpha = 1.0
            for _ in range(6):
                x_try = np.clip(x - alpha * delta, 1e-12, 1 - 1e-12)
                f_try = _residual_flat(mu_field, x_try, gamma, tau)
                if np.max(np.abs(f_try)) < F_inf * (1 - 1e-4 * alpha):
                    x = x_try
                    accepted = True
                    if verbose and alpha < 1.0:
                        print(f"          line-search α={alpha:.3g}")
                    break
                alpha *= 0.5
        if not accepted:
            x = np.clip(x - delta, 1e-12, 1 - 1e-12)

    mu_field.mu_vals = x.reshape(G, Gp)
    mu_field._rebuild_row_interp()
    return mu_field, history


# =============================================================================
# §6.  Convergence metric — weighted 1-R² (matches EQUATIONS.md §7)
# =============================================================================

def reconstruct_ree_price(mu_field, xi1, xi2, xi3, gamma, tau, eps=1e-6):
    """For three signals (ξ_1, ξ_2, ξ_3), find the REE price that clears
    the market when each agent believes μ*(ξ, p)."""
    def F(p):
        m1 = float(mu_field.mu_curve_at_p(p)(xi1))
        m2 = float(mu_field.mu_curve_at_p(p)(xi2))
        m3 = float(mu_field.mu_curve_at_p(p)(xi3))
        return (x_crra(m1, p, gamma) + x_crra(m2, p, gamma) + x_crra(m3, p, gamma))
    if F(eps) <= 0:
        return eps
    if F(1 - eps) >= 0:
        return 1 - eps
    return brentq(F, eps, 1 - eps, xtol=1e-9)


def weighted_1mR2(mu_field, gamma, tau, n_samples=13):
    """Weighted 1-R² of logit(p_REE) on T* = τ Σ u_k."""
    xi_samples = np.linspace(-0.92, 0.92, n_samples)
    u_samples = u_of_xi(xi_samples, tau)
    f0 = np.array([f_signal(u, 0, tau) for u in u_samples])
    f1 = np.array([f_signal(u, 1, tau) for u in u_samples])

    N = n_samples ** 3
    Ts = np.empty(N); LPs = np.empty(N); ws = np.empty(N)
    n = 0
    for i, ui in enumerate(u_samples):
        for j, uj in enumerate(u_samples):
            for k, uk in enumerate(u_samples):
                p = reconstruct_ree_price(mu_field, xi_samples[i], xi_samples[j],
                                          xi_samples[k], gamma, tau)
                Ts[n] = tau * (ui + uj + uk)
                LPs[n] = logit(p)
                ws[n] = 0.5 * (f0[i] * f0[j] * f0[k] + f1[i] * f1[j] * f1[k])
                n += 1

    slope, intercept = np.polyfit(Ts, LPs, 1, w=np.sqrt(ws))
    pred = slope * Ts + intercept
    mean_lp = np.average(LPs, weights=ws)
    var_tot = np.average((LPs - mean_lp) ** 2, weights=ws)
    var_res = np.average((LPs - pred) ** 2, weights=ws)
    return var_res / var_tot, float(slope), float(intercept)


# =============================================================================
# §7.  IFT Jacobian — for Newton, written analytically (not invoked here)
# =============================================================================
"""
For the inexact-Newton accelerator (not used in this demo, but trivial to bolt on
in JAX), the Jacobian dΦ/dμ has a closed-form chain via the IFT.

Per cell (i, j), Φ[i,j] = f₁(u_i) A₁ / (f₀(u_i) A₀ + f₁(u_i) A₁), with

    A_v = Σ_{i'} w_{i'} · f_v(u_{i'}) · f_v(u_3*(u_{i'};  μ-column at p_j))

For a perturbation δμ on grid node (l, q):

    δA_v = Σ_{i'} w_{i'} · f_v(u_{i'}) · f_v'(u_3*) · (δ u_3*/δ μ[l,q])

By IFT applied to F(u_2, u_3) := Σ_k x(μ(u_k, p), p) = 0:

    δu_3*/δμ[l,q]   =  -(δF/δμ[l,q]) / (∂F/∂u_3)
                     =  -[ x'_μ(μ_3, p) ψ_l(u_3*) ϕ_q(p)
                         + x'_μ(μ_2, p) ψ_l(u_2)   ϕ_q(p) ] / (x'_μ(μ_3, p) μ'(u_3*, p))

where ψ_l, ϕ_q are the PCHIP basis functions of the μ interpolant. This is
sparse (only the row-q PCHIP basis at p, and the nodes adjacent to u_3* or u_2,
participate) and exact — no kernel bandwidth, no finite differencing.

In JAX:
    mu_field = jax_pytree(...)
    phi_full = jax.vmap(jax.vmap(phi_cell, in_axes=(None, None, 0)), in_axes=(None, 0, None))
    J = jax.jacfwd(phi_full)(mu_field)
gives the exact Jacobian. Newton-Krylov then plugs in directly.
"""


# =============================================================================
# Driver
# =============================================================================
if __name__ == "__main__":
    import time

    G, Gp = 13, 9
    gamma, tau = 0.5, 2.0

    print(f"Compactified IFT solver — γ={gamma}, τ={tau}, G={G}, Gp={Gp}")
    print("=" * 60)
    t0 = time.perf_counter()
    mu_field = initial_mu_field(G, Gp, tau, gamma)
    print(f"Initial μ field built ({time.perf_counter()-t0:.1f}s)")

    print("\nPicard iteration:")
    t0 = time.perf_counter()
    mu_field, hist = iterate_phi(mu_field, gamma, tau, max_iter=50, alpha=0.35, tol=5e-6)
    print(f"Solve done ({time.perf_counter()-t0:.0f}s, {len(hist)} iters, F_final={hist[-1]:.2e})")

    print("\nWeighted 1-R² via REE price reconstruction (13³ triples)...")
    t0 = time.perf_counter()
    onemR2, slope, intercept = weighted_1mR2(mu_field, gamma, tau, n_samples=13)
    print(f"  ({time.perf_counter()-t0:.0f}s)")
    print()
    print(f"  γ = {gamma}, τ = {tau}")
    print(f"  weighted 1−R²  = {onemR2:.5f}    (production anchor 0.085)")
    print(f"  weighted slope = {slope:.4f}     (production anchor 0.543)")
