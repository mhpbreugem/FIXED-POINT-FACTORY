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
        """μ(ξ_i, p) for every i — row-PCHIP inside [p_lo, p_hi], else flat
        (grid-edge value). The flat fallback is biased near the boundary but
        much safer than CARA-asymptote extrapolation (μ → p), which makes
        all demands vanish and breaks the REE-price reconstruction."""
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

        Boundary extension: if the grid doesn't already reach the boundary
        (|xi_max| < 0.99), add anchor points at ξ = ±0.99 set to the
        no-learning values. If the grid does extend close to ±1 (e.g.,
        Chebyshev with xi_max → 1), skip the anchors entirely."""
        col = self.col_at_p(p)
        xi_anchor = 0.99
        if self.xi_grid[-1] < xi_anchor and self.xi_grid[0] > -xi_anchor:
            u_anchor = u_of_xi(xi_anchor, self.tau)
            xi_ext = np.concatenate([[-xi_anchor], self.xi_grid, [xi_anchor]])
            col_ext = np.concatenate([
                [mu_no_learning(-u_anchor, self.tau)], col,
                [mu_no_learning( u_anchor, self.tau)],
            ])
        else:
            xi_ext = self.xi_grid
            col_ext = col
        return PchipInterpolator(xi_ext, col_ext, extrapolate=True)


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

def initial_mu_field(G, Gp, tau, gamma, xi_max=0.88, init_kind="no_learning",
                     alpha_tilt=1.0, beta_tilt=1.0):
    """Initialise μ on a shared p-grid.

    init_kind:
      "no_learning"   — μ(ξ, p) = μ_NL(ξ) (constant in p). Default.
      "cara_tilted"   — μ(ξ, p) = σ(α·atanh(ξ) + β·logit p). Parallel-line
                        contours with negative slope (the CRRA REE shape).
                        Pure CARA (α=0, β=1) is singular for this solver
                        because all demands vanish; tilted CARA breaks the
                        degeneracy with a finite α and converges nicely.
    """
    xi_grid = np.linspace(-xi_max, xi_max, G)
    u_grid = u_of_xi(xi_grid, tau)
    mu_NL = mu_no_learning(u_grid, tau)

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

    if init_kind == "no_learning":
        mu_vals = np.tile(mu_NL[:, None], (1, Gp))
    elif init_kind == "cara_tilted":
        # logit μ = α·atanh(ξ) + β·logit(p)
        atanh_xi = np.arctanh(np.clip(xi_grid, -1+1e-15, 1-1e-15))
        lp = np.log(p_shared / (1 - p_shared))
        # Outer combination, axes (ξ, p)
        L = alpha_tilt * atanh_xi[:, None] + beta_tilt * lp[None, :]
        mu_vals = 1.0 / (1.0 + np.exp(-L))
        mu_vals = np.clip(mu_vals, 1e-9, 1 - 1e-9)
    else:
        raise ValueError(f"unknown init_kind: {init_kind!r}")

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
# §4b. PAVA isotonic projection — enforce μ monotone in both u and p
# =============================================================================

def _pava_1d(y):
    """Pool-adjacent-violators: monotone non-decreasing isotonic regression."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    means = list(y); lens = [1] * n; starts = list(range(n))
    i = 1
    while i < len(means):
        if means[i-1] > means[i]:
            new_len = lens[i-1] + lens[i]
            new_mean = (means[i-1]*lens[i-1] + means[i]*lens[i]) / new_len
            means.pop(i); lens.pop(i); starts.pop(i)
            means[i-1] = new_mean; lens[i-1] = new_len
            if i > 1:
                i -= 1
        else:
            i += 1
    out = np.empty(n)
    for k in range(len(means)):
        end = starts[k+1] if k+1 < len(starts) else n
        out[starts[k]:end] = means[k]
    return out


def project_monotone_2d(mu_vals, max_passes=4, atol=1e-13):
    """Project μ-array onto the cone of bivariate-monotone (non-decreasing)
    matrices: μ[i, j] non-decreasing in both i and j. Alternates PAVA in each
    axis until stable. The true REE μ(u, p) is bivariate-monotone (Bayes +
    market clearing), so this projection keeps the iterate on the physical
    manifold and rules out spurious non-monotone fixed points."""
    mu = mu_vals.copy()
    for _ in range(max_passes):
        old = mu.copy()
        for j in range(mu.shape[1]):
            mu[:, j] = _pava_1d(mu[:, j])
        for i in range(mu.shape[0]):
            mu[i, :] = _pava_1d(mu[i, :])
        if np.max(np.abs(mu - old)) < atol:
            break
    return mu


# =============================================================================
# §5a. Analytic IFT Jacobian — derivatives of demand, contour, and Bayes
# =============================================================================

def x_crra_mu(mu, p, gamma, W=1.0):
    """∂x_crra/∂μ at fixed (p, γ, W). Derivation:
       R = exp((logit μ − logit p)/γ)
       x = W (R−1)/((1−p) + R p)
       dx/dR = W / ((1−p) + R p)²
       dR/dμ = R / (γ μ (1−μ))
    """
    L_mu = np.log(mu / (1 - mu))
    L_p = np.log(p / (1 - p))
    R = np.exp((L_mu - L_p) / gamma)
    D = (1 - p) + R * p
    return (W / (D * D)) * (R / (gamma * mu * (1 - mu)))


def f_signal_prime(u, v, tau):
    """∂f_v(u)/∂u = −τ(u − (v−½)) f_v(u)."""
    mean = 0.5 if v == 1 else -0.5
    return -tau * (u - mean) * f_signal(u, v, tau)


def _build_basis_funcs(xi_full):
    """For each interior node l (i.e. indices 1..G in xi_full which includes
    the two anchors), return the PCHIP basis function B_l(ξ): the spline that
    is 1 at xi_full[1+l] and 0 at every other node of xi_full."""
    n = len(xi_full)
    G_int = n - 2
    basis = []
    for l in range(G_int):
        e = np.zeros(n)
        e[1 + l] = 1.0
        basis.append(PchipInterpolator(xi_full, e, extrapolate=True))
    return basis


def phi_cell_with_jac(mu_field, i, j, gamma, tau, basis_funcs):
    """Evaluate Φ[i,j] AND the i-th row of the j-th Jacobian block:

        J^(j)[i, l] = ∂Φ[i, j] / ∂μ[l, j]   for l = 0, …, G−1

    Returns (phi_new, jac_row) where jac_row is a length-G vector.

    The Jacobian is block-diagonal in j because mu_curve_at_p(p_shared[j]) only
    depends on the j-th column of μ (PCHIP passes through nodes at grid prices).
    """
    G = mu_field.G
    xi_grid = mu_field.xi_grid
    xi_i = xi_grid[i]
    p_j = mu_field.p_grids[i, j]
    u_i = u_of_xi(xi_i, tau)

    mu_xi = mu_field.mu_curve_at_p(p_j)
    mu_xi_d = mu_xi.derivative()        # PCHIP derivative is analytic

    # Cached evaluations
    def mu_at(xi):  return float(np.clip(mu_xi(xi), 1e-12, 1 - 1e-12))
    def mu_prime_at(xi): return float(mu_xi_d(xi))

    mu_i = mu_at(xi_i)
    d_own = x_crra(mu_i, p_j, gamma)
    xmu_i = x_crra_mu(mu_i, p_j, gamma)

    dxi = xi_grid[1] - xi_grid[0]
    w_trap = np.full(G, dxi); w_trap[0] *= 0.5; w_trap[-1] *= 0.5

    eps = 1e-4
    d_lo = x_crra(mu_at(-1 + eps), p_j, gamma)
    d_hi = x_crra(mu_at( 1 - eps), p_j, gamma)

    A0 = 0.0; A1 = 0.0
    dA0 = np.zeros(G)
    dA1 = np.zeros(G)

    # Precompute B_l(ξ_i) for every l — cell-independent across sweep
    B_i = np.array([float(basis_funcs[l](xi_i)) for l in range(G)])

    for ip in range(G):
        xi_2 = xi_grid[ip]
        u_2 = u_of_xi(xi_2, tau)
        mu_2 = mu_at(xi_2)
        d_2  = x_crra(mu_2, p_j, gamma)
        xmu_2 = x_crra_mu(mu_2, p_j, gamma)

        target = -d_own - d_2
        if target < d_lo or target > d_hi:
            continue
        try:
            xi_3 = brentq(lambda x: x_crra(mu_at(x), p_j, gamma) - target,
                          -1 + eps, 1 - eps, xtol=1e-14, rtol=1e-14)
        except ValueError:
            continue

        u_3 = u_of_xi(xi_3, tau)
        mu_3 = mu_at(xi_3)
        xmu_3 = x_crra_mu(mu_3, p_j, gamma)
        mu3_prime = mu_prime_at(xi_3)

        # IFT denominator: ∂d/∂ξ at ξ_3
        dd_dxi_3 = xmu_3 * mu3_prime
        if abs(dd_dxi_3) < 1e-300:
            continue

        # Φ-evaluation contribution to A_v
        J_2 = dudxi(xi_2, tau)
        f0_u2 = f_signal(u_2, 0, tau); f1_u2 = f_signal(u_2, 1, tau)
        f0_u3 = f_signal(u_3, 0, tau); f1_u3 = f_signal(u_3, 1, tau)
        weight = w_trap[ip] * J_2

        A0 += weight * f0_u2 * f0_u3
        A1 += weight * f1_u2 * f1_u3

        # Jacobian contribution: ∂(f_v(u_3))/∂μ[l, j] for each l
        f0p_u3 = f_signal_prime(u_3, 0, tau)
        f1p_u3 = f_signal_prime(u_3, 1, tau)
        du3_dxi3 = dudxi(xi_3, tau)

        # B_l at ξ_2 and ξ_3 — vectorise over l
        B2 = np.array([float(basis_funcs[l](xi_2)) for l in range(G)])
        B3 = np.array([float(basis_funcs[l](xi_3)) for l in range(G)])

        # IFT chain: ∂ξ_3/∂μ[l, j]
        dxi3_dmu = -(xmu_3 * B3 + xmu_i * B_i + xmu_2 * B2) / dd_dxi_3
        du3_dmu = du3_dxi3 * dxi3_dmu

        dA0 += weight * f0_u2 * f0p_u3 * du3_dmu
        dA1 += weight * f1_u2 * f1p_u3 * du3_dmu

    f0_i = f_signal(u_i, 0, tau)
    f1_i = f_signal(u_i, 1, tau)
    D = f0_i * A0 + f1_i * A1
    if D <= 0:
        # Contour is empty at this (u_i, p_j) — the cell is informationally
        # undefined. Pin it to the no-learning posterior so it can still serve
        # as an interpolation node for neighbouring cells. ∂Φ/∂μ = 0 in this
        # branch (Φ is a constant in μ).
        return mu_no_learning(u_i, tau), np.zeros(G)
    phi_new = f1_i * A1 / D
    # Bayes ratio derivative: ∂(f1·A1/D)/∂μ = f0·f1/D² · (A0·∂A1 − A1·∂A0)
    jac_row = (f0_i * f1_i) / (D * D) * (A0 * dA1 - A1 * dA0)
    return phi_new, jac_row


def newton_polish_analytic(mu_field, gamma, tau,
                            max_iter=10, tol=1e-12,
                            line_search=True, project_monotone=False,
                            verbose=True):
    """Newton iteration using the analytic IFT Jacobian.

    The Jacobian is block-diagonal in the price index j (rigorously, because
    PCHIP-in-p passes through every grid node and p_j is a grid node). Each
    G×G block is built analytically and solved by np.linalg.solve. No
    finite differences, no Krylov, no eps_fd tuning.

    If `project_monotone=True`, projects each iterate onto the bivariate-
    monotone cone via PAVA after each accepted step. This rules out the
    spurious non-monotone fixed points reachable from degenerate-cell
    fallbacks and matches the PAVA invariants enforced by the production
    halo solver (see projects/REZN/CHECKPOINT_FORMAT.md §3.4–5).
    """
    G, Gp = mu_field.G, mu_field.Gp
    # Build basis funcs once (depends only on xi_grid + anchors)
    xi_anchor = 0.99
    xi_full = np.concatenate([[-xi_anchor], mu_field.xi_grid, [xi_anchor]])
    basis_funcs = _build_basis_funcs(xi_full)

    history = []
    for it in range(max_iter):
        # Build full Φ-output and Jacobian blocks in one pass
        F = np.zeros((G, Gp))
        J_blocks = np.zeros((Gp, G, G))
        for j in range(Gp):
            for i in range(G):
                phi_ij, jac_row = phi_cell_with_jac(mu_field, i, j, gamma, tau, basis_funcs)
                F[i, j] = phi_ij - mu_field.mu_vals[i, j]
                J_blocks[j, i, :] = jac_row

        F_inf = float(np.max(np.abs(F)))
        history.append(F_inf)
        if verbose:
            print(f"  ana-newton {it+1:3d}  ||F||_∞ = {F_inf:.3e}", flush=True)
        if F_inf < tol:
            break

        # Solve  (I − J_Φ^(j)) · δ^(j) = F^(j)   per block.
        # Newton convention:  F = Φ − μ,   ∂F/∂μ = J_Φ − I,
        # so the Newton step is  μ_new = μ + δ  where  δ = (I − J_Φ)^{−1} F.
        delta = np.zeros_like(F)
        I_G = np.eye(G)
        for j in range(Gp):
            A = I_G - J_blocks[j]
            try:
                delta[:, j] = np.linalg.solve(A, F[:, j])
            except np.linalg.LinAlgError:
                delta[:, j] = F[:, j]  # fallback = pure Picard step

        accepted = False
        if line_search:
            alpha = 1.0
            mu_old = mu_field.mu_vals.copy()
            for _ in range(8):
                trial = np.clip(mu_old + alpha * delta, 1e-12, 1 - 1e-12)
                if project_monotone:
                    trial = project_monotone_2d(trial)
                mu_field.mu_vals = trial
                mu_field._rebuild_row_interp()
                F_try_inf = 0.0
                for j in range(Gp):
                    for i in range(G):
                        phi_ij, _ = phi_cell_with_jac(mu_field, i, j, gamma, tau, basis_funcs)
                        diff = abs(phi_ij - mu_field.mu_vals[i, j])
                        if diff > F_try_inf:
                            F_try_inf = diff
                if F_try_inf < F_inf * (1 - 1e-4 * alpha):
                    accepted = True
                    if verbose and alpha < 1.0:
                        print(f"             line-search α={alpha:.3g}", flush=True)
                    break
                alpha *= 0.5
            if not accepted:
                mu_field.mu_vals = mu_old
                mu_field._rebuild_row_interp()
        if not accepted:
            new_mu = np.clip(mu_field.mu_vals + delta, 1e-12, 1 - 1e-12)
            if project_monotone:
                new_mu = project_monotone_2d(new_mu)
            mu_field.mu_vals = new_mu
            mu_field._rebuild_row_interp()

    return mu_field, history


# =============================================================================
# §5c. Ragged (per-row) p-grid — full analytic Jacobian, no block-diagonal
# =============================================================================

def _build_row_basis(p_grids):
    """For each row l and each price-grid node q, build the PCHIP basis function
    B^(l)_q(p) — the spline that's 1 at p_grids[l, q] and 0 at every other node
    of row l's grid. Returns a list of lists."""
    G, Gp = p_grids.shape
    row_basis = []
    for l in range(G):
        bases = []
        for q in range(Gp):
            e = np.zeros(Gp); e[q] = 1.0
            bases.append(PchipInterpolator(p_grids[l], e, extrapolate=False))
        row_basis.append(bases)
    return row_basis


def initial_mu_field_ragged(G, Gp, tau, gamma, xi_max=0.88,
                             init_kind="no_learning", alpha_tilt=1.0, beta_tilt=1.0,
                             pad_factor=1.5, grid_kind="uniform"):
    """Per-row p-grids: each row covers its own no-learning price range.

    Eliminates the degenerate-cell issue from a shared p-grid: every (i, j)
    cell has a meaningful contour because p_grids[i, :] only covers prices
    achievable for own-signal u_i.

    grid_kind:
      "uniform" — equispaced ξ in [-xi_max, +xi_max] (default).
      "chebyshev" — Chebyshev nodes of the first kind, clustered toward the
                    boundary ±xi_max. With xi_max → 1 this puts dense
                    resolution where the no-learning posterior saturates.
    """
    if grid_kind == "uniform":
        xi_grid = np.linspace(-xi_max, xi_max, G)
    elif grid_kind == "chebyshev":
        k = np.arange(1, G + 1)
        xi_grid = xi_max * np.cos((2*k - 1) * np.pi / (2 * G))[::-1]
    else:
        raise ValueError(f"unknown grid_kind: {grid_kind!r}")
    u_grid = u_of_xi(xi_grid, tau)
    mu_NL = mu_no_learning(u_grid, tau)

    p_grids = np.empty((G, Gp))
    mu_vals = np.empty((G, Gp))
    for i in range(G):
        prices = np.empty(G * G); idx = 0
        for j_ in range(G):
            for k_ in range(G):
                prices[idx] = clear_K3(mu_NL[i], mu_NL[j_], mu_NL[k_], gamma); idx += 1
        # logit-uniform p-grid covering the no-learning range with padding
        lp_lo = logit(max(prices.min(), 1e-5)) - 0.4
        lp_hi = logit(min(prices.max(), 1 - 1e-5)) + 0.4
        lp_row = np.linspace(lp_lo, lp_hi, Gp)
        p_grids[i] = sigmoid(lp_row)

        if init_kind == "no_learning":
            mu_vals[i, :] = mu_NL[i]
        elif init_kind == "cara_tilted":
            atanh_xi_i = np.arctanh(np.clip(xi_grid[i], -1 + 1e-15, 1 - 1e-15))
            L = alpha_tilt * atanh_xi_i + beta_tilt * lp_row
            mu_vals[i, :] = np.clip(1.0 / (1.0 + np.exp(-L)), 1e-9, 1 - 1e-9)
        else:
            raise ValueError(f"unknown init_kind: {init_kind!r}")

    return MuField(xi_grid, p_grids, mu_vals, tau)


def phi_cell_with_jac_ragged(mu_field, i, j, gamma, tau, basis_funcs, row_basis):
    """Compute Φ[i, j] AND the full (G, Gp) Jacobian ∂Φ[i, j]/∂μ[l, q].

    With ragged p-grids, p_j = p_grids[i, j] is NOT generally at row l's
    grid node for l ≠ i. So the basis function B^(l)_q(p_j) is non-trivial.
    """
    G, Gp = mu_field.G, mu_field.Gp
    xi_grid = mu_field.xi_grid
    xi_i = xi_grid[i]
    p_j = mu_field.p_grids[i, j]
    u_i = u_of_xi(xi_i, tau)

    mu_xi = mu_field.mu_curve_at_p(p_j)
    mu_xi_d = mu_xi.derivative()

    def mu_at(xi):
        return float(np.clip(mu_xi(xi), 1e-12, 1 - 1e-12))
    def mu_prime_at(xi):
        return float(mu_xi_d(xi))

    mu_i = mu_at(xi_i)
    d_own = x_crra(mu_i, p_j, gamma)
    xmu_i = x_crra_mu(mu_i, p_j, gamma)

    dxi = xi_grid[1] - xi_grid[0]
    w_trap = np.full(G, dxi); w_trap[0] *= 0.5; w_trap[-1] *= 0.5

    eps = 1e-4
    d_lo = x_crra(mu_at(-1 + eps), p_j, gamma)
    d_hi = x_crra(mu_at( 1 - eps), p_j, gamma)

    # B^(l)_q(p_j) for each (l, q). Row i is exactly diagonal at q = j.
    # For l ≠ i with p_j inside row l's range: full PCHIP basis at p_j.
    # For l ≠ i with p_j outside row l's range: col_at_p falls back to the
    # grid-edge value µ[l, 0] or µ[l, −1], so the basis is δ_{q=0} or δ_{q=Gp−1}.
    p_basis = np.zeros((G, Gp))
    p_basis[i, j] = 1.0
    for l in range(G):
        if l == i:
            continue
        p_lo_l, p_hi_l = mu_field.p_grids[l, 0], mu_field.p_grids[l, -1]
        if p_j <= p_lo_l:
            p_basis[l, 0] = 1.0
        elif p_j >= p_hi_l:
            p_basis[l, -1] = 1.0
        else:
            for q in range(Gp):
                p_basis[l, q] = float(row_basis[l][q](p_j))

    A0 = 0.0; A1 = 0.0
    dA0 = np.zeros((G, Gp))
    dA1 = np.zeros((G, Gp))

    for ip in range(G):
        xi_2 = xi_grid[ip]
        u_2 = u_of_xi(xi_2, tau)
        mu_2 = mu_at(xi_2)
        d_2 = x_crra(mu_2, p_j, gamma)
        xmu_2 = x_crra_mu(mu_2, p_j, gamma)

        target = -d_own - d_2
        if target < d_lo or target > d_hi:
            continue
        try:
            xi_3 = brentq(lambda x: x_crra(mu_at(x), p_j, gamma) - target,
                          -1 + eps, 1 - eps, xtol=1e-14, rtol=1e-14)
        except ValueError:
            continue

        u_3 = u_of_xi(xi_3, tau)
        mu_3 = mu_at(xi_3)
        xmu_3 = x_crra_mu(mu_3, p_j, gamma)
        mu3_prime = mu_prime_at(xi_3)
        dd_dxi_3 = xmu_3 * mu3_prime
        if abs(dd_dxi_3) < 1e-300:
            continue

        J_2 = dudxi(xi_2, tau)
        f0_u2 = f_signal(u_2, 0, tau); f1_u2 = f_signal(u_2, 1, tau)
        f0_u3 = f_signal(u_3, 0, tau); f1_u3 = f_signal(u_3, 1, tau)
        weight = w_trap[ip] * J_2
        A0 += weight * f0_u2 * f0_u3
        A1 += weight * f1_u2 * f1_u3

        f0p_u3 = f_signal_prime(u_3, 0, tau)
        f1p_u3 = f_signal_prime(u_3, 1, tau)
        du3_dxi3 = dudxi(xi_3, tau)

        # ξ-basis B_l(xi_2), B_l(xi_3). Both are exactly e_ip-like at the grid
        # nodes for xi_2 (since xi_2 = xi_grid[ip]) and general at xi_3.
        B2_xi = np.array([float(basis_funcs[l](xi_2)) for l in range(G)])
        B3_xi = np.array([float(basis_funcs[l](xi_3)) for l in range(G)])

        # IFT: ∂ξ_3/∂μ[l, q] = -coef[l, q]
        #   coef[l, q] = ( (xmu_3 · B3_xi[l] + xmu_2 · B2_xi[l]) · p_basis[l, q]
        #                + xmu_i · δ_{l=i} · δ_{q=j} ) / dd_dxi_3
        coef = ((xmu_3 * B3_xi[:, None] + xmu_2 * B2_xi[:, None]) * p_basis) / dd_dxi_3
        coef[i, j] += xmu_i / dd_dxi_3

        # ∂A_v/∂μ[l, q] += weight · f_v(u_2) · f_v'(u_3) · du3_dxi3 · (-coef[l, q])
        factor0 = weight * f0_u2 * f0p_u3 * du3_dxi3
        factor1 = weight * f1_u2 * f1p_u3 * du3_dxi3
        dA0 -= factor0 * coef
        dA1 -= factor1 * coef

    f0_i = f_signal(u_i, 0, tau)
    f1_i = f_signal(u_i, 1, tau)
    D = f0_i * A0 + f1_i * A1
    if D <= 0:
        return mu_no_learning(u_i, tau), np.zeros((G, Gp))
    phi_new = f1_i * A1 / D
    jac = (f0_i * f1_i) / (D * D) * (A0 * dA1 - A1 * dA0)
    return phi_new, jac


def newton_polish_analytic_ragged(mu_field, gamma, tau,
                                   max_iter=15, tol=1e-12,
                                   line_search=True, project_monotone=False,
                                   verbose=True):
    """Newton with full (n × n) analytic Jacobian for ragged p-grids.

    n = G · Gp. For G=50, Gp=15 that's a 750×750 dense solve per step.
    """
    G, Gp = mu_field.G, mu_field.Gp
    n = G * Gp
    # Match mu_curve_at_p's anchor logic exactly
    xi_anchor = 0.99
    if mu_field.xi_grid[-1] < xi_anchor and mu_field.xi_grid[0] > -xi_anchor:
        xi_full = np.concatenate([[-xi_anchor], mu_field.xi_grid, [xi_anchor]])
        basis_funcs = _build_basis_funcs(xi_full)
    else:
        # Grid already reaches boundary — no anchors; one basis function per interior node
        xi_full = mu_field.xi_grid.copy()
        basis_funcs = []
        for l in range(G):
            e = np.zeros(G); e[l] = 1.0
            basis_funcs.append(PchipInterpolator(xi_full, e, extrapolate=True))
    row_basis = _build_row_basis(mu_field.p_grids)

    history = []
    for it in range(max_iter):
        F = np.zeros((G, Gp))
        J4 = np.zeros((G, Gp, G, Gp))
        for j in range(Gp):
            for i in range(G):
                phi_ij, jac_ij = phi_cell_with_jac_ragged(
                    mu_field, i, j, gamma, tau, basis_funcs, row_basis,
                )
                F[i, j] = phi_ij - mu_field.mu_vals[i, j]
                J4[i, j] = jac_ij

        F_inf = float(np.max(np.abs(F)))
        history.append(F_inf)
        if verbose:
            print(f"  ana-newton-rag {it+1:3d}  ||F||_∞ = {F_inf:.3e}", flush=True)
        if F_inf < tol:
            break

        J_mat = J4.reshape(n, n)
        A = np.eye(n) - J_mat
        try:
            delta_flat = np.linalg.solve(A, F.ravel())
        except np.linalg.LinAlgError:
            delta_flat = F.ravel()
        delta = delta_flat.reshape(G, Gp)

        accepted = False
        if line_search:
            alpha = 1.0
            mu_old = mu_field.mu_vals.copy()
            for _ in range(8):
                trial = np.clip(mu_old + alpha * delta, 1e-12, 1 - 1e-12)
                if project_monotone:
                    trial = project_monotone_2d(trial)
                mu_field.mu_vals = trial
                mu_field._rebuild_row_interp()
                # Quick re-eval of F (without Jacobian)
                F_try_inf = 0.0
                for j in range(Gp):
                    for i in range(G):
                        phi_ij, _ = phi_cell_with_jac_ragged(
                            mu_field, i, j, gamma, tau, basis_funcs, row_basis)
                        d = abs(phi_ij - mu_field.mu_vals[i, j])
                        if d > F_try_inf: F_try_inf = d
                if F_try_inf < F_inf * (1 - 1e-4 * alpha):
                    accepted = True
                    if verbose and alpha < 1.0:
                        print(f"               line-search α={alpha:.3g}", flush=True)
                    break
                alpha *= 0.5
            if not accepted:
                mu_field.mu_vals = mu_old
                mu_field._rebuild_row_interp()
        if not accepted:
            new_mu = np.clip(mu_field.mu_vals + delta, 1e-12, 1 - 1e-12)
            if project_monotone:
                new_mu = project_monotone_2d(new_mu)
            mu_field.mu_vals = new_mu
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
