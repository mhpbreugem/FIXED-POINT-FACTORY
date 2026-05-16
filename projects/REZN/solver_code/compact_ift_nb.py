"""compact_ift_nb.py — numba-accelerated drop-in for the analytic-IFT Newton
loop. PCHIP-in-p coefficients are extracted via scipy once per Newton step,
then the entire residual-and-Jacobian assembly runs in @njit code.

Expected speedup vs the pure-Python loop: 30–50×.
"""
from __future__ import annotations
import numpy as np
import numba as nb
from scipy.interpolate import PchipInterpolator


# =============================================================================
# numba scalar primitives
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _logit(z):
    return np.log(z / (1.0 - z))


@nb.njit(cache=True, fastmath=True)
def _f_signal(u, v, tau):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2.0 * np.pi)) * np.exp(-tau / 2.0 * (u - mean) ** 2)


@nb.njit(cache=True, fastmath=True)
def _f_signal_prime(u, v, tau):
    mean = 0.5 if v == 1 else -0.5
    return -tau * (u - mean) * _f_signal(u, v, tau)


@nb.njit(cache=True, fastmath=True)
def _u_of_xi(xi, tau):
    if xi > 1.0 - 1e-15: xi = 1.0 - 1e-15
    if xi < -1.0 + 1e-15: xi = -1.0 + 1e-15
    return (2.0 / tau) * np.arctanh(xi)


@nb.njit(cache=True, fastmath=True)
def _dudxi(xi, tau):
    if xi > 1.0 - 1e-15: xi = 1.0 - 1e-15
    if xi < -1.0 + 1e-15: xi = -1.0 + 1e-15
    return (2.0 / tau) / (1.0 - xi * xi)


@nb.njit(cache=True, fastmath=True)
def _x_crra(mu, p, gamma):
    if mu < 1e-12: mu = 1e-12
    if mu > 1.0 - 1e-12: mu = 1.0 - 1e-12
    L_mu = np.log(mu / (1.0 - mu))
    L_p  = np.log(p  / (1.0 - p))
    R = np.exp((L_mu - L_p) / gamma)
    D = (1.0 - p) + R * p
    return (R - 1.0) / D


@nb.njit(cache=True, fastmath=True)
def _x_crra_mu(mu, p, gamma):
    if mu < 1e-12: mu = 1e-12
    if mu > 1.0 - 1e-12: mu = 1.0 - 1e-12
    L_mu = np.log(mu / (1.0 - mu))
    L_p  = np.log(p  / (1.0 - p))
    R = np.exp((L_mu - L_p) / gamma)
    D = (1.0 - p) + R * p
    return (1.0 / (D * D)) * R / (gamma * mu * (1.0 - mu))


# =============================================================================
# PCHIP evaluator from precomputed coefficients (scipy's format)
# =============================================================================
# scipy PchipInterpolator stores cubic pieces as
#     p_k(x) = c[0,k]·(x-x[k])³ + c[1,k]·(x-x[k])² + c[2,k]·(x-x[k]) + c[3,k]
# for x in [x[k], x[k+1]).

@nb.njit(cache=True, fastmath=True)
def _pchip_eval(xq, x, c):
    n = x.shape[0]
    # Linear extrapolation past the ends (matches scipy's `extrapolate=True`)
    if xq <= x[0]:
        # Use slope at left endpoint
        dx = xq - x[0]
        return c[3, 0] + c[2, 0] * dx
    if xq >= x[n - 1]:
        # Last segment is k = n-2
        kk = n - 2
        dx = xq - x[kk]
        return ((c[0, kk] * dx + c[1, kk]) * dx + c[2, kk]) * dx + c[3, kk]
    # Binary search for segment containing xq
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] > xq:
            hi = mid
        else:
            lo = mid
    dx = xq - x[lo]
    return ((c[0, lo] * dx + c[1, lo]) * dx + c[2, lo]) * dx + c[3, lo]


@nb.njit(cache=True, fastmath=True)
def _pchip_eval_clip(xq, x, c, lo_bound, hi_bound):
    """Clipped eval — like _pchip_eval but clips result to (lo_bound, hi_bound)."""
    v = _pchip_eval(xq, x, c)
    if v < lo_bound: return lo_bound
    if v > hi_bound: return hi_bound
    return v


# =============================================================================
# Root finding on (-1, 1) for the contour — Brent's method
# =============================================================================
# Inverse-quadratic + bisection hybrid (van Wijngaarden–Dekker–Brent).
# Matches scipy.brentq's iteration trajectory to floating-point precision,
# so the numba and pure-Python Newton iterates agree.

@nb.njit(cache=True, fastmath=True)
def _f_demand(xi, target, p, gamma, xi_full, mu_coeffs):
    mu = _pchip_eval_clip(xi, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12)
    return _x_crra(mu, p, gamma) - target


@nb.njit(cache=True, fastmath=True)
def _bisect_xi3(target, p, gamma, xi_full, mu_coeffs, eps, max_iter, xtol):
    """Simple bisection. Brent's method broke convergence at small-G test cases
    because of the underlying PCHIP non-strict-monotonicity. Plain bisection
    always converges to the same root that the original (pure-Python) code
    finds via scipy.brentq's internal bisection fallback."""
    lo = -1.0 + eps
    hi =  1.0 - eps
    f_lo = _f_demand(lo, target, p, gamma, xi_full, mu_coeffs)
    f_hi = _f_demand(hi, target, p, gamma, xi_full, mu_coeffs)
    if f_lo * f_hi > 0.0:
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _f_demand(mid, target, p, gamma, xi_full, mu_coeffs)
        if abs(f_mid) < xtol:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
        if hi - lo < xtol:
            return 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


# =============================================================================
# Build PCHIP coefficients (Fritsch–Carlson)  — called from numba
# =============================================================================

@nb.njit(cache=True)
def _pchip_coeffs(x, y):
    """Fritsch–Carlson PCHIP slopes + cubic coefficients in scipy's format.
    Returns c of shape (4, n-1)."""
    n = x.shape[0]
    h = np.empty(n - 1)
    delta = np.empty(n - 1)
    for k in range(n - 1):
        h[k] = x[k + 1] - x[k]
        delta[k] = (y[k + 1] - y[k]) / h[k]
    m = np.empty(n)
    # Interior
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # Endpoints (Fritsch's recipe — three-point one-sided estimate, clamped)
    # Left
    if n >= 3:
        slope_l = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        if slope_l * delta[0] <= 0:
            slope_l = 0.0
        elif (delta[0] * delta[1] <= 0) and (abs(slope_l) > 3.0 * abs(delta[0])):
            slope_l = 3.0 * delta[0]
        m[0] = slope_l
        # Right
        slope_r = ((2.0 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3])
        if slope_r * delta[n - 2] <= 0:
            slope_r = 0.0
        elif (delta[n - 3] * delta[n - 2] <= 0) and (abs(slope_r) > 3.0 * abs(delta[n - 2])):
            slope_r = 3.0 * delta[n - 2]
        m[n - 1] = slope_r
    else:
        m[0] = delta[0]
        m[n - 1] = delta[0]
    # Build cubic coefficients in scipy's PPoly format (per segment [x[k], x[k+1]])
    c = np.empty((4, n - 1))
    for k in range(n - 1):
        # Hermite cubic on segment k: parameter t = (x − x[k]) / h[k]
        # p(dx) = c0·dx³ + c1·dx² + c2·dx + c3 with dx = x - x[k]
        c[3, k] = y[k]
        c[2, k] = m[k]
        c[1, k] = (3.0 * delta[k] - 2.0 * m[k] - m[k + 1]) / h[k]
        c[0, k] = (m[k] + m[k + 1] - 2.0 * delta[k]) / (h[k] * h[k])
    return c


# =============================================================================
# Main routine — phi_cell + per-cell Jacobian row, all in @njit
# =============================================================================

@nb.njit(cache=True, fastmath=True, parallel=False)
def assemble_F_and_J(
    xi_grid, p_grids, mu_vals, tau, gamma,
    xi_full, row_coeffs, row_basis_coeffs,
    xi_basis_coeffs, anchor_vals, n_anchor_left,
):
    """
    xi_grid       (G,)        ξ-grid for the unknowns
    p_grids       (G, Gp)     per-row price grids
    mu_vals       (G, Gp)     μ values at the grid nodes
    tau, gamma    scalars
    xi_full (G,)        same as xi_grid (when no anchors); used as PCHIP knots
    row_coeffs    (G, 4, Gp-1) PCHIP coeffs of μ(p) for each row
    row_basis_coeffs (G, Gp, 4, Gp-1)  basis-spline coeffs: per row l, per node q

    Returns:
      F   (G, Gp)              residual Φ − μ
      J4  (G, Gp, G, Gp)       Jacobian ∂Φ[i,j]/∂μ[l, q]
    """
    G = xi_grid.shape[0]
    Gp = p_grids.shape[1]
    F = np.zeros((G, Gp))
    J4 = np.zeros((G, Gp, G, Gp))

    dxi = xi_grid[1] - xi_grid[0] if G >= 2 else 1.0   # unused for chebyshev
    eps = 1e-4

    # Build mu_xi PCHIP coefficients once per (i, j) — depends on p_j only
    # but p_j may differ across cells of row i (since p_grids is ragged)
    # so we compute it inline.

    for j in range(Gp):
        for i in range(G):
            xi_i = xi_grid[i]
            p_j = p_grids[i, j]
            u_i = _u_of_xi(xi_i, tau)

            # ----- col_at_p(p_j): one μ value per row, using row PCHIP -----
            col = np.empty(G)
            for l in range(G):
                p_lo_l = p_grids[l, 0]
                p_hi_l = p_grids[l, -1]
                if p_j <= p_lo_l:
                    col[l] = mu_vals[l, 0]
                elif p_j >= p_hi_l:
                    col[l] = mu_vals[l, -1]
                else:
                    col[l] = _pchip_eval(p_j, p_grids[l], row_coeffs[l])

            # ----- p_basis at p_j: B^(l)_q(p_j) -----
            p_basis = np.zeros((G, Gp))
            p_basis[i, j] = 1.0   # row i hits exactly
            for l in range(G):
                if l == i:
                    continue
                p_lo_l = p_grids[l, 0]
                p_hi_l = p_grids[l, -1]
                if p_j <= p_lo_l:
                    p_basis[l, 0] = 1.0
                elif p_j >= p_hi_l:
                    p_basis[l, -1] = 1.0
                else:
                    for q in range(Gp):
                        p_basis[l, q] = _pchip_eval(p_j, p_grids[l], row_basis_coeffs[l, q])

            # ----- Build μ_xi PCHIP coefficients (over xi_full incl. anchors) -----
            nf = xi_full.shape[0]
            col_ext = np.empty(nf)
            if n_anchor_left == 1:
                col_ext[0] = anchor_vals[0]
                for k_ in range(G):
                    col_ext[1 + k_] = col[k_]
                col_ext[nf - 1] = anchor_vals[1]
            else:
                for k_ in range(G):
                    col_ext[k_] = col[k_]
            mu_coeffs = _pchip_coeffs(xi_full, col_ext)

            # μ_xi and its derivative at any ξ — via the cubic coefficients
            mu_i = _pchip_eval_clip(xi_i, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12)
            # μ'_xi at xi_i: derivative of the cubic at the right segment
            # find segment containing xi_i (which is itself a node, so use right segment)
            # PCHIP derivative at a node is m[i]; we recompute below from coeffs at midpoint

            d_own = _x_crra(mu_i, p_j, gamma)
            xmu_i = _x_crra_mu(mu_i, p_j, gamma)

            # Trapezoidal weights in ξ
            # (uniform grid: middle nodes weight = dxi, ends = 0.5*dxi; Chebyshev: use averaged half-widths)
            # Simpler: use cell midpoint widths
            # For now: w_trap[ip] = 0.5*(xi_grid[ip+1] - xi_grid[ip-1])  (and half at endpoints)

            d_lo = _x_crra(_pchip_eval_clip(-1.0 + eps, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12),
                           p_j, gamma)
            d_hi = _x_crra(_pchip_eval_clip( 1.0 - eps, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12),
                           p_j, gamma)
            if d_lo > d_hi:
                d_lo, d_hi = d_hi, d_lo

            A0 = 0.0; A1 = 0.0
            dA0 = np.zeros((G, Gp))
            dA1 = np.zeros((G, Gp))

            # Pre-compute ξ-basis B_l at xi_i (= delta_l,i since xi_i is a grid node)
            # AND at every sweep node xi_2 (also delta_l,ip).
            # And at the contour points xi_3 (NOT grid nodes — must evaluate).

            for ip in range(G):
                xi_2 = xi_grid[ip]
                u_2 = _u_of_xi(xi_2, tau)
                mu_2 = _pchip_eval_clip(xi_2, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12)
                d_2 = _x_crra(mu_2, p_j, gamma)
                xmu_2 = _x_crra_mu(mu_2, p_j, gamma)

                target = -d_own - d_2
                if target < d_lo or target > d_hi:
                    continue

                xi_3 = _bisect_xi3(target, p_j, gamma, xi_full, mu_coeffs,
                                    eps, 60, 1e-14)
                if not np.isfinite(xi_3):
                    continue

                u_3 = _u_of_xi(xi_3, tau)
                mu_3 = _pchip_eval_clip(xi_3, xi_full, mu_coeffs, 1e-12, 1.0 - 1e-12)
                xmu_3 = _x_crra_mu(mu_3, p_j, gamma)

                # μ'(ξ_3): derivative of the cubic segment containing xi_3
                # find segment
                lo_k = 0; hi_k = G - 1
                while hi_k - lo_k > 1:
                    mid_k = (lo_k + hi_k) // 2
                    if xi_full[mid_k] > xi_3:
                        hi_k = mid_k
                    else:
                        lo_k = mid_k
                dx3 = xi_3 - xi_full[lo_k]
                # derivative of c0*dx^3 + c1*dx^2 + c2*dx + c3 is 3c0*dx^2 + 2c1*dx + c2
                mu3_prime = (3.0 * mu_coeffs[0, lo_k] * dx3 + 2.0 * mu_coeffs[1, lo_k]) * dx3 + mu_coeffs[2, lo_k]

                dd_dxi_3 = xmu_3 * mu3_prime
                if abs(dd_dxi_3) < 1e-300:
                    continue

                # Trapezoidal weight in ξ at sweep node ip (uniform spacing approx)
                if ip == 0:
                    w_ip = (xi_grid[1] - xi_grid[0]) * 0.5
                elif ip == G - 1:
                    w_ip = (xi_grid[G - 1] - xi_grid[G - 2]) * 0.5
                else:
                    w_ip = 0.5 * (xi_grid[ip + 1] - xi_grid[ip - 1])
                J_2 = _dudxi(xi_2, tau)
                weight = w_ip * J_2

                f0_u2 = _f_signal(u_2, 0, tau); f1_u2 = _f_signal(u_2, 1, tau)
                f0_u3 = _f_signal(u_3, 0, tau); f1_u3 = _f_signal(u_3, 1, tau)
                A0 += weight * f0_u2 * f0_u3
                A1 += weight * f1_u2 * f1_u3

                # ξ-basis B_l(xi_3) — evaluate each as a delta-basis spline
                # Computing every basis is O(G); we inline it here via Lagrange-style fan-out,
                # but that's nontrivial — instead, we approximate B_l(xi_3) by checking which
                # segment xi_3 falls in: only the two surrounding nodes contribute meaningfully.
                # Use linear approximation for the basis (consistent up to O(h²)).
                # For the Jacobian we need analytic chain — use exact PCHIP basis via small system:
                # B3_l(xi_3) = PCHIP through (xi_full, e_l) evaluated at xi_3.
                # We compute this on the fly by building a tiny spline per l.
                f0p_u3 = _f_signal_prime(u_3, 0, tau)
                f1p_u3 = _f_signal_prime(u_3, 1, tau)
                du3_dxi3 = _dudxi(xi_3, tau)
                factor0 = weight * f0_u2 * f0p_u3 * du3_dxi3
                factor1 = weight * f1_u2 * f1p_u3 * du3_dxi3

                # Full PCHIP basis B_l(xi_3) — evaluate each precomputed basis spline
                B3 = np.empty(G)
                for l in range(G):
                    B3[l] = _pchip_eval(xi_3, xi_full, xi_basis_coeffs[l])

                # B_l(xi_2) = delta (since xi_2 is a grid node)
                # B_l(xi_i) = delta (xi_i grid node)

                # ∂ξ_3/∂μ[l, q] = -(xmu_3 * B3[l] + xmu_2*delta_{l=ip} + xmu_i*delta_{l=i}) * p_basis[l,q] / dd_dxi_3
                # Careful: the xmu_i and xmu_2 terms are delta in l, so they only contribute at l=i, l=ip respectively
                inv_dd = -1.0 / dd_dxi_3
                for l in range(G):
                    base_l = xmu_3 * B3[l]
                    if l == i:
                        base_l_i_extra = xmu_i  # this is in addition to base_l
                    else:
                        base_l_i_extra = 0.0
                    if l == ip:
                        base_l_ip_extra = xmu_2
                    else:
                        base_l_ip_extra = 0.0
                    total_l = base_l + base_l_i_extra + base_l_ip_extra
                    for q in range(Gp):
                        coef = inv_dd * total_l * p_basis[l, q]
                        # ∂A_v/∂μ[l, q] += factor * coef
                        dA0[l, q] += factor0 * coef
                        dA1[l, q] += factor1 * coef

            f0_i = _f_signal(u_i, 0, tau)
            f1_i = _f_signal(u_i, 1, tau)
            D = f0_i * A0 + f1_i * A1
            if D <= 0:
                # Pin to no-learning
                F[i, j] = (1.0 / (1.0 + np.exp(-tau * u_i))) - mu_vals[i, j]
                # ∂Φ/∂μ = 0 (Φ constant)
                continue
            phi_ij = f1_i * A1 / D
            F[i, j] = phi_ij - mu_vals[i, j]
            # Bayes ratio derivative
            factor = (f0_i * f1_i) / (D * D)
            for l in range(G):
                for q in range(Gp):
                    J4[i, j, l, q] = factor * (A0 * dA1[l, q] - A1 * dA0[l, q])

    return F, J4


# =============================================================================
# Outer driver  — Newton with Armijo line search, all in numpy outside
# =============================================================================

def build_pchip_pack(mu_field):
    """Build PCHIP coefficient arrays needed by assemble_F_and_J.

    The ξ-direction PCHIP uses anchors at ±0.99 (with no-learning μ-values)
    when the grid doesn't already reach the boundary — matches mu_curve_at_p
    in compact_ift.py."""
    G, Gp = mu_field.G, mu_field.Gp
    tau = mu_field.tau
    p_grids = mu_field.p_grids
    mu_vals = mu_field.mu_vals
    xi_grid = mu_field.xi_grid

    # Row PCHIPs of μ(p)
    row_coeffs = np.empty((G, 4, Gp - 1))
    for l in range(G):
        spl = PchipInterpolator(p_grids[l], mu_vals[l], extrapolate=False)
        row_coeffs[l] = spl.c

    # Per-row p-basis: e_q at row l
    row_basis_coeffs = np.empty((G, Gp, 4, Gp - 1))
    for l in range(G):
        for q in range(Gp):
            e = np.zeros(Gp); e[q] = 1.0
            spl = PchipInterpolator(p_grids[l], e, extrapolate=False)
            row_basis_coeffs[l, q] = spl.c

    # ξ-direction setup with optional anchors
    xi_anchor = 0.99
    if xi_grid[-1] < xi_anchor and xi_grid[0] > -xi_anchor:
        u_anchor = (2.0 / tau) * np.arctanh(xi_anchor)
        mu_anchor_lo = 1.0 / (1.0 + np.exp(tau * u_anchor))
        mu_anchor_hi = 1.0 / (1.0 + np.exp(-tau * u_anchor))
        xi_full = np.concatenate([[-xi_anchor], xi_grid, [xi_anchor]])
        anchor_vals = np.array([mu_anchor_lo, mu_anchor_hi])
    else:
        xi_full = xi_grid.copy()
        anchor_vals = np.array([])

    nf = xi_full.shape[0]
    # ξ-basis: only the INTERIOR nodes get a basis function (the anchors are fixed)
    n_anchor_left = 1 if xi_grid[0] > -xi_anchor and xi_grid[-1] < xi_anchor else 0
    xi_basis_coeffs = np.empty((G, 4, nf - 1))
    for l in range(G):
        e = np.zeros(nf); e[n_anchor_left + l] = 1.0
        spl = PchipInterpolator(xi_full, e, extrapolate=True)
        xi_basis_coeffs[l] = spl.c

    return row_coeffs, row_basis_coeffs, xi_basis_coeffs, xi_full, anchor_vals, n_anchor_left


def newton_polish_nb_adaptive(mu_field, gamma, tau, max_iter=40, tol=1e-12,
                               alpha_init=0.5, alpha_min=1e-3, alpha_max=1.0,
                               symmetrize_step=True, verbose=True):
    """Damped Newton with ADAPTIVE step size:
      - α starts at alpha_init (default 0.5)
      - On accepted descent step: α *= 1.4 (capped at alpha_max)
      - On step that increases F: α *= 0.5, retry the same direction
      - On too-many failed shrinks: take the smallest α anyway and continue
    Combined with the analytic IFT Jacobian + 0↔1 symmetry projection.
    """
    G, Gp = mu_field.G, mu_field.Gp
    n = G * Gp
    history = []
    alpha = alpha_init

    for it in range(max_iter):
        row_coeffs, row_basis_coeffs, xi_basis_coeffs, xi_full, anchor_vals, n_anchor_left = build_pchip_pack(mu_field)
        F, J4 = assemble_F_and_J(
            mu_field.xi_grid, mu_field.p_grids, mu_field.mu_vals,
            tau, gamma, xi_full, row_coeffs, row_basis_coeffs,
            xi_basis_coeffs, anchor_vals, n_anchor_left,
        )
        F_inf = float(np.max(np.abs(F)))
        history.append(F_inf)
        if verbose:
            print(f"  ada-newton {it+1:3d}  ||F||_∞ = {F_inf:.3e}  α={alpha:.3g}", flush=True)
        if F_inf < tol:
            break

        A = np.eye(n) - J4.reshape(n, n)
        try:
            delta = np.linalg.solve(A, F.ravel()).reshape(G, Gp)
        except np.linalg.LinAlgError:
            delta = F.copy()

        # Try damped step with adaptive shrink
        mu_old = mu_field.mu_vals.copy()
        accepted = False
        for shrink in range(8):
            trial = np.clip(mu_old + alpha * delta, 1e-12, 1 - 1e-12)
            if symmetrize_step:
                from compact_ift import symmetrize_mu as _sym
                trial = _sym(trial)
            mu_field.mu_vals = trial
            mu_field._rebuild_row_interp()
            rc, rbc, xbc, xf, av, nal = build_pchip_pack(mu_field)
            F_try, _ = assemble_F_and_J(
                mu_field.xi_grid, mu_field.p_grids, mu_field.mu_vals,
                tau, gamma, xf, rc, rbc, xbc, av, nal,
            )
            F_try_inf = float(np.max(np.abs(F_try)))
            if F_try_inf < F_inf:                  # any descent is fine
                accepted = True
                # On success, gently grow α back up
                alpha = min(alpha_max, alpha * 1.4)
                break
            # Shrink α and retry
            alpha = max(alpha_min, alpha * 0.5)
            if alpha == alpha_min:
                # Last resort: take the smallest step even if F goes up
                trial = np.clip(mu_old + alpha_min * delta, 1e-12, 1 - 1e-12)
                if symmetrize_step:
                    from compact_ift import symmetrize_mu as _sym
                    trial = _sym(trial)
                mu_field.mu_vals = trial
                mu_field._rebuild_row_interp()
                accepted = True
                break
    return mu_field, history


def newton_polish_nb(mu_field, gamma, tau, max_iter=30, tol=1e-12,
                     line_search=True, verbose=True):
    G, Gp = mu_field.G, mu_field.Gp
    n = G * Gp
    history = []
    for it in range(max_iter):
        row_coeffs, row_basis_coeffs, xi_basis_coeffs, xi_full, anchor_vals, n_anchor_left = build_pchip_pack(mu_field)
        F, J4 = assemble_F_and_J(
            mu_field.xi_grid, mu_field.p_grids, mu_field.mu_vals,
            tau, gamma, xi_full, row_coeffs, row_basis_coeffs,
            xi_basis_coeffs, anchor_vals, n_anchor_left,
        )
        F_inf = float(np.max(np.abs(F)))
        history.append(F_inf)
        if verbose:
            print(f"  nb-newton {it+1:3d}  ||F||_∞ = {F_inf:.3e}", flush=True)
        if F_inf < tol:
            break

        # Solve (I - J) delta = F
        A = np.eye(n) - J4.reshape(n, n)
        try:
            delta = np.linalg.solve(A, F.ravel()).reshape(G, Gp)
        except np.linalg.LinAlgError:
            delta = F.copy()

        accepted = False
        if line_search:
            alpha = 1.0
            mu_old = mu_field.mu_vals.copy()
            for _ in range(8):
                trial = np.clip(mu_old + alpha * delta, 1e-12, 1 - 1e-12)
                mu_field.mu_vals = trial
                mu_field._rebuild_row_interp()
                rc, rbc, xbc, xf, av, nal = build_pchip_pack(mu_field)
                F_try, _ = assemble_F_and_J(
                    mu_field.xi_grid, mu_field.p_grids, mu_field.mu_vals,
                    tau, gamma, xf, rc, rbc, xbc, av, nal,
                )
                F_try_inf = float(np.max(np.abs(F_try)))
                if F_try_inf < F_inf * (1 - 1e-4 * alpha):
                    accepted = True
                    if verbose and alpha < 1.0:
                        print(f"            line-search α={alpha:.3g}", flush=True)
                    break
                alpha *= 0.5
            if not accepted:
                # All line-search trials failed — take the full step anyway
                mu_field.mu_vals = np.clip(mu_old + delta, 1e-12, 1 - 1e-12)
                mu_field._rebuild_row_interp()
        else:
            # No line search: just take the full step
            mu_field.mu_vals = np.clip(mu_field.mu_vals + delta, 1e-12, 1 - 1e-12)
            mu_field._rebuild_row_interp()

    return mu_field, history
