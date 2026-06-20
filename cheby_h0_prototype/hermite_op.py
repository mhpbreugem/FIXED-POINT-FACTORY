"""STRICT h=0 (no kernel) co-area operator on a Gauss-Hermite grid.

logit(P)(u_1, u_2, u_3) = sum_{ijk} c_{ijk} \\hat He_i(u_1) \\hat He_j(u_2) \\hat He_k(u_3)

where \\hat He_n = He_n / sqrt(n!) are orthonormal probabilist Hermite
polynomials w.r.t. weight w(u) = exp(-u^2/2)/sqrt(2*pi).

- Grid: G Gauss-Hermite nodes (probabilist; quadrature nodes)
- Mass matrix is identity for orthonormal basis -> simple vals/coeffs
- Co-area: strict h=0 line integral, POU over both axes,
  Gauss-Hermite quadrature in u (NO atanh stretching needed)
- Tabulated mu(p, u_k) + per-cube CRRA clearing
"""
import time, math
import numpy as np
from numpy.polynomial.hermite_e import hermeroots, hermevander, hermeval, hermeder
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

EPS_PRICE = 1e-9


def f_signal(u, v, tau):
    """N(±0.5, 1/tau) signal density."""
    vm = 0.5 if v == 1 else -0.5
    return math.sqrt(tau / (2*math.pi)) * math.exp(-0.5 * tau * (u - vm)**2)


def crra_clear(mu0, mu1, mu2, gamma, steps=200):
    """Same CRRA clearing as cheby_numba."""
    eps_p = EPS_PRICE
    m0 = max(min(mu0, 1-eps_p), eps_p)
    m1 = max(min(mu1, 1-eps_p), eps_p)
    m2 = max(min(mu2, 1-eps_p), eps_p)
    lm0 = math.log(m0/(1-m0))
    lm1 = math.log(m1/(1-m1))
    lm2 = math.log(m2/(1-m2))
    a = 1e-30; b = 1 - 1e-30
    for _ in range(steps):
        m = 0.5*(a+b)
        lp = math.log(m/(1-m))
        e = 0.0
        for lmk in (lm0, lm1, lm2):
            arg = (lmk - lp)/gamma
            if arg > 700: e += 1.0/m
            elif arg < -700: pass
            else:
                R = math.exp(arg)
                e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)


def make_hermite_grid(G):
    """Gauss-Hermite (probabilist) nodes and weights at G points.
    np.polynomial.hermite_e.hermegauss returns probabilist weights/nodes."""
    from numpy.polynomial.hermite_e import hermegauss
    return hermegauss(G)  # u_nodes, weights


# ===== Vandermonde for probabilist Hermite =====
def hermite_vandermonde(G, u_nodes):
    """V[i, j] = \\hat He_j(u_i) for j = 0..G-1.
    \\hat He_j = He_j / sqrt(j!)."""
    Vhe = hermevander(u_nodes, G - 1)  # unnormalized: V[i, j] = He_j(u_i)
    norms = np.array([math.sqrt(math.factorial(j)) for j in range(G)])
    return Vhe / norms[None, :]


# ===== All roots of normalized Hermite polynomial in u-space =====
def hermite_all_roots(c_norm, deg):
    """All real roots of sum_n c_norm[n] * \\hat He_n(u) = 0 over R.
    Convert to un-normalized He coeffs, use hermeroots."""
    norms = np.array([math.sqrt(math.factorial(j)) for j in range(deg + 1)])
    c_unnorm = c_norm / norms
    # Trim trailing zeros
    n_eff = deg
    while n_eff > 0 and abs(c_unnorm[n_eff]) < 1e-300:
        n_eff -= 1
    if n_eff < 1:
        return np.array([])
    roots = hermeroots(c_unnorm[:n_eff+1])
    # Keep real roots
    real = roots[np.abs(roots.imag) < 1e-10].real
    return np.sort(real)


# ===== Slice extraction =====
def eval_hermite_3d_at(coeffs_norm, u1, u2, u3, G):
    """Evaluate sum_{ijk} c_{ijk} \\hat He_i(u_1) \\hat He_j(u_2) \\hat He_k(u_3)."""
    He1 = hermevander(np.array([u1]), G - 1)[0]
    He2 = hermevander(np.array([u2]), G - 1)[0]
    He3 = hermevander(np.array([u3]), G - 1)[0]
    norms = np.array([math.sqrt(math.factorial(j)) for j in range(G)])
    H1 = He1 / norms; H2 = He2 / norms; H3 = He3 / norms
    return float(np.einsum('ijk,i,j,k->', coeffs_norm, H1, H2, H3))


def slice_at_u3(coeffs_norm, u3, G):
    """Return c_2d[i, j] such that slice(u_1, u_2) = sum_{ij} c_2d[i,j]
       * \\hat He_i(u_1) \\hat He_j(u_2)."""
    He3 = hermevander(np.array([u3]), G - 1)[0]
    norms = np.array([math.sqrt(math.factorial(j)) for j in range(G)])
    H3 = He3 / norms
    return np.einsum('ijk,k->ij', coeffs_norm, H3)


# ===== POU strict h=0 co-area integral =====
def co_area_pou_hermite(c_2d, p_target, gh_nodes, gh_weights, tau, G):
    """POU strict h=0 co-area integral on Hermite representation.

    c_2d[i, j]: 2D Hermite coefficient slice (already extracted at u_3 = fixed)

    A_v = A^(a) + A^(b), each weighted by partition-of-unity.
    Quadrature: Gauss-Hermite (weighted by w(u) = exp(-u^2/2)/sqrt(2pi)).
    """
    A0 = 0.0; A1 = 0.0
    norms = np.array([math.sqrt(math.factorial(j)) for j in range(G)])

    # Build evaluator at given u_a or u_b
    # Term A^(b): fix u_a at GH nodes, find roots in u_b
    for q in range(gh_nodes.size):
        u_a = gh_nodes[q]
        w_a_quad = gh_weights[q]  # Gauss-Hermite weight (probabilist)
        # The GH integration approximates:
        # int g(u) * exp(-u^2/2)/sqrt(2pi) du ~ sum w_q g(u_q) * sqrt(2pi)
        # Actually probabilist hermegauss weights w_q are such that
        # int f(u) * exp(-u^2/2) du ~ sum w_q f(u_q)
        # So to integrate int f(u) du we need to divide out the weight:
        # int f(u) du = int (f(u) * exp(u^2/2)) * exp(-u^2/2) du
        #            ~ sum w_q f(u_q) * exp(u_q^2/2)
        # So actual weight for plain Lebesgue: w_q * exp(u_q^2/2)
        w_actual_a = w_a_quad * math.exp(0.5 * u_a * u_a)
        f0a = f_signal(u_a, 0, tau)
        f1a = f_signal(u_a, 1, tau)
        # 1D polynomial in u_b at fixed u_a: c1d_b[j] = sum_i c_2d[i,j] * \hat He_i(u_a)
        He_a = hermevander(np.array([u_a]), G - 1)[0] / norms
        c1d_b = np.einsum('ij,i->j', c_2d, He_a)
        # Find roots
        # Shift: roots of c1d_b - p_target = 0 in u_b
        c1d_b_shift = c1d_b.copy()
        c1d_b_shift[0] -= p_target * norms[0]  # constant in norm basis
        # Actually shifting by p_target in PROBABILITY space means subtracting p_target
        # from the function value. Since \hat He_0(u) = 1, the constant term shifts by -p_target.
        c1d_b_shift[0] = c1d_b[0] - p_target
        roots = hermite_all_roots(c1d_b_shift, G - 1)
        if roots.size == 0: continue
        # d_b derivative coefs (in normalized basis)
        # Convert to unnormalized, take chebder, convert back
        c1d_b_un = c1d_b / norms
        c1d_db_un = hermeder(c1d_b_un)
        # d_a derivative slice: at fixed u_a, slice of dP/du_a in u_b basis
        # dP/du_a = sum_i c_2d[i,j] * d\hat He_i/du(u_a) * \hat He_j(u_b)
        # d He_i/du = i * He_{i-1}, so d \hat He_i / du = i/sqrt(i!) * He_{i-1}
        # = sqrt(i)/sqrt((i-1)!) * He_{i-1} = sqrt(i) * \hat He_{i-1}
        # So d \hat He_i / du = sqrt(i) * \hat He_{i-1}
        # Thus (d/du_a) c_2d[i, j] \hat He_i(u_a) = sqrt(i) c_2d[i, j] \hat He_{i-1}(u_a)
        # Let dHe_a[i] = (d \hat He_i / du)(u_a). Then dHe_a[i] = sqrt(i) * \hat He_{i-1}(u_a) for i>=1
        dHe_a = np.zeros(G)
        for i in range(1, G):
            dHe_a[i] = math.sqrt(i) * He_a[i-1]
        # d_a slice at fixed u_a: c1d_dap[j] = sum_i c_2d[i, j] * dHe_a[i]
        c1d_dap = np.einsum('ij,i->j', c_2d, dHe_a)
        # Now for each root u_b:
        for u_b in roots:
            He_b = hermevander(np.array([u_b]), G - 1)[0] / norms
            # dP/du_b = chebval (un-normalized) at u_b
            dPdu_b = float(hermeval(u_b, c1d_db_un))
            # dP/du_a = sum_j c1d_dap[j] * \hat He_j(u_b)
            dPdu_a = float(np.dot(c1d_dap, He_b))
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal(u_b, 0, tau); f1b = f_signal(u_b, 1, tau)
            wt = w_actual_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b

    # Term A^(a): fix u_b at GH nodes, find roots in u_a (swap roles)
    for q in range(gh_nodes.size):
        u_b = gh_nodes[q]
        w_b_quad = gh_weights[q]
        w_actual_b = w_b_quad * math.exp(0.5 * u_b * u_b)
        f0b = f_signal(u_b, 0, tau); f1b = f_signal(u_b, 1, tau)
        He_b = hermevander(np.array([u_b]), G - 1)[0] / norms
        c1d_a = np.einsum('ij,j->i', c_2d, He_b)
        c1d_a_shift = c1d_a.copy(); c1d_a_shift[0] -= p_target
        roots = hermite_all_roots(c1d_a_shift, G - 1)
        if roots.size == 0: continue
        c1d_a_un = c1d_a / norms
        c1d_da_un = hermeder(c1d_a_un)
        dHe_b = np.zeros(G)
        for j in range(1, G):
            dHe_b[j] = math.sqrt(j) * He_b[j-1]
        c1d_dbp = np.einsum('ij,j->i', c_2d, dHe_b)
        for u_a in roots:
            He_a = hermevander(np.array([u_a]), G - 1)[0] / norms
            dPdu_a = float(hermeval(u_a, c1d_da_un))
            dPdu_b = float(np.dot(c1d_dbp, He_a))
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal(u_a, 0, tau); f1a = f_signal(u_a, 1, tau)
            wt = w_actual_b * w_a_pou / abs(dPdu_a)
            A0 += wt * f0a * f0b
            A1 += wt * f1a * f1b
    return A0, A1


# ===== Main Phi =====
def phi_hermite(P_vals, gh_nodes, gh_weights, V, V_inv, gamma=1.0, tau=1.0,
                  G_p=121):
    """Strict h=0 K=3 CRRA REE operator on Hermite (GH) grid."""
    G = gh_nodes.size
    # Vals -> coeffs (normalized basis): coeffs = V_inv @ P_vals along each axis
    coeffs = np.einsum('ai,bj,ck,abc->ijk', V_inv, V_inv, V_inv, P_vals)
    # p-grid (logit-uniform, same as Cheb)
    p_grid = 1.0 / (1.0 + np.exp(-np.linspace(-8, 8, G_p)))
    # Build mu table
    mu_table = np.empty((G_p, G))
    for k_node in range(G):
        u_k = gh_nodes[k_node]
        f0k = f_signal(u_k, 0, tau); f1k = f_signal(u_k, 1, tau)
        c_2d = slice_at_u3(coeffs, u_k, G)
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pou_hermite(c_2d, p, gh_nodes, gh_weights, tau, G)
            den = f0k*A0 + f1k*A1
            mu_table[ip, k_node] = f1k*A1 / den if den > 1e-300 else 0.5
    # Per-cube clearing
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = P_vals[i, j, k]
                p_cell = max(min(p_cell, 1-1e-9), 1e-9)
                # Linear interp in p
                # binary search
                if p_cell <= p_grid[0]:
                    mu0 = mu_table[0, i]; mu1 = mu_table[0, j]; mu2 = mu_table[0, k]
                elif p_cell >= p_grid[-1]:
                    mu0 = mu_table[-1, i]; mu1 = mu_table[-1, j]; mu2 = mu_table[-1, k]
                else:
                    idx = np.searchsorted(p_grid, p_cell) - 1
                    w = (p_cell - p_grid[idx]) / (p_grid[idx+1] - p_grid[idx])
                    mu0 = (1-w)*mu_table[idx, i] + w*mu_table[idx+1, i]
                    mu1 = (1-w)*mu_table[idx, j] + w*mu_table[idx+1, j]
                    mu2 = (1-w)*mu_table[idx, k] + w*mu_table[idx+1, k]
                P_new[i, j, k] = crra_clear(mu0, mu1, mu2, gamma)
    return P_new


def setup(G):
    """Returns (u_nodes, gh_weights, V, V_inv) for Hermite grid of size G."""
    u_nodes, gh_weights = make_hermite_grid(G)
    V = hermite_vandermonde(G, u_nodes)
    V_inv = np.linalg.inv(V)
    return u_nodes, gh_weights, V, V_inv


if __name__ == '__main__':
    print('=== Hermite strict h=0 POU operator self-test ===\n')
    G = 7
    u_nodes, gh_weights, V, V_inv = setup(G)
    print(f'Gauss-Hermite nodes (G={G}):')
    print(f'  {u_nodes}')
    print(f'GH weights:')
    print(f'  {gh_weights}')
    print(f'  sum_q w_q exp(u_q^2/2) ~ sqrt(2pi) = {math.sqrt(2*math.pi):.4f}: '
          f'got {np.sum(gh_weights * np.exp(0.5*u_nodes**2)):.4f}')

    U1, U2, U3 = np.meshgrid(u_nodes, u_nodes, u_nodes, indexing='ij')
    TAU = 1.0; GAMMA = 1.0
    T_full = TAU*(U1+U2+U3)
    sg = lambda x: 1/(1+np.exp(-x))
    P_in = sg(0.5*T_full)
    print(f'\nP_in range: [{P_in.min():.4f}, {P_in.max():.4f}]')

    # Try single Phi
    print('\nFirst Phi call (this is slow, all numpy)...', flush=True)
    t0 = time.time()
    P_out = phi_hermite(P_in, u_nodes, gh_weights, V, V_inv,
                          gamma=GAMMA, tau=TAU)
    dt = time.time() - t0
    F = float(np.max(np.abs(P_out - P_in)))
    print(f'  t={dt:.1f}s, |F|={F:.3e}, P_out range [{P_out.min():.4f}, {P_out.max():.4f}]')
