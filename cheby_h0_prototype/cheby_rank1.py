"""Rank-1 (h ≡ 0) solver. P(u) = σ(α·T) with T = τ(u_1+u_2+u_3).

No Chebyshev basis. No root-finding. No eigvals.
- Contour {P = p} is exact hyperplane T = logit(p)/α.
- Co-area integral A_v(p, u_k) = g_v(s_k) / (ατ·p(1-p)) with
  s_k = logit(p)/(ατ) - u_k and g_v = closed-form Gaussian convolution.
- Clearing for K=3 CRRA agents: 1D bisection in p.
- Fixed-point in α: 1D Newton on α' - α (or Anderson-accelerated iter).
"""
import numpy as np
import math, time
from numba import njit, prange

# =====  Closed-form components  =====

@njit(cache=True, inline='always')
def f_v_density(u, v_is_one, tau):
    """f_v(u) = N(±0.5, 1/τ) density, v_is_one=True for v=1."""
    vm = 0.5 if v_is_one else -0.5
    return math.sqrt(tau/(2.0*math.pi)) * math.exp(-0.5*tau*(u-vm)*(u-vm))

@njit(cache=True, inline='always')
def g_v_conv(s, v_is_one, tau):
    """g_v(s) = (f_v ⋆ f_v)(s) closed form. Var = 2/τ, mean = ±1."""
    twovm = 1.0 if v_is_one else -1.0
    return math.sqrt(tau/(4.0*math.pi)) * math.exp(-tau*(s - twovm)*(s - twovm)/4.0)

@njit(cache=True)
def posterior_v1(u_k, s_k, tau):
    """μ_k = Pr(v=1 | u_k, u_a+u_b = s_k)."""
    f1 = f_v_density(u_k, True, tau); g1 = g_v_conv(s_k, True, tau)
    f0 = f_v_density(u_k, False, tau); g0 = g_v_conv(s_k, False, tau)
    num = f1 * g1
    return num / (num + f0 * g0 + 1e-300)

@njit(cache=True)
def crra_excess_demand(p, mu0, mu1, mu2, gamma):
    e = 0.0
    eps = 1e-12
    p_c = min(max(p, eps), 1-eps)
    lp = math.log(p_c/(1-p_c))
    for mu in (mu0, mu1, mu2):
        m = min(max(mu, 1e-9), 1-1e-9)
        lm = math.log(m/(1-m))
        arg = (lm - lp)/gamma
        if arg > 500: e += 1.0/p_c
        elif arg < -500: pass
        else:
            R = math.exp(arg)
            e += (R-1.0)/((1-p_c) + R*p_c)
    return e

@njit(cache=True)
def clear_one_cell_rank1(u0, u1, u2, tau, gamma):
    """At (u0, u1, u2), apply Φ given the rank-1 conjecture σ(αT).
    Key insight: agents extract s_k = u_a + u_b from p_conj = σ(αT(u))
    EXACTLY (independent of α!) because logit(σ(αT))/(ατ) = Σu_k.
    Then μ_k and clearing are α-independent → Φ output is α-independent."""
    # Bayes: extract s_k directly (no α dependence)
    s0 = u1 + u2
    s1 = u0 + u2
    s2 = u0 + u1
    mu0 = posterior_v1(u0, s0, tau)
    mu1 = posterior_v1(u1, s1, tau)
    mu2 = posterior_v1(u2, s2, tau)
    # CRRA clearing: bisection in p
    a = 1e-9; b = 1.0 - 1e-9
    for _ in range(80):
        p = 0.5*(a + b)
        e = crra_excess_demand(p, mu0, mu1, mu2, gamma)
        if e > 0: a = p
        else: b = p
    return 0.5*(a + b)

@njit(cache=True)
def operator_rank1(u_nodes, tau, gamma):
    """Apply Φ_R1: P_new on (G, G, G). NO α argument — operator is α-invariant
    under the rank-1 ansatz σ(αT). Returns P_new such that the rank-1 FP is
    α* = <logit P_new, T>/<T, T> in ONE shot."""
    G = u_nodes.shape[0]
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                P_new[i, j, k] = clear_one_cell_rank1(u_nodes[i], u_nodes[j], u_nodes[k],
                                                       tau, gamma)
    return P_new

def fit_alpha(P, T_grid):
    """Regress α' from logit(P) on T."""
    Pc = np.clip(P, 1e-15, 1 - 1e-15)
    L = np.log(Pc / (1 - Pc))
    return float(np.sum(L * T_grid) / np.sum(T_grid**2))

def deficit_R2(P, T_grid):
    """1 − R² of the rank-1 regression (how much structure escapes rank-1 model)."""
    Pc = np.clip(P, 1e-15, 1 - 1e-15)
    L = np.log(Pc / (1 - Pc)).ravel()
    Tf = T_grid.ravel()
    a = np.polyfit(Tf, L, 1)
    pred = a[0]*Tf + a[1]
    ss_res = float(np.sum((L - pred)**2))
    ss_tot = float(np.sum((L - L.mean())**2))
    return ss_res / max(ss_tot, 1e-30), float(a[0])


def solve_alpha(tau, gamma, G=7, verbose=False):
    """ONE-SHOT rank-1 solve. Operator is α-invariant; α* = best-fit slope of Φ output."""
    C_STRETCH = 2.0
    LOBATTO = -np.cos(np.pi * np.arange(G) / (G-1)) if G > 1 else np.array([0.0])
    U_NODES = C_STRETCH * np.arctanh(np.clip(LOBATTO, -0.9999, 0.9999))
    U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
    T_grid = tau * (U1 + U2 + U3)
    P = operator_rank1(U_NODES, tau, gamma)
    alpha_star = fit_alpha(P, T_grid)
    d_R2, _ = deficit_R2(P, T_grid)
    if verbose:
        print(f'  Direct: alpha* = {alpha_star:.10f}, deficit = {d_R2:.4e}')
    return dict(alpha=alpha_star, deficit_R2=d_R2, P=P, U_NODES=U_NODES, T_grid=T_grid)


if __name__ == '__main__':
    print('=== Rank-1 solver self-test ===\n')
    print('tau=1, gamma=1, G=7:')
    t = time.time()
    r = solve_alpha(1.0, 1.0, G=7, verbose=True)
    print(f'  alpha*={r["alpha"]:.8f}, deficit={r["deficit_R2"]:.4e}, time={time.time()-t:.3f}s\n')

    # Grid invariance check (key test: rank-1 should give SAME alpha* at any G)
    print('=== Grid invariance check (rank-1 should be G-independent) ===')
    print(f'{"G":>4} {"alpha*":>14} {"deficit":>12} {"time(s)":>8}')
    for G in [3, 5, 7, 9, 11, 15, 21, 31]:
        t = time.time()
        r = solve_alpha(1.0, 1.0, G=G, verbose=False)
        print(f'{G:>4} {r["alpha"]:>14.10f} {r["deficit_R2"]:>12.4e} {time.time()-t:>8.3f}')
