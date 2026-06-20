"""Refit the additive ansatz against STRICT-h=0 P (not against R4).

If best-fit additive sum_i h(u_i) still has large residual vs L_strict,
the FP is genuinely non-additive and v3 needs a correction term.
If best-fit additive is small, v2's residual was just R4-bias leakage and
v3 isn't needed — we should rebuild v2 against strict instead.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from scipy.stats import norm
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
from lin_cdf_pchip import make_cdf_uniform_grid

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v3"
os.makedirs(f"{OUT}/figs", exist_ok=True)

U_MAX = 2.33


def sigma2_at(coefs, s, U=U_MAX):
    """sigma^2(s) = (sum c_n T_{2n}(s/U))^2 -- guarantees >= 0."""
    d = len(coefs) - 1
    xi = s / U
    T = [1.0, xi]
    for k in range(1, 2*d):
        T.append(2*xi*T[-1] - T[-2])
    return sum(coefs[n] * T[2*n] for n in range(d + 1))**2


def h_at(coefs, u_val, U=U_MAX, NQ=24):
    if u_val == 0.0: return 0.0
    sign = 1.0 if u_val > 0 else -1.0
    a, b = 0.0, abs(u_val); half = (b - a) / 2; mid = (a + b) / 2
    xi, w = np.polynomial.legendre.leggauss(NQ)
    return sign * sum(w[i] * half * sigma2_at(coefs, mid + half*xi[i], U) for i in range(NQ))


def fit_additive_to_strict(L_strict, u_grid, n_params=5, gamma_pen=0.0):
    """Fit coefs to minimize |L_strict - sum_i h_coefs(u_i)|_2."""
    G = len(u_grid)

    def residuals(coefs):
        h_vals = np.array([h_at(coefs, u) for u in u_grid])
        L_add = h_vals[:, None, None] + h_vals[None, :, None] + h_vals[None, None, :]
        return (L_strict - L_add).ravel()

    x0 = np.array([1.0] + [0.1]*(n_params-1))  # initial guess
    result = least_squares(residuals, x0, method='trf', max_nfev=500, verbose=0)
    return result.x, result


def main():
    # Load strict FP at (g=100, t=1)
    d = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g100_t1.0000.npz")
    P_strict = d["P_strict"].astype(np.float64)
    G = P_strict.shape[0]
    u_grid = make_cdf_uniform_grid(G)
    eps = 1e-9
    L_strict = np.log(np.clip(P_strict, eps, 1-eps) / (1 - np.clip(P_strict, eps, 1-eps)))
    print(f"=== strict FP (g=100, tau=1): G={G}, |L|_inf={np.max(np.abs(L_strict)):.3f} ===")

    # Try fits with increasing n_params
    print("\n--- Best additive fits at increasing n_params ---")
    results = {}
    for n in [3, 5, 7, 9, 11]:
        coefs, res = fit_additive_to_strict(L_strict, u_grid, n_params=n)
        L_add_max = np.max(np.abs(res.fun))
        L_add_rms = np.sqrt(np.mean(res.fun**2))
        print(f"  n={n:2d}: |residual|_inf = {L_add_max:.3e}, "
              f"|residual|_rms = {L_add_rms:.3e}, "
              f"cost={res.cost:.3e}, nfev={res.nfev}")
        results[n] = (coefs, res)

    # Plot the best fit (n=11) and compute residual structure
    coefs_best, res_best = results[11]
    h_vals = np.array([h_at(coefs_best, u) for u in u_grid])
    L_add = h_vals[:, None, None] + h_vals[None, :, None] + h_vals[None, None, :]
    R = L_strict - L_add
    print(f"\n--- Best fit (n=11) structural residual ---")
    print(f"|R|_inf = {np.max(np.abs(R)):.3e}")
    print(f"|R|_rms = {np.sqrt(np.mean(R**2)):.3e}")
    print(f"|R|_inf / |L|_inf = {np.max(np.abs(R))/np.max(np.abs(L_strict)):.3e}")

    R_diag = R[np.arange(G), np.arange(G), np.arange(G)]
    print(f"\nResidual on diagonal R(s,s,s):")
    for i, (s, r) in enumerate(zip(u_grid, R_diag)):
        print(f"  s={s:+.3f}: R={r:+.4f}")
    print(f"|R on diag|_inf = {np.max(np.abs(R_diag)):.3e}")

    R_diss = np.array([R[i, i, G-1-i] for i in range(G)])
    print(f"\nResidual on dissenter R(s,s,-s):")
    for i, (s, r) in enumerate(zip(u_grid, R_diss)):
        print(f"  s={s:+.3f}: R={r:+.4f}")
    print(f"|R on diss|_inf = {np.max(np.abs(R_diss)):.3e}")

    # Test if any low-degree polynomial g captures R well
    print("\n--- 1-param fits at n=11 best-additive residual ---")
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    candidates = {
        "u1*u2*u3 (triple)":               U1*U2*U3,
        "sum (ui-uj)^2 (pairwise-var)":    (U1-U2)**2 + (U2-U3)**2 + (U1-U3)**2,
        "u1*u2 + u2*u3 + u1*u3":           U1*U2 + U2*U3 + U1*U3,
        "(u1+u2+u3)^3 (sum-cubed)":         (U1+U2+U3)**3,
        "sum(ui^3)":                       U1**3 + U2**3 + U3**3,
        "sum(ui^5)":                       U1**5 + U2**5 + U3**5,
    }
    for name, g in candidates.items():
        g_flat = g.ravel(); R_flat = R.ravel()
        gnorm2 = np.dot(g_flat, g_flat)
        if gnorm2 < 1e-30: continue
        kappa = float(np.dot(R_flat, g_flat) / gnorm2)
        residual = R_flat - kappa * g_flat
        captured = 1 - np.linalg.norm(residual)**2 / np.linalg.norm(R_flat)**2
        print(f"  {name:40s}: kappa = {kappa:+.4f}, captures {captured*100:5.1f}% of |R|^2")

    # Multi-term: try all combinations
    print("\n--- multi-term LSQ fits at n=11 residual ---")
    bases = {
        'tri': U1*U2*U3,
        'pvar': (U1-U2)**2 + (U2-U3)**2 + (U1-U3)**2,
        'pprod': U1*U2 + U2*U3 + U1*U3,
        'sumcube': (U1+U2+U3)**3,
        'sum_ui3': U1**3 + U2**3 + U3**3,
        'sum_ui5': U1**5 + U2**5 + U3**5,
    }
    R_flat = R.ravel()
    for combo in [
        ['tri', 'pvar'],
        ['tri', 'sumcube'],
        ['tri', 'pvar', 'sumcube'],
        ['tri', 'pvar', 'pprod', 'sumcube'],
        ['tri', 'pvar', 'pprod', 'sumcube', 'sum_ui3'],
        ['tri', 'pvar', 'pprod', 'sumcube', 'sum_ui3', 'sum_ui5'],
    ]:
        M = np.column_stack([bases[k].ravel() for k in combo])
        kappas, _, _, _ = np.linalg.lstsq(M, R_flat, rcond=None)
        residual = R_flat - M @ kappas
        captured = 1 - np.linalg.norm(residual)**2 / np.linalg.norm(R_flat)**2
        print(f"  {'+'.join(combo):50s}: captures {captured*100:5.1f}% of |R|^2, "
              f"|R_after|_inf = {np.max(np.abs(residual)):.3e}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Plot 1: residual norm vs n_params
    ns = sorted(results.keys())
    rs_inf = [np.max(np.abs(results[n][1].fun)) for n in ns]
    rs_rms = [np.sqrt(np.mean(results[n][1].fun**2)) for n in ns]
    axes[0, 0].semilogy(ns, rs_inf, 'b-o', label='|R|_inf')
    axes[0, 0].semilogy(ns, rs_rms, 'r-s', label='|R|_rms')
    axes[0, 0].set_xlabel('n_params in h-fit')
    axes[0, 0].set_ylabel('residual magnitude')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_title('Best additive fit: residual vs h-flexibility')

    # Plot 2: h(u) at n=11
    h_dense = np.array([h_at(coefs_best, u) for u in np.linspace(-U_MAX, U_MAX, 100)])
    axes[0, 1].plot(np.linspace(-U_MAX, U_MAX, 100), h_dense, 'b-')
    axes[0, 1].plot(u_grid, h_vals, 'ro')
    axes[0, 1].set_xlabel('u'); axes[0, 1].set_ylabel('h(u)')
    axes[0, 1].set_title(f'Best-fit h(u), n_params={11}')
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: R on diagonal and dissenter
    axes[1, 0].plot(u_grid, R_diag, 'b-o', label='R on diagonal')
    axes[1, 0].plot(u_grid, R_diss, 'g-s', label='R on dissenter')
    axes[1, 0].axhline(0, color='k', lw=0.5)
    axes[1, 0].set_xlabel('s'); axes[1, 0].set_ylabel('R')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('Structural residual at best additive fit')

    # Plot 4: heatmap
    im = axes[1, 1].imshow(R[:, :, G//2], extent=[u_grid[0],u_grid[-1],u_grid[0],u_grid[-1]],
                              origin='lower', cmap='RdBu_r')
    axes[1, 1].set_title('R(u1, u2, u3=0)')
    axes[1, 1].set_xlabel('u1'); axes[1, 1].set_ylabel('u2')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/refit.png", dpi=120)
    plt.close()
    print(f"\nsaved {OUT}/figs/refit.png")

    # Save best coefs
    np.savez(f"{OUT}/best_additive_g100_t1.npz",
             coefs=coefs_best, R=R, h_vals=h_vals, u_grid=u_grid,
             L_strict=L_strict, L_add=L_add)
    print(f"saved best additive fit to {OUT}/best_additive_g100_t1.npz")


if __name__ == "__main__":
    main()
