"""Fingerprint test for Method C v3 non-additive correction.

At (gamma=100, tau=1.0):
  - Reconstruct P_v2(u1,u2,u3) = sigmoid(h(u1)+h(u2)+h(u3)) from saved 5 coefs.
  - Compute R(u1,u2,u3) = logit(P_strict) - (h(u1)+h(u2)+h(u3))
    (residual in logit space, not P space, since the ansatz lives there).
  - Look at R on:
      (a) Diagonal: u1 = u2 = u3 = s. Both triple-product and pairwise-var
          conjectures give 0 here for the form g(u1,u2,u3) - g(s,s,s)?
          Actually for triple-product g = u1*u2*u3, on diagonal g = s^3 (nonzero).
          For pairwise-var g = sum (ui-uj)^2, on diagonal g = 0.
          So R on diagonal: nonzero ~ s^3 → triple product; ~0 → pairwise var.
      (b) Dissenter: u1 = u2 = s, u3 = -s.
          Triple product g = -s^3
          Pairwise var g = 0 + 4s^2 + 4s^2 = 8 s^2
  - Fit each candidate g to R via 1-param kappa optimization at the tested cell;
    compare residuals.
"""
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb
from scipy.optimize import minimize_scalar
from scipy.stats import norm

sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
from lin_cdf_pchip import make_cdf_uniform_grid

REPO = "/home/user/FIXED-POINT-FACTORY"
OUT = f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v3"
os.makedirs(f"{OUT}/figs", exist_ok=True)

U_MAX = 2.33
D_SIGMA = 4


def sigma2_at(coefs, s, d_sigma=D_SIGMA, U=U_MAX):
    """sigma^2(s) = (sum_n c_n T_{2n}(s/U))^2."""
    xi = s / U
    T = [1.0, xi]
    for k in range(1, 2*d_sigma):
        T.append(2*xi*T[-1] - T[-2])
    sigma = sum(coefs[n] * T[2*n] for n in range(d_sigma + 1))
    return sigma * sigma


def h_at(coefs, u_val, d_sigma=D_SIGMA, U=U_MAX, NQ=24):
    """h(u) = int_0^u sigma^2(s) ds via GL quadrature."""
    if u_val == 0.0: return 0.0
    sign = 1.0 if u_val > 0 else -1.0
    a, b = 0.0, abs(u_val)
    half = (b - a) / 2; mid = (a + b) / 2
    xi, w = np.polynomial.legendre.leggauss(NQ)
    return sign * np.sum(w * half * np.array([sigma2_at(coefs, mid + half*x, d_sigma, U) for x in xi]))


def L_v2(u1, u2, u3, coefs):
    """logit P under Method C v2 ansatz."""
    return h_at(coefs, u1) + h_at(coefs, u2) + h_at(coefs, u3)


def main():
    # Load Method C v2 coefs at (gamma=100, tau=1)
    grid = json.load(open(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2_grid/grid.json"))
    cell = grid["g100_t1.0000"]
    coefs = np.array(cell["coefs"])
    print(f"Method C v2 coefs at (g=100, tau=1): {coefs}")
    print(f"  F at this cell: {cell['F']:.3e}")

    # Load strict FP
    d = np.load(f"{REPO}/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_k3_strict_fp_g100_t1.0000.npz")
    P_strict = d["P_strict"].astype(np.float64)  # (11,11,11)
    G = P_strict.shape[0]
    u_grid = make_cdf_uniform_grid(G)
    print(f"  G={G}, u_grid range [{u_grid[0]:.3f}, {u_grid[-1]:.3f}]")

    # logit transform
    eps = 1e-9
    L_strict = np.log(np.clip(P_strict, eps, 1-eps) / (1 - np.clip(P_strict, eps, 1-eps)))

    # Compute L_v2 on the same grid
    h_vals = np.array([h_at(coefs, u) for u in u_grid])
    L_v2_grid = h_vals[:, None, None] + h_vals[None, :, None] + h_vals[None, None, :]

    # Residual in logit space
    R = L_strict - L_v2_grid
    print(f"\n|L_strict|_inf = {np.max(np.abs(L_strict)):.3f}")
    print(f"|L_v2|_inf     = {np.max(np.abs(L_v2_grid)):.3f}")
    print(f"|R|_inf        = {np.max(np.abs(R)):.3f}")
    print(f"|R|_rms        = {np.sqrt(np.mean(R**2)):.3f}")

    # --- Test 1: R on the diagonal ---
    diag_idx = np.arange(G)
    R_diag = R[diag_idx, diag_idx, diag_idx]  # R(s, s, s) for s in u_grid
    print(f"\n--- DIAGONAL: u1=u2=u3=s ---")
    for i, (s, r) in enumerate(zip(u_grid, R_diag)):
        print(f"  s={s:+.3f}: R = {r:+.4f}")
    print(f"|R on diag|_inf = {np.max(np.abs(R_diag)):.3e}")

    # --- Test 2: R on dissenter slice ---
    # u1 = u2 = s, u3 = -s, for s on u_grid (find paired indices)
    # G=11 CDF-uniform is symmetric: u_grid[i] = -u_grid[G-1-i]
    R_diss = np.zeros(G)
    for i in range(G):
        j = G - 1 - i  # u_grid[j] = -u_grid[i]
        R_diss[i] = R[i, i, j]  # u1=u_grid[i], u2=u_grid[i], u3=u_grid[j]
    print(f"\n--- DISSENTER: u1=u2=s, u3=-s ---")
    for i, (s, r) in enumerate(zip(u_grid, R_diss)):
        print(f"  s={s:+.3f}: R = {r:+.4f}")
    print(f"|R on diss|_inf = {np.max(np.abs(R_diss)):.3e}")

    # --- Candidate non-additive terms g(u1, u2, u3) ---
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    g_triple = U1 * U2 * U3  # triple product
    g_pairvar = (U1 - U2)**2 + (U2 - U3)**2 + (U1 - U3)**2  # pairwise variance
    g_pairprod = U1*U2 + U2*U3 + U1*U3  # pairwise product (Newton's 2nd)
    g_q = U1**2 + U2**2 + U3**2  # symmetric quadratic (likely already absorbed by h)

    # --- For each g, find optimal kappa: minimize |R - kappa*g|_2 ---
    print("\n--- 1-param fit kappa for each candidate g ---")
    candidates = {
        "triple_product u1*u2*u3": g_triple,
        "pairwise_var Sum(ui-uj)^2": g_pairvar,
        "pairwise_prod u1*u2+u2*u3+u1*u3": g_pairprod,
    }
    for name, g in candidates.items():
        g_flat = g.ravel(); R_flat = R.ravel()
        kappa = float(np.dot(R_flat, g_flat) / np.dot(g_flat, g_flat))
        residual = R_flat - kappa * g_flat
        improvement = 1 - np.linalg.norm(residual) / np.linalg.norm(R_flat)
        print(f"  {name:40s}: kappa = {kappa:+.4f}, |R-kg|/|R| = {np.linalg.norm(residual)/np.linalg.norm(R_flat):.4f}, captures {improvement*100:.1f}%")

    # --- Multi-term fit: try kappa1*triple + kappa2*pairvar ---
    print("\n--- 2-param fit ---")
    G_mat = np.column_stack([g_triple.ravel(), g_pairvar.ravel()])
    R_flat = R.ravel()
    kappas, _, _, _ = np.linalg.lstsq(G_mat, R_flat, rcond=None)
    residual = R_flat - G_mat @ kappas
    print(f"  triple + pairvar: kappas = {kappas}, residual_norm/R_norm = {np.linalg.norm(residual)/np.linalg.norm(R_flat):.4f}")

    G_mat = np.column_stack([g_triple.ravel(), g_pairvar.ravel(), g_pairprod.ravel()])
    kappas, _, _, _ = np.linalg.lstsq(G_mat, R_flat, rcond=None)
    residual = R_flat - G_mat @ kappas
    print(f"  triple+pairvar+pairprod: kappas={kappas}, residual_norm/R_norm = {np.linalg.norm(residual)/np.linalg.norm(R_flat):.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(u_grid, R_diag, 'b-o', label='R on diagonal (u,u,u)', ms=6)
    axes[0].plot(u_grid, u_grid**3 * (R_diag.mean()/(u_grid**3).mean() if (u_grid**3).any() else 1),
                 'r--', alpha=0.5, label='~s^3 (triple-product fingerprint)')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('s'); axes[0].set_ylabel('R(s,s,s)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title('Diagonal slice (rules out pairwise-var if nonzero)')

    axes[1].plot(u_grid, R_diss, 'g-s', label='R on dissenter (u,u,-u)', ms=6)
    axes[1].plot(u_grid, -u_grid**3 * (R_diss.mean()/(-u_grid**3).mean() if (u_grid**3).any() else 1),
                 'r--', alpha=0.5, label='~-s^3 (triple)')
    axes[1].plot(u_grid, 8*u_grid**2 * (R_diss.mean()/(8*u_grid**2).mean() if (u_grid**2).any() else 1),
                 'm--', alpha=0.5, label='~8s^2 (pairwise-var)')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('s'); axes[1].set_ylabel('R(s,s,-s)'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title('Dissenter slice')

    # Heatmap of R at u3 = 0 (middle slice)
    k0 = G // 2
    im = axes[2].imshow(R[:,:,k0], extent=[u_grid[0],u_grid[-1],u_grid[0],u_grid[-1]],
                          origin='lower', cmap='RdBu_r')
    axes[2].set_title(f'R(u1, u2, u3=0)')
    axes[2].set_xlabel('u1'); axes[2].set_ylabel('u2')
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/fingerprint.png", dpi=120)
    plt.close()
    print(f"\nsaved {OUT}/figs/fingerprint.png")


if __name__ == "__main__":
    main()
