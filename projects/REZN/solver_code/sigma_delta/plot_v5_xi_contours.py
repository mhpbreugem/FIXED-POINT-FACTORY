"""Contour plots in the BOUNDED xi-coordinate (Sigma_hat, delta_hat) plane --
the natural uniform xi-grid. Same FP data but axes are xi_Sigma and xi_delta
in (-1, +1) rather than physical Sigma, delta (which are non-uniform via tanh).

This makes the boundary structure (xi = +-1 -> P=0 or 1) visible at the panel
edges, and shows the contour scan / kernel evaluation in its native coords.
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta')
OUT = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/plots')
TAU = 2.0; TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0

def grid_for_G(G):
    G_FULL = G + 2
    dxi = 2.0 / (G_FULL - 1)
    xi_inner = np.linspace(-1+dxi, 1-dxi, G)
    return xi_inner, TOT_u*np.arctanh(xi_inner), TOT_S*np.arctanh(xi_inner), TOT_d*np.arctanh(xi_inner)

def find_P_files(G):
    out = {}
    if G == 10:
        prefix = SD / 'flint_v5_gamma'
    else:
        prefix = SD / f'flint_v5_G{G}_gamma'
    for f in sorted(glob.glob(str(prefix) + '*_P.npy')):
        stem = Path(f).stem
        try:
            tail = stem.split('gamma')[-1]
            g = float(tail.split('_')[0])
            out[g] = np.load(f)
        except Exception:
            continue
    return out

def make_xi_grid(Pdict, gammas_list, xi_inner, u_phys, S_phys, d_phys, xi_u1_target, fname, title_suffix):
    """P(xi_Sigma, xi_delta) at fixed xi_u1.  Solid black = price contour;
    dashed lime = FR P=0.5 (where Sigma=-u_1, i.e. xi_Sigma=tanh(-u_1/TOT_S))."""
    n = min(9, len(gammas_list))
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=140)
    axes = axes.flatten()
    levels = np.linspace(0.02, 0.98, 17)
    cmap = plt.cm.RdBu_r
    last_cs = None
    i_u = int(np.argmin(np.abs(xi_inner - xi_u1_target)))
    u_actual = u_phys[i_u]
    xi_u_actual = xi_inner[i_u]
    for idx in range(9):
        ax = axes[idx]
        if idx >= n:
            ax.axis('off'); continue
        g = gammas_list[idx]
        slc = Pdict[g][i_u, :, :]   # (xi_S, xi_d)
        cs = ax.contourf(xi_inner, xi_inner, slc, levels=levels, cmap=cmap, extend='both')
        ax.contour(xi_inner, xi_inner, slc, levels=[0.5], colors='k', linewidths=1.2)
        # FR overlay: P_FR(u_1, Sigma_hat, delta_hat) = sigma(tau*(u_1 + Sigma(xi_S)))
        XS, XD = np.meshgrid(xi_inner, xi_inner, indexing='ij')
        S_phys_grid = TOT_S * np.arctanh(np.clip(XS, -0.999999, 0.999999))
        FR_p = 1/(1+np.exp(-TAU*(u_actual + S_phys_grid)))
        ax.contour(xi_inner, xi_inner, FR_p, levels=[0.5], colors='lime', linewidths=1.2, linestyles='--')
        ax.set_title(f'γ = {g:g}  ({title_suffix}, ξ_u₁={xi_u_actual:+.2f} ⇒ u₁≈{u_actual:.2f})', fontsize=10)
        ax.set_xlabel('ξ_Σ (Σ̂)'); ax.set_ylabel('ξ_δ (δ̂)')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        last_cs = cs
    if last_cs is not None:
        fig.colorbar(last_cs, ax=axes, shrink=0.85, location='right', label='P')
    plt.suptitle(f'σ-δ V5: P(ξ_Σ, ξ_δ) at fixed ξ_u₁={xi_u_actual:+.2f}. '
                 f'Solid black = price P=0.5; dashed lime = FR P=0.5. ({title_suffix})',
                 weight='bold', fontsize=11)
    plt.savefig(OUT/fname, dpi=140, bbox_inches='tight'); plt.close()
    print(f'wrote {fname}')

def make_xi_logit_grid(Pdict, gammas_list, xi_inner, u_phys, S_phys, d_phys, xi_u1_target, fname, title_suffix):
    """logit(P)(xi_Sigma, xi_delta) at fixed xi_u1."""
    n = min(9, len(gammas_list))
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=140)
    axes = axes.flatten()
    levels = np.linspace(-8, 8, 17)
    cmap = plt.cm.RdBu_r
    last_cs = None
    i_u = int(np.argmin(np.abs(xi_inner - xi_u1_target)))
    u_actual = u_phys[i_u]; xi_u_actual = xi_inner[i_u]
    XS, XD = np.meshgrid(xi_inner, xi_inner, indexing='ij')
    S_phys_grid = TOT_S * np.arctanh(np.clip(XS, -0.999999, 0.999999))
    for idx in range(9):
        ax = axes[idx]
        if idx >= n:
            ax.axis('off'); continue
        g = gammas_list[idx]
        slc = Pdict[g][i_u, :, :]
        Pc = np.clip(slc, 1e-9, 1-1e-9)
        lp = np.log(Pc/(1-Pc))
        cs = ax.contourf(xi_inner, xi_inner, lp, levels=levels, cmap=cmap, extend='both')
        # FP iso-logit
        ax.contour(xi_inner, xi_inner, lp, levels=[-4, -2, 0, 2, 4], colors='k', linewidths=0.8)
        # FR iso-logit (perfectly δ̂-flat -> horizontal lines in (xi_S, xi_d))
        lp_FR = TAU*(u_actual + S_phys_grid)
        ax.contour(xi_inner, xi_inner, lp_FR, levels=[-4, -2, 0, 2, 4], colors='lime', linewidths=0.8, linestyles='--', alpha=0.85)
        ax.set_title(f'γ = {g:g}  (ξ_u₁={xi_u_actual:+.2f})', fontsize=10)
        ax.set_xlabel('ξ_Σ'); ax.set_ylabel('ξ_δ')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        last_cs = cs
    if last_cs is not None:
        fig.colorbar(last_cs, ax=axes, shrink=0.85, location='right', label='logit(P)')
    plt.suptitle(f'σ-δ V5: logit(P) in (ξ_Σ, ξ_δ) at ξ_u₁={xi_u_actual:+.2f}. Solid black = FP iso-logit; dashed lime = FR (δ̂-flat).',
                 weight='bold', fontsize=11)
    plt.savefig(OUT/fname, dpi=140, bbox_inches='tight'); plt.close()
    print(f'wrote {fname}')

P10 = find_P_files(10); g10 = sorted(P10.keys())
P15 = find_P_files(15); g15 = sorted(P15.keys())

if g10:
    xi10, u10, S10, d10 = grid_for_G(10)
    for xi_t, tag in [(0.0, 'center'), (-0.7, 'lowxi_u1'), (+0.7, 'highxi_u1')]:
        make_xi_grid(P10, g10, xi10, u10, S10, d10, xi_t,
                     f'v5_contours_xiSxid_{tag}_G10.png', 'G=10')
    make_xi_logit_grid(P10, g10, xi10, u10, S10, d10, 0.0,
                       'v5_logitP_xiSxid_center_G10.png', 'G=10')

if g15:
    xi15, u15, S15, d15 = grid_for_G(15)
    for xi_t, tag in [(0.0, 'center'), (-0.7, 'lowxi_u1'), (+0.7, 'highxi_u1')]:
        make_xi_grid(P15, g15, xi15, u15, S15, d15, xi_t,
                     f'v5_contours_xiSxid_{tag}_G15.png', 'G=15')
    make_xi_logit_grid(P15, g15, xi15, u15, S15, d15, 0.0,
                       'v5_logitP_xiSxid_center_G15.png', 'G=15')

print('DONE')
