"""Additional (Sigma, delta) contour plots -- the natural agent 1 slice
at fixed u_1. FR predicts these slices to be exactly delta-flat (P depends
only on u_1+Sigma). Any delta-curvature reveals the Jensen-gap signature.
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
    return TOT_u * np.arctanh(xi_inner), TOT_S * np.arctanh(xi_inner), TOT_d * np.arctanh(xi_inner)

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

def make_Sd_grid(Pdict, gammas_list, u, S, d, u1_target, fname, title_suffix):
    n = min(9, len(gammas_list))
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=140)
    axes = axes.flatten()
    levels = np.linspace(0.02, 0.98, 17)
    cmap = plt.cm.RdBu_r
    last_cs = None
    i_u = int(np.argmin(np.abs(u - u1_target)))
    u_actual = u[i_u]
    for idx in range(9):
        ax = axes[idx]
        if idx >= n:
            ax.axis('off'); continue
        g = gammas_list[idx]
        slc = Pdict[g][i_u, :, :]  # P(Sigma, delta) at fixed u_1
        cs = ax.contourf(d, S, slc, levels=levels, cmap=cmap, extend='both')
        ax.contour(d, S, slc, levels=[0.5], colors='k', linewidths=1.2)
        # FR overlay: line Sigma = -u_1 (where u_1+Sigma=0 i.e. P=0.5)
        Dg, Sg = np.meshgrid(d, S)
        FR_p = 1/(1+np.exp(-TAU*(u_actual + Sg)))
        ax.contour(d, S, FR_p, levels=[0.5], colors='lime', linewidths=1.2, linestyles='--')
        ax.set_title(f'γ = {g:g}  ({title_suffix}, u₁≈{u_actual:.2f})', fontsize=10)
        ax.set_xlabel('δ'); ax.set_ylabel('Σ')
        last_cs = cs
    if last_cs is not None:
        fig.colorbar(last_cs, ax=axes, shrink=0.85, location='right', label='P')
    plt.suptitle(f'σ-δ V5: P(Σ, δ) at fixed u₁≈{u_actual:.2f}. '
                 'Solid black = price P=0.5; dashed lime = FR P=0.5 (horizontal Σ=−u₁ line, δ-flat).',
                 weight='bold', fontsize=11)
    plt.savefig(OUT/fname, dpi=140, bbox_inches='tight'); plt.close()
    print(f'wrote {fname}')

# G=10 panel grids at 3 u_1 values
P10 = find_P_files(10); gammas10 = sorted(P10.keys())
P15 = find_P_files(15); gammas15 = sorted(P15.keys())

if gammas10:
    u10, S10, d10 = grid_for_G(10)
    for u1_t, tag in [(0.0, 'center'), (-1.5, 'low_u1'), (+1.5, 'high_u1')]:
        make_Sd_grid(P10, gammas10, u10, S10, d10, u1_t,
                     f'v5_contours_Sd_at_u1_{tag}_G10.png', 'G=10')

if gammas15:
    u15, S15, d15 = grid_for_G(15)
    for u1_t, tag in [(0.0, 'center'), (-1.5, 'low_u1'), (+1.5, 'high_u1')]:
        make_Sd_grid(P15, gammas15, u15, S15, d15, u1_t,
                     f'v5_contours_Sd_at_u1_{tag}_G15.png', 'G=15')

# Also: per-γ overlay of P(Σ, δ) at center, showing how the contour curves with γ
fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=140)
axes = axes.flatten()
if gammas10:
    i_u = int(np.argmin(np.abs(u10)))
    for idx, g in enumerate(gammas10[:9]):
        ax = axes[idx]
        slc = P10[g][i_u, :, :]
        # Show logit P
        Pc = np.clip(slc, 1e-9, 1-1e-9)
        lp = np.log(Pc/(1-Pc))
        cs = ax.contourf(d10, S10, lp, levels=np.linspace(-8, 8, 17), cmap='RdBu_r', extend='both')
        # logit FR = tau*(u_1+Sigma): horizontal lines at Sigma values
        Dg, Sg = np.meshgrid(d10, S10)
        lp_FR = TAU*(u10[i_u] + Sg)
        ax.contour(d10, S10, lp_FR, levels=[-4, -2, 0, 2, 4], colors='lime', linewidths=0.8, linestyles='--', alpha=0.8)
        # FP logit contour
        ax.contour(d10, S10, lp, levels=[-4, -2, 0, 2, 4], colors='k', linewidths=0.8)
        ax.set_title(f'γ = {g:g}  (logit P, u₁=0)', fontsize=10)
        ax.set_xlabel('δ'); ax.set_ylabel('Σ')
    fig.colorbar(cs, ax=axes, shrink=0.85, location='right', label='logit(P)')
    plt.suptitle('V5 σ-δ G=10: logit(P) heat-maps at u₁=0. Solid black = FP iso-logit; dashed lime = FR iso-logit (horizontal, δ-flat).',
                 weight='bold', fontsize=11)
    plt.savefig(OUT/'v5_logitP_Sd_at_center_G10.png', dpi=140, bbox_inches='tight'); plt.close()
    print('wrote v5_logitP_Sd_at_center_G10.png')

print('DONE')
