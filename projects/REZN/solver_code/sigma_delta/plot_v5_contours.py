"""Contour plots of the V5 sigma-delta gamma-sweep price surfaces.

Loads the saved P arrays (G_inner=10 and G_inner=15 where available) for each
gamma and produces:

  1. v5_contours_uS_at_delta0.png
     P(u_1, Sigma) at delta=0 (the FR-relevant slice) -- 3x3 grid one per
     gamma. Includes FR reference contours.
  2. v5_contours_uS_at_deltamax.png
     Same but at delta = +max (tests delta-dependence; FR is delta-flat).
  3. v5_delta_dependence.png
     P at the center (u_1=0, Sigma=0) plotted along the delta axis -- shows
     how the FP departs from FR's delta-flatness.
  4. v5_logitP_vs_Tstar.png
     Scatter plots of logit(P) vs T* for each gamma -- shows slope going
     from ~0.5 (low gamma) to ~0.6 (G=10 ceiling) or higher (G=15).
  5. v5_diagonal_slice.png
     P along the antidiagonal u_1+Sigma constant (signal-density weighted)
     for each gamma -- effectively the "informativeness curve".
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta')
OUT = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/plots')
OUT.mkdir(exist_ok=True)

TAU = 2.0
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0

def grid_for_G(G):
    """Return u, S, d physical axes for G_inner=G (G_FULL=G+2, inner xi grid)."""
    G_FULL = G + 2
    dxi = 2.0 / (G_FULL - 1)
    xi_inner = np.linspace(-1+dxi, 1-dxi, G)
    u = TOT_u * np.arctanh(xi_inner)
    S = TOT_S * np.arctanh(xi_inner)
    d = TOT_d * np.arctanh(xi_inner)
    return u, S, d

def find_P_files(G):
    """Find saved {gamma: P_array} for given G."""
    out = {}
    if G == 10:
        prefix = SD / 'flint_v5_gamma'
    else:
        prefix = SD / f'flint_v5_G{G}_gamma'
    for f in sorted(glob.glob(str(prefix) + '*_P.npy')):
        # parse gamma from filename
        stem = Path(f).stem  # e.g. flint_v5_G15_gamma0.05_G10_P or flint_v5_gamma0.1_G10_P
        # try to extract gamma after 'gamma'
        try:
            tail = stem.split('gamma')[-1]
            g = float(tail.split('_')[0])
            out[g] = np.load(f)
        except Exception:
            continue
    return out

P10 = find_P_files(10)
P15 = find_P_files(15)
gammas10 = sorted(P10.keys())
gammas15 = sorted(P15.keys())
print(f'G=10 gammas: {gammas10}')
print(f'G=15 gammas: {gammas15}')

# ===================================================================
# 1. P(u_1, Sigma) at delta=0 heatmap, 3x3 grid per gamma
# ===================================================================
def heat_at_delta(P, u, S, d, delta_target=0.0):
    """Return P(:, :, k) at the k closest to delta_target."""
    k = int(np.argmin(np.abs(d - delta_target)))
    return P[:, :, k], d[k]

def make_heat_grid(Pdict, gammas_list, u, S, d, delta_target, fname, title_suffix):
    n = min(9, len(gammas_list))
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=140)
    axes = axes.flatten()
    levels = np.linspace(0.02, 0.98, 17)
    cmap = plt.cm.RdBu_r
    last_cs = None
    for idx in range(9):
        ax = axes[idx]
        if idx >= n:
            ax.axis('off'); continue
        g = gammas_list[idx]
        slc, d_actual = heat_at_delta(Pdict[g], u, S, d, delta_target)
        cs = ax.contourf(S, u, slc, levels=levels, cmap=cmap, extend='both')
        ax.contour(S, u, slc, levels=[0.5], colors='k', linewidths=1.2)
        # FR overlay: line u_1+Sigma=0
        Sg, Ug = np.meshgrid(S, u)
        FR_p = 1/(1+np.exp(-TAU*(Ug+Sg)))
        ax.contour(S, u, FR_p, levels=[0.5], colors='lime', linewidths=1.2, linestyles='--')
        ax.set_title(f'γ = {g:g}  ({title_suffix}, δ≈{d_actual:.2f})', fontsize=10)
        ax.set_xlabel('Σ'); ax.set_ylabel('u₁')
        last_cs = cs
    if last_cs is not None:
        cbar = fig.colorbar(last_cs, ax=axes, shrink=0.85, location='right', label='P')
    plt.suptitle(f'σ-δ V5 kernel co-area: P(u₁, Σ) heat-maps at δ≈{delta_target}. '
                 f'Solid black = price contour P=0.5; dashed lime = FR P=0.5.', weight='bold', fontsize=11)
    plt.savefig(OUT/fname, dpi=140, bbox_inches='tight'); plt.close()
    print(f'wrote {fname}')

if gammas10:
    u10, S10, d10 = grid_for_G(10)
    make_heat_grid(P10, gammas10, u10, S10, d10, 0.0,
                   'v5_contours_uS_at_delta0_G10.png', 'G=10')
    make_heat_grid(P10, gammas10, u10, S10, d10, d10[-1],
                   'v5_contours_uS_at_deltamax_G10.png', 'G=10')

if gammas15:
    u15, S15, d15 = grid_for_G(15)
    make_heat_grid(P15, gammas15, u15, S15, d15, 0.0,
                   'v5_contours_uS_at_delta0_G15.png', 'G=15')

# ===================================================================
# 2. delta-dependence: P at center (u_1=0, Sigma=0) along delta axis
# ===================================================================
def delta_curve(P, u, S, d, u1_target=0.0, S_target=0.0):
    i = int(np.argmin(np.abs(u - u1_target)))
    j = int(np.argmin(np.abs(S - S_target)))
    return d, P[i, j, :]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
cmap = plt.cm.viridis
n = len(gammas10) if gammas10 else 1
for idx, g in enumerate(gammas10):
    c = cmap(idx / max(n-1, 1))
    d_axis, p_curve = delta_curve(P10[g], u10, S10, d10)
    axes[0].plot(d_axis, p_curve, '-', color=c, lw=1.5, label=f'γ={g:g}')
axes[0].axhline(0.5, color='k', ls='--', alpha=0.5, label='FR (δ-flat at center)')
axes[0].set_xlabel('δ'); axes[0].set_ylabel('P(u₁=0, Σ=0, δ)')
axes[0].set_title('δ-dependence at center (G=10): FR predicts FLAT')
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(ls=':')

if gammas15:
    n15 = len(gammas15)
    for idx, g in enumerate(gammas15):
        c = cmap(idx / max(n15-1, 1))
        d_axis, p_curve = delta_curve(P15[g], u15, S15, d15)
        axes[1].plot(d_axis, p_curve, '-', color=c, lw=1.5, label=f'γ={g:g}')
    axes[1].axhline(0.5, color='k', ls='--', alpha=0.5, label='FR (δ-flat)')
    axes[1].set_xlabel('δ'); axes[1].set_ylabel('P(u₁=0, Σ=0, δ)')
    axes[1].set_title('δ-dependence at center (G=15)')
    axes[1].legend(fontsize=8); axes[1].grid(ls=':')
else:
    axes[1].text(0.5, 0.5, 'G=15 data not ready yet', ha='center', va='center')
    axes[1].axis('off')

plt.suptitle('δ-curvature of FP: a Jensen-gap signature (FR is exactly δ-flat)', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'v5_delta_dependence.png', dpi=140, bbox_inches='tight'); plt.close()
print('wrote v5_delta_dependence.png')

# ===================================================================
# 3. logit(P) vs T* scatter for several gammas (G=10)
# ===================================================================
def logit_vs_Tstar(P, u, S, d):
    U1, SI, DE = np.meshgrid(u, S, d, indexing='ij')
    U2 = 0.5*(SI+DE); U3 = 0.5*(SI-DE)
    Tstar = TAU*(U1 + U2 + U3)
    Pc = np.clip(P, 1e-30, 1-1e-30)
    lp = np.log(Pc/(1-Pc))
    # weights
    def fa(uu, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(uu-vm)**2)
    Wd = 0.5*(fa(U1,-0.5)*fa(U2,-0.5)*fa(U3,-0.5) + fa(U1,0.5)*fa(U2,0.5)*fa(U3,0.5))
    Wd /= Wd.sum()
    return Tstar.flatten(), lp.flatten(), Wd.flatten()

fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
n = len(gammas10)
for idx, g in enumerate(gammas10):
    c = cmap(idx / max(n-1, 1))
    Tf, lpf, wf = logit_vs_Tstar(P10[g], u10, S10, d10)
    slope, intc = np.polyfit(Tf, lpf, 1, w=np.sqrt(wf))
    # subsample for plot
    sort = np.argsort(Tf); Tf_s = Tf[sort][::25]; lpf_s = lpf[sort][::25]
    ax.scatter(Tf_s, lpf_s, color=c, s=10, alpha=0.5)
    # regression line
    Tline = np.linspace(Tf.min(), Tf.max(), 50)
    ax.plot(Tline, slope*Tline+intc, '-', color=c, lw=2, label=f'γ={g:g}  slope={slope:.3f}')
ax.plot([-15, 15], [-15, 15], 'k--', alpha=0.7, lw=1.5, label='FR slope=1')
ax.set_xlabel('T* = τ(u₁+u₂+u₃)'); ax.set_ylabel('logit(P)')
ax.set_title('logit(P) vs T* (G=10): slope rises with γ, approaches FR=1')
ax.legend(fontsize=8, ncol=2); ax.grid(ls=':')
ax.set_xlim(-15, 15); ax.set_ylim(-12, 12)
plt.tight_layout(); plt.savefig(OUT/'v5_logitP_vs_Tstar.png', dpi=140, bbox_inches='tight'); plt.close()
print('wrote v5_logitP_vs_Tstar.png')

# ===================================================================
# 4. T* binned profile -- logit(P) vs T*, signal-weighted average per bin
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
for idx, g in enumerate(gammas10):
    c = cmap(idx / max(n-1, 1))
    Tf, lpf, wf = logit_vs_Tstar(P10[g], u10, S10, d10)
    bins = np.linspace(Tf.min(), Tf.max(), 25)
    binc = 0.5*(bins[1:]+bins[:-1])
    binmean = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        msk = (Tf >= b0) & (Tf < b1)
        if msk.sum() == 0 or wf[msk].sum() == 0: binmean.append(np.nan)
        else: binmean.append(float(np.average(lpf[msk], weights=wf[msk])))
    ax.plot(binc, binmean, 'o-', color=c, lw=2, ms=6, label=f'γ={g:g}')
ax.plot([-15, 15], [-15, 15], 'k--', alpha=0.7, lw=2, label='FR slope=1')
ax.set_xlabel('T*'); ax.set_ylabel('⟨logit(P)⟩ (signal-weighted in T* bin)')
ax.set_title('Binned logit(P) vs T* (G=10): the σ-δ V5 FP as γ varies')
ax.legend(fontsize=8, ncol=2); ax.grid(ls=':')
ax.set_xlim(-15, 15); ax.set_ylim(-10, 10)
plt.tight_layout(); plt.savefig(OUT/'v5_logitP_vs_Tstar_binned.png', dpi=140, bbox_inches='tight'); plt.close()
print('wrote v5_logitP_vs_Tstar_binned.png')

print('DONE')
