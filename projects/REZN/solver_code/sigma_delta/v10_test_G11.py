"""V10 = numba-compiled V9. Test at G_inner=11. CARA + CRRA γ=0.1 + plots."""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from v10_hfree_gl_numba import (phi_v10, set_boundary, crra_clear_nb,
                                  XI_GL, W_GL, NQ,
                                  TAU, TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, EPS_PRICE)

G_FULL = 13; INNER_LO, INNER_HI = 1, G_FULL - 1; G_INNER = INNER_HI - INNER_LO
xi_arr = np.linspace(-1.0, 1.0, G_FULL); dxi = float(xi_arr[1]-xi_arr[0])
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe)
S_arr = TOT_S * np.arctanh(safe)
d_arr = TOT_d * np.arctanh(safe)
u_in = u_arr[INNER_LO:INNER_HI]; S_in = S_arr[INNER_LO:INNER_HI]; d_in = d_arr[INNER_LO:INNER_HI]
xi_in = xi_arr[INNER_LO:INNER_HI]
U1m, SIm, DEm = np.meshgrid(u_in, S_in, d_in, indexing='ij')
Sfull = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm); Tstar = TAU*Sfull
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(Tstar)
def fa(u,vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
Wd = 0.5*(fa(U1m,-0.5)*fa(0.5*(SIm+DEm),-0.5)*fa(0.5*(SIm-DEm),-0.5)+
          fa(U1m,+0.5)*fa(0.5*(SIm+DEm),+0.5)*fa(0.5*(SIm-DEm),+0.5))
Wd /= Wd.sum()
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl_x=Tstar.flatten(); fl_lp=lp.flatten(); fl_w=Wd.flatten()
    slope, intc = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred=slope*fl_x+intc
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    d_FR_w = float(np.sqrt(np.sum((P - P_FR_in)**2 * Wd)))
    return slope, intc, vr/vt if vt>0 else float('nan'), d_FR_w

def run_picard(P_ic, clearing_cara, gamma_val, tag, max_iter=10):
    P = np.zeros((G_FULL,)*3); P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_ic
    P = set_boundary(P)
    s0, i0, o0, d0 = metrics(P_ic)
    print(f'\n--- {tag}  IC: slope={s0:.5f} 1-R²={o0:.3e} d_FR_w={d0:.3e}', flush=True)
    omega = 1.0; res_prev = 1e100
    for it in range(1, max_iter+1):
        t = time.time()
        P_phi = phi_v10(P, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
                         XI_GL, W_GL, NQ, gamma_val, clearing_cara,
                         TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
        P_damp = (1-omega)*P + omega*P_phi
        P_damp = set_boundary(P_damp)
        res = float(np.max(np.abs((P_damp - P)[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI])))
        if it > 3:
            if res > res_prev*0.99: omega = max(omega*0.7, 0.05)
            elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
        P = P_damp; res_prev = res
        inner = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI]
        s, i, o, d = metrics(inner)
        print(f'  it {it}  ω={omega:.3f}  ferr={res:.3e}  slope={s:.5f}  1-R²={o:.3e}  d_FR_w={d:.3e}  ({time.time()-t:.1f}s)', flush=True)
    return inner

# JIT warmup
print('JIT warmup...', flush=True)
t0 = time.time()
P_warm = np.zeros((G_FULL,)*3); P_warm[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_warm = set_boundary(P_warm)
_ = phi_v10(P_warm, INNER_LO, INNER_HI, xi_arr, dxi, u_arr, S_arr, d_arr,
              XI_GL, W_GL, NQ, 0.1, True,
              TOT_u, TOT_S, TOT_d, VM0, VM1, COEF, TAU, EPS_PRICE)
print(f'  JIT done {time.time()-t0:.0f}s', flush=True)

# CARA from FR-IC
inner_cara = run_picard(P_FR_in.copy(), True, 0.1, 'CARA from FR-IC', max_iter=8)
np.save(os.path.join(HERE, 'v10_G11_cara_FR_IC_P.npy'), inner_cara)

# CRRA γ=0.1 from NL-IC
mu1 = sg(TAU*U1m); mu2 = sg(TAU*0.5*(SIm+DEm)); mu3 = sg(TAU*0.5*(SIm-DEm))
P_NL = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL[i,j,k] = crra_clear_nb(mu1[i,j,k], mu2[i,j,k], mu3[i,j,k], 0.1, 120)
inner_crra = run_picard(P_NL, False, 0.1, 'CRRA γ=0.1 from NL-IC', max_iter=12)
np.save(os.path.join(HERE, 'v10_G11_crra_g0p1_NL_IC_P.npy'), inner_crra)

# === Plot contours ===
i_mid = G_INNER // 2
print(f'\nPlotting contours at xi_u_1 ≈ {xi_in[i_mid]:.3f}', flush=True)
levels = np.linspace(0.02, 0.98, 17)
cmap = plt.cm.RdBu_r
Sphys, Dphys = np.meshgrid(S_in, d_in, indexing='ij')

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
for ax, (P_inner, runtag) in zip(axes, [(inner_cara, 'CARA'), (inner_crra, 'CRRA γ=0.1')]):
    slc = P_inner[i_mid, :, :]
    cs = ax.contourf(xi_in, xi_in, slc.T, levels=levels, cmap=cmap, extend='both')
    ax.contour(xi_in, xi_in, slc.T, levels=[0.5], colors='k', linewidths=1.2)
    FR_p = 1/(1+np.exp(-TAU*(u_in[i_mid] + Sphys)))
    ax.contour(xi_in, xi_in, FR_p.T, levels=[0.5], colors='lime', linewidths=1.2, linestyles='--')
    ax.set_xlabel('ξ_Σ'); ax.set_ylabel('ξ_δ')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
    ax.set_title(f'V10 {runtag}, ξ_u₁={xi_in[i_mid]:+.3f}\n(black=FP, lime=FR)', fontsize=11)
    plt.colorbar(cs, ax=ax, shrink=0.85, label='P')
plt.suptitle(f'V10 (numba h-free smooth + GL) σ-δ FP at G_inner={G_INNER}', weight='bold', fontsize=12)
plt.tight_layout()
out = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/plots/v10_G11_contours_xi_plane.png'
plt.savefig(out, dpi=140, bbox_inches='tight'); plt.close()
print(f'wrote {out}', flush=True)

# 3-slice grid
fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=140)
for col, i_show in enumerate([1, i_mid, G_INNER-2]):
    for row, (P_in, runtag) in enumerate(zip([inner_cara, inner_crra], ['CARA', 'CRRA γ=0.1'])):
        ax = axes[row, col]
        slc = P_in[i_show, :, :]
        cs = ax.contourf(xi_in, xi_in, slc.T, levels=levels, cmap=cmap, extend='both')
        ax.contour(xi_in, xi_in, slc.T, levels=[0.5], colors='k', linewidths=1.2)
        FR_p = 1/(1+np.exp(-TAU*(u_in[i_show] + Sphys)))
        ax.contour(xi_in, xi_in, FR_p.T, levels=[0.5], colors='lime', linewidths=1.2, linestyles='--')
        ax.set_xlabel('ξ_Σ'); ax.set_ylabel('ξ_δ')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        ax.set_title(f'{runtag}  ξ_u₁={xi_in[i_show]:+.2f}', fontsize=10)
plt.suptitle(f'V10 σ-δ FP at G_inner={G_INNER}: P(ξ_Σ, ξ_δ) at 3 ξ_u₁ slices', weight='bold', fontsize=12)
plt.tight_layout()
out2 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/plots/v10_G11_contours_xi_3slices.png'
plt.savefig(out2, dpi=140, bbox_inches='tight'); plt.close()
print(f'wrote {out2}', flush=True)
print('DONE.', flush=True)
