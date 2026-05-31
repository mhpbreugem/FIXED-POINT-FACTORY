"""OVERNIGHT: solve CRRA at γ=0.1, τ=2 with the strict h=0, change-of-variables,
PCHIP architecture at G ∈ {9, 11, 13}, using Anderson(m=8) for faster convergence.
Then do POINT-BY-POINT comparison vs the u-grid kernel-smooth PR FP at G=17
(P_g0.1_t2.0.npy in k3_coarea_sweep) using trilinear interp to map u-coords.

Saves at each G:
  crra_xi_pchip_OVR_G{G}.npy        — final FP on the ξ-grid
  crra_xi_pchip_OVR_G{G}.json       — Anderson history + metrics
  comparison_G{G}.csv               — per-cell P_ours, P_ugrid_interp, residual
  comparison_G{G}_summary.json      — stats of the comparison

The "code in which the line smoothly changes with a change of variables, no h"
applied to CRRA, with strict A/B comparison to the already-pushed PR-promising
u-grid co-area FP.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

from crra_xi_pchip import (
    phi_crra_xi, crra_clear_proper, metrics, f_signal, sigmoid,
    TAU, GAMMA, EPS_PRICE
)

# u-grid kernel PR FP (already pushed)
P_ugrid = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
G_UG = P_ugrid.shape[0]
UMAX_UG = 4.0
ui_UG = np.linspace(-UMAX_UG, UMAX_UG, G_UG)
interp_ugrid = RegularGridInterpolator((ui_UG, ui_UG, ui_UG), P_ugrid,
                                          bounds_error=False, fill_value=None,
                                          method='linear')
print(f'Loaded u-grid PR FP: G={G_UG}, u∈[±{UMAX_UG}], P∈[{P_ugrid.min():.4f},{P_ugrid.max():.4f}]', flush=True)

# Confirm u-grid metrics
u_in = ui_UG
U1, U2, U3 = np.meshgrid(u_in, u_in, u_in, indexing='ij')
T = TAU*(U1+U2+U3); P_FR = sigmoid(T)
Pc = np.clip(P_ugrid, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel()+aa[1]
slope_ug = float(aa[0])
omr_ug = float(((y-pr)**2).mean()/max(((y-y.mean())**2).mean(),1e-30))
dfr_ug = float(np.sqrt(np.mean((P_ugrid-P_FR)**2)))
print(f'u-grid PR FP metrics: slope={slope_ug:.4f} 1-R²={omr_ug:.4f} d_FR={dfr_ug:.4f}', flush=True)
print('(This is the PR equilibrium we are comparing against)\n', flush=True)

def anderson_pchip(G_inner, gamma, tau, epsb=0.10, m_max=8, max_iter=50,
                    tol=1e-7, verbose=True):
    """Anderson(m_max)-accelerated Picard for phi_crra_xi."""
    dxi = 2*(1-epsb)/(G_inner-1)
    xi_full = np.linspace(-(1-epsb), +(1-epsb), G_inner)
    u_full = (2.0/tau) * np.arctanh(xi_full)
    dudxi = (2.0/tau) / (1.0 - xi_full**2)
    U1, U2, U3 = np.meshgrid(u_full, u_full, u_full, indexing='ij')

    # IC: proper-CRRA no-learning ansatz
    mu1 = sigmoid(tau*U1); mu2 = sigmoid(tau*U2); mu3 = sigmoid(tau*U3)
    Pf = np.empty_like(U1)
    for ii in range(G_inner):
        for jj in range(G_inner):
            for kk in range(G_inner):
                Pf[ii,jj,kk] = crra_clear_proper(mu1[ii,jj,kk], mu2[ii,jj,kk], mu3[ii,jj,kk], gamma)
    s0, om0, d0 = metrics(Pf, u_full, 0, G_inner, tau)
    print(f'IC (NL proper-CRRA): slope={s0:.4f} 1-R²={om0:.4f} d_FR={d0:.4f}', flush=True)

    x = Pf.ravel().copy()
    G_h=[]; F_h=[]; hist=[]
    best = dict(ferr=np.inf, x=x.copy())
    for it in range(1, max_iter+1):
        ts = time.time()
        P_in = x.reshape((G_inner,)*3)
        P_out = phi_crra_xi(P_in, xi_full, u_full, dudxi, 0, G_inner, tau, gamma)
        g = P_out.ravel()
        f = g - x
        ferr = float(np.max(np.abs(f)))
        s, om, d = metrics(P_out, u_full, 0, G_inner, tau)
        rec = dict(it=it, ferr=ferr, slope=s, omr=om, d_FR=d, dt=time.time()-ts)
        hist.append(rec)
        if ferr < best['ferr']:
            best = dict(ferr=ferr, x=g.copy(), slope=s, omr=om, d_FR=d, it=it)
        if verbose:
            print(f'  it {it:3d} ferr={ferr:.3e} slope={s:.4f} 1-R²={om:.4f} d_FR={d:.4f} ({rec["dt"]:.1f}s)', flush=True)
        if ferr < tol:
            print(f'  CONVERGED', flush=True); break
        G_h.append(g.copy()); F_h.append(f.copy())
        if len(F_h) > m_max + 1:
            G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0:
            x_new = g
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x_new = g - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g
        x = np.clip(x_new, 1e-12, 1-1e-12)
    P_final = best['x'].reshape((G_inner,)*3)
    return P_final, hist, dict(G=G_inner, gamma=gamma, tau=tau, epsb=epsb,
                                 ferr=best['ferr'], slope=best['slope'],
                                 omr=best['omr'], d_FR=best['d_FR'],
                                 best_iter=best['it'], total_iters=len(hist),
                                 u_full=u_full.tolist(), xi_full=xi_full.tolist())

def compare_pointwise(P_ours, u_full, summary, tag):
    """Map ξ-grid cell to u-coords, interp u-grid PR FP there, compare cell-by-cell."""
    G_inner = P_ours.shape[0]
    rows = []
    for i in range(G_inner):
        for j in range(G_inner):
            for k in range(G_inner):
                u1 = u_full[i]; u2 = u_full[j]; u3 = u_full[k]
                # clamp to u-grid range
                u1c = max(ui_UG[0], min(ui_UG[-1], u1))
                u2c = max(ui_UG[0], min(ui_UG[-1], u2))
                u3c = max(ui_UG[0], min(ui_UG[-1], u3))
                p_ours = float(P_ours[i,j,k])
                p_pr   = float(interp_ugrid(np.array([u1c, u2c, u3c]))[0])
                r = p_ours - p_pr
                rows.append(dict(i=i, j=j, k=k, u1=u1, u2=u2, u3=u3,
                                  P_ours=p_ours, P_ugrid_PR=p_pr,
                                  resid=r, abs_resid=abs(r)))
    # Save CSV
    import csv
    csv_path = os.path.join(HERE, f'comparison_G{G_inner}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # Stats
    arr_r = np.array([r['resid'] for r in rows])
    arr_a = np.array([r['abs_resid'] for r in rows])
    arr_o = np.array([r['P_ours'] for r in rows])
    arr_p = np.array([r['P_ugrid_PR'] for r in rows])
    summ = dict(
        n=len(rows),
        max_abs=float(arr_a.max()),
        mean_abs=float(arr_a.mean()),
        rms=float(np.sqrt((arr_r**2).mean())),
        signed_mean=float(arr_r.mean()),
        ours_range=[float(arr_o.min()), float(arr_o.max())],
        pr_range=[float(arr_p.min()), float(arr_p.max())],
        worst_cell=sorted(rows, key=lambda r:-r['abs_resid'])[0],
        best_cell=sorted(rows, key=lambda r:r['abs_resid'])[0],
        operator_summary=summary,
    )
    json.dump(summ, open(os.path.join(HERE, f'comparison_G{G_inner}_summary.json'),'w'),
              indent=2, default=str)
    print(f'\n=== POINT-BY-POINT COMPARISON G={G_inner} ===', flush=True)
    print(f'  n_cells = {len(rows)}', flush=True)
    print(f'  max |P_ours - P_ugrid_PR| = {arr_a.max():.4f}', flush=True)
    print(f'  mean |.| = {arr_a.mean():.4f}    rms = {np.sqrt((arr_r**2).mean()):.4f}', flush=True)
    print(f'  signed mean = {arr_r.mean():+.4f}', flush=True)
    print(f'  P_ours range = [{arr_o.min():.4f}, {arr_o.max():.4f}]', flush=True)
    print(f'  P_ugrid_PR range = [{arr_p.min():.4f}, {arr_p.max():.4f}]', flush=True)
    print(f'  worst cell: ({summ["worst_cell"]["i"]},{summ["worst_cell"]["j"]},{summ["worst_cell"]["k"]}) '
          f'u=({summ["worst_cell"]["u1"]:+.2f},{summ["worst_cell"]["u2"]:+.2f},{summ["worst_cell"]["u3"]:+.2f}) '
          f'P_ours={summ["worst_cell"]["P_ours"]:.4f} P_PR={summ["worst_cell"]["P_ugrid_PR"]:.4f} '
          f'r={summ["worst_cell"]["resid"]:+.4f}', flush=True)
    return summ

if __name__ == '__main__':
    LOG = os.path.join(HERE, 'overnight_crra_xi_vs_ugrid.log')
    open(LOG, 'w').close()

    all_summaries = []
    t_total = time.time()
    for G in [9, 11, 13]:
        print(f'\n\n======== G = {G}  (γ={GAMMA}, τ={TAU}, h=0, PCHIP, ξ-stretch) ========', flush=True)
        try:
            P, hist, summary = anderson_pchip(G, GAMMA, TAU, epsb=0.10,
                                                m_max=8, max_iter=40, tol=1e-7)
        except Exception as e:
            print(f'EXCEPTION at G={G}: {e}', flush=True)
            continue
        np.save(os.path.join(HERE, f'crra_xi_pchip_OVR_G{G}.npy'), P)
        json.dump({'summary':summary, 'history':hist},
                  open(os.path.join(HERE, f'crra_xi_pchip_OVR_G{G}.json'),'w'),
                  indent=2, default=str)
        u_full = np.array(summary['u_full'])
        cmp_summ = compare_pointwise(P, u_full, summary, f'G{G}')
        all_summaries.append(dict(G=G, operator=summary, comparison=cmp_summ))
        json.dump({'all':all_summaries, 'ugrid_target':dict(slope=slope_ug, omr=omr_ug, d_FR=dfr_ug)},
                  open(os.path.join(HERE, 'overnight_all_summaries.json'),'w'),
                  indent=2, default=str)
        # auto-push
        os.system(f'cd /home/user/FIXED-POINT-FACTORY && git add -A projects/REZN/solver_code/sigma_delta/crra_xi_pchip_OVR_G{G}.* projects/REZN/solver_code/sigma_delta/comparison_G{G}* projects/REZN/solver_code/sigma_delta/overnight_all_summaries.json projects/REZN/solver_code/sigma_delta/overnight_crra_xi_vs_ugrid.log && git commit -m "overnight crra_xi_pchip G={G}: slope={summary["slope"]:.3f} d_FR={summary["d_FR"]:.3f} ferr={summary["ferr"]:.2e}; max|P-P_ugrid_PR|={cmp_summ["max_abs"]:.3f}" 2>&1 | tail -2 && git push -u origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2')
        print(f'\n--- G={G} done; cumulative {(time.time()-t_total)/60:.1f} min ---', flush=True)
    print(f'\nOVERNIGHT DONE in {(time.time()-t_total)/60:.1f} min', flush=True)
