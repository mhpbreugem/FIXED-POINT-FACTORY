"""sigma-delta CARA G-sweep (float64) starting from analytic FR-ansatz.

The exact BCs (u_1, Sigma = +-inf -> P=0,1 FR limit; delta = +-inf zero-order
extrap) make FR analytically self-consistent. We test: does Picard stay near
FR as G grows? And if d_FR grows with G, in WHICH cells does the operator
err the most?

Sweep: G = 6, 8, 10, 12, 14, 16. For each:
  - run float64 sigma-delta CARA Picard from FR-ansatz with the same proper BCs
  - report d_FR, slope, 1-R^2, the cell with max |Phi(P_FR) - P_FR|

Also: per-cell residual map at G=10 vs G=15 to identify systematic errors.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from phi_sigma_delta import set_boundary, finf_interior, fsig
from phi_sigma_delta_cara import phi_sigmadelta_cara

TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; W = 1.0
MAX_ITER = 100
REPORT_EVERY = 20

def run_at_G(G, ic='FR'):
    G_FULL = G + 2
    INNER_LO, INNER_HI = 1, G + 1
    dxi = 2.0 / (G + 1)
    xi_inner = np.linspace(-1+dxi, 1-dxi, G)
    xi_full = np.concatenate([[-1.0], xi_inner, [1.0]])
    xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()
    u1_inner = TOT_u * np.arctanh(xi_inner)
    S_inner = TOT_S * np.arctanh(xi_inner)
    d_inner = TOT_d * np.arctanh(xi_inner)
    U1m, SIm, DEm = np.meshgrid(u1_inner, S_inner, d_inner, indexing='ij')
    S_full = U1m + 0.5*(SIm+DEm) + 0.5*(SIm-DEm)  # = U1m + SIm (δ-flat)
    Tstar = TAU * S_full
    P_FR_inner = 1.0 / (1.0 + np.exp(-Tstar))

    def f_arr(u, vm): return np.sqrt(TAU/(2*np.pi))*np.exp(-0.5*TAU*(u-vm)**2)
    Wd = 0.5*(f_arr(U1m,-0.5)*f_arr(0.5*(SIm+DEm),-0.5)*f_arr(0.5*(SIm-DEm),-0.5) +
              f_arr(U1m,+0.5)*f_arr(0.5*(SIm+DEm),+0.5)*f_arr(0.5*(SIm-DEm),+0.5))
    Wd = Wd / max(Wd.sum(), 1e-30)

    def dist_to_FR(Pi):
        return float(np.sqrt(np.sum((Pi - P_FR_inner)**2 * Wd)))

    if ic == 'FR':
        P_inner = P_FR_inner.copy()
    else:
        # no-learning IC
        def sg(x): return 1.0/(1.0+np.exp(-x))
        m1 = sg(TAU*U1m); m2 = sg(TAU*0.5*(SIm+DEm)); m3 = sg(TAU*0.5*(SIm-DEm))
        eps=1e-12
        l1 = np.log(np.clip(m1,eps,1-eps)/(1-np.clip(m1,eps,1-eps)))
        l2 = np.log(np.clip(m2,eps,1-eps)/(1-np.clip(m2,eps,1-eps)))
        l3 = np.log(np.clip(m3,eps,1-eps)/(1-np.clip(m3,eps,1-eps)))
        P_inner = sg((l1+l2+l3)/3)

    P = np.zeros((G_FULL,)*3)
    P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_inner
    P = set_boundary(P, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)

    # one step BEFORE Picard: residual at P_FR
    P_one = phi_sigmadelta_cara(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, W,
                                 INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
    P_one = set_boundary(P_one, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
    res_FR = float(np.max(np.abs((P_one - P)[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])))
    # locate where the max residual sits
    delta_block = np.abs((P_one - P)[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI])
    i_max, j_max, k_max = np.unravel_index(np.argmax(delta_block), delta_block.shape)
    cell_loc = (u1_inner[i_max], S_inner[j_max], d_inner[k_max])
    cell_residual = float(delta_block[i_max, j_max, k_max])
    fr_at_cell = float(P_FR_inner[i_max, j_max, k_max])
    phi_at_cell = float(P_one[INNER_LO+i_max, INNER_LO+j_max, INNER_LO+k_max])

    # Picard loop
    res = res_FR
    for it in range(1, MAX_ITER+1):
        P_new = phi_sigmadelta_cara(P, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, W,
                                     INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P_new = set_boundary(P_new, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
        res = finf_interior(P_new, P, INNER_LO, INNER_HI, INNER_LO, INNER_HI, INNER_LO, INNER_HI)
        P = P_new
        if res < 1e-15:
            break
    P_inner_final = P[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI].copy()
    d_FR = dist_to_FR(P_inner_final)
    # 1-R² fit
    eps = 1e-30
    Pc = np.clip(P_inner_final, eps, 1-eps); lp = np.log(Pc/(1-Pc))
    fl_x = Tstar.flatten(); fl_lp = lp.flatten(); fl_w = Wd.flatten()
    slope, intercept = np.polyfit(fl_x, fl_lp, 1, w=np.sqrt(fl_w))
    pred = slope*fl_x + intercept
    m = float(np.average(fl_lp, weights=fl_w))
    vt = float(np.average((fl_lp-m)**2, weights=fl_w))
    vr = float(np.average((fl_lp-pred)**2, weights=fl_w))
    omr2 = vr/vt if vt > 0 else float('nan')
    return dict(G=G, ic=ic, res_FR=res_FR, res_final=res, d_FR_final=d_FR,
                slope=float(slope), intercept=float(intercept), one_minus_R2=omr2,
                iters=it,
                max_res_cell=dict(i=int(i_max), j=int(j_max), k=int(k_max),
                                  u_1=cell_loc[0], Sigma=cell_loc[1], delta=cell_loc[2],
                                  P_FR=fr_at_cell, Phi_P_FR=phi_at_cell, residual=cell_residual))

if __name__ == '__main__':
    rows = []
    print('sigma-delta CARA G-sweep (float64), FR-ansatz IC, exact BCs:', flush=True)
    print(f'{"G":>4} | {"res(Phi(P_FR))":>16} | {"final ferr":>14} | {"d_FR":>10} | {"slope":>8} | {"1-R^2":>10} | iters', flush=True)
    print('-'*100, flush=True)
    for G in [6, 8, 10, 12, 14, 16]:
        t = time.time()
        r = run_at_G(G, ic='FR')
        sec = time.time() - t
        r['sec'] = round(sec, 1)
        rows.append(r)
        print(f'{r["G"]:>4} | {r["res_FR"]:>16.3e} | {r["res_final"]:>14.3e} | {r["d_FR_final"]:>10.3e} | '
              f'{r["slope"]:>8.4f} | {r["one_minus_R2"]:>10.3e} | {r["iters"]:>3}  ({sec:.0f}s)', flush=True)
        print(f'      worst cell: (u_1={r["max_res_cell"]["u_1"]:+.2f}, '
              f'Σ={r["max_res_cell"]["Sigma"]:+.2f}, δ={r["max_res_cell"]["delta"]:+.2f}) '
              f'P_FR={r["max_res_cell"]["P_FR"]:.4f}  Φ(P_FR)={r["max_res_cell"]["Phi_P_FR"]:.4f}  '
              f'diff={r["max_res_cell"]["residual"]:.3e}', flush=True)
        json.dump({'tau': TAU, 'rows': rows}, open(os.path.join(HERE, 'sd_cara_G_sweep.json'), 'w'), indent=2)
    print('\nDONE.', flush=True)
