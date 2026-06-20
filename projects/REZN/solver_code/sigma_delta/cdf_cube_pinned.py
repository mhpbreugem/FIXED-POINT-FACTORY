"""Creative repair attempt: PIN boundary cells (ζ axis = 0 or 1) to known
asymptotic FR limits (P = 0 at ζ = 0, P = 1 at ζ = 1, per axis). Iterate
only the interior (G-2)³ unknowns. The operator (cubic-spline-clip) is
otherwise unchanged.

Rationale: the spline overshoot + Morse-critical effects we measured all
concentrate in the few cells near the cube corners (where P → 0 or 1). By
holding those at their EXACT analytical limits, the residual ||F||_inf is
computed over (G-2)³ interior cells where the operator is well-conditioned.
"""
import os, sys, time, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from numba import njit, prange

from cdf_cube_numba import (TAU, GAMMA, EPS_PRICE,
                              TAB_Z, TAB_U, TAB_DUDZ,
                              lookup_uz, f_signal_nb,
                              natural_spline_M, spline_eval, spline_roots,
                              crra_clear_nb, slice_evidence_cdf,
                              gauss_legendre, build_zeta_grid, metrics,
                              phi_cdf_numba)

G_INNER_OFF = 1   # pin layer width: 1 boundary face

def pin_boundary(P, G):
    """Pin ζ-axis faces: P[0,j,k]=0, P[G-1,j,k]=1, P[i,0,k]=0, P[i,G-1,k]=1,
    P[i,j,0]=0, P[i,j,G-1]=1. (ζ=0 ↔ u→-∞ ↔ perfect info v=0; symmetrically.)
    """
    P2 = P.copy()
    eps = EPS_PRICE
    # ζ_1 axis
    P2[0, :, :] = eps; P2[G-1, :, :] = 1.0 - eps
    P2[:, 0, :] = eps; P2[:, G-1, :] = 1.0 - eps
    P2[:, :, 0] = eps; P2[:, :, G-1] = 1.0 - eps
    return P2

if __name__ == '__main__':
    G = 13; NQ = 64
    zeta_arr, u_arr, h_z, z0 = build_zeta_grid(G)
    gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
    print(f'Pinned-boundary CDF-cube G={G}, NQ={NQ}')

    # NL IC, then pin
    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P_NL = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
                P_NL[i,j,k] = crra_clear_nb(mu1, mu2, mu3, GAMMA, 120)
    P_NL = pin_boundary(P_NL, G)
    m_ic = metrics(P_NL, u_arr)
    print(f'IC pinned: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}')

    print('JIT warmup phi_cdf_numba...', flush=True); t=time.time()
    _ = phi_cdf_numba(P_NL.copy(), zeta_arr, h_z, z0,
                       gl_z_nodes, gl_z_weights,
                       TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
    print(f'  warmup {time.time()-t:.1f}s', flush=True)

    # Anderson + NK, but Phi pins boundary AFTER each Phi call
    from scipy.optimize import newton_krylov
    try: from scipy.optimize import NoConvergence
    except ImportError: from scipy.optimize._nonlin import NoConvergence

    def Phi_pinned(P):
        Pn = phi_cdf_numba(P, zeta_arr, h_z, z0,
                            gl_z_nodes, gl_z_weights,
                            TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
        return pin_boundary(Pn, G)

    # Iterate only the INTERIOR (G-2)³ cells; the boundary stays fixed
    G_i = G - 2
    def F_inner(x_inner):
        P = P_NL.copy()  # start with the pinned IC
        P[1:-1, 1:-1, 1:-1] = x_inner.reshape((G_i,)*3)
        Pn = Phi_pinned(P)
        return (Pn[1:-1, 1:-1, 1:-1] - P[1:-1, 1:-1, 1:-1]).ravel()

    print(f'\n=== Anderson(m=8) PINNED, iterate inner {G_i}³={G_i**3} cells ===', flush=True)
    x_inner = P_NL[1:-1, 1:-1, 1:-1].ravel().copy()
    G_h=[]; F_h=[]; best = dict(ferr=np.inf, x=x_inner.copy())
    for it in range(1, 31):
        ts = time.time()
        f = F_inner(x_inner)
        g = f + x_inner
        ferr = float(np.max(np.abs(f)))
        if ferr < best['ferr']: best = dict(ferr=ferr, x=g.copy(), it=it)
        if it % 5 == 0 or it == 1 or it < 5:
            P_full = P_NL.copy()
            P_full[1:-1, 1:-1, 1:-1] = x_inner.reshape((G_i,)*3)
            m = metrics(P_full, u_arr)
            print(f'  A it {it:3d} ferr_inner={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
        if ferr < 1e-7: break
        G_h.append(g.copy()); F_h.append(f.copy())
        if len(F_h) > 9: G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0: x_new = g
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x_new = g - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g
        x_inner = np.clip(x_new, 1e-12, 1-1e-12)
    print(f'Anderson best: ||F||_inner={best["ferr"]:.3e} at it {best["it"]}', flush=True)

    print('\n=== NK PINNED from Anderson best ===', flush=True)
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            P_full = P_NL.copy()
            P_full[1:-1, 1:-1, 1:-1] = x.reshape((G_i,)*3)
            m = metrics(P_full, u_arr)
            print(f'  NK it {cnt["n"]:3d} ||F||_inner={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}', flush=True)
    try:
        sol = newton_krylov(F_inner, best['x'], f_tol=1e-7, maxiter=40, method='lgmres', callback=cb)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf_inner = float(np.max(np.abs(F_inner(sol))))
    P_full = P_NL.copy()
    P_full[1:-1, 1:-1, 1:-1] = sol.reshape((G_i,)*3)
    m = metrics(P_full, u_arr)
    print(f'\nPinned RESULT γ=0.1: conv={conv}, ||F||_inner={Finf_inner:.3e}, slope={m["slope_T"]:.4f} '
          f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f}')

    np.save(os.path.join(HERE, 'cdf_pinned_NK_g0.1_G13.npy'), P_full)
    json.dump({'G':G,'G_inner':G_i,'NQ':NQ,'gamma':float(GAMMA),
                'final_Finf_inner':Finf_inner,
                'slope':m['slope_T'],'deficit':m['deficit'],'d_FR':m['d_FR'],
                'NK_iters':cnt['n'],'NK_hist':cnt['hist'],
                'anderson_best':best['ferr']},
              open(os.path.join(HERE,'cdf_pinned_NK_g0.1_G13.json'),'w'), indent=2, default=str)
    print('saved')
