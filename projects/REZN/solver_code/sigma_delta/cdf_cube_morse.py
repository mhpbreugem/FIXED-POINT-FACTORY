"""Test 4: CDF-cube CRRA operator with proper Morse-adaptive partition-of-unity.

Same as cdf_cube_numba.py (cubic spline + clip + GL + ζ axes) BUT slice_evidence
now uses point-wise partition-of-unity weights:
  pass-0 (vary ζ_a at GL, root-find ζ_b): weight contribution by w = dA²/(dA²+dB²)
  pass-1 (swap):                             weight contribution by w = dB²/(dA²+dB²)
where dA = ∂P/∂ζ_a (from a-direction spline) and dB = ∂P/∂ζ_b (from b-direction
spline), evaluated at the contour root. This is the same partition-of-unity
used in hfree_morse_operator: at Morse-critical points where one derivative
vanishes, the weight transfers smoothly to the other direction, killing the
1/|grad P| singularity. The two passes are added (NOT averaged), because the
weights sum to 1 per root.
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from numba import njit, prange
from scipy.stats import norm
from scipy.optimize import brentq

from cdf_cube_numba import (
    TAU, GAMMA, EPS_PRICE, SQRT_TAU,
    F_bar, f_bar, F_bar_inv, TAB_Z, TAB_U, TAB_DUDZ,
    lookup_uz, f_signal_nb,
    natural_spline_M, spline_eval, spline_roots,
    crra_clear_nb,
    gauss_legendre, build_zeta_grid, metrics
)

@njit(cache=True)
def slice_evidence_morse(P_slice, zeta_grid, h, z0,
                          gl_z_nodes, gl_z_weights,
                          TAB_Z, TAB_U, TAB_DUDZ, tau, p_target):
    """Co-area evidence on 2D slice with Morse-adaptive partition-of-unity.

    For each contour root (ζ_a, ζ_b), compute both dA = ∂P/∂ζ_a and
    dB = ∂P/∂ζ_b via the row/column splines, and weight the contribution by
    w = (other-deriv)² / (dA² + dB² + ε). The two passes sum (not average)
    because the per-root weights already sum to 1.
    """
    G = zeta_grid.size; nq = gl_z_nodes.size
    out_roots = np.empty(20); out_ders = np.empty(20)
    A0 = 0.0; A1 = 0.0
    eps_grad = 1e-30

    # Cache column splines (along axis a for each b)
    M_cols = np.empty((G, G))
    for kb in range(G):
        M_cols[kb, :] = natural_spline_M(P_slice[:, kb], h)
    # Cache row splines (along axis b for each a)
    M_rows = np.empty((G, G))
    for ka in range(G):
        M_rows[ka, :] = natural_spline_M(P_slice[ka, :], h)

    # PASS 0: ζ_a at GL nodes, root-find ζ_b
    for ia in range(nq):
        za_gl = gl_z_nodes[ia]; wa = gl_z_weights[ia]
        ua, dudza = lookup_uz(za_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
        # Build P-line in b-direction at fixed za_gl (use column splines along a)
        P_line = np.empty(G)
        for kb in range(G):
            v, _ = spline_eval(P_slice[:, kb], M_cols[kb, :], h, z0, za_gl)
            P_line[kb] = v
        M_line = natural_spline_M(P_line, h)
        nr = spline_roots(P_line, M_line, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            zb = out_roots[r]; dB = out_ders[r]  # ∂P/∂ζ_b at (za_gl, zb)
            # Compute dA = ∂P/∂ζ_a at (za_gl, zb):
            # Build cross-line along a-direction at fixed zb using row splines
            P_cross = np.empty(G)
            for ka in range(G):
                v, _ = spline_eval(P_slice[ka, :], M_rows[ka, :], h, z0, zb)
                P_cross[ka] = v
            M_cross = natural_spline_M(P_cross, h)
            _, dA = spline_eval(P_cross, M_cross, h, z0, za_gl)
            grad_sq = dA*dA + dB*dB
            if grad_sq < eps_grad: continue
            # Partition-of-unity weight for pass-0: w0 = dA² / (dA² + dB²)
            w0 = dA*dA / grad_sq
            # f at root (za_gl, zb) — uses ua already, and ub
            ub, dudzb = lookup_uz(zb, TAB_Z, TAB_U, TAB_DUDZ)
            f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
            # 1/|grad P_u| with chain rule: |grad_u P|² = (dA/dudza)² + (dB/dudzb)²
            # but Morse-cancel: contribution is w0 * f / |∂P/∂u_b|
            dPdu_b = dB / dudzb
            if abs(dPdu_b) < eps_grad: continue
            A0 += w0 * wa * dudza * f0a * f0b / abs(dPdu_b)
            A1 += w0 * wa * dudza * f1a * f1b / abs(dPdu_b)

    # PASS 1: ζ_b at GL nodes, root-find ζ_a
    for ib in range(nq):
        zb_gl = gl_z_nodes[ib]; wb = gl_z_weights[ib]
        ub, dudzb = lookup_uz(zb_gl, TAB_Z, TAB_U, TAB_DUDZ)
        f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
        P_line = np.empty(G)
        for ka in range(G):
            v, _ = spline_eval(P_slice[ka, :], M_rows[ka, :], h, z0, zb_gl)
            P_line[ka] = v
        M_line = natural_spline_M(P_line, h)
        nr = spline_roots(P_line, M_line, h, z0, p_target, out_roots, out_ders, 4)
        for r in range(nr):
            za = out_roots[r]; dA = out_ders[r]  # ∂P/∂ζ_a at (za, zb_gl)
            # Compute dB = ∂P/∂ζ_b at (za, zb_gl):
            P_cross = np.empty(G)
            for kb in range(G):
                v, _ = spline_eval(P_slice[:, kb], M_cols[kb, :], h, z0, za)
                P_cross[kb] = v
            M_cross = natural_spline_M(P_cross, h)
            _, dB = spline_eval(P_cross, M_cross, h, z0, zb_gl)
            grad_sq = dA*dA + dB*dB
            if grad_sq < eps_grad: continue
            w1 = dB*dB / grad_sq
            ua, dudza = lookup_uz(za, TAB_Z, TAB_U, TAB_DUDZ)
            f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
            dPdu_a = dA / dudza
            if abs(dPdu_a) < eps_grad: continue
            A0 += w1 * wb * dudzb * f0a * f0b / abs(dPdu_a)
            A1 += w1 * wb * dudzb * f1a * f1b / abs(dPdu_a)

    return A0, A1  # NOT averaged: per-root weights sum to 1

@njit(cache=True, parallel=True)
def phi_cdf_morse(P, zeta_grid, h, z0,
                   gl_z_nodes, gl_z_weights,
                   TAB_Z, TAB_U, TAB_DUDZ,
                   gamma, tau):
    G = zeta_grid.size
    P_new = P.copy()
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                A0a, A1a = slice_evidence_morse(P[i, :, :], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_i, _ = lookup_uz(zeta_grid[i], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_i, -0.5, tau); f1o = f_signal_nb(u_i, 0.5, tau)
                den = f0o*A0a + f1o*A1a
                mu0 = (f1o*A1a)/den if den>1e-30 else 0.5
                A0b, A1b = slice_evidence_morse(P[:, j, :], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_j, _ = lookup_uz(zeta_grid[j], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_j, -0.5, tau); f1o = f_signal_nb(u_j, 0.5, tau)
                den = f0o*A0b + f1o*A1b
                mu1 = (f1o*A1b)/den if den>1e-30 else 0.5
                A0c, A1c = slice_evidence_morse(P[:, :, k], zeta_grid, h, z0,
                                                  gl_z_nodes, gl_z_weights,
                                                  TAB_Z, TAB_U, TAB_DUDZ, tau, p)
                u_k, _ = lookup_uz(zeta_grid[k], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_k, -0.5, tau); f1o = f_signal_nb(u_k, 0.5, tau)
                den = f0o*A0c + f1o*A1c
                mu2 = (f1o*A1c)/den if den>1e-30 else 0.5
                P_new[i, j, k] = crra_clear_nb(mu0, mu1, mu2, gamma, 120)
    return P_new

if __name__ == '__main__':
    G = 13; NQ = 64
    zeta_arr, u_arr, h, z0 = build_zeta_grid(G)
    gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
    print(f'Morse-PoU CDF-cube G={G}, NQ={NQ}')

    def sg(x): return 1/(1+np.exp(-x))
    U1, U2, U3 = np.meshgrid(u_arr, u_arr, u_arr, indexing='ij')
    P_NL = np.empty_like(U1)
    for i in range(G):
        for j in range(G):
            for k in range(G):
                mu1 = sg(TAU*u_arr[i]); mu2 = sg(TAU*u_arr[j]); mu3 = sg(TAU*u_arr[k])
                P_NL[i,j,k] = crra_clear_nb(mu1, mu2, mu3, GAMMA, 120)
    m_ic = metrics(P_NL, u_arr)
    print(f'IC: deficit={m_ic["deficit"]:.4f} slope={m_ic["slope_T"]:.4f} d_FR={m_ic["d_FR"]:.4f}')

    print('JIT warmup phi_cdf_morse...', flush=True); t=time.time()
    _ = phi_cdf_morse(P_NL.copy(), zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                        TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
    print(f'  warmup {time.time()-t:.1f}s', flush=True)

    # Anderson + NK
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize._nonlin import NoConvergence

    def F_of(x):
        P = x.reshape((G,G,G))
        return (phi_cdf_morse(P, zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                               TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU) - P).ravel()

    x = P_NL.ravel().copy()
    G_h=[]; F_h=[]
    best = dict(ferr=np.inf, x=x.copy())
    print('\n=== Anderson(m=8) Morse-PoU ===', flush=True)
    for it in range(1, 31):
        ts = time.time()
        g = (F_of(x) + x); f = g - x
        ferr = float(np.max(np.abs(f)))
        if ferr < best['ferr']: best = dict(ferr=ferr, x=g.copy(), it=it)
        if it % 5 == 0 or it == 1 or it < 5:
            m = metrics(x.reshape((G,)*3), u_arr)
            print(f'  A it {it:3d} ferr={ferr:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f} ({time.time()-ts:.1f}s)', flush=True)
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
        x = np.clip(x_new, 1e-12, 1-1e-12)
    print(f'Anderson best: ||F||={best["ferr"]:.3e} at it {best["it"]}', flush=True)

    print('\n=== NK Morse-PoU from Anderson best ===', flush=True)
    cnt = {'n':0, 'hist':[]}
    def cb(x, fx):
        cnt['n'] += 1; cnt['hist'].append(float(np.max(np.abs(fx))))
        if cnt['n'] % 5 == 0 or cnt['n'] == 1:
            m = metrics(x.reshape((G,)*3), u_arr)
            print(f'  NK it {cnt["n"]:3d} ||F||={cnt["hist"][-1]:.3e} slope={m["slope_T"]:.4f} d_FR={m["d_FR"]:.4f}', flush=True)
    try:
        sol = newton_krylov(F_of, best['x'], f_tol=1e-7, maxiter=40, method='lgmres', callback=cb)
        conv = True
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(F_of(sol))))
    m = metrics(sol.reshape((G,)*3), u_arr)
    print(f'\nMorse-PoU RESULT γ=0.1: conv={conv}, ||F||={Finf:.3e}, slope={m["slope_T"]:.4f} '
          f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f}')
    print(f'Compare to cubic-spline-clip (no PoU): ||F||≈0.04')

    np.save(os.path.join(HERE, 'morse_NK_g0.1_G13.npy'), sol.reshape((G,)*3))
    json.dump({'G':G,'NQ':NQ,'gamma':float(GAMMA),'final_Finf':Finf,
                'slope':m['slope_T'],'deficit':m['deficit'],'d_FR':m['d_FR'],
                'NK_iters':cnt['n'],'NK_hist':cnt['hist'],
                'anderson_best':best['ferr']},
              open(os.path.join(HERE,'morse_NK_g0.1_G13.json'),'w'), indent=2, default=str)
    print('saved')
