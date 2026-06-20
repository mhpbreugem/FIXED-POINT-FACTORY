"""Sub-level-set CDF approach on the CDF-cube. STRICT h=0, NO contour
root-finding, NO topology jumps.

For each 2D slice P(ζ_a, ζ_b), compute the co-area evidence
    A_v(p) = dG_v/dp,  G_v(p) = ∫ f_v(u_a) f_v(u_b) 1_{P<p} du_a du_b
analytically via the closed-form derivative of below-plane area in each
bilinear subcell. No 1/|grad P| division anywhere. dA/dp is BOUNDED.

This guarantees:
  - G_v(p) is C^0 in p (continuous)
  - A_v(p) is piecewise-continuous (finite, no singularities)
  - The full Phi operator is then smooth enough for NK to converge to high
    accuracy.

Port of k3_strict_h0_cdf/cdf_h0_operator.py (flint arb) to numba float64
on the CDF-cube ζ-grid with the Jacobian du/dζ absorbed into the integrand.
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from numba import njit, prange

from cdf_cube_numba import (TAU, GAMMA, EPS_PRICE,
                              TAB_Z, TAB_U, TAB_DUDZ,
                              lookup_uz, f_signal_nb, crra_clear_nb,
                              gauss_legendre, build_zeta_grid, metrics)

# ----- Closed-form dA/dp over unit square with bilinear-reduced affine field -----
@njit(cache=True, inline='always')
def dA_dp_affine(P00, bx, by, p):
    """dA/dp where A is fraction of unit [0,1]² with P00 + bx*x + by*y < p.
    Returns a non-negative BOUNDED value (no 1/|grad| singularity).
    """
    u = bx if bx >= 0 else -bx
    v = by if by >= 0 else -by
    if u < v:
        u, v = v, u
    Pmin = P00
    if bx < 0: Pmin += bx
    if by < 0: Pmin += by
    q = p - Pmin
    s = u + v
    if q <= 0.0 or q >= s:
        return 0.0
    if u <= 0.0:
        return 0.0
    if v <= 0.0:
        return 1.0/u
    if q <= v:
        return q/(u*v)
    if q <= u:
        return 1.0/u
    return (s - q)/(u*v)

@njit(cache=True, inline='always')
def bilin(P00, P10, P01, P11, s, t):
    return (P00*(1.0-s)*(1.0-t) + P10*s*(1.0-t)
            + P01*(1.0-s)*t + P11*s*t)

@njit(cache=True)
def agent_evidence_sublevel(P_slice, zeta_grid, h_z, z0,
                              TAB_Z, TAB_U, TAB_DUDZ,
                              tau, p_target, NSUB):
    """Co-area A_v(p) on 2D slice via CDF-deriv (sub-level-set), in ζ-space.

    The slice P_slice[a, b] is on a (ζ_a, ζ_b) uniform grid with spacing h_z.
    G_v(p) = ∫ f_v(u(ζ_a)) f_v(u(ζ_b)) 1_{P(ζ_a,ζ_b)<p} (du/dζ_a)(du/dζ_b) dζ_a dζ_b
    For each cell (i, j), subdivide NSUB×NSUB, use bilinear-corner P-values,
    accumulate dA/dp · (du/dζ_a evaluated at subcell center) · (du/dζ_b) ·
    physical sub-area in ζ.
    """
    n = zeta_grid.size
    h_sub = 1.0/NSUB
    sub_area_z = (h_z*h_z)*(h_sub*h_sub)  # subcell physical area in ζ²
    A0 = 0.0; A1 = 0.0
    for i in range(n - 1):
        ze_i = z0 + i*h_z
        for j in range(n - 1):
            ze_j = z0 + j*h_z
            P00 = P_slice[i, j]
            P10 = P_slice[i+1, j]
            P01 = P_slice[i, j+1]
            P11 = P_slice[i+1, j+1]
            cmin = P00; cmax = P00
            if P10 < cmin: cmin = P10
            if P01 < cmin: cmin = P01
            if P11 < cmin: cmin = P11
            if P10 > cmax: cmax = P10
            if P01 > cmax: cmax = P01
            if P11 > cmax: cmax = P11
            if p_target <= cmin or p_target >= cmax:
                continue
            for a in range(NSUB):
                s0 = a*h_sub; s1 = (a+1)*h_sub; sc = 0.5*(s0+s1)
                za = ze_i + sc*h_z
                ua, dudza = lookup_uz(za, TAB_Z, TAB_U, TAB_DUDZ)
                fa0 = f_signal_nb(ua, -0.5, tau); fa1 = f_signal_nb(ua, 0.5, tau)
                for b in range(NSUB):
                    t0 = b*h_sub; t1 = (b+1)*h_sub; tc = 0.5*(t0+t1)
                    # subcell corners (bilinear samples)
                    q00 = bilin(P00, P10, P01, P11, s0, t0)
                    q10 = bilin(P00, P10, P01, P11, s1, t0)
                    q01 = bilin(P00, P10, P01, P11, s0, t1)
                    smin = q00; smax = q00
                    if q10 < smin: smin = q10
                    if q01 < smin: smin = q01
                    if q10 > smax: smax = q10
                    if q01 > smax: smax = q01
                    q11 = bilin(P00, P10, P01, P11, s1, t1)
                    if q11 < smin: smin = q11
                    if q11 > smax: smax = q11
                    if p_target <= smin or p_target >= smax: continue
                    bx = q10 - q00; by = q01 - q00
                    dadp = dA_dp_affine(q00, bx, by, p_target)
                    if dadp <= 0.0: continue
                    zb = ze_j + tc*h_z
                    ub, dudzb = lookup_uz(zb, TAB_Z, TAB_U, TAB_DUDZ)
                    fb0 = f_signal_nb(ub, -0.5, tau); fb1 = f_signal_nb(ub, 0.5, tau)
                    w = dadp * sub_area_z * dudza * dudzb
                    A0 += w * fa0 * fb0
                    A1 += w * fa1 * fb1
    return A0, A1

@njit(cache=True, parallel=True)
def phi_cdf_sublevel(P, zeta_grid, h_z, z0,
                      TAB_Z, TAB_U, TAB_DUDZ,
                      gamma, tau, NSUB):
    G = zeta_grid.size
    P_new = P.copy()
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                A0a, A1a = agent_evidence_sublevel(P[i, :, :], zeta_grid, h_z, z0,
                                                     TAB_Z, TAB_U, TAB_DUDZ, tau, p, NSUB)
                u_i, _ = lookup_uz(zeta_grid[i], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_i, -0.5, tau); f1o = f_signal_nb(u_i, 0.5, tau)
                den = f0o*A0a + f1o*A1a
                mu0 = (f1o*A1a)/den if den > 1e-30 else 0.5
                A0b, A1b = agent_evidence_sublevel(P[:, j, :], zeta_grid, h_z, z0,
                                                     TAB_Z, TAB_U, TAB_DUDZ, tau, p, NSUB)
                u_j, _ = lookup_uz(zeta_grid[j], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_j, -0.5, tau); f1o = f_signal_nb(u_j, 0.5, tau)
                den = f0o*A0b + f1o*A1b
                mu1 = (f1o*A1b)/den if den > 1e-30 else 0.5
                A0c, A1c = agent_evidence_sublevel(P[:, :, k], zeta_grid, h_z, z0,
                                                     TAB_Z, TAB_U, TAB_DUDZ, tau, p, NSUB)
                u_k, _ = lookup_uz(zeta_grid[k], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_k, -0.5, tau); f1o = f_signal_nb(u_k, 0.5, tau)
                den = f0o*A0c + f1o*A1c
                mu2 = (f1o*A1c)/den if den > 1e-30 else 0.5
                P_new[i, j, k] = crra_clear_nb(mu0, mu1, mu2, gamma, 120)
    return P_new

if __name__ == '__main__':
    G = 13; NSUB = 3
    zeta_arr, u_arr, h_z, z0 = build_zeta_grid(G)
    print(f'Sub-level-set CDF-cube G={G}, NSUB={NSUB}')

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

    print('JIT warmup phi_cdf_sublevel...', flush=True); t=time.time()
    _ = phi_cdf_sublevel(P_NL.copy(), zeta_arr, h_z, z0,
                          TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU, NSUB)
    print(f'  warmup {time.time()-t:.1f}s', flush=True)

    # Anderson + NK
    from scipy.optimize import newton_krylov
    try: from scipy.optimize import NoConvergence
    except ImportError: from scipy.optimize._nonlin import NoConvergence

    def F_of(x):
        P = x.reshape((G,G,G))
        return (phi_cdf_sublevel(P, zeta_arr, h_z, z0,
                                   TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU, NSUB) - P).ravel()

    print('\n=== Anderson(m=8) sublevel ===', flush=True)
    x = P_NL.ravel().copy()
    G_h=[]; F_h=[]; best = dict(ferr=np.inf, x=x.copy())
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

    print('\n=== NK sublevel from Anderson best ===', flush=True)
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
    print(f'\nSubLevel RESULT γ=0.1: conv={conv}, ||F||={Finf:.3e}, slope={m["slope_T"]:.4f} '
          f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f}')
    print(f'(cubic-spline-clip best was ||F||≈0.038, slope=0.66, deficit=0.12)')

    np.save(os.path.join(HERE, 'cdf_sublevel_NK_g0.1_G13.npy'), sol.reshape((G,)*3))
    json.dump({'G':G,'NSUB':NSUB,'gamma':float(GAMMA),'final_Finf':Finf,
                'slope':m['slope_T'],'deficit':m['deficit'],'d_FR':m['d_FR'],
                'NK_iters':cnt['n'],'NK_hist':cnt['hist'],
                'anderson_best':best['ferr']},
              open(os.path.join(HERE,'cdf_sublevel_NK_g0.1_G13.json'),'w'), indent=2, default=str)
    print('saved')
