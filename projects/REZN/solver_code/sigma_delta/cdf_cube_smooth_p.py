"""Test 6: exact line integral A_v(p), but make A_v(p) smooth in p by sampling
many p-values per slice and fitting a cubic spline.

Per slice (i fixed) we precompute A_v(p_n) for v ∈ {0,1} at a uniform p-grid
of P_N samples on (eps, 1-eps), using the EXACT contour line integral (cubic
spline of P + root-find + 1/|grad P|). This is h=0, EXACT line integral.

The discrete topology jumps in p (root counts changing across critical
values) generate small bumps in A_v(p_n). We then fit a natural cubic spline
of A_v as a function of p (in 1D), evaluated at each cell's p_cell. The
spline-in-p averages the bumps out, giving a smooth Phi as a function of
the iterate.
"""
import os, sys, math, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from numba import njit, prange

from cdf_cube_numba import (TAU, GAMMA, EPS_PRICE,
                              TAB_Z, TAB_U, TAB_DUDZ,
                              lookup_uz, f_signal_nb,
                              natural_spline_M, spline_eval, spline_roots,
                              crra_clear_nb,
                              gauss_legendre, build_zeta_grid, metrics)

P_N = 64          # number of p-samples per slice
EPS_PCURVE = 1e-3
P_GRID = np.linspace(EPS_PCURVE, 1.0 - EPS_PCURVE, P_N)
DP = (1.0 - 2*EPS_PCURVE)/(P_N - 1)
P0 = P_GRID[0]

@njit(cache=True)
def slice_Avs_p(P_slice, zeta_grid, h_z, z0,
                 gl_z_nodes, gl_z_weights,
                 TAB_Z, TAB_U, TAB_DUDZ, tau, P_grid):
    """Compute A_0(p), A_1(p) at each p in P_grid for the 2D slice.
    Uses the SAME co-area integration as cdf_cube_numba.slice_evidence_cdf
    (cubic spline + GL + 2-pass average + clip-to-[0,1]).
    Returns (A0_vec[P_N], A1_vec[P_N]).
    """
    G = zeta_grid.size; nq = gl_z_nodes.size; pn = P_grid.size
    out_roots = np.empty(20); out_ders = np.empty(20)
    A0_vec = np.zeros(pn); A1_vec = np.zeros(pn)
    # Pre-cache splines
    Mb_cache = np.empty((G, G))
    for kb in range(G):
        Mb_cache[kb, :] = natural_spline_M(P_slice[:, kb], h_z)
    Ma_cache = np.empty((G, G))
    for ka in range(G):
        Ma_cache[ka, :] = natural_spline_M(P_slice[ka, :], h_z)
    for ip in range(pn):
        p_target = P_grid[ip]
        A0 = 0.0; A1 = 0.0
        # PASS 0: ζ_a at GL, root ζ_b
        for ia in range(nq):
            za_gl = gl_z_nodes[ia]; wa = gl_z_weights[ia]
            ua, dudza = lookup_uz(za_gl, TAB_Z, TAB_U, TAB_DUDZ)
            f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
            P_line = np.empty(G)
            for kb in range(G):
                v, _ = spline_eval(P_slice[:, kb], Mb_cache[kb, :], h_z, z0, za_gl)
                P_line[kb] = v
            Ma = natural_spline_M(P_line, h_z)
            nr = spline_roots(P_line, Ma, h_z, z0, p_target, out_roots, out_ders, 4)
            for r in range(nr):
                zb = out_roots[r]; dPdz_b = out_ders[r]
                if abs(dPdz_b) < 1e-30: continue
                ub, dudzb = lookup_uz(zb, TAB_Z, TAB_U, TAB_DUDZ)
                f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
                dPdu_b = dPdz_b / dudzb
                A0 += wa * dudza * f0a * f0b / abs(dPdu_b)
                A1 += wa * dudza * f1a * f1b / abs(dPdu_b)
        # PASS 1: ζ_b at GL, root ζ_a
        for ib in range(nq):
            zb_gl = gl_z_nodes[ib]; wb = gl_z_weights[ib]
            ub, dudzb = lookup_uz(zb_gl, TAB_Z, TAB_U, TAB_DUDZ)
            f0b = f_signal_nb(ub, -0.5, tau); f1b = f_signal_nb(ub, 0.5, tau)
            P_line = np.empty(G)
            for ka in range(G):
                v, _ = spline_eval(P_slice[ka, :], Ma_cache[ka, :], h_z, z0, zb_gl)
                P_line[ka] = v
            Mb = natural_spline_M(P_line, h_z)
            nr = spline_roots(P_line, Mb, h_z, z0, p_target, out_roots, out_ders, 4)
            for r in range(nr):
                za = out_roots[r]; dPdz_a = out_ders[r]
                if abs(dPdz_a) < 1e-30: continue
                ua, dudza = lookup_uz(za, TAB_Z, TAB_U, TAB_DUDZ)
                f0a = f_signal_nb(ua, -0.5, tau); f1a = f_signal_nb(ua, 0.5, tau)
                dPdu_a = dPdz_a / dudza
                A0 += wb * dudzb * f0a * f0b / abs(dPdu_a)
                A1 += wb * dudzb * f1a * f1b / abs(dPdu_a)
        A0_vec[ip] = 0.5*A0; A1_vec[ip] = 0.5*A1
    return A0_vec, A1_vec

@njit(cache=True, inline='always')
def spline_eval_1d(y, M, h, x0, x):
    i = int((x - x0)/h)
    n = y.size
    if i < 0: i = 0
    if i > n - 2: i = n - 2
    xL = x0 + i*h; xR = xL + h
    A = (xR - x)/h; B = (x - xL)/h
    val = A*y[i] + B*y[i+1] + ((A**3-A)*M[i] + (B**3-B)*M[i+1])*(h*h)/6.0
    return val

@njit(cache=True, parallel=True)
def phi_cdf_smooth_p(P, zeta_grid, h_z, z0,
                      gl_z_nodes, gl_z_weights,
                      TAB_Z, TAB_U, TAB_DUDZ,
                      gamma, tau, P_grid, h_p, p0):
    G = zeta_grid.size
    P_new = P.copy()
    # Precompute A_v(p) spline coefficients for each of the 3·G slice types
    # Agent 0: slice P[i, :, :] for each i (varies in ζ_2, ζ_3); G slices
    # Agent 1: slice P[:, j, :] for each j; G slices
    # Agent 2: slice P[:, :, k] for each k; G slices
    # Each slice has 2 A_v(p) curves -> 2 splines (A0, A1).
    A0_0 = np.empty((G, P_grid.size)); A1_0 = np.empty((G, P_grid.size))
    A0_1 = np.empty((G, P_grid.size)); A1_1 = np.empty((G, P_grid.size))
    A0_2 = np.empty((G, P_grid.size)); A1_2 = np.empty((G, P_grid.size))
    for i in prange(G):
        A0_0[i, :], A1_0[i, :] = slice_Avs_p(P[i, :, :], zeta_grid, h_z, z0,
                                                gl_z_nodes, gl_z_weights,
                                                TAB_Z, TAB_U, TAB_DUDZ, tau, P_grid)
        A0_1[i, :], A1_1[i, :] = slice_Avs_p(P[:, i, :], zeta_grid, h_z, z0,
                                                gl_z_nodes, gl_z_weights,
                                                TAB_Z, TAB_U, TAB_DUDZ, tau, P_grid)
        A0_2[i, :], A1_2[i, :] = slice_Avs_p(P[:, :, i], zeta_grid, h_z, z0,
                                                gl_z_nodes, gl_z_weights,
                                                TAB_Z, TAB_U, TAB_DUDZ, tau, P_grid)
    # Build cubic spline of A_v(p) for each slice
    M_A0_0 = np.empty((G, P_grid.size)); M_A1_0 = np.empty((G, P_grid.size))
    M_A0_1 = np.empty((G, P_grid.size)); M_A1_1 = np.empty((G, P_grid.size))
    M_A0_2 = np.empty((G, P_grid.size)); M_A1_2 = np.empty((G, P_grid.size))
    for i in range(G):
        M_A0_0[i, :] = natural_spline_M(A0_0[i, :], h_p)
        M_A1_0[i, :] = natural_spline_M(A1_0[i, :], h_p)
        M_A0_1[i, :] = natural_spline_M(A0_1[i, :], h_p)
        M_A1_1[i, :] = natural_spline_M(A1_1[i, :], h_p)
        M_A0_2[i, :] = natural_spline_M(A0_2[i, :], h_p)
        M_A1_2[i, :] = natural_spline_M(A1_2[i, :], h_p)
    # Phi per cell using smoothed A_v(p_cell)
    for i in prange(G):
        for j in range(G):
            for k in range(G):
                p = P[i, j, k]
                pc = p
                if pc < p0: pc = p0
                if pc > p0 + (P_grid.size-1)*h_p: pc = p0 + (P_grid.size-1)*h_p
                # Agent 0 uses slice index i
                A0a = spline_eval_1d(A0_0[i, :], M_A0_0[i, :], h_p, p0, pc)
                A1a = spline_eval_1d(A1_0[i, :], M_A1_0[i, :], h_p, p0, pc)
                u_i, _ = lookup_uz(zeta_grid[i], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_i, -0.5, tau); f1o = f_signal_nb(u_i, 0.5, tau)
                den = f0o*A0a + f1o*A1a
                mu0 = (f1o*A1a)/den if den>1e-30 else 0.5
                # Agent 1 uses slice index j
                A0b = spline_eval_1d(A0_1[j, :], M_A0_1[j, :], h_p, p0, pc)
                A1b = spline_eval_1d(A1_1[j, :], M_A1_1[j, :], h_p, p0, pc)
                u_j, _ = lookup_uz(zeta_grid[j], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_j, -0.5, tau); f1o = f_signal_nb(u_j, 0.5, tau)
                den = f0o*A0b + f1o*A1b
                mu1 = (f1o*A1b)/den if den>1e-30 else 0.5
                # Agent 2 uses slice index k
                A0c = spline_eval_1d(A0_2[k, :], M_A0_2[k, :], h_p, p0, pc)
                A1c = spline_eval_1d(A1_2[k, :], M_A1_2[k, :], h_p, p0, pc)
                u_k, _ = lookup_uz(zeta_grid[k], TAB_Z, TAB_U, TAB_DUDZ)
                f0o = f_signal_nb(u_k, -0.5, tau); f1o = f_signal_nb(u_k, 0.5, tau)
                den = f0o*A0c + f1o*A1c
                mu2 = (f1o*A1c)/den if den>1e-30 else 0.5
                P_new[i, j, k] = crra_clear_nb(mu0, mu1, mu2, gamma, 120)
    return P_new

if __name__ == '__main__':
    G = 13; NQ = 64
    zeta_arr, u_arr, h_z, z0 = build_zeta_grid(G)
    gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
    print(f'Smooth-A_v(p) CDF-cube G={G}, NQ={NQ}, P_N={P_N}')

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

    print('JIT warmup phi_cdf_smooth_p...', flush=True); t=time.time()
    _ = phi_cdf_smooth_p(P_NL.copy(), zeta_arr, h_z, z0,
                          gl_z_nodes, gl_z_weights,
                          TAB_Z, TAB_U, TAB_DUDZ,
                          GAMMA, TAU, P_GRID, DP, P0)
    print(f'  warmup {time.time()-t:.1f}s', flush=True)

    # Anderson + NK
    from scipy.optimize import newton_krylov
    try: from scipy.optimize import NoConvergence
    except ImportError: from scipy.optimize._nonlin import NoConvergence

    def F_of(x):
        P = x.reshape((G,G,G))
        return (phi_cdf_smooth_p(P, zeta_arr, h_z, z0,
                                   gl_z_nodes, gl_z_weights,
                                   TAB_Z, TAB_U, TAB_DUDZ,
                                   GAMMA, TAU, P_GRID, DP, P0) - P).ravel()

    print('\n=== Anderson(m=8) smooth-p ===', flush=True)
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

    print('\n=== NK smooth-p from Anderson best ===', flush=True)
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
    print(f'\nSmooth-p RESULT γ=0.1: conv={conv}, ||F||={Finf:.3e}, slope={m["slope_T"]:.4f} '
          f'deficit={m["deficit"]:.4f} d_FR={m["d_FR"]:.4f}')

    np.save(os.path.join(HERE, 'cdf_smooth_p_NK_g0.1_G13.npy'), sol.reshape((G,)*3))
    json.dump({'G':G,'NQ':NQ,'P_N':P_N,'gamma':float(GAMMA),'final_Finf':Finf,
                'slope':m['slope_T'],'deficit':m['deficit'],'d_FR':m['d_FR'],
                'NK_iters':cnt['n'],'NK_hist':cnt['hist'],
                'anderson_best':best['ferr']},
              open(os.path.join(HERE,'cdf_smooth_p_NK_g0.1_G13.json'),'w'), indent=2, default=str)
    print('saved')
