"""Bicubic-Hermite strict-h=0 K=3 operator (proof of concept).

Replaces piecewise-LINEAR interpolation of P along the sweep direction
with monotone Hermite cubic (PCHIP) interpolation. The function and its
derivative are C^0 in the position; the derivative is therefore smooth
in P (no piecewise-constant slope jumps as roots cross cell boundaries).

Root finding becomes nonlinear (cubic per cell), solved by Newton seeded
from the linear-interp root.

Test: cold start at (tau=2, gamma=0.1), G=9, compare residual trajectory
and best-iterate deficit vs the linear-interp strict operator.
"""
import os, sys, time
import numpy as np
from scipy.stats import norm
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
from scipy.interpolate import PchipInterpolator
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, phi_lin_strict_jit, f_signal_jit, _find_interval


def pchip_along(u_grid, P_along):
    """Build a PCHIP for P along u_grid (1D)."""
    return PchipInterpolator(u_grid, P_along, extrapolate=False)


def pchip_eval(p, q):
    if q <= p.x[0]: return float(p.c[-1, 0])  # constant extrapolation
    if q >= p.x[-1]: return float(p.c[-1, -1])
    return float(p(q))


def pchip_slope(p, q):
    if q <= p.x[0]: return float(p.derivative()(p.x[0]))
    if q >= p.x[-1]: return float(p.derivative()(p.x[-1]))
    return float(p.derivative()(q))


def find_pchip_roots(u_grid, P_along, p_target):
    """Roots of pchip(u) = p_target. Per cell, try Newton seeded from the
    linear-interp root if there is a sign change."""
    pc = pchip_along(u_grid, P_along)
    dpc = pc.derivative()
    n = len(u_grid)
    roots = []
    for j in range(n - 1):
        a = P_along[j] - p_target
        b = P_along[j+1] - p_target
        if a * b > 0: continue  # no crossing in linear -- accept this approximation
        # Newton seeded from linear-interp root
        slope = (P_along[j+1] - P_along[j]) / (u_grid[j+1] - u_grid[j])
        if abs(slope) < 1e-14: continue
        u = u_grid[j] + (p_target - P_along[j]) / slope
        u = max(u_grid[j], min(u_grid[j+1], u))
        for _ in range(8):
            Pv = float(pc(u)) - p_target
            sl = float(dpc(u))
            if abs(sl) < 1e-14: break
            du = -Pv / sl
            u_new = u + du
            if u_new < u_grid[j]: u_new = u_grid[j]
            if u_new > u_grid[j+1]: u_new = u_grid[j+1]
            if abs(u_new - u) < 1e-12: u = u_new; break
            u = u_new
        roots.append((float(u), float(dpc(u))))
    return roots


def co_area_pchip(u_grid, slice2d, p_target, gl_u, gl_du, tau, n, nq):
    """Same POU co-area as the linear strict op, but with PCHIP interp."""
    A0, A1 = 0.0, 0.0
    # u_a sweep, find u_b roots
    for q in range(nq):
        u_a = gl_u[q]; w_a = gl_du[q]
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        # slice along u_b at u_a (linear interp of vertex grid in u_a, PCHIP in u_b)
        if u_a <= u_grid[0]: P_along_b = slice2d[0, :].copy()
        elif u_a >= u_grid[-1]: P_along_b = slice2d[-1, :].copy()
        else:
            i = _find_interval(u_grid, u_a, n)
            w = (u_a - u_grid[i]) / (u_grid[i+1] - u_grid[i])
            P_along_b = (1 - w) * slice2d[i, :] + w * slice2d[i+1, :]
        roots = find_pchip_roots(u_grid, P_along_b, p_target)
        for u_b, dPdu_b in roots:
            # dP/du_a at (u_a, u_b): slice along u_a at u_b is PCHIP
            if u_b <= u_grid[0]: P_along_a = slice2d[:, 0].copy()
            elif u_b >= u_grid[-1]: P_along_a = slice2d[:, -1].copy()
            else:
                j = _find_interval(u_grid, u_b, n)
                w = (u_b - u_grid[j]) / (u_grid[j+1] - u_grid[j])
                P_along_a = (1 - w) * slice2d[:, j] + w * slice2d[:, j+1]
            pc_a = pchip_along(u_grid, P_along_a)
            dPdu_a = pchip_slope(pc_a, u_a)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    # u_b sweep, find u_a roots
    for q in range(nq):
        u_b = gl_u[q]; w_b = gl_du[q]
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        if u_b <= u_grid[0]: P_along_a = slice2d[:, 0].copy()
        elif u_b >= u_grid[-1]: P_along_a = slice2d[:, -1].copy()
        else:
            j = _find_interval(u_grid, u_b, n)
            w = (u_b - u_grid[j]) / (u_grid[j+1] - u_grid[j])
            P_along_a = (1 - w) * slice2d[:, j] + w * slice2d[:, j+1]
        roots = find_pchip_roots(u_grid, P_along_a, p_target)
        for u_a, dPdu_a in roots:
            if u_a <= u_grid[0]: P_along_b = slice2d[0, :].copy()
            elif u_a >= u_grid[-1]: P_along_b = slice2d[-1, :].copy()
            else:
                i = _find_interval(u_grid, u_a, n)
                w = (u_a - u_grid[i]) / (u_grid[i+1] - u_grid[i])
                P_along_b = (1 - w) * slice2d[i, :] + w * slice2d[i+1, :]
            pc_b = pchip_along(u_grid, P_along_b)
            dPdu_b = pchip_slope(pc_b, u_b)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            wt = w_b * w_a_pou / abs(dPdu_a)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    return A0, A1


def build_mu_pchip(P_vals, u_grid, p_grid, gl_u, gl_du, tau, n, nq):
    G_p = p_grid.size
    mu = np.empty((G_p, n))
    for k in range(n):
        u_k = u_grid[k]
        f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
        slice2d = P_vals[k, :, :].copy()
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1 = co_area_pchip(u_grid, slice2d, p, gl_u, gl_du, tau, n, nq)
            den = f0k*A0 + f1k*A1
            if den > 1e-300:
                mu[ip, k] = f1k*A1 / den
            else:
                mu[ip, k] = 0.5
    return mu


def crra_clear(m0, m1, m2, gamma, steps=80):
    eps = 1e-15
    lm0 = np.log(max(min(m0,1-eps),eps)/(1-max(min(m0,1-eps),eps)))
    lm1 = np.log(max(min(m1,1-eps),eps)/(1-max(min(m1,1-eps),eps)))
    lm2 = np.log(max(min(m2,1-eps),eps)/(1-max(min(m2,1-eps),eps)))
    a, b = 1e-12, 1-1e-12
    for _ in range(steps):
        m = 0.5*(a+b); lp = np.log(m/(1-m))
        e = 0.0
        for lmk in (lm0, lm1, lm2):
            ar = (lmk - lp)/gamma
            if ar > 60: ar = 60
            if ar < -60: ar = -60
            R = np.exp(ar)
            e += (R-1)/((1-m) + R*m)
        if e > 0: a = m
        else: b = m
    return 0.5*(a+b)


def phi_pchip(P_vals, u_grid, p_grid, gl_u, gl_du, tau, gamma, n, nq):
    mu = build_mu_pchip(P_vals, u_grid, p_grid, gl_u, gl_du, tau, n, nq)
    P_new = np.empty_like(P_vals)
    G_p = p_grid.size
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p_cell = P_vals[i,j,k]
                if p_cell <= p_grid[0]:
                    m0,m1,m2 = mu[0,i], mu[0,j], mu[0,k]
                elif p_cell >= p_grid[-1]:
                    m0,m1,m2 = mu[-1,i], mu[-1,j], mu[-1,k]
                else:
                    lo, hi = 0, G_p-1
                    while hi - lo > 1:
                        mid = (lo+hi)//2
                        if p_grid[mid] <= p_cell: lo = mid
                        else: hi = mid
                    w = (p_cell - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
                    m0 = (1-w)*mu[lo,i] + w*mu[hi,i]
                    m1 = (1-w)*mu[lo,j] + w*mu[hi,j]
                    m2 = (1-w)*mu[lo,k] + w*mu[hi,k]
                P_new[i,j,k] = crra_clear(m0, m1, m2, gamma)
    return P_new


def main():
    TAU, GAMMA, G, G_p, NQK = 2.0, 0.1, 9, 81, 12
    u = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)
    sd = 1/np.sqrt(TAU)
    f = 0.5*norm.pdf(u,-0.5,sd) + 0.5*norm.pdf(u,0.5,sd)
    W3 = f[:,None,None]*f[None,:,None]*f[None,None,:]; W3 /= W3.max()

    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    P0 = 1/(1+np.exp(-0.5*(U1+U2+U3)))

    def deficit_of(P):
        T = (U1+U2+U3).ravel()
        L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
        uT, inv = np.unique(np.round(T, 10), return_inverse=True)
        ss = float(np.sum((L - L.mean())**2)); w = 0.0
        for g in range(len(uT)):
            m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
        return w/ss

    print(f"PCHIP-strict K=3 test: tau={TAU}, gamma={GAMMA}, G={G}, G_p={G_p}, NQK={NQK}")
    print(f"Anchor (k3_coarea_limit): deficit ~ 0.282\n")
    print("=== LINEAR strict-h=0 (reference) ===")
    P_l = P0.copy(); best_l = (1e9, None)
    t0 = time.time()
    for it in range(40):
        Pn = phi_lin_strict_jit(P_l, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK)
        F = float(np.max(np.abs(Pn - P_l) * W3))
        if F < best_l[0]: best_l = (F, P_l.copy())
        if it % 5 == 0:
            print(f"  it{it:3d}: F_w={F:.3e}", flush=True)
        P_l = 0.5*P_l + 0.5*Pn
    print(f"  LINEAR best F_w = {best_l[0]:.3e} deficit={deficit_of(best_l[1]):.4f} ({time.time()-t0:.0f}s)\n")
    print("=== PCHIP strict-h=0 (smoother dP/du) ===")
    P_p = P0.copy(); best_p = (1e9, None)
    t0 = time.time()
    for it in range(40):
        Pn = phi_pchip(P_p, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK)
        F = float(np.max(np.abs(Pn - P_p) * W3))
        if F < best_p[0]: best_p = (F, P_p.copy())
        if it % 5 == 0:
            print(f"  it{it:3d}: F_w={F:.3e}", flush=True)
        P_p = 0.5*P_p + 0.5*Pn
    print(f"  PCHIP  best F_w = {best_p[0]:.3e} deficit={deficit_of(best_p[1]):.4f} ({time.time()-t0:.0f}s)\n")
    print(f"deficit improvement: {abs(deficit_of(best_p[1]) - 0.282):.4f} vs {abs(deficit_of(best_l[1]) - 0.282):.4f}")
    np.savez('/tmp/pchip_test.npz', P_lin=best_l[1], P_pchip=best_p[1])

if __name__ == "__main__":
    main()
