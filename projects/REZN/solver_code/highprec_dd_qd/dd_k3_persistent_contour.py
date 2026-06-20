"""Persistent-contour strict-h=0 K=3 operator.

Same co-area integrand as `lin_cdf_strict.co_area_pou_lin`, but the
roots {P_along(u) = p} are TRACKED across Phi-calls instead of
re-enumerated from scratch. Each call:
  1. Take previous root list for each (sweep axis, q, p-index, u_k-index).
  2. Newton-advance each root on the NEW P_along (smooth in P by IFT).
  3. Detect topology changes: drop roots that leave the domain,
     create roots for sign-changing cells that have no tracked root,
     merge roots that converge to a common point.
  4. Compute the POU co-area integral from the (advected, smooth) roots.

This eliminates the cell-membership flip discontinuity in Phi. The only
remaining non-smoothness is the Morse critical-value codim-1 event
(integrable, recoverable by adaptive damping).

Test: cold start at (tau=2, gamma=0.1) where the original strict op
stalls at F=0.99 (k3_strict_h0_exact). If persistent contour drives F
substantially lower and the deficit moves toward the anchor (0.282),
the diagnosis is confirmed.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
from scipy.stats import norm
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import (make_gl_for_u, phi_lin_strict_jit,
                                  f_signal_jit, _find_interval)


def linterp_along(u_grid, P_along, q, n):
    """Linear interp of piecewise-linear P_along at u=q."""
    if q <= u_grid[0]: return P_along[0]
    if q >= u_grid[n-1]: return P_along[n-1]
    i = _find_interval(u_grid, q, n)
    w = (q - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    return (1.0 - w) * P_along[i] + w * P_along[i+1]


def slope_along(u_grid, P_along, q, n):
    if q <= u_grid[0]:
        return (P_along[1] - P_along[0]) / (u_grid[1] - u_grid[0])
    if q >= u_grid[n-1]:
        return (P_along[n-1] - P_along[n-2]) / (u_grid[n-1] - u_grid[n-2])
    i = _find_interval(u_grid, q, n)
    return (P_along[i+1] - P_along[i]) / (u_grid[i+1] - u_grid[i])


def linterp_slice_at_a(u_grid, slice2d, u_a, n):
    if u_a <= u_grid[0]: return slice2d[0, :].copy()
    if u_a >= u_grid[n-1]: return slice2d[n-1, :].copy()
    i = _find_interval(u_grid, u_a, n)
    w = (u_a - u_grid[i]) / (u_grid[i+1] - u_grid[i])
    return (1.0 - w) * slice2d[i, :] + w * slice2d[i+1, :]


def linterp_slice_at_b(u_grid, slice2d, u_b, n):
    if u_b <= u_grid[0]: return slice2d[:, 0].copy()
    if u_b >= u_grid[n-1]: return slice2d[:, n-1].copy()
    j = _find_interval(u_grid, u_b, n)
    w = (u_b - u_grid[j]) / (u_grid[j+1] - u_grid[j])
    return (1.0 - w) * slice2d[:, j] + w * slice2d[:, j+1]


def advect_roots(u_grid, P_along, p_target, prev_roots, n, n_newton=5, tol=1e-12):
    """Newton-advance previous roots; drop those leaving domain."""
    new = []
    for u in prev_roots:
        # Initial bracket: keep within domain
        u = float(min(max(u, u_grid[0]), u_grid[-1]))
        for _ in range(n_newton):
            Pv = linterp_along(u_grid, P_along, u, n)
            sl = slope_along(u_grid, P_along, u, n)
            if abs(sl) < 1e-14: break
            du = -(Pv - p_target) / sl
            u_new = u + du
            # Project to domain
            if u_new < u_grid[0]: u_new = u_grid[0]
            elif u_new > u_grid[-1]: u_new = u_grid[-1]
            if abs(u_new - u) < tol: u = u_new; break
            u = u_new
        # Accept if final point is at a real root
        Pv = linterp_along(u_grid, P_along, u, n)
        if abs(Pv - p_target) < 1e-8:
            new.append(u)
    return new


def detect_new_roots(u_grid, P_along, p_target, existing, n):
    """Detect cells that have a sign-change but no tracked root in them."""
    existing_cells = set()
    for u in existing:
        if u <= u_grid[0]: existing_cells.add(0); continue
        if u >= u_grid[-1]: existing_cells.add(n-2); continue
        existing_cells.add(_find_interval(u_grid, u, n))
    new = list(existing)
    for j in range(n - 1):
        if j in existing_cells: continue
        a = P_along[j] - p_target
        b = P_along[j+1] - p_target
        if a * b < 0:
            slope = (P_along[j+1] - P_along[j]) / (u_grid[j+1] - u_grid[j])
            if abs(slope) > 1e-14:
                u_root = u_grid[j] + (p_target - P_along[j]) / slope
                new.append(u_root)
    return new


def dedupe_and_sort(roots, tol=1e-7):
    if not roots: return roots
    roots = sorted(roots)
    out = [roots[0]]
    for u in roots[1:]:
        if u - out[-1] > tol:
            out.append(u)
    return out


def co_area_persistent(u_grid, slice2d, p_target, gl_u_a, gl_du_a,
                          tau, n, nq, prev_state=None):
    """Persistent-root co-area; prev_state is dict with 'b' and 'a' root lists per q."""
    if prev_state is None:
        prev_state = {'b': [[] for _ in range(nq)], 'a': [[] for _ in range(nq)]}
    new_state = {'b': [], 'a': []}
    A0, A1 = 0.0, 0.0
    # Term over u_b roots, sweeping u_a
    for q in range(nq):
        u_a = gl_u_a[q]; w_a = gl_du_a[q]
        f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
        P_along_b = linterp_slice_at_a(u_grid, slice2d, u_a, n)
        prev = prev_state['b'][q] if q < len(prev_state['b']) else []
        advected = advect_roots(u_grid, P_along_b, p_target, prev, n)
        with_new = detect_new_roots(u_grid, P_along_b, p_target, advected, n)
        roots = dedupe_and_sort(with_new)
        new_state['b'].append(roots)
        for u_b in roots:
            dPdu_b = slope_along(u_grid, P_along_b, u_b, n)
            P_along_a = linterp_slice_at_b(u_grid, slice2d, u_b, n)
            dPdu_a = slope_along(u_grid, P_along_a, u_a, n)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_b_pou = dPdu_b*dPdu_b / denom
            if abs(dPdu_b) < 1e-300: continue
            f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
            wt = w_a * w_b_pou / abs(dPdu_b)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    # Term over u_a roots, sweeping u_b
    for q in range(nq):
        u_b = gl_u_a[q]; w_b = gl_du_a[q]
        f0b = f_signal_jit(u_b, 0, tau); f1b = f_signal_jit(u_b, 1, tau)
        P_along_a = linterp_slice_at_b(u_grid, slice2d, u_b, n)
        prev = prev_state['a'][q] if q < len(prev_state['a']) else []
        advected = advect_roots(u_grid, P_along_a, p_target, prev, n)
        with_new = detect_new_roots(u_grid, P_along_a, p_target, advected, n)
        roots = dedupe_and_sort(with_new)
        new_state['a'].append(roots)
        for u_a in roots:
            dPdu_a = slope_along(u_grid, P_along_a, u_a, n)
            P_along_b = linterp_slice_at_a(u_grid, slice2d, u_a, n)
            dPdu_b = slope_along(u_grid, P_along_b, u_b, n)
            denom = dPdu_a*dPdu_a + dPdu_b*dPdu_b
            if denom < 1e-300: continue
            w_a_pou = dPdu_a*dPdu_a / denom
            if abs(dPdu_a) < 1e-300: continue
            f0a = f_signal_jit(u_a, 0, tau); f1a = f_signal_jit(u_a, 1, tau)
            wt = w_b * w_a_pou / abs(dPdu_a)
            A0 += wt * f0a * f0b; A1 += wt * f1a * f1b
    return A0, A1, new_state


def build_mu_persistent(P_vals, u_grid, p_grid, gl_u, gl_du, tau, n, nq, prev_states=None):
    """Build mu_table with persistent contour state per (k, ip)."""
    G_p = p_grid.size
    mu_table = np.empty((G_p, n))
    if prev_states is None:
        prev_states = [[None for _ in range(G_p)] for _ in range(n)]
    new_states = [[None for _ in range(G_p)] for _ in range(n)]
    for k_node in range(n):
        u_k = u_grid[k_node]
        f0k = f_signal_jit(u_k, 0, tau); f1k = f_signal_jit(u_k, 1, tau)
        slice2d = P_vals[k_node, :, :].copy()
        for ip in range(G_p):
            p = p_grid[ip]
            A0, A1, st = co_area_persistent(u_grid, slice2d, p, gl_u, gl_du,
                                                tau, n, nq, prev_states[k_node][ip])
            new_states[k_node][ip] = st
            den = f0k*A0 + f1k*A1
            if den > 1e-300:
                mu_table[ip, k_node] = f1k*A1 / den
            else:
                mu_table[ip, k_node] = 0.5
    return mu_table, new_states


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


def phi_persistent(P_vals, u_grid, p_grid, gl_u, gl_du, tau, gamma, n, nq, prev_states=None):
    """One Phi-pass using persistent contour."""
    mu_table, new_states = build_mu_persistent(P_vals, u_grid, p_grid, gl_u, gl_du,
                                                       tau, n, nq, prev_states)
    P_new = np.empty_like(P_vals)
    G_p = p_grid.size
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p_cell = P_vals[i,j,k]
                # Interp mu at p_cell for each u_k slice
                if p_cell <= p_grid[0]: m0 = mu_table[0, i]; m1 = mu_table[0, j]; m2 = mu_table[0, k]
                elif p_cell >= p_grid[-1]: m0 = mu_table[-1, i]; m1 = mu_table[-1, j]; m2 = mu_table[-1, k]
                else:
                    lo, hi = 0, G_p - 1
                    while hi - lo > 1:
                        mid = (lo+hi)//2
                        if p_grid[mid] <= p_cell: lo = mid
                        else: hi = mid
                    w = (p_cell - p_grid[lo]) / (p_grid[hi] - p_grid[lo])
                    m0 = (1-w)*mu_table[lo, i] + w*mu_table[hi, i]
                    m1 = (1-w)*mu_table[lo, j] + w*mu_table[hi, j]
                    m2 = (1-w)*mu_table[lo, k] + w*mu_table[hi, k]
                P_new[i,j,k] = crra_clear(m0, m1, m2, gamma)
    return P_new, new_states


def main():
    TAU, GAMMA, G, G_p, NQK = 2.0, 0.1, 9, 81, 12  # smaller G_p, NQK for speed
    u = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)

    print(f"Persistent contour strict-h=0 K=3 test: tau={TAU}, gamma={GAMMA}, "
          f"G={G}, G_p={G_p}, NQK={NQK}")
    print(f"Anchor (k3_coarea_limit): deficit ~ 0.282")
    print()

    # cold start: P = sigmoid(0.5 * sum(u))
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    P = 1/(1+np.exp(-0.5*(U1+U2+U3)))

    # First call against ORIGINAL strict (no persistence)
    print("=== ORIGINAL strict-h=0 ===")
    P_orig = P.copy(); best_o = (1e9, None)
    sd = 1/np.sqrt(TAU)
    f = 0.5*norm.pdf(u,-0.5,sd) + 0.5*norm.pdf(u,0.5,sd)
    W3 = f[:,None,None]*f[None,:,None]*f[None,None,:]; W3 /= W3.max()
    t0 = time.time()
    for it in range(40):
        Pn = phi_lin_strict_jit(P_orig, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK)
        F = float(np.max(np.abs(Pn - P_orig) * W3))
        F_inf = float(np.max(np.abs(Pn - P_orig)))
        if F < best_o[0]: best_o = (F, P_orig.copy())
        if it % 5 == 0 or it == 39:
            print(f"  it{it:3d}: F_w={F:.3e} F_inf={F_inf:.3e}", flush=True)
        P_orig = 0.5*P_orig + 0.5*Pn
    print(f"  ORIGINAL best F_w = {best_o[0]:.3e} ({time.time()-t0:.0f}s)")

    print()
    print("=== PERSISTENT-CONTOUR strict-h=0 ===")
    P_p = P.copy(); best_p = (1e9, None)
    state = None
    t0 = time.time()
    for it in range(40):
        Pn, state = phi_persistent(P_p, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK, state)
        F = float(np.max(np.abs(Pn - P_p) * W3))
        F_inf = float(np.max(np.abs(Pn - P_p)))
        if F < best_p[0]: best_p = (F, P_p.copy())
        if it % 5 == 0 or it == 39:
            print(f"  it{it:3d}: F_w={F:.3e} F_inf={F_inf:.3e}", flush=True)
        P_p = 0.5*P_p + 0.5*Pn

    print(f"  PERSISTENT best F_w = {best_p[0]:.3e} ({time.time()-t0:.0f}s)")

    # Deficit at each best iterate
    def deficit_of(P):
        T = (U1+U2+U3).ravel()
        L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
        uT, inv = np.unique(np.round(T, 10), return_inverse=True)
        ss = float(np.sum((L - L.mean())**2)); w = 0.0
        for g in range(len(uT)):
            m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
        return w/ss
    print()
    print(f"deficit(original best) = {deficit_of(best_o[1]):.4f}")
    print(f"deficit(persistent best) = {deficit_of(best_p[1]):.4f}")
    print(f"anchor deficit          = 0.282 (k3_coarea_limit)")
    np.savez('/tmp/persistent_test.npz', P_orig=best_o[1], P_pers=best_p[1],
             F_orig=best_o[0], F_pers=best_p[0])

if __name__ == "__main__":
    main()
