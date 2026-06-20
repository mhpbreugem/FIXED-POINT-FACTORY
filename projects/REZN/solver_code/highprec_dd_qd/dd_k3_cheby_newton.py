"""Pure Newton on the Cheby-native operator with symmetric reduction.
Tests: does Newton converge cleanly to a FP (even if non-monotone)?
If yes, the Cheby-native operator HAS a clean FP and the alternating-
projection failure was just method choice; if no, the operator is genuinely
non-smooth.
"""
import os, sys, time, json
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from itertools import combinations_with_replacement, permutations
import scipy.linalg as sla
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          lookup_analytic_lobatto)
from cheby_numba import crra_clear_jit


class SymRed3:
    def __init__(self, G):
        self.G = G
        self.triples = list(combinations_with_replacement(range(G), 3))
        self.n_red = len(self.triples)
    def reduce(self, P):
        return np.array([P[t[0], t[1], t[2]] for t in self.triples])
    def expand(self, vec):
        P = np.empty((self.G, self.G, self.G))
        for i, t in enumerate(self.triples):
            v = vec[i]
            for p in set(permutations(t)):
                P[p[0], p[1], p[2]] = v
        return P


def phi_cheby(P, u_grid, U_MAX, tau, gamma, NQK=24):
    G = u_grid.size
    coefs = cheby_fit_lobatto_3d(P, U_MAX)
    P_new = np.empty((G, G, G))
    for i in range(G):
        for j in range(G):
            for k in range(G):
                p_cell = cheb.chebval3d(u_grid[i]/U_MAX, u_grid[j]/U_MAX,
                                              u_grid[k]/U_MAX, coefs)
                p_cell = float(min(max(p_cell, 1e-9), 1.0 - 1e-9))
                mu0 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[i], tau, NQK)
                mu1 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[j], tau, NQK)
                mu2 = lookup_analytic_lobatto(coefs, U_MAX, p_cell, u_grid[k], tau, NQK)
                P_new[i, j, k] = crra_clear_jit(mu0, mu1, mu2, gamma)
    return P_new


def F_call(red, x_red, u_grid, U_MAX, tau, gamma, NQK=24):
    P = red.expand(x_red)
    P_new = phi_cheby(P, u_grid, U_MAX, tau, gamma, NQK)
    return red.reduce(P_new) - x_red


def build_jacobian(red, x_red, u_grid, U_MAX, tau, gamma, eps=1e-7, NQK=20):
    n = red.n_red
    F0 = F_call(red, x_red, u_grid, U_MAX, tau, gamma, NQK)
    J = np.empty((n, n))
    t0 = time.time()
    for k in range(n):
        xp = x_red.copy(); xp[k] += eps
        Fp = F_call(red, xp, u_grid, U_MAX, tau, gamma, NQK)
        J[:, k] = (Fp - F0) / eps
        if k % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n - k - 1) / max(1, k+1)
            print(f"    Jacobian col {k+1}/{n}, ETA {eta:.0f}s", flush=True)
    return J, F0


def main(gamma=100.0, tau=0.1, G=7, max_newton=10, target=1e-10):
    print(f"=== Cheby-native pure Newton at gamma={gamma}, tau={tau}, G={G} ===",
          flush=True)
    u_grid, U_MAX = lobatto_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    red = SymRed3(G)
    x = red.reduce(P)
    print(f"sym-reduced unknowns: {red.n_red}", flush=True)
    print(f"\nBuilding FD Jacobian (this is the slow part)...", flush=True)
    t0 = time.time()
    J, F0 = build_jacobian(red, x, u_grid, U_MAX, tau, gamma, NQK=16)
    print(f"  Jacobian built in {time.time()-t0:.0f}s, init |F|={float(np.max(np.abs(F0))):.3e}",
          flush=True)
    # Chord Newton (reuse J)
    try: lu, piv = sla.lu_factor(J)
    except Exception as e:
        print(f"LU FAIL: {e}"); return
    Jinv_F = sla.lu_solve((lu, piv), F0)
    x_best = x.copy(); F_best = float(np.max(np.abs(F0)))
    print(f"\nChord Newton steps:", flush=True)
    for it in range(max_newton):
        F = F_call(red, x, u_grid, U_MAX, tau, gamma, NQK=24)
        Fi = float(np.max(np.abs(F)))
        if Fi < F_best: F_best = Fi; x_best = x.copy()
        print(f"  step {it+1}: |F|={Fi:.3e}", flush=True)
        if Fi < target: break
        # Newton step: solve J dx = -F
        dx = sla.lu_solve((lu, piv), -F)
        # Damped step
        x = x + 0.7 * dx
        x = np.clip(x, 1e-9, 1-1e-9)
        if it >= 2:
            F_now = F_call(red, x, u_grid, U_MAX, tau, gamma, NQK=24)
            Fi_now = float(np.max(np.abs(F_now)))
            if Fi_now > Fi:
                # Refresh Jacobian
                print(f"    F diverging; refresh Jacobian...", flush=True)
                J, F0 = build_jacobian(red, x, u_grid, U_MAX, tau, gamma, NQK=16)
                try: lu, piv = sla.lu_factor(J)
                except: break
    print(f"\nFinal best |F|={F_best:.3e}", flush=True)
    P_final = red.expand(x_best)
    np.savez(f"/tmp/dd_k3_cheby_newton_g{gamma}_t{tau}_G{G}.npz",
                P=P_final, gamma=gamma, tau=tau, G=G, F=F_best)


if __name__ == "__main__":
    main()
