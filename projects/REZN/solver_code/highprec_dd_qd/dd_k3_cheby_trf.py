"""Cheby-native operator solved by trust-region reflective least squares
(scipy.optimize.least_squares with method='trf').
"""
import os, sys, time
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from itertools import combinations_with_replacement, permutations
from scipy.optimize import least_squares
from numpy.polynomial import chebyshev as cheb
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


def phi_cheby(P, u_grid, U_MAX, tau, gamma, NQK=20):
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


def main(gamma=100.0, tau=0.1, G=7):
    print(f"=== Cheby-native TRF at gamma={gamma}, tau={tau}, G={G} ===", flush=True)
    u_grid, U_MAX = lobatto_grid(G)
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P_init = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    red = SymRed3(G)
    x0 = red.reduce(P_init)
    print(f"sym-reduced unknowns: {red.n_red}", flush=True)

    n_eval = [0]
    t_start = time.time()
    def F(x_red):
        n_eval[0] += 1
        P = red.expand(x_red)
        P_new = phi_cheby(P, u_grid, U_MAX, tau, gamma)
        residual = (red.reduce(P_new) - x_red)
        if n_eval[0] % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  call #{n_eval[0]}: |F|={float(np.max(np.abs(residual))):.3e} "
                  f"({elapsed:.0f}s elapsed)", flush=True)
        return residual

    # trf with 2-point FD jacobian, bounds [0, 1]
    t0 = time.time()
    res = least_squares(F, x0, jac="2-point", method="trf",
                              bounds=(1e-9, 1.0 - 1e-9),
                              max_nfev=200, ftol=1e-14, xtol=1e-14, gtol=1e-14,
                              verbose=2)
    print(f"\nTRF done in {time.time()-t0:.0f}s, status={res.status}, "
          f"|F|_inf={float(np.max(np.abs(res.fun))):.3e}, "
          f"|F|_2={float(np.linalg.norm(res.fun)):.3e}", flush=True)
    print(f"  message: {res.message}", flush=True)
    print(f"  n_eval: {n_eval[0]}", flush=True)
    P_final = red.expand(res.x)
    # Monotonicity check
    diffs = [np.diff(P_final, axis=ax) for ax in range(3)]
    viols = sum(int((d < 0).sum()) for d in diffs)
    min_d = min(float(d.min()) for d in diffs)
    print(f"  monotonicity: {viols} viol, min_diff={min_d:.3e}", flush=True)
    # Cheby fit + edge
    coefs = cheby_fit_lobatto_3d(P_final, U_MAX)
    edge = float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))
    print(f"  Cheby edge: {edge:.3e}", flush=True)
    np.savez(f"/tmp/dd_k3_trf_g{gamma}_t{tau}_G{G}.npz",
                P=P_final, gamma=gamma, tau=tau, G=G,
                F=float(np.max(np.abs(res.fun))), edge=edge)


if __name__ == "__main__":
    main()
