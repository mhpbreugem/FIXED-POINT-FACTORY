"""FAST solver for the h-FREE SMOOTH co-area K=3 CRRA REE.

SAME operator as hfree_operator.py (NO kernel / bandwidth / smoothing param);
ONLY the solver is changed. Two orthogonal speedups over the original
dense-FD Newton (~90s/step at G=9):

  1. MATRIX-FREE Newton-Krylov (scipy newton_krylov + lgmres) at ALL grids,
     including G=9. Instead of building a dense N=729 finite-difference
     Jacobian (729 operator evals / Newton step), the Krylov solver only
     needs ~20-30 Jacobian-vector products / step, each one a single FD
     directional derivative = 1 operator eval (vs 729).

  2. SYMMETRY REDUCTION. The K=3 agents are identical (tau,gamma,W equal) so
     the price P(u1,u2,u3) is invariant under permuting the 3 axes, and the
     operator Phi commutes with permutations (verified asym ~1e-16). We solve
     on the reduced set of sorted-multiset cells i<=j<=l: 729 -> C(11,3)=165
     unknowns. expand(reduced)->full, apply Phi, symmetrize (orbit-average),
     read reduced. This both shrinks the Krylov problem and exactly projects
     onto the symmetric subspace where the PR fixed point lives.

The h-free RESULT is unchanged: same fixed point, same deficit (~0.291 at
G=9, tau=2, gamma=0.1) as k3_coarea_limit / k3_hfree_smooth.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import sys
import time
from itertools import combinations_with_replacement
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp")
import hfree_operator as H

UMAX = 4.0
NQ = 40
SUB = 4


# ----------------------------------------------------------------------
# Symmetry reducer for K=3 over the inner grid (sorted-multiset cells).
# ----------------------------------------------------------------------
class SymReducer3:
    K = 3

    def __init__(self, G):
        self.G = G
        self.multisets = list(combinations_with_replacement(range(G), 3))
        self.n_red = len(self.multisets)
        ms_to_red = {ms: r for r, ms in enumerate(self.multisets)}
        red_of_cell = np.empty((G, G, G), dtype=np.int64)
        orbit_count = np.zeros(self.n_red, dtype=np.int64)
        for i in range(G):
            for j in range(G):
                for l in range(G):
                    r = ms_to_red[tuple(sorted((i, j, l)))]
                    red_of_cell[i, j, l] = r
                    orbit_count[r] += 1
        self.red_of_cell = red_of_cell
        self.orbit_count = orbit_count

    def expand(self, vec_red):
        """reduced (n_red,) -> full inner block (G,G,G) by broadcasting orbit."""
        return vec_red[self.red_of_cell]

    def reduce(self, full):
        """full (G,G,G) -> reduced via orbit-average (symmetrize + read)."""
        acc = np.zeros(self.n_red)
        np.add.at(acc, self.red_of_cell.ravel(), full.ravel())
        return acc / self.orbit_count

    def read_rep(self, full):
        """read reduced at sorted representatives (no averaging)."""
        out = np.empty(self.n_red)
        for r, ms in enumerate(self.multisets):
            out[r] = full[ms]
        return out


# ----------------------------------------------------------------------
# Residual builders.
# ----------------------------------------------------------------------
def make_full_resid(G, ui, gn, gw, tau, gam, W):
    def F(x):
        P = x.reshape((G, G, G))
        return (H.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB) - P).ravel()
    return F


def make_red_resid(red: SymReducer3, ui, gn, gw, tau, gam, W):
    G = red.G

    def Fred(vec_red):
        P = red.expand(vec_red)
        Pn = H.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB)
        return red.reduce(Pn) - vec_red
    return Fred


# ----------------------------------------------------------------------
# Matrix-free Newton-Krylov solve (symmetry-reduced).
# ----------------------------------------------------------------------
def nk_solve(G, x0_red, red, ui, gn, gw, tau, gam, W,
             f_tol=1e-10, maxiter=80, verbose=None):
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize._nonlin import NoConvergence

    Fred = make_red_resid(red, ui, gn, gw, tau, gam, W)
    cnt = {"n": 0}

    def cb(x, fx):
        cnt["n"] += 1
        if verbose:
            verbose(cnt["n"], float(np.max(np.abs(fx))))

    conv = True
    try:
        # inner_maxiter caps the Krylov (lgmres) inner products per outer step:
        # each inner product is ONE FD directional-derivative = one operator
        # eval (vs the dense FD Jacobian's N=729 evals/step). rdiff sets the FD
        # step for the Jacobian-vector products.
        sol = newton_krylov(Fred, x0_red, f_tol=f_tol, maxiter=maxiter,
                            method="lgmres", inner_maxiter=20, outer_k=8,
                            rdiff=1e-6, callback=cb)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel()
        conv = False
    Finf = float(np.max(np.abs(Fred(sol))))
    return sol, Finf, cnt["n"], conv


# ----------------------------------------------------------------------
# Metrics (revelation deficit etc.) -- identical formulas to hfree_nail.
# ----------------------------------------------------------------------
def metrics(P, ui, TAU):
    G = ui.size
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    deficit = float(np.sum((y - pr) ** 2) /
                    max(np.sum((y - y.mean()) ** 2), 1e-30))
    P_FR = 1.0 / (1.0 + np.exp(-T))
    d_FR = float(np.sqrt(np.mean((P - P_FR) ** 2)))
    X = np.column_stack([U1.ravel(), U2.ravel(), U3.ravel(), np.ones(U1.size)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]),
                b1=float(coef[0]), b2=float(coef[1]), b3=float(coef[2]))


def warm_start_red(G, red, ui, TAU):
    """Warm start reduced vector from the kernel-nailed co-area PR if present,
    else from a fully-revealing seed (symmetric)."""
    path = os.path.join(
        "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points",
        "k3_coarea_limit", f"P_inner_G{G}.npy")
    if os.path.exists(path):
        P0 = np.load(path)
        if P0.shape == (G, G, G):
            # symmetrize then read reduced
            return red.reduce(P0)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = TAU * (U1 + U2 + U3)
    P_fr = np.clip(1.0 / (1.0 + np.exp(-T)), 1e-9, 1 - 1e-9)
    return red.reduce(P_fr)


def solve_hfree_pr(G, TAU, GAMMA, x0_red=None, f_tol=1e-10, verbose=None):
    """Nail the h-free PR fixed point. Returns (P_full, Finf, info)."""
    ui = np.linspace(-UMAX, UMAX, G)
    gn, gw = H.gauss_legendre(NQ, -UMAX, UMAX)
    tau = np.full(3, TAU)
    gam = np.full(3, GAMMA)
    W = np.full(3, 1.0)
    red = SymReducer3(G)
    if x0_red is None:
        x0_red = warm_start_red(G, red, ui, TAU)
    sol, Finf, iters, conv = nk_solve(
        G, x0_red, red, ui, gn, gw, tau, gam, W,
        f_tol=f_tol, verbose=verbose)
    P_full = red.expand(sol)
    m = metrics(P_full, ui, TAU)
    info = dict(G=G, tau=TAU, gamma=GAMMA, Finf=Finf, iters=iters,
                converged=conv, n_red=red.n_red, **m)
    return P_full, sol, Finf, info


if __name__ == "__main__":
    # Quick standalone nail at the reference point.
    t = time.time()
    P, sol, Finf, info = solve_hfree_pr(9, 2.0, 0.1,
                                        verbose=lambda n, f: print(
                                            f"  nk it {n} ||F||={f:.3e}"))
    info["walltime_s"] = round(time.time() - t, 2)
    print("RESULT", info)
