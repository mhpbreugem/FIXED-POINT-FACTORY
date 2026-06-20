"""Solver for the MORSE-ROBUST h-free (no-kernel) K=3 CRRA REE.

Same solver pattern as ../k3_hfree_fast/hfree_fast_solver.py:
  - SymReducer3 symmetry reduction (729 -> C(G+2,3) reduced cells),
  - ORBIT-AVERAGE residual  F(v) = reduce(Phi(expand(v))) - v   (REQUIRED for
    convergence; the operator is not exactly axis-symmetric pointwise, so the
    read-representative residual fails to converge -- orbit-average projects
    onto the symmetric subspace where the PR fixed point lives),
  - matrix-free Newton-Krylov (lgmres) polish, Picard pre-stage fallback.

ONLY the operator is swapped: hfree_morse_operator (eps_c-softened geometric
weight) in place of hfree_operator.  NO kernel / bandwidth / smoothing param --
eps_c*h^2 is a grid-tied numerical regularization that -> 0 as G -> inf.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import sys
import time
from itertools import combinations_with_replacement
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfree_morse_operator as H

UMAX = 4.0
NQ = 40
SUB = 4
REFINE = 1          # adaptive transverse refinement off (eps_c softening suffices)
EPS_C = 1.0e-3      # Morse softening: geo = |dB| / (|gradP|^2 + eps_c*h^2).
                    # eps_c=1e-3 nails G=9 (7.9e-15) and G=13 (1.8e-10) where the
                    # unregularized operator floors; eps2=eps_c*h^2 -> 0 as G->inf.


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
        return vec_red[self.red_of_cell]

    def reduce(self, full):
        """orbit-average: symmetrize (average over the permutation orbit) + read."""
        acc = np.bincount(self.red_of_cell.ravel(), weights=full.ravel(),
                          minlength=self.n_red)
        return acc / self.orbit_count


def make_red_resid(red, ui, gn, gw, tau, gam, W, eps_c=EPS_C):
    def Fred(vec_red):
        P = red.expand(vec_red)
        Pn = H.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB, REFINE, eps_c)
        return red.reduce(Pn) - vec_red
    return Fred


def picard_prestage(x0_red, red, ui, gn, gw, tau, gam, W, iters=60, eps_c=EPS_C):
    v = x0_red.copy()
    best = v.copy()
    Fn = red.reduce(H.phi_hfree(red.expand(v), ui, gn, gw, tau, gam, W,
                                SUB, REFINE, eps_c))
    best_norm = float(np.max(np.abs(Fn - v)))
    for _ in range(iters):
        vn = red.reduce(H.phi_hfree(red.expand(v), ui, gn, gw, tau, gam, W,
                                    SUB, REFINE, eps_c))
        nrm = float(np.max(np.abs(vn - v)))
        v = vn
        if nrm < best_norm:
            best_norm = nrm
            best = v.copy()
        if best_norm < 1e-10:
            break
    return best, best_norm


def nk_solve(x0_red, red, ui, gn, gw, tau, gam, W,
             f_tol=1e-9, maxiter=200, inner_maxiter=40, verbose=None,
             eps_c=EPS_C):
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize._nonlin import NoConvergence
    Fred = make_red_resid(red, ui, gn, gw, tau, gam, W, eps_c)
    cnt = {"n": 0}

    def cb(x, fx):
        cnt["n"] += 1
        if verbose:
            verbose(cnt["n"], float(np.max(np.abs(fx))))
    conv = True
    try:
        sol = newton_krylov(Fred, x0_red, f_tol=f_tol, maxiter=maxiter,
                            method="lgmres", inner_maxiter=inner_maxiter,
                            callback=cb)
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel()
        conv = False
    Finf = float(np.max(np.abs(Fred(sol))))
    return sol, Finf, cnt["n"], conv


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
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))


def solve_morse_pr(G, TAU, GAMMA, x0_red=None, f_tol=1e-9, nk_maxiter=60,
                   nk_probe_iter=15, picard_iters=40, n_cycles=5,
                   inner_maxiter=60, verbose=None, eps_c=EPS_C):
    """Nail the Morse-robust h-free PR fixed point.  Returns (P_full, sol_red,
    Finf, info).

    Robust strategy: a short NK probe from the warm start, then up to n_cycles
    of (Picard pre-stage -> NK polish), always RE-SEEDING from the global-best
    iterate and keeping the global best.  Newton-Krylov can stall at a Picard
    endpoint when the Jacobian I-Phi' is locally ill-conditioned (a few cells
    near a critical price); re-running a short Picard relaxation off the stalled
    point perturbs into a better-conditioned neighbourhood from which NK
    descends.  Each cycle starts a FRESH Krylov subspace.  Stops as soon as
    f_tol is met."""
    ui = np.linspace(-UMAX, UMAX, G)
    gn, gw = H.gauss_legendre(NQ, -UMAX, UMAX)
    tau = np.full(3, TAU)
    gam = np.full(3, GAMMA)
    W = np.full(3, 1.0)
    red = SymReducer3(G)
    if x0_red is None:
        x0_red = warm_start_red(G, red, ui, TAU)

    sol, Finf, iters, conv = nk_solve(
        x0_red, red, ui, gn, gw, tau, gam, W,
        f_tol=f_tol, maxiter=nk_probe_iter, inner_maxiter=inner_maxiter,
        verbose=verbose, eps_c=eps_c)
    best_sol, best_Finf = sol.copy(), Finf
    used_picard = False
    picard_norm = None
    total_iters = iters

    c = 0
    while best_Finf > f_tol and c < n_cycles:
        c += 1
        used_picard = True
        # Picard relaxation off the current best (slides into a smoother basin)
        v0, pn = picard_prestage(
            best_sol, red, ui, gn, gw, tau, gam, W, iters=picard_iters,
            eps_c=eps_c)
        picard_norm = pn
        if pn < best_Finf:
            best_sol, best_Finf = v0.copy(), pn
        # NK polish from the freshly-relaxed Picard endpoint (better-conditioned
        # neighbourhood than a stalled NK iterate)
        sol2, Finf2, it2, conv2 = nk_solve(
            v0, red, ui, gn, gw, tau, gam, W,
            f_tol=f_tol, maxiter=nk_maxiter, inner_maxiter=inner_maxiter,
            verbose=verbose, eps_c=eps_c)
        total_iters += it2
        if Finf2 < best_Finf:
            best_sol, best_Finf, conv = sol2.copy(), Finf2, conv2

    sol, Finf, iters = best_sol, best_Finf, total_iters
    P_full = red.expand(sol)
    m = metrics(P_full, ui, TAU)
    info = dict(G=G, tau=TAU, gamma=GAMMA, Finf=Finf, iters=iters,
                converged=bool(Finf <= f_tol), n_red=red.n_red,
                used_picard=used_picard, picard_norm=picard_norm,
                eps_c=eps_c, **m)
    return P_full, sol, Finf, info


def warm_start_red(G, red, ui, TAU):
    """Warm start at the BASE grid: prefer the kernel-nailed co-area PR (a good
    basin seed), else the fully-revealing symmetric seed.  Finer grids use
    self-continuation from the previous nailed Morse solution (passed in)."""
    kp = os.path.join(BASE, "k3_coarea_limit", f"P_inner_G{G}.npy")
    if os.path.exists(kp):
        P0 = np.load(kp)
        if P0.shape == (G, G, G):
            return red.reduce(np.clip(P0, 1e-9, 1 - 1e-9))
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = TAU * (U1 + U2 + U3)
    P_fr = np.clip(1.0 / (1.0 + np.exp(-T)), 1e-9, 1 - 1e-9)
    return red.reduce(P_fr)
