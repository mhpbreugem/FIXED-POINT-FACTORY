"""Sanity tests for the K=4 implementation. Cheap: G=5 throughout.

Run:  taskset -c 0 python -m code.smoke
"""

from __future__ import annotations

import numpy as np

from .config import Config, SolverConfig
from .contour_K4 import init_no_learning, phi_K4, residual_inf
from .demand import x_crra, clear_crra
from .metrics import revelation_deficit
from .solver import solve
from .symmetry import is_symmetric, symmetrize


def test_demand_monotone_in_p() -> None:
    gam = 0.5
    W = 1.0
    mu = 0.7
    ps = np.linspace(0.05, 0.95, 50)
    xs = np.array([x_crra(mu, p, gam, W) for p in ps])
    diffs = np.diff(xs)
    assert np.all(diffs <= 1.0e-12), "x_crra must be non-increasing in p"
    print("[smoke] x_crra monotone-decreasing in p:           OK")


def test_market_clearing_homogeneous() -> None:
    """Homogeneous posteriors -> p = mu (no trade clears at the common belief)."""
    mu = 0.62
    gam_vec = np.full(4, 0.5)
    W_vec = np.full(4, 1.0)
    mu_vec = np.full(4, mu)
    p = clear_crra(mu_vec, gam_vec, W_vec)
    assert abs(p - mu) < 1.0e-10, f"expected p == mu, got {p}"
    print(f"[smoke] homogeneous mu clears at mu (err={abs(p - mu):.2e}): OK")


def test_no_learning_init_symmetric() -> None:
    """No-learning P should be invariant under axis permutations."""
    cfg = Config(K=4, G=5, gamma=0.5, tau=2.0, cara=False)
    P0 = init_no_learning(cfg.u_grid(), cfg.tau_vec(),
                          cfg.gamma_vec(), cfg.W_vec(), cfg.cara)
    assert is_symmetric(P0, atol=1.0e-10), "P0 not symmetric"
    print("[smoke] no-learning P0 symmetric in axes:         OK")


def test_cara_full_revelation_no_learning() -> None:
    """Under CARA, the no-learning equilibrium IS fully revealing."""
    cfg = Config(K=4, G=5, gamma=1.0, tau=2.0, cara=True)
    P0 = init_no_learning(cfg.u_grid(), cfg.tau_vec(),
                          cfg.gamma_vec(), cfg.W_vec(), cfg.cara)
    deficit = revelation_deficit(P0, cfg.u_grid(), cfg.tau_vec(), cfg.K)
    print(f"[smoke] CARA no-learning 1-R^2 = {deficit:.3e}    (expect ~0)")
    assert deficit < 1.0e-8, f"CARA should give 1-R^2 ~ 0, got {deficit}"
    print("[smoke] CARA no-learning is fully revealing:      OK")


def test_phi_one_step_runs_and_symmetric() -> None:
    """One Phi step must execute and (after symmetrisation) preserve symmetry."""
    cfg = Config(K=4, G=5, gamma=0.5, tau=2.0, cara=False)
    u, tv, gv, Wv = cfg.u_grid(), cfg.tau_vec(), cfg.gamma_vec(), cfg.W_vec()
    P0 = init_no_learning(u, tv, gv, Wv, cfg.cara)
    P1 = phi_K4(P0, u, tv, gv, Wv, cfg.cara)
    P1s = symmetrize(P1)
    r = residual_inf(P0, P1s)
    print(f"[smoke] one Phi step: ||P0 - sym(Phi(P0))||inf = {r:.3e}")
    assert is_symmetric(P1s, atol=1.0e-10), "symmetrized step not symmetric"
    print("[smoke] one Phi step runs and symmetrises:        OK")


def test_cara_phi_preserves_full_revelation() -> None:
    """Under CARA, Phi maps the FR manifold {logit p = c T*} into itself.

    The no-learning seed sits at c = 1/K; a single Phi step ought to
    move it (toward c = 1, the REE FR price), but the new array must
    still be logit-affine in T* (1 - R^2 ~= 0).
    """
    cfg = Config(K=4, G=5, gamma=1.0, tau=2.0, cara=True)
    u, tv, gv, Wv = cfg.u_grid(), cfg.tau_vec(), cfg.gamma_vec(), cfg.W_vec()
    P0 = init_no_learning(u, tv, gv, Wv, cfg.cara)
    P1 = symmetrize(phi_K4(P0, u, tv, gv, Wv, cfg.cara))
    deficit_after = revelation_deficit(P1, u, tv, cfg.K)
    print(f"[smoke] CARA: 1-R^2 after one Phi step = {deficit_after:.3e}")
    assert deficit_after < 1.0e-6, \
        f"CARA Phi step left FR manifold: 1-R^2 = {deficit_after}"
    print("[smoke] CARA: Phi preserves full revelation:      OK")


def test_cara_anderson_converges_to_full_revelation() -> None:
    """Anderson under CARA must converge to a fully-revealing price."""
    cfg = Config(K=4, G=5, gamma=1.0, tau=2.0, cara=True)
    scfg = SolverConfig(method="anderson", max_iters=30, tol=1.0e-8,
                        anderson_m=6, symmetrize=True, verbose=False,
                        checkpoint_every=0)
    u, tv, gv, Wv = cfg.u_grid(), cfg.tau_vec(), cfg.gamma_vec(), cfg.W_vec()
    P0 = init_no_learning(u, tv, gv, Wv, cfg.cara)

    def phi_fn(P: np.ndarray) -> np.ndarray:
        return phi_K4(P, u, tv, gv, Wv, cfg.cara)

    P, hist = solve(phi_fn, P0, scfg)
    deficit = revelation_deficit(P, u, tv, cfg.K)
    print(f"[smoke] CARA Anderson: {len(hist)} iters, "
          f"residual {hist[-1]:.3e}, 1-R^2 = {deficit:.3e}")
    assert deficit < 1.0e-6, f"CARA REE not FR: 1-R^2 = {deficit}"
    print("[smoke] CARA Anderson converges to FR:            OK")


def test_solver_converges_at_g5() -> None:
    """Anderson at G=5 should converge in well under max_iters."""
    cfg = Config(K=4, G=5, gamma=0.5, tau=2.0, cara=False)
    scfg = SolverConfig(method="anderson", max_iters=30, tol=1.0e-7,
                        anderson_m=6, symmetrize=True, verbose=False,
                        checkpoint_every=0)
    u, tv, gv, Wv = cfg.u_grid(), cfg.tau_vec(), cfg.gamma_vec(), cfg.W_vec()
    P0 = init_no_learning(u, tv, gv, Wv, cfg.cara)

    def phi_fn(P: np.ndarray) -> np.ndarray:
        return phi_K4(P, u, tv, gv, Wv, cfg.cara)

    P, hist = solve(phi_fn, P0, scfg)
    drop = hist[0] / hist[-1] if hist[-1] > 0 else float("inf")
    print(f"[smoke] solver: {len(hist)} iters, "
          f"hist[0]={hist[0]:.3e} -> hist[-1]={hist[-1]:.3e}  "
          f"({drop:.1f}x drop)")
    # G=5 has a coarse interpolation floor for K=4; require >=5x drop
    # and a final residual under 0.05 (safely above the floor).
    assert drop >= 5.0, f"solver made too little progress: drop={drop}"
    assert hist[-1] < 0.05, f"final residual too large: {hist[-1]}"
    print("[smoke] solver makes progress at G=5:             OK")


def main() -> None:
    print("=== K=4 smoke tests ===")
    test_demand_monotone_in_p()
    test_market_clearing_homogeneous()
    test_no_learning_init_symmetric()
    test_cara_full_revelation_no_learning()
    test_phi_one_step_runs_and_symmetric()
    test_cara_phi_preserves_full_revelation()
    test_cara_anderson_converges_to_full_revelation()
    test_solver_converges_at_g5()
    print("=== all smoke tests passed ===")


if __name__ == "__main__":
    main()
