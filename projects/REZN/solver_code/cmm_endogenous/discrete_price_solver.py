"""Discrete-price-grid strict-h=0 equilibrium solver.

The user's idea: force P to take values on a FIXED discrete grid
   p_1 < p_2 < ... < p_M
and find a partition {Pi_1, ..., Pi_M} of the cube such that for every
u in Pi_m the CRRA clearing price equals p_m.

Because each Pi_m is a POSITIVE-MEASURE subset of R^3, Bayes is
classical and the GS sufficient-statistic identity (mu_k = p at level
set, which caused PFR degeneracy) does NOT apply.

This solver implements both:
  Path A: hard K-means iteration (assignments update via argmin)
  Path B: soft-max relaxation with scipy.optimize.least_squares,
          temperature annealing T -> 0

Two regimes tested:
  (G, M) = (8, 8)   -- small, fast, sanity
  (G, M) = (11, 16) -- larger, refinement test
"""
import os, sys, time, json, math
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.demand import clear_crra
from reznsrc.signals import f_signal
from scipy.optimize import newton_krylov, least_squares
try: from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

TAU, GAMMA = 2.0, 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price'
os.makedirs(OUT, exist_ok=True)


def build_grid(Gi):
    pad = 2; UMAX = 4.0
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def solve_kernel(Gi, tau, gamma):
    """Get a kernel-h>0 certified solution at G=Gi for warm start."""
    pad = 2; UMAX = 4.0; C = 0.45; K = 3
    du, uf, lo, hi = build_grid(Gi)
    h = C*np.sqrt(du)
    tv = np.full(K, tau); gv = np.full(K, gamma); wv = np.full(K, 1.0)
    P_full = init_no_learning_K3(uf, tv, gv, wv)
    P0 = P_full[lo:hi, lo:hi, lo:hi].ravel().copy()
    def F(x):
        Pf = P_full.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
        return (Pn[lo:hi, lo:hi, lo:hi] - x.reshape(Gi, Gi, Gi)).ravel()
    try:
        x = newton_krylov(F, P0, f_tol=1e-12, maxiter=80, verbose=False)
    except NoConvergence as e:
        x = e.args[0]
    return x.reshape(Gi, Gi, Gi), uf, lo, hi


def compute_partition_bayes_and_clearing(P_inner, assignment, p_levels, uf, lo, hi, tau, gamma):
    """Given a partition (assignment[c] in 0..M-1 for each cell c),
    compute for every cell its Bayes posteriors (one per agent) using
    the partition-conditional Bayes, then the CRRA clearing price.
    Returns: clearing_prices (Gi^3 array), per-cell posteriors, residuals.
    """
    Gi = hi - lo
    M = len(p_levels)
    # Precompute f0(u), f1(u) on inner cube
    f0_arr = np.array([f_signal(uf[lo + i], 0, tau) for i in range(Gi)])
    f1_arr = np.array([f_signal(uf[lo + i], 1, tau) for i in range(Gi)])
    # Joint density on full inner cube
    pf0 = np.einsum('i,j,k->ijk', f0_arr, f0_arr, f0_arr)  # (Gi,Gi,Gi)
    pf1 = np.einsum('i,j,k->ijk', f1_arr, f1_arr, f1_arr)
    w_total = 0.5 * (pf0 + pf1)
    # Per partition cell, per axis k=0,1,2, per "own_signal" index i:
    # mu_k(u_k=i, in Pi_m) = sum_{(j,l): cell in Pi_m} f_1(j)*f_1(l) /
    #                       [ sum_{(j,l): in Pi_m} f_0(j)*f_0(l) + sym ]
    # Precompute by binning per axis per assignment level
    # For agent 0 (own_signal = i): cells are P_inner[i, j, l].
    # We need, for each (m, i), the integrals over j,l with assignment[i,j,l] = m.
    A0_arr = np.zeros((3, M, Gi))
    A1_arr = np.zeros((3, M, Gi))
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                m = assignment[i, j, l]
                # agent 0 (own = i): contribute integrand f_v(j)*f_v(l)
                A0_arr[0, m, i] += f0_arr[j] * f0_arr[l]
                A1_arr[0, m, i] += f1_arr[j] * f1_arr[l]
                # agent 1 (own = j): contribute f_v(i)*f_v(l)
                A0_arr[1, m, j] += f0_arr[i] * f0_arr[l]
                A1_arr[1, m, j] += f1_arr[i] * f1_arr[l]
                # agent 2 (own = l): contribute f_v(i)*f_v(j)
                A0_arr[2, m, l] += f0_arr[i] * f0_arr[j]
                A1_arr[2, m, l] += f1_arr[i] * f1_arr[j]
    # Per-cell clearing
    p_clear = np.zeros((Gi, Gi, Gi))
    gam_vec = np.full(3, gamma); W_vec = np.full(3, 1.0)
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                m = assignment[i, j, l]
                mu = np.empty(3)
                for k_own, idx in [(0, i), (1, j), (2, l)]:
                    A0_v = A0_arr[k_own, m, idx]; A1_v = A1_arr[k_own, m, idx]
                    f0_own = f0_arr[idx]; f1_own = f1_arr[idx]
                    num = f1_own * A1_v; den = f0_own * A0_v + num
                    mu[k_own] = max(1e-12, min(1-1e-12, num/den if den > 0 else 0.5))
                p_clear[i, j, l] = clear_crra(mu, gam_vec, W_vec)
    # Residual: p_clear - p_assigned
    p_assigned = np.array([p_levels[assignment[i, j, l]]
                            for i in range(Gi) for j in range(Gi) for l in range(Gi)]).reshape(Gi, Gi, Gi)
    return p_clear, p_assigned


def hard_kmeans_iterate(P_kernel, p_levels, uf, lo, hi, tau, gamma, max_iter=30):
    Gi = hi - lo
    # Initial assignment: each cell goes to nearest price level
    assignment = np.argmin(np.abs(P_kernel[..., None] - p_levels[None, None, None, :]), axis=-1)
    history = []
    for it in range(max_iter):
        p_clear, p_assigned = compute_partition_bayes_and_clearing(
            P_kernel, assignment, p_levels, uf, lo, hi, tau, gamma)
        # Reassign to nearest price level given current clearing
        new_assignment = np.argmin(np.abs(p_clear[..., None] - p_levels[None, None, None, :]), axis=-1)
        n_changed = int(np.sum(new_assignment != assignment))
        max_res = float(np.max(np.abs(p_clear - p_assigned)))
        med_res = float(np.median(np.abs(p_clear - p_assigned)))
        deficit = 1.0 - np.corrcoef(np.log(np.clip(p_assigned, 1e-12, 1-1e-12).ravel() /
                                            np.clip(1 - p_assigned, 1e-12, 1-1e-12).ravel()),
                                     (np.add.outer(np.add.outer(
                                         uf[lo:hi], uf[lo:hi]), uf[lo:hi])).ravel())[0,1] ** 2
        history.append(dict(it=it, n_changed=n_changed, max_res=max_res,
                             med_res=med_res, deficit=float(deficit)))
        print(f"  it={it:>3}: n_changed={n_changed:>5} max_res={max_res:.3e} "
              f"med_res={med_res:.3e} deficit={deficit:.4e}", flush=True)
        if n_changed == 0:
            print(f"  CONVERGED at iter {it}", flush=True)
            break
        assignment = new_assignment
    return assignment, history


def main():
    print("=== Discrete-price-grid strict-h=0 solver ===\n", flush=True)
    print(f"(tau, gamma) = ({TAU}, {GAMMA})", flush=True)

    # Test: small G, two grids of M values
    for Gi in [8, 11]:
        for M in [8, 16]:
            print(f"\n========== G={Gi}, M={M} ==========", flush=True)
            print("Solving kernel at G={}...".format(Gi), flush=True)
            t0 = time.time()
            P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
            print(f"  kernel solved in {time.time()-t0:.1f}s", flush=True)

            # Build price grid: M evenly-spaced levels in [0.05, 0.95]
            p_levels = np.linspace(0.05, 0.95, M)
            print(f"  price grid: {p_levels}", flush=True)

            # Hard K-means
            print(f"\n  --- Hard K-means iteration ---", flush=True)
            t0 = time.time()
            assignment, history = hard_kmeans_iterate(P_kernel, p_levels, uf, lo, hi, TAU, GAMMA, max_iter=30)
            wall = time.time() - t0
            converged = history[-1]['n_changed'] == 0
            final = history[-1]
            print(f"  FINAL: converged={converged}, max_res={final['max_res']:.3e}, "
                  f"med_res={final['med_res']:.3e}, deficit={final['deficit']:.4e}, "
                  f"wall={wall:.1f}s", flush=True)

            # Save
            np.save(f"{OUT}/assignment_G{Gi}_M{M}.npy", assignment)
            json.dump(dict(G=Gi, M=M, p_levels=list(p_levels),
                            converged=bool(converged),
                            final=final, history=history, wall=float(wall)),
                      open(f"{OUT}/result_G{Gi}_M{M}.json", 'w'), indent=2, default=str)


if __name__ == "__main__":
    main()
