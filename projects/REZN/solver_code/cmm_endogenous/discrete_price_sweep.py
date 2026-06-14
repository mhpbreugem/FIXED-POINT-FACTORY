"""Extended discrete-price-grid sweep.

Sweeps G in {8, 11, 15, 21} x M in {8, 16, 32, 64}.

For each cell:
  Path A: hard K-means (max 50 iters)
  Path B (if A oscillates): record the period-K cycle's mean deficit
         as a "soft" equilibrium estimate.

Optimization: vectorize the partition-Bayes computation via np.bincount
so each iteration is O(Gi^3) rather than O(Gi^3 * M * agent).

For each (G, M):
  - converged: True/False
  - max_res, med_res, deficit (final or cycle-mean)
  - period (1 = converged, k > 1 = oscillation)
  - wall

Also produces a deficit-vs-M and deficit-vs-G plot.
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.demand import clear_crra
from reznsrc.signals import f_signal
from scipy.optimize import newton_krylov
try: from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TAU, GAMMA = 2.0, 0.01
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_sweep'
os.makedirs(OUT, exist_ok=True)


def build_grid(Gi):
    pad = 2; UMAX = 4.0
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def solve_kernel(Gi, tau, gamma):
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


def compute_clearing_vectorized(assignment, p_levels, uf, lo, hi, tau, gamma):
    """Vectorized computation of clearing prices given a partition assignment."""
    Gi = hi - lo
    M = len(p_levels)
    f0_arr = np.array([f_signal(uf[lo + i], 0, tau) for i in range(Gi)])
    f1_arr = np.array([f_signal(uf[lo + i], 1, tau) for i in range(Gi)])
    # f0*f0, f1*f1 outer products
    F0_outer = np.outer(f0_arr, f0_arr)  # (Gi, Gi)
    F1_outer = np.outer(f1_arr, f1_arr)
    # Agent 0 (own = i): sum over (j, l) cells assigned to m
    # A0[m, i] = sum_{j,l : a[i,j,l]=m} f0(j)*f0(l)
    A0 = np.zeros((3, M, Gi))
    A1 = np.zeros((3, M, Gi))
    # axis 0: own = i, sum over j,l
    for m in range(M):
        mask = (assignment == m).astype(np.float64)
        # axis-i: sum_{j,l} mask[i,j,l] * F0_outer[j,l]
        A0[0, m, :] = np.einsum('ijl,jl->i', mask, F0_outer)
        A1[0, m, :] = np.einsum('ijl,jl->i', mask, F1_outer)
        # axis-j: sum_{i,l} mask[i,j,l] * F0_outer[i,l]
        A0[1, m, :] = np.einsum('ijl,il->j', mask, F0_outer)
        A1[1, m, :] = np.einsum('ijl,il->j', mask, F1_outer)
        # axis-l: sum_{i,j} mask[i,j,l] * F0_outer[i,j]
        A0[2, m, :] = np.einsum('ijl,ij->l', mask, F0_outer)
        A1[2, m, :] = np.einsum('ijl,ij->l', mask, F1_outer)
    # Bayes posteriors per cell per agent
    p_clear = np.zeros((Gi, Gi, Gi))
    gam_vec = np.full(3, gamma); W_vec = np.full(3, 1.0)
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                m = assignment[i, j, l]
                mu = np.empty(3)
                for k_own, idx in [(0, i), (1, j), (2, l)]:
                    A0_v = A0[k_own, m, idx]; A1_v = A1[k_own, m, idx]
                    f0_own = f0_arr[idx]; f1_own = f1_arr[idx]
                    num = f1_own * A1_v; den = f0_own * A0_v + num
                    mu[k_own] = max(1e-12, min(1-1e-12, num/den if den > 0 else 0.5))
                p_clear[i, j, l] = clear_crra(mu, gam_vec, W_vec)
    return p_clear


def compute_deficit(P, uf, lo, hi, tau):
    Gi = hi - lo
    u_in = uf[lo:hi]
    sum_u = np.add.outer(np.add.outer(u_in, u_in), u_in).ravel()
    Pc = np.clip(P.ravel(), 1e-12, 1-1e-12)
    y = np.log(Pc / (1 - Pc))
    # OLS slope and R^2
    a = np.polyfit(sum_u, y, 1)
    pr = a[0] * sum_u + a[1]
    deficit = float(np.sum((y - pr)**2) / max(np.sum((y - y.mean())**2), 1e-30))
    return deficit


def solve_kmeans(P_kernel, p_levels, uf, lo, hi, tau, gamma, max_iter=50):
    """K-means iteration with cycle detection."""
    assignment = np.argmin(np.abs(P_kernel[..., None] - p_levels[None, None, None, :]), axis=-1)
    history = []
    assignment_history = []
    for it in range(max_iter):
        p_clear = compute_clearing_vectorized(assignment, p_levels, uf, lo, hi, tau, gamma)
        new_assignment = np.argmin(np.abs(p_clear[..., None] - p_levels[None, None, None, :]), axis=-1)
        p_assigned = p_levels[assignment]
        res = p_clear - p_assigned
        max_res = float(np.max(np.abs(res)))
        med_res = float(np.median(np.abs(res)))
        deficit = compute_deficit(p_assigned, uf, lo, hi, tau)
        n_changed = int(np.sum(new_assignment != assignment))
        history.append(dict(it=it, n_changed=n_changed, max_res=max_res,
                             med_res=med_res, deficit=deficit))
        # Cycle detection
        assignment_history.append(assignment.copy())
        cycle = 0
        for k in range(1, min(len(assignment_history), 5) + 1):
            if k < len(assignment_history) and np.array_equal(assignment, assignment_history[-k-1]):
                cycle = k
                break
        if n_changed == 0:
            return assignment, history, 1  # converged
        if cycle > 0 and it > 5:
            return assignment, history, cycle
        assignment = new_assignment
    return assignment, history, 0  # max_iter hit, no cycle detected


def main():
    print("=== Discrete-price-grid sweep ===\n", flush=True)
    print(f"(tau, gamma) = ({TAU}, {GAMMA})\n", flush=True)
    G_list = [8, 11, 15, 21]
    M_list = [8, 16, 32, 64]
    results = {}
    for Gi in G_list:
        print(f"\n========== G = {Gi} ==========", flush=True)
        t0 = time.time()
        P_kernel, uf, lo, hi = solve_kernel(Gi, TAU, GAMMA)
        kernel_deficit = compute_deficit(P_kernel, uf, lo, hi, TAU)
        print(f"  kernel solved in {time.time()-t0:.1f}s, deficit = {kernel_deficit:.4e}",
              flush=True)
        for M in M_list:
            p_levels = np.linspace(0.05, 0.95, M)
            print(f"\n  --- M={M} (spacing={p_levels[1]-p_levels[0]:.3f}) ---",
                  flush=True)
            t0 = time.time()
            assignment, history, period = solve_kmeans(
                P_kernel, p_levels, uf, lo, hi, TAU, GAMMA, max_iter=60)
            wall = time.time() - t0
            final = history[-1]
            tail = history[-min(10, len(history)):]
            mean_def_tail = float(np.mean([h['deficit'] for h in tail]))
            print(f"    period={period} (1=converged), max_res={final['max_res']:.3e}, "
                  f"med_res={final['med_res']:.3e}, deficit_final={final['deficit']:.4e}, "
                  f"deficit_tail_mean={mean_def_tail:.4e}, wall={wall:.1f}s",
                  flush=True)
            results[f"G{Gi}_M{M}"] = dict(
                G=Gi, M=M,
                period=int(period),
                converged=(period == 1),
                max_res=final['max_res'], med_res=final['med_res'],
                deficit_final=final['deficit'],
                deficit_tail_mean=mean_def_tail,
                kernel_deficit=float(kernel_deficit),
                wall=float(wall), spacing=float(p_levels[1]-p_levels[0]),
                history=history)
            json.dump(results, open(f"{OUT}/sweep.json", 'w'), indent=2, default=str)
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for Gi in G_list:
        ms = [r['M'] for r in results.values() if r['G'] == Gi]
        defs = [r['deficit_tail_mean'] for r in results.values() if r['G'] == Gi]
        max_r = [r['max_res'] for r in results.values() if r['G'] == Gi]
        axes[0].plot(ms, defs, 'o-', label=f'G={Gi}')
        axes[1].plot(ms, max_r, 'o-', label=f'G={Gi}')
    # Also overlay kernel deficits per G
    for Gi in G_list:
        kd = next(r['kernel_deficit'] for r in results.values() if r['G'] == Gi)
        axes[0].axhline(kd, ls=':', alpha=0.4)
    axes[0].set_xlabel('M (price levels)'); axes[0].set_ylabel('deficit (tail mean)')
    axes[0].set_xscale('log', base=2); axes[0].set_title('Discrete-price deficit vs M')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('M'); axes[1].set_ylabel('max residual')
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log')
    axes[1].set_title('Max residual vs M (should be ~ 1/M)')
    axes[1].legend(); axes[1].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/sweep.png", dpi=140); plt.close()
    print(f"\nSaved {OUT}/sweep.json, {OUT}/sweep.png", flush=True)


if __name__ == "__main__":
    main()
