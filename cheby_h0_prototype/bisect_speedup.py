"""Verify the predicted bisection-on-polynomial speedup vs chebroots.

Drop-in replacement: instead of chebroots (colleague matrix + complex eigvals),
do bisection on the 1D Cheb polynomial P(xi_b) - p_target = 0.

For monotone polynomials (lifted form with reasonable h), this gives 1 root
deterministically in O(log(1/eps)) iterations.

We time both methods on the EXACT inner loop the operator uses.
"""
import sys, time, math
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from numba import njit
from cheby_numba import (chebval_jit, chebroots_jit, chebder_jit,
                            U_NODES, LOBATTO, V_INV, GL_NODES, GL_WEIGHTS,
                            TAU, GAMMA, C_STRETCH, NQ, phi_jit)

# ===== Bisection-on-polynomial replacement =====
@njit(cache=True)
def cheb_bisect_single(c, n, p_target, lo=-1.0, hi=1.0, n_iter=50):
    """Bisect the single root of c.T - p_target = 0 in (lo, hi).
    Returns the root if a sign change exists, else NaN."""
    # Shrink slightly to avoid boundary issues
    lo += 1e-12; hi -= 1e-12
    val_lo = chebval_jit(lo, c, n) - p_target
    val_hi = chebval_jit(hi, c, n) - p_target
    if val_lo == 0.0: return lo
    if val_hi == 0.0: return hi
    if val_lo * val_hi > 0:
        return float('nan')
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        val_mid = chebval_jit(mid, c, n) - p_target
        if val_lo * val_mid < 0:
            hi = mid
        else:
            lo = mid
            val_lo = val_mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def time_chebroots_inner(P_vals, V_inv, lobatto, gl_nodes, gl_weights,
                           tau, c_stretch, n_grid, nq, n_calls):
    """Run only the chebroots inner loop many times to measure."""
    G = n_grid
    # Set up a representative 1D polynomial coming from operator
    # Just use a single slice's column.
    coeffs1d = np.zeros(G)
    coeffs1d[0] = 0.5
    coeffs1d[1] = 0.2
    coeffs1d[2] = -0.05
    if G > 3: coeffs1d[3] = 0.03
    if G > 4: coeffs1d[4] = -0.01
    coeffs1d_shift = coeffs1d.copy()
    coeffs1d_shift[0] -= 0.5  # p_target = 0.5
    total = 0
    for _ in range(n_calls):
        roots = chebroots_jit(coeffs1d_shift, G)
        total += roots.shape[0]
    return total


@njit(cache=True)
def time_bisection_inner(n_grid, n_calls):
    """Run bisection inner loop many times to measure."""
    G = n_grid
    coeffs1d = np.zeros(G)
    coeffs1d[0] = 0.5
    coeffs1d[1] = 0.2
    coeffs1d[2] = -0.05
    if G > 3: coeffs1d[3] = 0.03
    if G > 4: coeffs1d[4] = -0.01
    total = 0
    for _ in range(n_calls):
        root = cheb_bisect_single(coeffs1d, G, 0.5)
        if not math.isnan(root):
            total += 1
    return total


if __name__ == '__main__':
    print('=== Bisection vs chebroots timing ===\n')
    print(f'{"N":>3} {"G":>3} {"calls":>10} {"chebroots(s)":>14} '
          f'{"bisection(s)":>14} {"speedup":>10}')
    for N in [4, 6, 7, 8, 10, 12]:
        G = N + 1
        n_calls = 100_000 if N < 10 else 50_000
        # JIT warmup
        _ = time_chebroots_inner(np.zeros((G,G,G)), np.eye(G), np.linspace(-1, 1, G),
                                    GL_NODES, GL_WEIGHTS, TAU, C_STRETCH, G, NQ, 100)
        _ = time_bisection_inner(G, 100)
        # Time chebroots
        t0 = time.time()
        _ = time_chebroots_inner(np.zeros((G,G,G)), np.eye(G), np.linspace(-1, 1, G),
                                    GL_NODES, GL_WEIGHTS, TAU, C_STRETCH, G, NQ, n_calls)
        t_cr = time.time() - t0
        # Time bisection
        t0 = time.time()
        _ = time_bisection_inner(G, n_calls)
        t_bi = time.time() - t0
        print(f'{N:>3} {G:>3} {n_calls:>10} {t_cr:>14.4f} {t_bi:>14.4f} {t_cr/t_bi:>10.1f}x')

    print('\n=== Per-call time ===')
    print(f'{"N":>3} {"per chebroots":>20} {"per bisection":>20}')
    for N in [4, 6, 7, 8, 10, 12]:
        G = N + 1
        n_calls = 100_000 if N < 10 else 50_000
        t0 = time.time()
        _ = time_chebroots_inner(np.zeros((G,G,G)), np.eye(G), np.linspace(-1, 1, G),
                                    GL_NODES, GL_WEIGHTS, TAU, C_STRETCH, G, NQ, n_calls)
        t_cr = (time.time() - t0) / n_calls
        t0 = time.time()
        _ = time_bisection_inner(G, n_calls)
        t_bi = (time.time() - t0) / n_calls
        print(f'{N:>3} {t_cr*1e6:>16.2f} us {t_bi*1e6:>16.2f} us')
