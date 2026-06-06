"""Numba-jit'd chebroots via companion-matrix eigenvalues.

For Cheb polynomial sum_k c_k T_k of degree d (deg = len(c)-1, c[d]!=0),
the comrade matrix M is the (d x d) matrix
  M[0, 1] = 1
  M[i, i-1] = 0.5, M[i, i+1] = 0.5  for 1 <= i <= d-2
  M[d-1, j] = -c[j]/(2*c[d])     for j = 0..d-2
  M[d-1, d-2] += 0.5
The real roots in [-1, 1] are the real parts of M's eigenvalues
with |imag| ~ 0 and real in [-1, 1].
"""
import numpy as np
from numba import njit


@njit(cache=True)
def chebroots_companion(c, n, out, tol_imag=1e-10):
    """Find all real roots in [-1, 1] of Cheb-coeff polynomial c[:n].
    Returns count; stores roots in out[]."""
    # Trim trailing zeros to find effective degree
    deg = n - 1
    while deg > 0 and abs(c[deg]) < 1e-300:
        deg -= 1
    if deg < 1:
        return 0
    if deg == 1:
        # c[0] + c[1] T_1(x) = c[0] + c[1] x = 0  =>  x = -c[0]/c[1]
        x = -c[0] / c[1]
        if -1.0 <= x <= 1.0:
            out[0] = x
            return 1
        return 0
    # Build comrade matrix M (deg x deg), complex for numba eigvals
    M = np.zeros((deg, deg), dtype=np.complex128)
    if deg >= 2:
        M[0, 1] = 1.0
    for i in range(1, deg - 1):
        M[i, i-1] = 0.5
        M[i, i+1] = 0.5
    if deg >= 2:
        for j in range(deg):
            M[deg-1, j] = -c[j] / (2.0 * c[deg])
        if deg - 2 >= 0:
            M[deg-1, deg-2] += 0.5
    # Eigvals (complex)
    eigs = np.linalg.eigvals(M)
    nr = 0
    for k in range(deg):
        if abs(eigs[k].imag) < tol_imag:
            r = eigs[k].real
            if -1.0 <= r <= 1.0:
                out[nr] = r
                nr += 1
    # Insertion sort for ascending output
    for i in range(1, nr):
        key = out[i]; j = i - 1
        while j >= 0 and out[j] > key:
            out[j+1] = out[j]; j -= 1
        out[j+1] = key
    return nr


if __name__ == '__main__':
    import time
    from cheby_numba import chebval_jit
    from numpy.polynomial import chebyshev as npc

    # Sanity test: known polynomial
    # P(x) = T_3(x) = 4x^3 - 3x has roots at +/- sqrt(3)/2, 0
    c = np.array([0.0, -0.75, 0.0, 0.25])  # coeffs in T basis: T_3 itself
    # Actually T_3(x) IS T_3 alone, so c = [0, 0, 0, 1]
    c = np.array([0.0, 0.0, 0.0, 1.0])  # T_3 = 4x^3 - 3x has roots x = 0, ±√3/2
    out = np.empty(16)
    n = chebroots_companion(c, 4, out)
    print(f'T_3 roots found: {n}')
    for i in range(n):
        print(f'  root[{i}] = {out[i]:.6f}  (expected: ±0.866 or 0)')
    print(f'numpy chebroots: {npc.chebroots(c)}')

    # Timing test
    np.random.seed(0)
    c_rand = np.random.randn(7)
    out2 = np.empty(16)
    _ = chebroots_companion(c_rand, 7, out2)  # warmup
    n_iters = 1000
    t0 = time.time()
    for _ in range(n_iters):
        n_r = chebroots_companion(c_rand, 7, out2)
    dt_jit = (time.time() - t0) / n_iters * 1e6
    t0 = time.time()
    for _ in range(n_iters):
        r = npc.chebroots(c_rand)
        r_real = r[np.abs(r.imag) < 1e-10].real
        r_in = r_real[(r_real >= -1) & (r_real <= 1)]
    dt_np = (time.time() - t0) / n_iters * 1e6
    print(f'\nTiming (per call, G=7):')
    print(f'  numba chebroots: {dt_jit:.1f} μs')
    print(f'  numpy chebroots: {dt_np:.1f} μs')
    print(f'  speedup: {dt_np/dt_jit:.1f}x')

    # Check agreement
    n_r = chebroots_companion(c_rand, 7, out2)
    np_r = npc.chebroots(c_rand)
    np_real = np.sort(np_r[np.abs(np_r.imag) < 1e-10].real)
    np_in = np_real[(np_real >= -1) & (np_real <= 1)]
    print(f'\nNumba roots ({n_r}): {[f"{out2[i]:.6f}" for i in range(n_r)]}')
    print(f'Numpy roots ({len(np_in)}): {[f"{r:.6f}" for r in np_in]}')
