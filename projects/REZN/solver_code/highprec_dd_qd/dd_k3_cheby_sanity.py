"""Sanity check: is the Cheby fit correct? Test on known analytic functions."""
import sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
from dd_k3_cheby_lobatto import lobatto_grid, cheby_fit_lobatto_3d


def edge_coef(coefs):
    return float(max(np.max(np.abs(coefs[-1, :, :])),
                        np.max(np.abs(coefs[:, -1, :])),
                        np.max(np.abs(coefs[:, :, -1]))))


def test_function(name, fn, Gs=(7, 11, 15, 21, 31)):
    print(f"\n--- {name} ---", flush=True)
    print(f"{'G':>5} {'edge_coef':>14} {'max_coef_decay':>15}")
    for G in Gs:
        u_grid, U_MAX = lobatto_grid(G)
        U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
        P = fn(U1, U2, U3)
        coefs = cheby_fit_lobatto_3d(P, U_MAX)
        ec = edge_coef(coefs)
        # Also check: ratio of last to first coef along axis 0
        cmax = float(np.max(np.abs(coefs[0, :, :])))
        ratio = ec / cmax if cmax > 0 else 0
        print(f"{G:>5d} {ec:>14.3e} {ratio:>15.3e}")


# Test 1: pure polynomial of low degree -> coefs should be exact (machine eps high modes)
test_function("polynomial degree 4: T_4(u/3) + 0.5*T_2(u/3)",
                  lambda U1, U2, U3: cheb.chebval(U1/3, [0,0,0.5,0,1])
                                          + cheb.chebval(U2/3, [0,0,0.5,0,1])
                                          + cheb.chebval(U3/3, [0,0,0.5,0,1]))


# Test 2: smooth analytic function
test_function("sigmoid(0.5*(u1+u2+u3))",
                  lambda U1, U2, U3: 1.0/(1.0+np.exp(-0.5*(U1+U2+U3))))


# Test 3: steep sigmoid
test_function("sigmoid(2.0*(u1+u2+u3))",
                  lambda U1, U2, U3: 1.0/(1.0+np.exp(-2.0*(U1+U2+U3))))


# Test 4: very steep sigmoid (approaching step)
test_function("sigmoid(10*(u1+u2+u3)) -- near-step",
                  lambda U1, U2, U3: 1.0/(1.0+np.exp(-10.0*(U1+U2+U3))))


# Test 5: actual saved R4 FP (the one we tested before)
import os, glob
fps = glob.glob("/tmp/dd_k3_sweep_fps/g100_t1.0000.npz")
if fps:
    d = np.load(fps[0])
    P_r4 = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
    if P_r4.shape[0] == 11:
        from lin_cdf_strict import make_cdf_uniform_grid
        u_grid_cdf = make_cdf_uniform_grid(11)
        U_MAX_cdf = max(abs(u_grid_cdf[0]), abs(u_grid_cdf[-1]))
        print(f"\n--- saved R4 FP at gamma=100, tau=1, G=11 (on CDF-uniform u-grid) ---")
        coefs = cheby_fit_lobatto_3d(P_r4, U_MAX_cdf)  # treats them as Lobatto (WRONG)
        ec = edge_coef(coefs)
        print(f"  treating CDF-uniform as Lobatto: edge_coef = {ec:.3e} (likely wrong!)")
        # Also fit with explicit LSQ
        deg = 10
        xi = u_grid_cdf / U_MAX_cdf
        V = cheb.chebvander(xi, deg)
        Vinv = np.linalg.pinv(V)   # pseudo-inverse for ill-conditioned
        c0 = np.tensordot(Vinv, P_r4, axes=([1], [0]))
        c1 = np.tensordot(Vinv, c0, axes=([1], [1])).transpose(1, 0, 2)
        c2 = np.tensordot(Vinv, c1, axes=([1], [2])).transpose(1, 2, 0)
        ec_lsq = edge_coef(c2)
        print(f"  LSQ fit on CDF-uniform: edge_coef = {ec_lsq:.3e}")
        # Now reconstruct the FP at Lobatto nodes via spectral interp from CDF-uniform
        # (this is the "wrong direction" path -- just to see what happens)
