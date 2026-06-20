"""Find the G where the kernel-band equilibrium becomes h-invariant at tau=0.1.
At each G, solve with two h-windows; h-invariance = range & slope drift small."""
import sys, time
sys.path.insert(0, 'cheby_h0_prototype')
sys.path.insert(0, 'projects/REZN/solver_code/highprec_dd_qd')
import numpy as np
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from dd_k3_solver import solve_warm_f64

TAU, GAMMA = 0.1, 10.0
G_p, NQK = 361, 24

def slope_of(P, u):
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    T = (U1+U2+U3).ravel()
    L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
    return float(np.sum(L*T)/np.sum(T*T))

p_grid = make_p_grid(G_p)
for G in [9, 11, 13, 15, 17]:
    u = make_cdf_uniform_grid(G)
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    P0 = 1/(1+np.exp(-0.5*(U1+U2+U3)))
    res = {}
    for tag, hs in [("hA", (0.12, 0.10, 0.08, 0.06)), ("hB", (0.08, 0.0667, 0.0533, 0.04))]:
        t0 = time.time()
        P, F = solve_warm_f64(P0.copy(), u, p_grid, hs, GAMMA, TAU, G_p, NQK, target=1e-12)
        res[tag] = (P.max()-P.min(), slope_of(P, u), F, time.time()-t0)
        P0 = P.copy()
    rA, sA, FA, tA = res["hA"]; rB, sB, FB, tB = res["hB"]
    drift_r = abs(rB-rA)/rA; drift_s = abs(sB-sA)/abs(sA)
    print(f"G={G:2d}: hA range={rA:.4f} slope={sA:.5f} (F={FA:.0e},{tA:.0f}s) | "
          f"hB range={rB:.4f} slope={sB:.5f} (F={FB:.0e},{tB:.0f}s) | "
          f"drift: range {100*drift_r:.1f}% slope {100*drift_s:.1f}%", flush=True)
