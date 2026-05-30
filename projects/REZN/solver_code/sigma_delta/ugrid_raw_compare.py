"""Cross-check: run the u-grid RAW CONTOUR SCAN op (phi_K3_halo, no kernel)
at the same physical setting (gamma=0.1, tau=2, G_inner~11, NL IC) and
compare the slope/d_FR/deficit to the sigma-delta V3 result.

If both give the same small deficit ~ 1e-2 -> raw contour scan
intrinsically misses the Jensen gap at this G (needs kernel smoothing or
higher order). sigma-delta is then CONSISTENT with the u-grid raw scan.

If u-grid raw scan gives big deficit ~ 0.28 -> the sigma-delta op is
under-resolving and there is a bug to fix in the port.
"""
import os, sys, time, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep")
import numpy as np
from reznsrc.contour_K3_halo import phi_K3_halo, init_no_learning_K3
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

TAU = 2.0; GAMMA = 0.1; W = 1.0; UMAX = 4.0; PAD = 2; G_INNER = 11
du = 2*UMAX/(G_INNER-1)
G_FULL = G_INNER + 2*PAD
uf = np.array([-UMAX + (q-PAD)*du for q in range(G_FULL)])
lo, hi = PAD, PAD + G_INNER
slc = (slice(lo, hi),)*3
tv = np.full(3, TAU); gv = np.full(3, GAMMA); Wv = np.full(3, W)

halo = init_no_learning_K3(uf, tv, gv, Wv)
print(f"u-grid raw contour scan -- CRRA gamma={GAMMA}, tau={TAU}, G_inner={G_INNER}, UMAX={UMAX}")
print(f"NL halo built")

# Picard with damping
P = halo.copy()
res_prev = np.inf
omega = 1.0
ferr_hist = []
for it in range(1, 81):
    t = time.time()
    P_phi = phi_K3_halo(P, uf, lo, hi, tv, gv, Wv)
    P_new = (1-omega)*P + omega*P_phi
    # halo stays NL
    P_new_h = halo.copy(); P_new_h[slc] = P_new[slc]
    res = float(np.max(np.abs((P_new_h - P)[slc])))
    if it > 3:
        if res > res_prev * 0.99: omega = max(omega*0.7, 0.05)
        elif res < res_prev*0.6: omega = min(omega*1.05, 1.0)
    P = P_new_h
    res_prev = res
    ferr_hist.append(res)
    if it % 5 == 0 or it == 1:
        # compute metrics
        Pin = P[slc]
        ui = uf[lo:hi]
        U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
        T = TAU*(U1+U2+U3)
        Pc = np.clip(Pin, 1e-12, 1-1e-12)
        y = np.log(Pc/(1-Pc)).ravel()
        aa = np.polyfit(T.ravel(), y, 1)
        pr = aa[0]*T.ravel() + aa[1]
        deficit = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
        d_FR_uw = float(np.sqrt(np.mean((Pin - 1/(1+np.exp(-T)))**2)))
        slope = float(aa[0]); intc = float(aa[1])
        print(f"  iter {it:3d}  ω={omega:.3f}  ferr={res:.3e}  deficit(1-R²)={deficit:.4e}  "
              f"slope={slope:.4f}  intc={intc:+.4f}  d_FR={d_FR_uw:.3e}  ({time.time()-t:.1f}s)")

# final
Pin = P[slc]
ui = uf[lo:hi]
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU*(U1+U2+U3)
Pc = np.clip(Pin, 1e-12, 1-1e-12)
y = np.log(Pc/(1-Pc)).ravel()
aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel() + aa[1]
deficit = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
d_FR = float(np.sqrt(np.mean((Pin - 1/(1+np.exp(-T)))**2)))
print(f"\nFINAL u-grid raw scan G={G_INNER}: deficit={deficit:.4e}  slope={float(aa[0]):.5f}  d_FR={d_FR:.3e}")
print(f"\nCompare to u-grid KERNEL co-area at G=17 (headline): deficit ~ 0.28, slope ~ 0.17")
print(f"Compare to sigma-delta V3 G=15 gamma=0.1: still running, expected slope ~ 0.95-1.0")

json.dump({'G_inner': G_INNER, 'tau': TAU, 'gamma': GAMMA, 'UMAX': UMAX,
           'deficit': deficit, 'slope': float(aa[0]), 'intercept': float(aa[1]),
           'd_FR': d_FR, 'ferr_hist': ferr_hist},
          open(os.path.join(HERE, 'ugrid_raw_compare.json'), 'w'), indent=2)
