"""h-ladder at (gamma=10, tau=0.1, G=9): slope(h), range(h) down in h;
   extrapolation cleanliness + strict best-iterate cross-check."""
import sys, time
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype')
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd')
import numpy as np
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, build_mu_table_lin_strict, phi_lin_strict_jit
from dd_k3_solver import solve_warm_f64

TAU, GAMMA, G = 0.1, 10.0, 9
G_p, NQK = 721, 32
u = make_cdf_uniform_grid(G)
p_grid = make_p_grid(G_p)
gl_u, gl_du = make_gl_for_u(u[0], u[-1], NQK)

def slope_of(P):
    U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
    T = (U1+U2+U3).ravel()
    L = np.log(np.clip(P,1e-15,1-1e-15)/(1-np.clip(P,1e-15,1-1e-15))).ravel()
    return float(np.sum(L*T)/np.sum(T*T))

U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
P0 = 1/(1+np.exp(-0.5*(U1+U2+U3)))
print("h_mid    F        slope    range   wall", flush=True)
hmids = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02]
out = []
for hm in hmids:
    hs = (hm*1.25, hm*1.0833, hm*0.9167, hm*0.75)
    t0 = time.time()
    P, F = solve_warm_f64(P0.copy(), u, p_grid, hs, GAMMA, TAU, G_p, NQK, target=1e-12)
    s = slope_of(P); r = P.max()-P.min()
    print(f"{hm:.3f}  {F:.1e}  {s:.5f}  {r:.4f}  {time.time()-t0:.0f}s", flush=True)
    out.append((hm, F, s, r)); P0 = P.copy()

ok = [(h,s,r) for h,F,s,r in out if F < 1e-10]
if len(ok) >= 3:
    H = np.array([h for h,_,_ in ok]); S = np.array([s for _,s,_ in ok]); R = np.array([r for _,_,r in ok])
    A = np.column_stack([np.ones_like(H), H**2])
    cs, *_ = np.linalg.lstsq(A, S, rcond=None)
    cr, *_ = np.linalg.lstsq(A, R, rcond=None)
    print(f"h->0 extrapolation: slope = {cs[0]:.5f}, range = {cr[0]:.4f}", flush=True)
    print(f"fit residuals: slope {np.max(np.abs(S - A@cs)):.5f}, range {np.max(np.abs(R - A@cr)):.4f}", flush=True)

P_s = P0.copy()
best = (1e9, None)
for it in range(60):
    Pn = phi_lin_strict_jit(P_s, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, NQK)
    Fs = float(np.max(np.abs(Pn - P_s)))
    if Fs < best[0]: best = (Fs, P_s.copy())
    P_s = 0.5*P_s + 0.5*Pn
print(f"strict best-iterate: F={best[0]:.2e}, slope={slope_of(best[1]):.5f}, "
      f"range={best[1].max()-best[1].min():.4f}", flush=True)
