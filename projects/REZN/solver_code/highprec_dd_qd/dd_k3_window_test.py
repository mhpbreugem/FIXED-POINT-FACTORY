import sys, time
sys.path.insert(0, 'cheby_h0_prototype')
sys.path.insert(0, 'projects/REZN/solver_code/highprec_dd_qd')
import numpy as np
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid
from lin_cdf_strict import make_gl_for_u, phi_lin_strict_jit
from lin_cdf_richardson import phi_lin_richardson
from dd_k3_solver import solve_warm_f64
from scipy.stats import norm

TAU, GAMMA = 0.1, 10.0

def assess(G, hs, G_p, nqk, P_init=None, label=""):
    u = make_cdf_uniform_grid(G)
    W3 = (lambda f: f[:,None,None]*f[None,:,None]*f[None,None,:])(
        0.5*norm.pdf(u,-0.5,1/np.sqrt(TAU)) + 0.5*norm.pdf(u,0.5,1/np.sqrt(TAU)))
    W3 /= W3.max()
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u[0], u[-1], nqk)
    if P_init is None:
        U1,U2,U3 = np.meshgrid(u,u,u,indexing='ij')
        P_init = 1/(1+np.exp(-0.5*(U1+U2+U3)))
    t0 = time.time()
    P, F = solve_warm_f64(P_init, u, p_grid, hs, GAMMA, TAU, G_p, nqk, target=1e-12)
    phis = [phi_lin_richardson(P, u, hs=(h,), gamma=GAMMA, tau=TAU, G_p=G_p, NQK=nqk, p_grid=p_grid) for h in hs]
    H2 = np.array([h*h for h in hs]); A = np.column_stack([np.ones(4), H2, H2*H2])
    Y = np.array([p.ravel() for p in phis])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    spread = Y.max(0) - Y.min(0)
    mask = (spread > 1e-10) & (W3.ravel() > 0.01)
    t5 = float(np.max(np.abs(Y - A@coef)[:, mask].max(0)/spread[mask])) if mask.any() else 0.0
    Pr4 = phi_lin_richardson(P, u, hs=tuple(hs), gamma=GAMMA, tau=TAU, G_p=G_p, NQK=nqk, p_grid=p_grid)
    Pst = phi_lin_strict_jit(P, u, p_grid, gl_u, gl_du, TAU, GAMMA, G, nqk)
    gap = float(np.max(np.abs(Pr4 - Pst)*W3))
    rng = P.max()-P.min()
    sp = float(np.median(np.abs(np.diff(P, axis=0))))
    print(f"{label}: G={G} hs={tuple(round(h,3) for h in hs)} G_p={G_p} NQK={nqk}: "
          f"F={F:.1e} T5={t5:.4f} gap_w={gap:.2e} range={rng:.4f} cellspace~{sp:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return P, F, t5, gap

P7, *_ = assess(7, (0.13, 0.11, 0.09, 0.07), 361, 24, label="G7-window")
assess(7, (0.16, 0.14, 0.12, 0.10), 361, 24, P_init=P7, label="G7-wide")
P9, *_ = assess(9, (0.12, 0.10, 0.08, 0.06), 361, 24, label="G9-window")
assess(9, (0.10, 0.085, 0.07, 0.055), 361, 24, P_init=P9, label="G9-tight")
