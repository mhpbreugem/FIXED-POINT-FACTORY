"""Diagnose Option B convergence on one cell."""
import os, sys, time
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import dd_k3_optB_numba as OB
from lin_cdf_strict import make_cdf_uniform_grid, make_p_grid

G = 11; G_p = 121
u_grid = make_cdf_uniform_grid(G); p_grid = make_p_grid(G_p)
U1,U2,U3 = np.meshgrid(u_grid,u_grid,u_grid, indexing="ij")
T = U1+U2+U3
P_cold = 1.0/(1.0+np.exp(-0.5*T))

# Try (gamma=100, tau=1.0)
gamma, tau = 100.0, 1.0
print("JIT...", flush=True); t0 = time.time()
OB.phi(P_cold, u_grid, p_grid, 1.0, 1.0)
print(f"  {time.time()-t0:.1f}s", flush=True)

# Plain Picard with damping
print(f"\n--- Picard (damped) from sigmoid cold at g={gamma}, t={tau} ---", flush=True)
P = P_cold.copy()
for it in range(30):
    Pn = OB.phi(P, u_grid, p_grid, tau, gamma)
    F = float(np.max(np.abs(Pn - P)))
    F_med = float(np.median(np.abs(Pn - P)))
    print(f"  it{it+1:3d}: |F|max={F:.3e} median={F_med:.3e}", flush=True)
    # damped step
    alpha = 0.3
    P = (1-alpha)*P + alpha*Pn

# Now try R4 FP as warm start
print(f"\n--- Anderson from R4 FP as warm ---", flush=True)
r4 = "/tmp/dd_k3_sweep_fps/g100_t1.0000.npz"
if os.path.exists(r4):
    d = np.load(r4)
    P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
    print(f"  warm |F0|={float(np.max(np.abs(OB.phi(P, u_grid, p_grid, tau, gamma) - P))):.3e}",
          flush=True)
    Xh, Gh = [], []
    for it in range(40):
        Pn = OB.phi(P, u_grid, p_grid, tau, gamma)
        Fv = (Pn - P).ravel()
        F = float(np.max(np.abs(Fv)))
        print(f"  it{it+1:3d}: |F|max={F:.3e}", flush=True)
        gx = Fv + P.ravel()
        Xh.append(P.ravel().copy()); Gh.append(gx.copy())
        if len(Xh) > 8: Xh.pop(0); Gh.pop(0)
        k = len(Xh)
        if k <= 1: P = (P + 0.5*Pn) / 1.5
        else:
            DR = np.column_stack([(Gh[i]-Xh[i])-(Gh[k-1]-Xh[k-1]) for i in range(k-1)])
            R_k = Gh[k-1] - Xh[k-1]
            try:
                A = DR.T @ DR + 1e-10*np.eye(DR.shape[1])
                ga = np.linalg.solve(A, -DR.T @ R_k)
                DG = np.column_stack([Gh[i]-Gh[k-1] for i in range(k-1)])
                # Damped Anderson
                x = Gh[k-1] + DG @ ga
                P = (0.4*P.ravel() + 0.6*x).reshape(G, G, G)
            except:
                P = ((1-0.3)*P.ravel() + 0.3*gx).reshape(G, G, G)
