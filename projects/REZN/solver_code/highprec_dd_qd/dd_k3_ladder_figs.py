"""Figures for the DD K=3 tau ladder."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/tmp/cheby_h0")

OUT = "/tmp/dd_k3_ladder_figs"
os.makedirs(OUT, exist_ok=True)
d = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/ladder.json"))
taus = sorted(float(k) for k in d.keys())
Fs = np.array([d[f"{t:.4f}"]["F"] for t in taus])
walls = np.array([d[f"{t:.4f}"]["wall"] for t in taus])
taus_arr = np.array(taus)

fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(taus_arr, Fs, ".", lw=0.6, ms=3, color="C0")
ax.axhline(1e-25, color="k", lw=0.5, alpha=0.5, label=r"DD target $10^{-25}$")
ax.axhline(1e-15, color="C2", lw=0.5, alpha=0.5, label="float64 eps")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$|F|_\infty$")
ax.set_title(f"DD K=3 ladder ({len(taus)} cells, $\\gamma=100$, $G=7$)")
ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_F_floor.png", dpi=140); plt.close()

# slope/deficit per cell
from lin_cdf_kern_tab import make_cdf_uniform_grid
u_grid = make_cdf_uniform_grid(7)
U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
T = U1 + U2 + U3
def fit(P):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel(); Tf = T.ravel()
    s = float(np.sum(L*Tf)/np.sum(Tf**2))
    uT, inv = np.unique(np.round(Tf, 10), return_inverse=True)
    ss = float(np.sum((L-L.mean())**2)); w = 0.0
    for g in range(len(uT)):
        m = (inv==g); w += float(np.sum((L[m]-L[m].mean())**2))
    return s, w/ss
slopes = []; deficits = []; tau_good = []
for t in taus:
    f = f"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/fps/tau{t:.4f}.npy"
    if os.path.exists(f):
        P = np.load(f)
        if P.shape == (7, 7, 7):
            s, dd_ = fit(P)
            slopes.append(s); deficits.append(dd_); tau_good.append(t)
tau_good = np.array(tau_good); slopes = np.array(slopes); deficits = np.array(deficits)

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(tau_good, slopes, "-", lw=1, color="C0")
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"slope $\alpha^*$")
axs[0].set_title("Slope along $\\tau$ ladder"); axs[0].grid(True, alpha=0.3)
axs[1].semilogy(tau_good, np.maximum(deficits, 1e-30), "-", lw=1, color="C2")
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel(r"deficit $1-R^2$")
axs[1].set_title("Deficit along $\\tau$ ladder"); axs[1].grid(True, alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_slope_deficit.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(taus_arr, walls, ".", lw=0.6, ms=2, color="C3")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("wall time per cell (s)")
ax.set_title(f"Per-cell wall (mean={walls.mean():.0f}s, max={walls.max():.0f}s)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_wall.png", dpi=140); plt.close()
print(f"cells: {len(taus)}, DD eps: {int((Fs<1e-25).sum())}")
print(f"figs in {OUT}")
