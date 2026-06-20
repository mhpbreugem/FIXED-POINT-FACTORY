"""Extended figures + PDF for the DD K=3 tau ladder."""
import os, json, sys
sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lin_cdf_kern_tab import make_cdf_uniform_grid

OUT = "/tmp/dd_k3_ladder_ext_figs"
os.makedirs(OUT, exist_ok=True)
ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder"
d = json.load(open(f"{ROOT}/ladder.json"))
taus = sorted(float(k) for k in d.keys())
Fs = np.array([d[f"{t:.4f}"]["F"] for t in taus])
walls = np.array([d[f"{t:.4f}"]["wall"] for t in taus])
t_warm = np.array([d[f"{t:.4f}"]["t_warm"] for t in taus])
t_nail = np.array([d[f"{t:.4f}"]["t_nail"] for t in taus])
taus_arr = np.array(taus)

# Compute slope/deficit per saved FP
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

# Monotonicity check
def mono_check(P):
    viol = 0; min_d = 0.0
    for ax in range(3):
        dd = np.diff(P, axis=ax)
        viol += int((dd < 0).sum())
        min_d = min(min_d, float(dd.min()))
    return viol, min_d

slopes = []; deficits = []; tau_good = []; violations = []; min_diffs = []
P_ranges = []
for t in taus:
    fp = f"{ROOT}/fps/tau{t:.4f}.npy"
    if os.path.exists(fp):
        P = np.load(fp)
        if P.shape == (7, 7, 7):
            s, dd = fit(P)
            v, md = mono_check(P)
            slopes.append(s); deficits.append(dd); tau_good.append(t)
            violations.append(v); min_diffs.append(md)
            P_ranges.append([float(P.min()), float(P.max())])
tau_good = np.array(tau_good); slopes = np.array(slopes)
deficits = np.array(deficits); violations = np.array(violations)
min_diffs = np.array(min_diffs); P_ranges = np.array(P_ranges)

# Fig 1: F floor
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(taus_arr, Fs, ".", lw=0.6, ms=3, color="C0")
ax.axhline(1e-25, color="k", lw=0.5, alpha=0.5, label="DD target $10^{-25}$")
ax.axhline(1e-15, color="C2", lw=0.5, alpha=0.5, label="float64 eps")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$|F|_\infty$")
ax.set_title(f"DD R4 ladder F floor ({len(taus)} cells, $\\gamma=100$, $G=7$, analytic-J Newton)")
ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_F.png", dpi=140); plt.close()

# Fig 2: slope + deficit
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(tau_good, slopes, "-", lw=1, color="C0")
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel(r"$\alpha^*$")
axs[0].set_title("Slope along ladder (R4 FP)"); axs[0].grid(True, alpha=0.3)
axs[1].semilogy(tau_good, np.maximum(deficits, 1e-30), "-", lw=1, color="C2")
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel(r"deficit $1-R^2$")
axs[1].set_title("Deficit along ladder (R4 FP — biased!)"); axs[1].grid(True, alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_slope_deficit.png", dpi=140); plt.close()

# Fig 3: monotonicity
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(tau_good, violations, "-", lw=1, color="C3")
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel("# violations of dP/du_i > 0")
axs[0].set_title("Monotonicity violations along R4 ladder")
axs[0].grid(True, alpha=0.3)
axs[1].plot(tau_good, min_diffs, "-", lw=1, color="C3")
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel("min dP/du (most negative)")
axs[1].set_title("Worst negative slope along ladder")
axs[1].grid(True, alpha=0.3); axs[1].axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_monotonicity.png", dpi=140); plt.close()

# Fig 4: P range
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(tau_good, P_ranges[:, 0], "-", lw=1, color="C0", label="$P_{\\min}$")
ax.plot(tau_good, P_ranges[:, 1], "-", lw=1, color="C2", label="$P_{\\max}$")
ax.fill_between(tau_good, P_ranges[:, 0], P_ranges[:, 1], alpha=0.2)
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("P values")
ax.set_title("P range along ladder ($P_{\\min}$ to $P_{\\max}$)")
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_Prange.png", dpi=140); plt.close()

# Fig 5: wall time + breakdown
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(taus_arr, walls, ".", lw=0.6, ms=2, color="C3")
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel("wall (s)")
axs[0].set_title(f"Per-cell wall (mean={walls.mean():.2f}s, max={walls.max():.2f}s)")
axs[0].grid(True, alpha=0.3)
axs[1].plot(taus_arr, t_warm, ".", lw=0.6, ms=2, color="C0", label="warm")
axs[1].plot(taus_arr, t_nail, ".", lw=0.6, ms=2, color="C3", label="nail")
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel("time (s)")
axs[1].set_title("Time breakdown"); axs[1].legend(); axs[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_wall.png", dpi=140); plt.close()

# Fig 6: contour samples at key tau
sample_taus = [0.001, 0.01, 0.05, 0.1, 0.2, 0.225]
existing = [t for t in sample_taus if os.path.exists(f"{ROOT}/fps/tau{t:.4f}.npy")]
fig, axs = plt.subplots(1, len(existing), figsize=(4*len(existing), 4))
if len(existing) == 1: axs = [axs]
for ax, t in zip(axs, existing):
    P = np.load(f"{ROOT}/fps/tau{t:.4f}.npy")
    mid = P.shape[2]//2
    cs = ax.contourf(P[:, :, mid], levels=15, cmap="RdBu_r")
    ax.contour(P[:, :, mid], levels=[0.5], colors="k", linewidths=1.2)
    ax.set_title(f"$\\tau={t}$")
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
plt.suptitle(f"P slice mid-$u_3$ along ladder", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_contours.png", dpi=140); plt.close()

print(f"figs in {OUT}: {sorted(os.listdir(OUT))}")
print(f"cells: {len(taus)}, F median: {np.median(Fs):.2e}, mono viol max: {violations.max()}")
