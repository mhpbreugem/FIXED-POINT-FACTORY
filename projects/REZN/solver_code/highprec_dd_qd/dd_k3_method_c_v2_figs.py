"""Figures for Method C v2 ladder."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/dd_k3_method_c_v2_figs"; os.makedirs(OUT, exist_ok=True)
d = json.load(open("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/method_c_v2/ladder.json"))
taus = sorted(float(k) for k in d.keys())
Fs = np.array([d[f"{t:.4f}"]["F"] for t in taus])
viols = np.array([d[f"{t:.4f}"]["viol"] for t in taus])
walls = np.array([d[f"{t:.4f}"]["wall"] for t in taus])
P_mins = np.array([d[f"{t:.4f}"]["P_min"] for t in taus])
P_maxs = np.array([d[f"{t:.4f}"]["P_max"] for t in taus])
coefs_all = np.array([d[f"{t:.4f}"]["coefs"] for t in taus])
taus = np.array(taus)

# Fig 1: structural deficit F vs tau
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(taus, Fs, "-", lw=1.0, color="C0")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"residual $|F|_\infty$ (= structural non-additivity)")
ax.set_title(f"Method C v2 structural-deficit residual along $\\tau$ ladder ({len(taus)} cells)")
ax.grid(True, alpha=0.3, which="both")
ax.axhline(1e-10, color="k", lw=0.5, alpha=0.5, label="machine eps")
ax.axhline(1e-5, color="C2", lw=0.5, alpha=0.5)
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_F.png", dpi=140); plt.close()

# Fig 2: log-log to see scaling
fig, ax = plt.subplots(figsize=(11, 5))
ax.loglog(taus, np.maximum(Fs, 1e-20), "-", lw=1.0, color="C0")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$|F|_\infty$")
ax.set_title("Same on log-log: detect any power-law scaling in tau")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_loglog.png", dpi=140); plt.close()

# Fig 3: P range across tau
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(taus, P_mins, "-", lw=1.0, color="C0", label="$P_{\\min}$")
ax.plot(taus, P_maxs, "-", lw=1.0, color="C2", label="$P_{\\max}$")
ax.fill_between(taus, P_mins, P_maxs, alpha=0.2)
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("P values")
ax.set_title("P range under additive-monotone ansatz")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_Prange.png", dpi=140); plt.close()

# Fig 4: coefs evolution
fig, ax = plt.subplots(figsize=(11, 5))
for j in range(coefs_all.shape[1]):
    ax.plot(taus, coefs_all[:, j], "-", lw=0.8, label=f"$a_{j}$")
ax.set_xlabel(r"$\tau$"); ax.set_ylabel("$\\sigma$ coefs")
ax.set_title("$\\sigma$(s) Cheby coefs along ladder (even basis)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_coefs.png", dpi=140); plt.close()

# Fig 5: wall time + monotonicity
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(taus, walls, ".", ms=2, color="C3")
axs[0].set_xlabel(r"$\tau$"); axs[0].set_ylabel("wall (s)")
axs[0].set_title(f"Per-cell wall (mean={walls.mean():.2f}s, max={walls.max():.2f}s)")
axs[0].grid(True, alpha=0.3)
axs[1].plot(taus, viols, "-", lw=1.0, color="C3")
axs[1].set_xlabel(r"$\tau$"); axs[1].set_ylabel("# monotonicity violations")
axs[1].set_title("Monotonicity violations (should be 0 by construction)")
axs[1].grid(True, alpha=0.3); axs[1].axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_diag.png", dpi=140); plt.close()

print(f"cells: {len(taus)}")
print(f"F range: [{Fs.min():.3e}, {Fs.max():.3e}]; median {np.median(Fs):.3e}")
print(f"max violations: {viols.max()}")
